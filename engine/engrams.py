"""Engrams: what a character remembers about an entity. GM-side only.

One record shape carries three kinds:

    fact      durable statement           "is a soldier"
    episode   a thing that happened       "saved me at the bridge"
    reading   inferred current state      invitation, threat — decays to neutral

The store is partitioned by owner — a character's engrams are that
character's perception and nothing else — so the information wall holds
without anyone checking it. Nothing in this module may be imported by
narrator.py; engrams reach the narrator only as the GM voices them.

The dynamics are deliberately dumb and deterministic. No model call touches
strength, valence, or eviction. The model supplies candidates (encoder) and
prose (renderer); everything between is arithmetic, which is what makes it
testable with a synthetic event stream and no API key (tools/test_engrams.py).

Design notes the numbers encode:

  * Salience at encoding = belief violation + |appraisal delta|, with a
    negativity multiplier: the bad thing enters at higher strength and
    outlives the pleasantries.
  * Strength and valence decay SEPARATELY. The memory of what happened
    persists; the feeling attached fades, and negative feeling fades faster
    than positive (fading affect bias). Per-character `affect_fade` scales
    that: below 1.0 is someone who can't let it go.
  * Decay is strength- and count-scaled — strong memories decay slowly,
    repeated ones barely at all. Facts have a floor; episodes go to zero.
  * Readings are asymmetric. A positive reading drifts back to neutral on
    quiet scenes; a negative one does not move except on a decisive event.
    Silence never becomes permission.
  * Capture: one high-salience event drags its neighbours up, at encode.
    Neighbours and weights come from the associative web
    (engine/engram_web.py): same scene, same entity, shared tags, same
    source, "right after" — so the bump is graded, not a flat scene sweep.
  * Recall spreads one hop over that web before the top-k cut, so a cue on
    the chapel surfaces what happened there and who was there. Spread-ins
    are never strength-bumped on recall; only direct hits are.
  * Reconsolidation is the repair path: a high-salience event landing while
    an engram is ACTIVE (recently recalled) rewrites that engram's valence.
    Forgiveness is earned in the moment the grudge is on the table.

The web is derived from the ledger and never saved: EngramStore builds it
from state.engrams at construction, keeps it in step on every write, and
rebuild_webs() re-derives it after anything that swaps the ledger wholesale.

Beliefs are NOT written here. The consolidator returns fact candidates; the
caller promotes them to the belief ledger under the card's mutability rules.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, asdict

from .engram_web import EngramWeb, RecallTrace

KINDS = ("fact", "episode", "reading")
SOURCES = ("told", "observed", "inferred")
TRUTHS = ("true", "false", "unknown")

# Default confidence by source. Told by the subject themselves is 1.0; the
# engine trusts the tell, whether or not the tell was honest (that is what
# `truth` is for, and truth never renders).
CONFIDENCE = {"told": 1.0, "observed": 0.9, "inferred": 0.5}

# Tier caps: how many engrams a character holds about ONE entity. The cap is
# per (owner, about), so a principal knows 20 things about the player and a
# separate 20 about each other NPC — knowing someone well is local.
CAP_BY_RUNG = {"principal": 20, "recurring": 12, "named": 12, "walk_on": 6}
DEFAULT_RUNG = "recurring"


@dataclass
class EngramConfig:
    """Every scalar names its direction. Presets at the bottom of the file."""
    d_base: float = 0.035          # per-scene decay of strength; higher = forgets faster
    fact_floor: float = 0.3        # facts never decay below this
    negativity: float = 1.3        # salience multiplier when valence < 0; >1 = bad things stick
    affect_fade_pos: float = 0.03  # per-scene fade of positive valence toward 0
    affect_fade_neg: float = 0.06  # per-scene fade of negative valence toward 0 (faster: fading affect bias)
    reading_fade: float = 0.12     # per-scene drift of a POSITIVE reading toward neutral (~8 scenes)
    # Capture (the encode-time neighbour bump) is tuned in engram_web.TUNING:
    # capture_threshold and capture_gain live there, beside the edge weights.
    reconsolidation_bar: float = 0.6  # salience at/above this rewrites ACTIVE engrams' valence
    recall_bump: float = 0.1       # strength added on recall
    mood_bonus: float = 0.15       # activation bonus for valence matching current mood
    recency_half_life: float = 12  # scenes; recency term halves every this many
    cap_growth_per: int = 10       # +1 cap per this many engrams ever encoded about the entity
    cap_growth_max: float = 2.0    # cap never exceeds base × this
    consolidate_min_count: int = 3 # repeated same-tag episodes needed to become a fact
    consolidate_neg_weight: float = 1.5  # negative episodes count this much toward promotion
    reconsolidation_window: int = 2      # scenes since recall for an engram to count as ACTIVE
    reconsolidation_gain: float = 0.6    # how far an active engram's valence moves toward the new event


MODES = ("normal", "realism")

PRESETS = {
    "normal": EngramConfig(),
    "realism": EngramConfig(
        negativity=1.5,
        affect_fade_neg=0.04,
        reconsolidation_window=1,
        reconsolidation_gain=0.4,
        consolidate_neg_weight=2.0,
    ),
}


def engram_id(owner: str, about: str, kind: str, canonical: str) -> str:
    key = f"{owner}|{about}|{kind}|{canonical.strip().lower()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


@dataclass
class Engram:
    id: str
    owner: str
    about: str
    kind: str
    canonical: str
    source: str
    told_by: str | None = None
    confidence: float = 0.5
    truth: str = "unknown"      # GM-only. Never rendered.
    valence: float = 0.0        # -1..1, feeling attached NOW (decays separately from strength)
    arousal: float = 0.0        # 0..1 at encoding
    salience: float = 0.0       # energy at encoding; read only by capture and reconsolidation
    strength: float = 0.0       # the live number; retrieval and eviction read this and nothing else
    count: int = 1
    scene_id: str = ""
    scene_first: int = 0
    scene_last: int = 0
    scene_recalled: int = -10_000
    turn_first: int = 0         # turn clock; the web's chain edge reads this ("right after")
    turn_last: int = 0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Engram":
        return Engram(**{k: d[k] for k in Engram.__dataclass_fields__ if k in d})


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class EngramStore:
    """Owns state.engrams: {owner: {about: {"n_encoded": int, "items": [dict]}}}.

    Scene clock: the caller advances `state.scene_count` on a scene break; all
    decay runs per scene, not per turn, because turn length varies wildly and
    a scene is the natural unit of "time passed".
    """

    def __init__(self, state, config: EngramConfig | None = None,
                 web_tuning: dict | None = None):
        self.state = state
        # A pinned config (tests, experiments) wins. Otherwise the store follows
        # state.mode, so /load of a realism save retunes without a rebuild and
        # a save never quietly changes mode under the player.
        self._pinned = config
        if not hasattr(state, "engrams"):
            state.engrams = {}
        if not hasattr(state, "scene_count"):
            state.scene_count = 0
        # The associative web (engine/engram_web.py), one per owner. Derived
        # from the ledger, never persisted: built here from whatever the
        # state holds, kept in step by _write, re-derived by rebuild_webs().
        self._web_tuning = dict(web_tuning or {})
        self._webs: dict[str, EngramWeb] = {}
        # What the web did on the most recent recall. For the debug panel.
        self.last_trace: RecallTrace | None = None
        self.rebuild_webs()

    @property
    def cfg(self) -> EngramConfig:
        if self._pinned is not None:
            return self._pinned
        return PRESETS.get(getattr(self.state, "mode", "normal"), PRESETS["normal"])

    # ── the web ──

    def _web(self, owner: str) -> EngramWeb:
        web = self._webs.get(owner)
        if web is None:
            web = EngramWeb(owner, self._web_tuning)
            per_owner = self.state.engrams.get(owner) or {}
            web.rebuild([d for b in per_owner.values() for d in b.get("items", [])])
            self._webs[owner] = web
        return web

    def web(self, owner: str) -> EngramWeb:
        """Read access, for the debug panel and tests."""
        return self._web(owner)

    def rebuild_webs(self) -> None:
        """Derive every owner's web from the ledger. Runs at construction and
        after anything that replaces state.engrams wholesale (/load, /new)."""
        self._webs.clear()
        for owner in list(self.state.engrams or {}):
            self._web(owner)

    @property
    def realism(self) -> bool:
        return self.cfg is PRESETS["realism"]

    def associativity(self, owner: str) -> float | None:
        """Per-character tangent-proneness, private card `associativity`.
        None lets the web fall back to its own default."""
        card_fn = getattr(self.state, "_card", None)
        if card_fn is None:
            return None
        try:
            raw = (card_fn(owner).get("private") or {}).get("associativity")
        except Exception:
            return None
        return None if raw is None else float(raw)

    # ── plumbing ──

    def _bucket(self, owner: str, about: str) -> dict:
        per_owner = self.state.engrams.setdefault(owner, {})
        return per_owner.setdefault(about, {"n_encoded": 0, "items": []})

    def items(self, owner: str, about: str | None = None) -> list[Engram]:
        per_owner = self.state.engrams.get(owner) or {}
        abouts = [about] if about else list(per_owner)
        out: list[Engram] = []
        for a in abouts:
            for d in (per_owner.get(a) or {}).get("items", []):
                out.append(Engram.from_dict(d))
        return out

    def _write(self, owner: str, about: str, engrams: list[Engram]) -> None:
        """The one write path. The web indexes the ledger's own dicts, so it
        is re-pointed here: everything the bucket held comes out, everything
        it holds now goes in. Buckets are small (the cap), so this is cheap."""
        bucket = self._bucket(owner, about)
        web = self._web(owner)
        for d in bucket["items"]:
            web.remove(d["id"])
        bucket["items"] = [e.to_dict() for e in engrams]
        for d in bucket["items"]:
            web.add(d)

    def get(self, owner: str, about: str, eid: str) -> Engram | None:
        for e in self.items(owner, about):
            if e.id == eid:
                return e
        return None

    # ── caps ──

    def rung(self, owner: str) -> str:
        """meta.rung from the card when present; recurring otherwise."""
        card_fn = getattr(self.state, "_card", None)
        if card_fn is None:
            return DEFAULT_RUNG
        try:
            card = card_fn(owner)
        except Exception:
            return DEFAULT_RUNG
        for side in ("public", "private"):
            meta = (card.get(side) or {}).get("meta") or {}
            if meta.get("rung") in CAP_BY_RUNG:
                return meta["rung"]
        return DEFAULT_RUNG

    def cap(self, owner: str, about: str, rung: str | None = None) -> int:
        base = CAP_BY_RUNG.get(rung or self.rung(owner), CAP_BY_RUNG[DEFAULT_RUNG])
        grown = base + self._bucket(owner, about)["n_encoded"] // self.cfg.cap_growth_per
        return int(min(grown, math.floor(base * self.cfg.cap_growth_max)))

    # ── encode ──

    def encode(
        self,
        owner: str,
        about: str,
        kind: str,
        canonical: str,
        *,
        source: str = "observed",
        told_by: str | None = None,
        confidence: float | None = None,
        truth: str = "unknown",
        valence: float = 0.0,
        arousal: float = 0.0,
        belief_violation: float = 0.0,
        appraisal_delta: float = 0.0,
        scene_id: str = "",
        tags: list[str] | None = None,
        rung: str | None = None,
    ) -> Engram:
        """Write or reinforce one engram. Returns the stored record.

        Salience is computed here from the two energy sources; callers pass
        the raw signals, never a salience number, so the multiplier policy
        lives in exactly one place.
        """
        assert kind in KINDS, kind
        assert source in SOURCES, source
        assert truth in TRUTHS, truth
        cfg = self.cfg
        scene_n = self.state.scene_count
        turn_n = int(getattr(self.state, "turn_count", 0) or 0)

        salience = _clamp(belief_violation + abs(appraisal_delta))
        if valence < 0:
            salience = _clamp(salience * cfg.negativity)

        eid = engram_id(owner, about, kind, canonical)
        engrams = self.items(owner, about)
        existing = next((e for e in engrams if e.id == eid), None)

        if existing is not None:
            existing.count += 1
            existing.strength = _clamp(existing.strength + salience * (1 - existing.strength))
            existing.salience = max(existing.salience, salience)
            if kind == "reading":
                # A reading is a read of NOW. One decisive turn lands the band;
                # it does not average against last week's read.
                existing.valence = _clamp(valence, -1, 1)
            else:
                # Blend feeling toward the new event; a repeated pleasant thing
                # warms, a repeated slight sours.
                existing.valence = _clamp(0.5 * existing.valence + 0.5 * valence, -1, 1)
            existing.arousal = max(existing.arousal, arousal)
            existing.scene_last = scene_n
            existing.turn_last = turn_n
            existing.scene_id = scene_id or existing.scene_id
            for t in tags or []:
                if t not in existing.tags:
                    existing.tags.append(t)
            # Confidence only ever rises on repetition from a better source.
            new_conf = CONFIDENCE[source] if confidence is None else confidence
            existing.confidence = max(existing.confidence, new_conf)
            stored = existing
        else:
            stored = Engram(
                id=eid, owner=owner, about=about, kind=kind, canonical=canonical.strip(),
                source=source, told_by=told_by,
                confidence=CONFIDENCE[source] if confidence is None else confidence,
                truth=truth, valence=_clamp(valence, -1, 1), arousal=_clamp(arousal),
                salience=salience,
                # A reading's strength is its certainty of the read; facts and
                # episodes enter at their salience.
                strength=salience if kind != "reading" else max(salience, 0.5),
                count=1, scene_id=scene_id, scene_first=scene_n, scene_last=scene_n,
                turn_first=turn_n, turn_last=turn_n,
                tags=list(tags or []),
            )
            engrams.append(stored)

        self._bucket(owner, about)["n_encoded"] += 1
        self._reconsolidate(engrams, stored)
        self._write(owner, about, engrams)
        self._capture(owner, eid)
        self.evict(owner, about, rung=rung)
        return self.get(owner, about, eid) or stored

    def _reconsolidate(self, engrams: list[Engram], new: Engram) -> None:
        """A high-salience event rewrites whatever was ACTIVE about this entity.

        Active = recalled within the window. This is the only way a strong
        negative engram's feeling moves by more than slow fade: she brought
        it up, you did the costly thing, the record re-encodes.
        """
        cfg = self.cfg
        if new.salience < cfg.reconsolidation_bar:
            return
        scene_n = self.state.scene_count
        for e in engrams:
            if e.id == new.id or e.kind == "reading":
                continue
            if scene_n - e.scene_recalled <= cfg.reconsolidation_window:
                e.valence = _clamp(
                    e.valence + (new.valence - e.valence) * cfg.reconsolidation_gain, -1, 1
                )

    # ── capture ──

    def _capture(self, owner: str, eid: str) -> list[tuple[str, float]]:
        """Encode-time. The web decides whether this engram's salience clears
        its threshold and how hard each neighbour is pulled — graded by edge
        weight, so the same scene, the same person, a shared tag all count
        and the strongest link counts most. The store only applies the
        bumps, with the usual headroom scaling so nothing saturates. Bumps
        can land in other entities' buckets; each touched bucket is written
        once. Returns the bumps as the web returned them."""
        bumps = self._web(owner).capture(eid)
        if not bumps:
            return []
        wanted = dict(bumps)
        for about, bucket in list((self.state.engrams.get(owner) or {}).items()):
            if not any(d["id"] in wanted for d in bucket.get("items", [])):
                continue
            engrams = self.items(owner, about)
            for e in engrams:
                delta = wanted.get(e.id)
                if delta:
                    e.strength = _clamp(e.strength + delta * (1 - e.strength))
            self._write(owner, about, engrams)
        return bumps

    # ── decay ──

    def affect_fade(self, owner: str) -> float:
        """Per-character multiplier on valence fade. 1.0 default; <1 ruminates.
        Read from private card `affect_fade` when authored."""
        card_fn = getattr(self.state, "_card", None)
        if card_fn is None:
            return 1.0
        try:
            return float((card_fn(owner).get("private") or {}).get("affect_fade", 1.0))
        except Exception:
            return 1.0

    def decay(self, owner: str, affect_fade: float | None = None) -> None:
        """One scene of quiet, for everything this owner holds."""
        cfg = self.cfg
        fade = self.affect_fade(owner) if affect_fade is None else affect_fade
        for about in list(self.state.engrams.get(owner) or {}):
            engrams = self.items(owner, about)
            for e in engrams:
                if e.kind == "reading":
                    # Asymmetric: positive readings drift to neutral, negative hold.
                    if e.valence > 0:
                        e.valence = max(0.0, e.valence - cfg.reading_fade)
                    continue
                d = cfg.d_base * (1 - e.strength) / max(1, e.count)
                floor = cfg.fact_floor if e.kind == "fact" else 0.0
                e.strength = max(floor, e.strength * (1 - d))
                if e.valence > 0:
                    e.valence = max(0.0, e.valence - cfg.affect_fade_pos * fade)
                elif e.valence < 0:
                    e.valence = min(0.0, e.valence + cfg.affect_fade_neg * fade)
            self._write(owner, about, engrams)

    def end_scene(self, owner: str) -> None:
        """Convenience for the turn loop: one scene of quiet. Capture does
        not wait for the scene break any more — it fires at encode, through
        the web. Caller bumps state.scene_count once per scene break, for
        all owners together."""
        self.decay(owner)

    # ── recall ──

    def recall(
        self,
        owner: str,
        cues: set[str],
        *,
        mood: float = 0.0,
        n: int = 5,
        kinds: tuple[str, ...] = ("fact", "episode"),
        about: str | None = None,
    ) -> list[Engram]:
        """Top-n by activation after one hop of spread over the web.

        Base activation = strength × cue_match × recency + mood bonus, for
        every engram the cues touch directly. Cues are entity ids on the
        scene map plus fired tags. The web then spreads from the strongest
        seeds along its edges, and the top-n cut runs over the FINAL
        activations — so a memory nothing cued can surface because it sits
        next to one that was, and can outrank a weak direct hit.

        Readings are never returned (they render as standing lines) but they
        do seed the spread, at the web's reduced weight: a hostile read of
        someone pulls up the episodes behind it.

        Recall stamps scene_recalled on everything returned (which is what
        makes an engram ACTIVE for reconsolidation). Only direct hits get
        the strength bump; a spread-in bumped on every recall of its
        neighbour would self-reinforce forever.
        """
        cfg = self.cfg
        scene_n = self.state.scene_count
        web = self._web(owner)

        by_id: dict[str, Engram] = {e.id: e for e in self.items(owner)}
        base: dict[str, float] = {}
        for e in by_id.values():
            if e.kind != "reading" and e.kind not in kinds:
                continue
            if about is not None and e.about != about:
                continue
            cue_match = 1.0 if e.about in cues else 0.0
            cue_match = max(cue_match, 0.6 if any(t in cues for t in e.tags) else 0.0)
            if cue_match == 0.0:
                continue
            recency = 0.5 ** ((scene_n - e.scene_last) / cfg.recency_half_life)
            activation = e.strength * cue_match * (0.5 + 0.5 * recency)
            if mood != 0.0 and e.valence != 0.0 and (mood > 0) == (e.valence > 0):
                activation += cfg.mood_bonus * min(abs(mood), abs(e.valence))
            base[e.id] = activation

        final, trace = web.spread(
            base, associativity=self.associativity(owner), realism=self.realism
        )
        self.last_trace = trace
        spread_ins = web.spread_ins(base, final)

        # Spread can reach any kind and any entity; the caller's filters
        # still bound what comes back.
        chosen: list[Engram] = []
        for eid, _ in sorted(final.items(), key=lambda p: (-p[1], p[0])):
            e = by_id.get(eid)
            if e is None or e.kind not in kinds:
                continue
            if about is not None and e.about != about:
                continue
            chosen.append(e)
            if len(chosen) >= n:
                break

        # Stamp everything; bump direct hits only.
        touched: dict[str, list[Engram]] = {}
        for e in chosen:
            if e.id not in spread_ins:
                e.strength = _clamp(e.strength + cfg.recall_bump * (1 - e.strength))
            e.scene_recalled = scene_n
            touched.setdefault(e.about, []).append(e)
        for a, hits in touched.items():
            engrams = self.items(owner, a)
            hit_by_id = {e.id: e for e in hits}
            engrams = [hit_by_id.get(e.id, e) for e in engrams]
            self._write(owner, a, engrams)
        return chosen

    # ── evict ──

    def evict(self, owner: str, about: str, rung: str | None = None) -> list[Engram]:
        """Drop the weakest beyond the cap. Readings are fixed slots, outside it."""
        engrams = self.items(owner, about)
        keep = [e for e in engrams if e.kind == "reading"]
        ranked = sorted((e for e in engrams if e.kind != "reading"),
                        key=lambda e: e.strength, reverse=True)
        cap = self.cap(owner, about, rung=rung)
        kept, dropped = ranked[:cap], ranked[cap:]
        self._write(owner, about, keep + kept)
        return dropped

    # ── consolidate ──

    def consolidate(self, owner: str, about: str | None = None) -> list[Engram]:
        """Scene break / time skip. Repeated same-tag episodes about one entity
        become an inferred fact. Returns the facts written this pass so the
        caller can offer them to the belief ledger (gated by mutability there,
        not here). Negative episodes count more toward promotion.
        """
        cfg = self.cfg
        promoted: list[Engram] = []
        for a in ([about] if about else list(self.state.engrams.get(owner) or {})):
            engrams = self.items(owner, a)
            by_tag: dict[str, list[Engram]] = {}
            for e in engrams:
                if e.kind != "episode":
                    continue
                for t in e.tags:
                    by_tag.setdefault(t, []).append(e)
            for tag, eps in by_tag.items():
                weight = sum(
                    e.count * (cfg.consolidate_neg_weight if e.valence < 0 else 1.0) for e in eps
                )
                if weight < cfg.consolidate_min_count:
                    continue
                canonical = f"tends to: {tag}"
                if any(e.kind == "fact" and e.canonical == canonical for e in engrams):
                    continue
                val = sum(e.valence for e in eps) / len(eps)
                fact = self.encode(
                    owner, a, "fact", canonical, source="inferred",
                    valence=val, belief_violation=0.0,
                    appraisal_delta=min(1.0, weight / (cfg.consolidate_min_count * 2)),
                    tags=[tag],
                )
                promoted.append(fact)
        return promoted

    # ── render ──

    def render(self, owner: str, about: str, *, cues: set[str] | None = None,
               mood: float = 0.0, episodes: int = 3) -> dict:
        """The only prompt-facing surface. No salience, no truth, no numbers.
        Confidence under 0.6 renders hedged. Facts ordered by strength.
        """
        facts = sorted((e for e in self.items(owner, about) if e.kind == "fact"),
                       key=lambda e: e.strength, reverse=True)
        readings = [e for e in self.items(owner, about) if e.kind == "reading"]

        def line(e: Engram) -> str:
            text = e.canonical
            if e.source == "inferred" and e.confidence < 0.6:
                text = f"suspects, is not sure: {text}"
            elif e.source == "told" and e.told_by and e.told_by != e.about:
                text = f"heard from {e.told_by}: {text}"
            return text

        recalled = self.recall(owner, cues or {about}, mood=mood, n=episodes,
                               kinds=("episode",), about=about)
        return {
            "knows": [line(e) for e in facts],
            "remembers": [line(e) for e in recalled],
            "readings": {e.canonical: round(e.valence, 2) for e in readings},
        }
