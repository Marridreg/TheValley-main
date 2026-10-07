"""Associative web over a character's engrams. GM-side only.

Memories are stored flat in the per-owner ledger (engine/engrams.py). This
module gives that ledger its *edges* — derived, never stored — and runs one
hop of spreading activation over them at recall time, so that asking a
character about the chapel surfaces the thing that happened in the chapel,
which surfaces the person who was there, without a model deciding any of it.

Edges between two engrams of the same owner. Several links between the
same pair combine noisy-OR style, w = 1 - prod(1 - w_k), so weak links stack
without ever saturating at 1 and flattening the ranking (a plain sum capped
at 1 did exactly that — entity alone hit the cap and chain could add nothing):

    entity   same `about`                                    0.8
    scene    same `scene_id`                                 0.5
    tag      per shared tag, capped                          0.3 / 0.6
    source   same `told_by`                                  0.2
    chain    `turn_first` within N turns, same scene         0.4

Edges never cross owners: a character cannot be reminded of someone else's
memory. `kind` never makes an edge — a reading beside an episode about the
same person is exactly the association wanted.

Recall seeds from the caller's base activations (strength × cue × recency
+ mood, computed in engrams.py), spreads once from the top seeds along the
edges, scaled by the owner's `associativity` trait, and returns the final
ranking plus a trace for the debug panel. Readings may *seed* spread (at a
reduced weight) but are never *targets*: they are fixed render slots, so
activation flowing into one changes nothing, and that is what keeps the
hostile-reading → hostile-episodes → hostile-reading loop from closing.

Every magic number lives in TUNING and nowhere else. `/tune` edits a copy
carried in the save; config sets defaults for a new game.

Nothing in this module may be imported by narrator.py.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

# ── tunables ──────────────────────────────────────────────────────────────
# Direction is named for every scalar: "higher = more X".
TUNING: dict[str, float | int | bool] = {
    # edge weights; higher = stronger association
    "w_entity": 0.8,
    "w_scene": 0.5,
    "w_tag": 0.3,
    "w_tag_cap": 0.6,
    "w_source": 0.2,
    "w_chain": 0.4,
    "chain_window": 2,          # turns; higher = looser sense of "right after"
    "chain_enabled": True,
    # spread
    "seed_count": 3,            # higher = more seeds spread
    "recall_floor": 0.05,       # higher = fewer memories eligible to seed
    "associativity_default": 0.5,   # per-owner trait fallback; higher = more tangent-prone
    "reading_seed_weight": 0.33,    # higher = readings pull harder on episodes
    "realism_negative_spread": 1.3, # multiplier on negative seeds in realism mode
    "neighbour_cap": 5,         # higher = wider fan-out per seed
    # capture (encode-time): a high-salience engram bumps its neighbours
    "capture_threshold": 0.6,   # higher = fewer events capture
    "capture_gain": 0.2,        # higher = stronger bump
}


@dataclass
class RecallTrace:
    """What the web did on one recall. Shown in the F5 panel; never rendered."""

    owner: str
    seeds: list[tuple[str, float]] = field(default_factory=list)
    spread: dict[str, list[tuple[str, float, float]]] = field(default_factory=dict)
    final: list[tuple[str, float]] = field(default_factory=list)

    def as_text(self) -> str:
        lines = [f"recall web for {self.owner}"]
        for eid, a in self.seeds:
            lines.append(f"  seed {eid} A0={a:.3f}")
            for nid, w, add in self.spread.get(eid, []):
                lines.append(f"       -> {nid} w={w:.2f} +{add:.3f}")
        lines.append("  final:")
        for eid, a in self.final:
            lines.append(f"    {eid} A={a:.3f}")
        return "\n".join(lines)


class EngramWeb:
    """Inverted indexes over one owner's ledger plus one-hop spread.

    Built incrementally at encode (add) and rebuilt from the ledger on load
    (rebuild). Never persisted: saves carry the ledger, indexes are derived.
    """

    def __init__(self, owner: str, tuning: dict | None = None):
        self.owner = owner
        self.tuning = dict(TUNING)
        if tuning:
            self.tuning.update(tuning)
        self._records: dict[str, dict] = {}
        self._by_about: dict[str, set[str]] = defaultdict(set)
        self._by_scene: dict[str, set[str]] = defaultdict(set)
        self._by_tag: dict[str, set[str]] = defaultdict(set)
        self._by_source: dict[str, set[str]] = defaultdict(set)

    # ── indexing ──

    def rebuild(self, records: list[dict]) -> None:
        self._records.clear()
        for idx in (self._by_about, self._by_scene, self._by_tag, self._by_source):
            idx.clear()
        for rec in records:
            self.add(rec)

    def add(self, rec: dict) -> None:
        if rec.get("owner") != self.owner:
            raise ValueError(f"engram {rec.get('id')} belongs to {rec.get('owner')}, not {self.owner}")
        eid = rec["id"]
        self._records[eid] = rec
        if rec.get("about"):
            self._by_about[rec["about"]].add(eid)
        if rec.get("scene_id"):
            self._by_scene[rec["scene_id"]].add(eid)
        for tag in rec.get("tags") or ():
            self._by_tag[tag].add(eid)
        if rec.get("told_by"):
            self._by_source[rec["told_by"]].add(eid)

    def remove(self, eid: str) -> None:
        rec = self._records.pop(eid, None)
        if not rec:
            return
        self._by_about.get(rec.get("about"), set()).discard(eid)
        self._by_scene.get(rec.get("scene_id"), set()).discard(eid)
        for tag in rec.get("tags") or ():
            self._by_tag.get(tag, set()).discard(eid)
        self._by_source.get(rec.get("told_by"), set()).discard(eid)

    def __len__(self) -> int:
        return len(self._records)

    # ── edges ──

    def weight(self, a: dict, b: dict) -> float:
        """Combined edge weight between two records in [0, 1). 0 = no edge.

        Links combine noisy-OR: w = 1 - prod(1 - w_k). Each extra link raises
        the weight by a share of the remaining headroom, so a pair joined by
        entity+scene+chain still ranks above a pair joined by entity alone.
        """
        t = self.tuning
        links: list[float] = []
        if a.get("about") and a.get("about") == b.get("about"):
            links.append(t["w_entity"])
        same_scene = bool(a.get("scene_id")) and a.get("scene_id") == b.get("scene_id")
        if same_scene:
            links.append(t["w_scene"])
        shared = set(a.get("tags") or ()) & set(b.get("tags") or ())
        if shared:
            links.append(min(t["w_tag"] * len(shared), t["w_tag_cap"]))
        if a.get("told_by") and a.get("told_by") == b.get("told_by"):
            links.append(t["w_source"])
        if t["chain_enabled"] and same_scene:
            ta, tb = a.get("turn_first"), b.get("turn_first")
            if ta is not None and tb is not None and 0 < abs(int(ta) - int(tb)) <= int(t["chain_window"]):
                links.append(t["w_chain"])
        if not links:
            return 0.0
        miss = 1.0
        for w in links:
            miss *= 1.0 - min(max(float(w), 0.0), 0.999)
        return 1.0 - miss

    def neighbours(self, eid: str) -> list[tuple[str, float]]:
        """Candidate neighbours via the indexes, weighted, strongest first.

        Readings are excluded as targets: they are fixed slots and spreading
        into them does nothing but open a feedback loop.
        """
        rec = self._records.get(eid)
        if not rec:
            return []
        cands: set[str] = set()
        if rec.get("about"):
            cands |= self._by_about[rec["about"]]
        if rec.get("scene_id"):
            cands |= self._by_scene[rec["scene_id"]]
        for tag in rec.get("tags") or ():
            cands |= self._by_tag[tag]
        if rec.get("told_by"):
            cands |= self._by_source[rec["told_by"]]
        cands.discard(eid)
        out = []
        for nid in cands:
            other = self._records[nid]
            if other.get("kind") == "reading":
                continue
            w = self.weight(rec, other)
            if w > 0:
                out.append((nid, w))
        out.sort(key=lambda p: (-p[1], p[0]))
        return out

    def degree(self, eid: str) -> float:
        """Weighted degree. High degree + repeated tag = consolidation candidate;
        low degree + low strength = eviction candidate."""
        return sum(w for _, w in self.neighbours(eid))

    # ── spread ──

    def spread(
        self,
        base: dict[str, float],
        *,
        associativity: float | None = None,
        realism: bool = False,
    ) -> tuple[dict[str, float], RecallTrace]:
        """One hop of spreading activation.

        `base` maps engram id -> A0 (strength × cue × recency + mood), computed
        by engrams.py. Returns final activations for every id in `base` plus
        any neighbour the spread reached, and a trace.
        """
        t = self.tuning
        assoc = t["associativity_default"] if associativity is None else float(associativity)
        trace = RecallTrace(owner=self.owner)
        final: dict[str, float] = dict(base)

        eligible = [(eid, a) for eid, a in base.items()
                    if a >= t["recall_floor"] and eid in self._records]
        eligible.sort(key=lambda p: (-p[1], p[0]))
        seeds = eligible[: int(t["seed_count"])]
        trace.seeds = list(seeds)

        for eid, a0 in seeds:
            rec = self._records[eid]
            gain = a0 * assoc
            if rec.get("kind") == "reading":
                gain *= t["reading_seed_weight"]
            if realism and (rec.get("valence") or 0) < 0:
                gain *= t["realism_negative_spread"]
            hops = []
            for nid, w in self.neighbours(eid)[: int(t["neighbour_cap"])]:
                add = gain * w
                final[nid] = final.get(nid, 0.0) + add
                hops.append((nid, w, add))
            trace.spread[eid] = hops

        trace.final = sorted(final.items(), key=lambda p: (-p[1], p[0]))
        return final, trace

    def spread_ins(self, base: dict[str, float], final: dict[str, float]) -> set[str]:
        """Ids whose activation came only from spread. Direct hits get the
        recall strength bump in engrams.py; these must not, or a chatty
        neighbour self-reinforces forever."""
        return {eid for eid in final if base.get(eid, 0.0) < self.tuning["recall_floor"]}

    # ── capture ──

    def capture(self, eid: str) -> list[tuple[str, float]]:
        """Encode-time: if this engram's salience clears the threshold, return
        (neighbour_id, strength_delta) bumps for engrams.py to apply. Same-scene
        capture falls out of the scene edge; entity capture comes free."""
        rec = self._records.get(eid)
        if not rec:
            return []
        sal = float(rec.get("salience") or 0.0)
        if sal < self.tuning["capture_threshold"]:
            return []
        gain = sal * self.tuning["capture_gain"]
        return [(nid, gain * w) for nid, w in self.neighbours(eid)]
