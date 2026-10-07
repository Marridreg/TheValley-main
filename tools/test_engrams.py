#!/usr/bin/env python3
"""Prove the engram dynamics. No API key, no network, no cost.

Feeds a synthetic event stream through engine/engrams.py and asserts the
behaviours the design promised: the lie outlives the pleasantries, a refusal
never decays into permission, the bridge survives twenty scenes, repetition
makes the mundane permanent, one bad night drags its neighbours into memory,
forgiveness only lands while the grudge is on the table, and nothing one
character holds is visible to another.

Then the web (engine/engram_web.py) as wired into the store: capture fires
at encode and its bumps land, recall spreads before the top-k cut and never
bumps a spread-in, and the web comes back identical from a save.

    python tools/test_engrams.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ["VALLEY_SAVES_DIR"] = tempfile.mkdtemp(prefix="valley_test_")
# The messages use arrows; a cp1252 console must not turn a pass into a crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.engrams import EngramStore, PRESETS  # noqa: E402
from engine.state import StateManager  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)
        print(f"  FAIL  {msg}")
    else:
        print(f"  ok    {msg}")


def fresh(preset: str = "normal") -> EngramStore:
    return EngramStore(SimpleNamespace(), PRESETS[preset])


def scenes(store: EngramStore, owner: str, n: int) -> None:
    for _ in range(n):
        store.state.scene_count += 1
        store.decay(owner)


# ── 1. the lie outlives the pleasantries ──
print("\n[1] negativity bias + eviction")
s = fresh()
lie = s.encode("alcina", "player", "episode", "lied about the ledger",
               valence=-0.8, arousal=0.7, belief_violation=0.3, appraisal_delta=0.4,
               scene_id="s1", tags=["deceit"], rung="walk_on")
for i in range(8):
    s.encode("alcina", "player", "episode", f"pleasant chat about the weather {i}",
             valence=0.3, appraisal_delta=0.2, scene_id=f"s{i+2}", rung="walk_on")
scenes(s, "alcina", 15)
items = s.items("alcina", "player")
check(any(e.id == lie.id for e in items), "lie survives walk_on cap of 6 against 8 pleasantries")
lie_now = s.get("alcina", "player", lie.id)
others = [e for e in items if e.id != lie.id]
check(all(lie_now.strength > e.strength for e in others), "lie is the strongest thing she holds")
check(len(items) <= 6, f"walk_on cap held ({len(items)} items)")

# ── 2. readings are asymmetric ──
print("\n[2] consent asymmetry")
s = fresh()
s.encode("alcina", "player", "reading", "invitation", valence=0.9, appraisal_delta=0.9, rung="principal")
s.encode("bela", "player", "reading", "invitation", valence=-0.9, appraisal_delta=0.9, rung="principal")
scenes(s, "alcina", 10)
scenes(s, "bela", 10)
yes = s.items("alcina", "player")[0]
no = s.items("bela", "player")[0]
check(yes.valence == 0.0, "an established yes drifts back to neutral over ~8 quiet scenes")
check(no.valence == -0.9, "a no does not move on quiet scenes")
s.encode("bela", "player", "reading", "invitation", valence=0.7, appraisal_delta=0.9, rung="principal")
check(s.items("bela", "player")[0].valence > 0, "a decisive turn the other way does move it")

# ── 3. the bridge survives twenty scenes; a weak one fades on schedule ──
print("\n[3] decay curve")
s = fresh()
bridge = s.encode("heisenberg", "player", "episode", "saved him at the bridge",
                  valence=0.9, arousal=0.9, belief_violation=0.5, appraisal_delta=0.9, rung="principal")
weak = s.encode("heisenberg", "player", "episode", "borrowed a lantern",
                valence=0.1, appraisal_delta=0.3, rung="principal")
fact = s.encode("heisenberg", "player", "fact", "is a soldier", source="told",
                appraisal_delta=0.2, rung="principal")
scenes(s, "heisenberg", 20)
b = s.get("heisenberg", "player", bridge.id)
w = s.get("heisenberg", "player", weak.id)
f = s.get("heisenberg", "player", fact.id)
check(b.strength > 0.9, f"strong episode barely moves in 20 scenes ({b.strength:.2f})")
check(0.5 < w.strength / weak.strength < 0.7, f"weak episode keeps 50-70% after 20 scenes ({w.strength/weak.strength:.2f})")
check(f.strength == PRESETS["normal"].fact_floor, "a told fact sits on the floor, never below")
scenes(s, "heisenberg", 200)
check(s.get("heisenberg", "player", fact.id).strength == PRESETS["normal"].fact_floor, "floor holds at 220 scenes")
check(b.valence > 0 and s.get("heisenberg", "player", bridge.id).valence < b.valence,
      "feeling fades even where the memory holds")

# ── 4. repetition makes the mundane permanent ──
print("\n[4] repetition")
s = fresh()
for i in range(6):
    s.state.scene_count += 1
    e = s.encode("duke", "player", "episode", "bought coffee at the cart", valence=0.2,
                 appraisal_delta=0.15, tags=["routine"], rung="named")
check(e.count == 6, "same canonical dedupes into one record with count 6")
before = e.strength
scenes(s, "duke", 30)
after = s.get("duke", "player", e.id).strength
check(after / before > 0.9, f"a six-times-repeated memory loses under 10% in 30 scenes ({after/before:.3f})")
check(before < 1.0, "strength saturates below 1")

# ── 5. capture ──
print("\n[5] tagging and capture")
s = fresh()
mundane = s.encode("donna", "player", "episode", "wore a green coat", valence=0.0,
                   appraisal_delta=0.1, scene_id="barn", rung="principal")
elsewhere = s.encode("donna", "player", "episode", "wore a blue coat", valence=0.0,
                     appraisal_delta=0.1, scene_id="market", rung="principal")
s.encode("donna", "player", "episode", "the barn burned", valence=-0.9, arousal=1.0,
         belief_violation=0.4, appraisal_delta=0.6, scene_id="barn", rung="principal")
green = s.get("donna", "player", mundane.id).strength
blue = s.get("donna", "player", elsewhere.id).strength
check(green > mundane.strength * 1.5, "the green coat is remembered because the barn burned")
check(blue > elsewhere.strength, "the blue coat is pulled too — same person, entity edge")
check(green > blue, "but the coat from the burning barn is pulled harder: scene edge stacks")
later = s.encode("donna", "player", "episode", "wore a red coat", valence=0.0,
                 appraisal_delta=0.1, scene_id="barn", rung="principal")
check(s.get("donna", "player", later.id).strength == later.strength,
      "capture fires at encode, not at scene end: a later mundane thing is not dragged")
s.end_scene("donna")
check(s.get("donna", "player", mundane.id).strength > mundane.strength * 1.5,
      "the green coat still holds through the scene break")

# ── 6. cap grows with interaction ──
print("\n[6] cap growth")
s = fresh()
check(s.cap("eugen", "player", rung="walk_on") == 6, "walk_on starts at 6")
for i in range(25):
    s.encode("eugen", "player", "episode", f"event {i}", appraisal_delta=0.3, rung="walk_on")
check(s.cap("eugen", "player", rung="walk_on") == 8, f"25 encodes grow it to 8 ({s.cap('eugen','player',rung='walk_on')})")
for i in range(200):
    s.encode("eugen", "player", "episode", f"event {i}", appraisal_delta=0.3, rung="walk_on")
check(s.cap("eugen", "player", rung="walk_on") == 12, "cap ceilings at 2× base")
check(s.cap("eugen", "player", rung="principal") == 40, "principal ceiling is 40")
check(len(s.web("eugen")) == len(s.items("eugen")), "the web drops what eviction drops")

# ── 7. reconsolidation is the repair path ──
print("\n[7] reconsolidation")
s = fresh()
lie = s.encode("alcina", "player", "episode", "lied about the ledger", valence=-0.8,
               belief_violation=0.3, appraisal_delta=0.5, tags=["deceit"], rung="principal")
s.state.scene_count += 1
# No recall: the big apology does nothing to the grudge.
s.encode("alcina", "player", "episode", "took a blade for her", valence=0.9,
         arousal=0.9, belief_violation=0.5, appraisal_delta=0.9, rung="principal")
check(s.get("alcina", "player", lie.id).valence == -0.8,
      "a grand gesture while the grudge is NOT on the table changes nothing")
# Recall first (she brings it up), then the costly thing in the window.
s.recall("alcina", {"player"})
s.encode("alcina", "player", "episode", "confessed the whole ledger", valence=0.8,
         belief_violation=0.4, appraisal_delta=0.8, rung="principal")
repaired = s.get("alcina", "player", lie.id)
check(repaired.valence > -0.2, f"while active, the lie's feeling is rewritten ({repaired.valence:.2f})")
check(repaired.strength > 0.5, "…but she still remembers it happened")

# ── 8. consolidation ──
print("\n[8] episodes become facts")
s = fresh()
for i in range(3):
    s.state.scene_count += 1
    s.encode("luiza", "player", "episode", f"arrived late to the {['hearth','field','church'][i]}",
             valence=-0.3, appraisal_delta=0.3, tags=["late"], rung="recurring")
promoted = s.consolidate("luiza")
check(len(promoted) == 1 and promoted[0].canonical == "tends to: late", "three late episodes → one inferred fact")
check(promoted[0].source == "inferred" and promoted[0].confidence == 0.5, "promoted fact is inferred, low confidence")
r = s.render("luiza", "player")
check(any(l.startswith("suspects") for l in r["knows"]), "low-confidence inferred fact renders hedged")
check(s.consolidate("luiza") == [], "consolidating again does not duplicate")

# ── 9. realism is harsher ──
print("\n[9] realism preset")
n, r = fresh("normal"), fresh("realism")
for st in (n, r):
    st.encode("miranda", "player", "episode", "defied her", valence=-0.9, belief_violation=0.3,
              appraisal_delta=0.3, rung="principal")
check(r.items("miranda", "player")[0].strength > n.items("miranda", "player")[0].strength,
      "realism encodes the slight harder")
scenes(n, "miranda", 10)
scenes(r, "miranda", 10)
check(r.items("miranda", "player")[0].valence < n.items("miranda", "player")[0].valence,
      "realism holds the feeling longer")
check(r.realism and not n.realism, "the mode reaches the web's spread")

# ── 10. rumination trait ──
print("\n[10] affect_fade trait")
s = fresh()
s.encode("donna", "player", "episode", "mocked angie", valence=-0.9, appraisal_delta=0.5, rung="principal")
s.encode("duke", "player", "episode", "mocked the wares", valence=-0.9, appraisal_delta=0.5, rung="principal")
for _ in range(10):
    s.state.scene_count += 1
    s.decay("donna", affect_fade=0.3)
    s.decay("duke", affect_fade=1.0)
check(s.items("donna", "player")[0].valence < s.items("duke", "player")[0].valence,
      "a character who can't let it go keeps the feeling")

# ── 11. the wall, and the save round-trip ──
print("\n[11] partition and persistence")
s = fresh()
s.encode("alcina", "player", "fact", "is named Lukas", source="told", rung="principal")
check(s.items("bela", "player") == [], "what Alcina was told, Bela does not hold")
blob = json.dumps(s.state.engrams)
s2 = EngramStore(SimpleNamespace(engrams=json.loads(blob), scene_count=s.state.scene_count))
check(s2.items("alcina", "player")[0].canonical == "is named Lukas", "engrams survive a JSON round-trip")
rendered = s2.render("alcina", "player")
check("truth" not in json.dumps(rendered) and "salience" not in json.dumps(rendered),
      "render carries no GM-only fields")

# ── 12. mood-congruent recall ──
print("\n[12] mood-congruent recall")
s = fresh()
# Equal strength on purpose: the negativity multiplier lives upstream and
# would otherwise decide this on its own. Here only mood decides.
s.encode("heisenberg", "player", "episode", "shared a drink", valence=0.6, appraisal_delta=0.65, rung="principal")
s.encode("heisenberg", "player", "episode", "broke his tool", valence=-0.6, appraisal_delta=0.5, rung="principal")
top_sour = s.recall("heisenberg", {"player"}, mood=-0.8, n=1)[0]
top_warm = s.recall("heisenberg", {"player"}, mood=0.8, n=1)[0]
check(top_sour.valence < 0 < top_warm.valence, "a sour mood surfaces the sour memory first, and vice versa")

# ── 13. the web: capture at encode, spread at recall ──
print("\n[13] associative web")
s = fresh()
s.state.turn_count = 3
candle = s.encode("bela", "chapel", "episode", "a candle guttered", appraisal_delta=0.4,
                  scene_id="chapel", tags=["chapel"], rung="principal")
check(candle.turn_first == 3 and candle.turn_last == 3, "encode stamps the turn clock for the chain edge")
s.state.turn_count = 4
prayed = s.encode("bela", "anton", "episode", "anton prayed", appraisal_delta=0.4,
                  scene_id="chapel", tags=["anton"], rung="principal")
s.encode("bela", "luiza", "episode", "luiza ground flour", appraisal_delta=0.4,
         scene_id="mill", tags=["mill"], rung="principal")
check(len(s.web("bela")) == 3, "the web indexes every record the ledger holds")

# capture crosses entities: a fire in the chapel sharpens the memory of who was there
anton_quiet = s.get("bela", "anton", prayed.id).strength
s.state.turn_count = 5
s.encode("bela", "chapel", "episode", "the altar cloth caught fire", valence=-0.8, arousal=1.0,
         belief_violation=0.4, appraisal_delta=0.6, scene_id="chapel", tags=["chapel", "fire"],
         rung="principal")
check(s.get("bela", "anton", prayed.id).strength > anton_quiet,
      "capture reaches a different entity through the scene edge")
check(s.items("bela", "luiza")[0].strength == 0.4, "and leaves the mill alone")

# spread: a cue on the chapel surfaces anton, who was there, though nothing cued him
anton_before = s.get("bela", "anton", prayed.id).strength
candle_before = s.get("bela", "chapel", candle.id).strength
hits = s.recall("bela", {"chapel"}, n=5)
ids = [e.id for e in hits]
check(prayed.id in ids, "recall surfaces anton via spread with no cue on anton")
check(ids[0] != prayed.id, "but the direct hits still come first")
check(s.get("bela", "anton", prayed.id).strength == anton_before, "a spread-in gets no recall bump")
check(s.get("bela", "chapel", candle.id).strength > candle_before, "a direct hit does")
check(s.get("bela", "anton", prayed.id).scene_recalled == s.state.scene_count,
      "a spread-in is still stamped ACTIVE — it was brought to mind")
check(s.last_trace is not None and "seed" in s.last_trace.as_text(),
      "the recall trace is kept for the debug panel")
scoped = s.recall("bela", {"chapel"}, about="chapel")
check(scoped and all(e.about == "chapel" for e in scoped), "an about-scoped recall stays scoped")

# a reading seeds the spread but is never itself returned
s = fresh()
debt = s.encode("bela", "anton", "episode", "anton lied about the debt", valence=-0.5,
                appraisal_delta=0.4, rung="principal")
s.encode("bela", "anton", "reading", "hostile", valence=-0.6, appraisal_delta=0.9,
         tags=["mask"], rung="principal")
hits = s.recall("bela", {"mask"}, kinds=("episode",))
check([e.id for e in hits] == [debt.id],
      "a cue that touches only the reading surfaces the episodes behind it")
check(all(e.kind != "reading" for e in hits), "and readings themselves are never recalled")

# ── 14. the web is rebuilt from the ledger at load, never saved ──
print("\n[14] rebuilt at load")
st = StateManager(ROOT / "data")
st.mode = "realism"
store = EngramStore(st)
st.turn_count = 7
store.encode("bela", "chapel", "episode", "a candle guttered", appraisal_delta=0.4,
             scene_id="chapel", tags=["chapel"])
store.encode("bela", "anton", "episode", "anton prayed", appraisal_delta=0.4,
             scene_id="chapel", tags=["anton"])
store.encode("bela", "chapel", "episode", "the altar cloth caught fire", valence=-0.8, arousal=1.0,
             belief_violation=0.4, appraisal_delta=0.6, scene_id="chapel", tags=["chapel", "fire"])
first = [e.id for e in store.recall("bela", {"chapel"})]
path = st.save("webtest")
raw = json.loads(path.read_text(encoding="utf-8"))
check("engrams" in raw and "web" not in raw and "_webs" not in raw,
      "the save carries the ledger and nothing derived")
st2 = StateManager(ROOT / "data")
st2.load("webtest")
store2 = EngramStore(st2)
check(st2.mode == "realism" and store2.realism, "mode comes back with the save and retunes the store")
check(len(store2.web("bela")) == len(store.web("bela")), "the web is rebuilt from the loaded ledger")
check(store2.web("bela").neighbours(first[0]) == store.web("bela").neighbours(first[0]),
      "with the same edges")
check([e.id for e in store2.recall("bela", {"chapel"})] == [e.id for e in store.recall("bela", {"chapel"})],
      "and recalls identically")
st2.load("webtest")           # the /load path: ledger swapped under the store
store2.rebuild_webs()
check(len(store2.web("bela")) == 3, "rebuild_webs re-derives after a load")

print()
if failures:
    print(f"{len(failures)} FAILED")
    sys.exit(1)
print("all engram checks passed")
