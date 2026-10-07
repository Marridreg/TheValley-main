#!/usr/bin/env python3
"""Prove the engram web spreads the way the design says. No API key, no
network, no cost. Builds a synthetic ledger for one owner and asserts:

  - the chapel case: a cue on the chapel surfaces what happened there and
    the person who was there, with nothing about the mill
  - chain: the event two turns after a seed outranks an unrelated same-scene
    memory from ten turns later; off when chain_enabled is false
  - readings seed at reduced weight and are never targets
  - edges never cross owners
  - realism mode spreads negative seeds harder
  - capture bumps neighbours only above the threshold
  - degree separates consolidation candidates from eviction candidates
  - indexes rebuild identically from the ledger

    python tools/test_engram_web.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.engram_web import EngramWeb, TUNING  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def rec(eid, about, kind="episode", scene="s1", turn=1, tags=(), told_by=None,
        valence=0.0, salience=0.3, strength=0.5, owner="bela"):
    return {
        "id": eid, "owner": owner, "about": about, "kind": kind,
        "source": "told" if told_by else "observed", "told_by": told_by,
        "canonical": eid, "confidence": 0.9, "truth": "unknown",
        "valence": valence, "arousal": 0.3, "salience": salience,
        "strength": strength, "count": 1, "scene_id": scene,
        "turn_first": turn, "turn_last": turn, "turn_recalled": None,
        "tags": list(tags),
    }


def ledger() -> list[dict]:
    return [
        # chapel scene, turns 3..5, plus one late straggler at turn 15
        rec("chapel_candle", "chapel", scene="chapel", turn=3, tags=["chapel", "fire"]),
        rec("anton_prayed", "anton", scene="chapel", turn=4, tags=["chapel", "anton"]),
        rec("anton_left", "anton", scene="chapel", turn=5, tags=["anton"]),
        rec("chapel_bell", "chapel", scene="chapel", turn=15, tags=["chapel"]),
        # mill scene, unrelated
        rec("mill_wheel", "mill", scene="mill", turn=8, tags=["mill"]),
        rec("luiza_flour", "luiza", scene="mill", turn=9, tags=["mill", "luiza"]),
        # a told fact, source edge
        rec("anton_debt", "anton", kind="fact", scene="hearth", turn=20,
            tags=["anton", "money"], told_by="luiza"),
        rec("eugen_debt", "eugen", kind="fact", scene="hearth", turn=21,
            tags=["eugen", "money"], told_by="luiza"),
        # a reading of anton
        rec("read_anton", "anton", kind="reading", scene="chapel", turn=5,
            tags=["anton"], valence=-0.6),
    ]


def main() -> int:
    web = EngramWeb("bela")
    web.rebuild(ledger())

    print("\nindexes")
    check("nine records indexed", len(web) == 9)
    nb = dict(web.neighbours("chapel_candle"))
    noisy_or = lambda *ws: 1 - __import__("math").prod(1 - w for w in ws)  # noqa: E731
    check("scene+tag+chain combine noisy-OR, no entity",
          abs(nb.get("anton_prayed", 0) - noisy_or(0.5, 0.3, 0.4)) < 1e-9,
          f"got {nb.get('anton_prayed')}")
    check("entity+scene+tag, far turn: no chain",
          abs(nb.get("chapel_bell", 0) - noisy_or(0.8, 0.5, 0.3)) < 1e-9,
          f"got {nb.get('chapel_bell')}")
    check("more links always outrank fewer (never saturates)",
          nb["chapel_bell"] < 1.0 and nb["chapel_bell"] > noisy_or(0.8, 0.5))
    check("mill is not a neighbour of the chapel", "mill_wheel" not in nb)
    check("readings are never targets", "read_anton" not in nb)

    print("\nchapel case")
    base = {"chapel_candle": 0.8}           # the cue hit one memory directly
    final, trace = web.spread(base)
    ranked = [eid for eid, _ in trace.final]
    check("direct hit stays first", ranked[0] == "chapel_candle")
    check("anton_prayed surfaces (entity-less but scene+tag+chain)",
          "anton_prayed" in final and final["anton_prayed"] > 0)
    check("anton_left surfaces via chain through the scene",
          "anton_left" in final and final["anton_left"] > 0)
    check("nothing about the mill",
          "mill_wheel" not in final and "luiza_flour" not in final)
    check("spread-ins identified for no-bump rule",
          web.spread_ins(base, final) >= {"anton_prayed"})

    print("\nchain")
    base = {"anton_prayed": 0.8}
    f_on, _ = web.spread(base)
    off = EngramWeb("bela", tuning={"chain_enabled": False})
    off.rebuild(ledger())
    f_off, _ = off.spread(base)
    check("anton_left (turn 5) outranks chapel_bell (turn 15) with chain on",
          f_on["anton_left"] > f_on["chapel_bell"],
          f"{f_on['anton_left']:.3f} vs {f_on['chapel_bell']:.3f}")
    check("chain off removes that advantage from the chain term",
          f_off["anton_left"] < f_on["anton_left"])

    print("\nreadings")
    base = {"read_anton": 0.9}
    f_r, tr = web.spread(base)
    check("a reading seeds spread", any(f_r.get(x, 0) > 0 for x in ("anton_prayed", "anton_left")))
    base_ep = {"anton_left": 0.9}
    f_e, _ = web.spread(base_ep)
    check("reading seeds at reduced weight vs an episode seed",
          f_r["anton_prayed"] < f_e["anton_prayed"],
          f"{f_r['anton_prayed']:.3f} vs {f_e['anton_prayed']:.3f}")
    ratio = f_r["anton_prayed"] / f_e["anton_prayed"] if f_e["anton_prayed"] else 0
    check("reduction equals reading_seed_weight",
          abs(ratio - TUNING["reading_seed_weight"]) < 1e-6, f"ratio {ratio:.3f}")

    print("\nassociativity and realism")
    f_lo, _ = web.spread({"anton_left": 0.9}, associativity=0.3)
    f_hi, _ = web.spread({"anton_left": 0.9}, associativity=0.7)
    check("tangent-prone character spreads harder", f_hi["anton_prayed"] > f_lo["anton_prayed"])
    f_n, _ = web.spread({"read_anton": 0.9})
    f_nr, _ = web.spread({"read_anton": 0.9}, realism=True)
    check("realism boosts negative seeds", f_nr["anton_prayed"] > f_n["anton_prayed"])
    f_p, _ = web.spread({"anton_left": 0.9})
    f_pr, _ = web.spread({"anton_left": 0.9}, realism=True)
    check("realism leaves neutral seeds alone", abs(f_p["anton_prayed"] - f_pr["anton_prayed"]) < 1e-9)

    print("\nowners")
    try:
        web.add(rec("stray", "chapel", owner="eugen"))
        check("edges never cross owners", False, "foreign record accepted")
    except ValueError:
        check("edges never cross owners", True)

    print("\ncapture")
    check("below threshold captures nothing", web.capture("chapel_candle") == [])
    hot = rec("chapel_fire", "chapel", scene="chapel", turn=6, tags=["chapel", "fire"], salience=0.9)
    web.add(hot)
    bumps = dict(web.capture("chapel_fire"))
    check("above threshold bumps neighbours", bumps.get("chapel_candle", 0) > 0)
    check("bump scales with edge weight",
          bumps.get("chapel_candle", 0) > bumps.get("chapel_bell", 0))
    check("capture never reaches the reading", "read_anton" not in bumps)
    web.remove("chapel_fire")

    print("\ndegree")
    check("well-connected episode beats isolated fact",
          web.degree("anton_prayed") > web.degree("eugen_debt"))

    print("\nrebuild")
    again = EngramWeb("bela")
    again.rebuild(ledger())
    a, _ = web.spread({"chapel_candle": 0.8})
    b, _ = again.spread({"chapel_candle": 0.8})
    check("rebuild from ledger is identical", a == b)

    print("\ntrace")
    text = trace.as_text()
    check("trace renders seeds and hops", "seed chapel_candle" in text and "->" in text)

    print()
    if failures:
        print(f"{len(failures)} failure(s): {failures}")
        return 1
    print("all engram web checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
