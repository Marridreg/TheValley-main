"""/new — the only way to start over from inside the app.

Checks, without a model call:
  1. bare /new refuses and names what it will delete
  2. /new confirm resets every piece of session state to turn 0
  3. the autosave is gone, named saves are not
  4. the bridge gets exactly one reset signal, then it is cleared
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["VALLEY_SAVES_DIR"] = tempfile.mkdtemp(prefix="valley_test_new_")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from engine.commands import CommandRouter
    from engine.state import StateManager
    from engine.wall import Wall

    saves = Path(os.environ["VALLEY_SAVES_DIR"])
    wall = Wall.__new__(Wall)
    wall.state = StateManager(ROOT / "data", saves)
    wall.last_turn = "not-none"
    wall.resumed_from = "turn 9"
    wall.reset_pending = False
    router = CommandRouter(wall)

    # Dirty the session the way a real game would.
    st = wall.state
    st.chat_history += [{"role": "user", "content": "look"},
                        {"role": "assistant", "content": "Snow."}]
    st.turn_count = 9
    st.revelation_log.append("moreau.drives")
    st.beliefs["heisenberg"] = {"player": {"stance": "useful"}}
    st.discovered.append("rumor_x")
    st.authors_note = "keep it grim"
    st.pc["vitals"] = {"health": {"current": 1}}
    st.autosave()
    st.save("keeper")
    assert (saves / "_autosave.json").exists()

    handled, text = router.execute("/new")
    assert handled and "DELETES" in text and "turn 9" in text, text
    assert st.turn_count == 9 and (saves / "_autosave.json").exists(), "bare /new must not act"
    assert not wall.reset_pending

    handled, text = router.execute("/new confirm")
    assert handled and "new game" in text, text
    assert wall.reset_pending, "bridge needs the reset signal"
    assert wall.last_turn is None and wall.resumed_from is None

    fresh = StateManager(ROOT / "data", saves)
    for attr in ("pc", "world", "vault", "fragments", "chat_history", "revelation_log",
                 "beliefs", "discovered", "offscreen", "authors_note", "turn_count"):
        assert getattr(st, attr) == getattr(fresh, attr), f"{attr} did not reset"
    assert st.turn_count == 0

    assert not (saves / "_autosave.json").exists(), "autosave survived /new confirm"
    assert (saves / "keeper.json").exists(), "named save was deleted"
    assert st.list_saves() == ["keeper"]

    print("test_new_game: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
