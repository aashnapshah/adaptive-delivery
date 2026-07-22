"""Load prompts from editable text files kept beside each phase's code.

Every agent's prompt lives in a `prompts/` folder next to its module, one `.txt` per prompt, so a
prompt can be edited without touching Python. Load one with:

    from ..shared.prompts import load
    GATEKEEPER_SYSTEM = load(__file__, "gatekeeper")   # -> reads <this dir>/prompts/gatekeeper.txt

Only a trailing newline is stripped (so files stay editor-friendly); leading and internal whitespace
is preserved verbatim, which keeps a design's prompt fingerprint stable.
"""

from __future__ import annotations

import os


def load(caller_file: str, name: str) -> str:
    """Read `<dir of caller_file>/prompts/<name>.txt`, stripping only trailing newlines."""
    d = os.path.join(os.path.dirname(os.path.abspath(caller_file)), "prompts")
    with open(os.path.join(d, f"{name}.txt"), encoding="utf-8") as f:
        return f.read().rstrip("\n")
