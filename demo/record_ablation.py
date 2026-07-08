"""Role ablation: which perspective in the LLM-Roles prompt actually matters?

The LLM-Roles design tells a single agent to weigh five perspectives before each action
(differential, stewardship, patient burden, clinician workload, checklist). We drop ONE
perspective from the prompt at a time, re-run the benchmark, and measure the accuracy change.
A big drop => that perspective was load-bearing.

Output: data/ablation.json  ->  {model: {variant: {label, runs: {case: run}}}}
Resumable. Local-model friendly (runs on whatever OLLAMA_MODEL is set).

Run:  OLLAMA_MODEL=qwen3:32b ABLATION_CASES=20 python3 record_ablation.py
"""

from __future__ import annotations

import json
import os
import re

import app
import record

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "ablation.json")

BASE = "maidxo"                                          # ablate the LLM-Roles prompt
# perspective number in the prompt -> short label for the variant
PERSPECTIVES = {1: "hypothesis", 2: "testchooser", 3: "challenger", 4: "stewardship", 5: "checklist"}
MODEL = os.environ.get("ABLATION_MODEL") or os.environ.get("OLLAMA_MODEL", "qwen3:32b")
LIMIT = int(os.environ.get("ABLATION_CASES", "20"))

_base_label, BASE_SYS, KIND = app.DESIGNS[BASE]
# the inline "(1) …; (2) …; … (5) …" list sits between "…out): " and ".\nThen take ONE action"
_LIST_RE = re.compile(r"(them out\):\s*)(.*?)(\.\nThen take ONE action)", re.S)


def ablate(system: str, n: int) -> str:
    """Drop the '(n) …' perspective clause from the LLM-Roles prompt's inline list."""
    m = _LIST_RE.search(system)
    if not m:
        return system
    items = [it.strip() for it in m.group(2).split("; ")]
    kept = [it for it in items if not it.lstrip().startswith(f"({n})")]
    return system[:m.start(2)] + "; ".join(kept) + system[m.end(2):]


def cpc_cases(limit: int) -> list[str]:
    out = []
    for o in app.all_case_options():
        try:
            _, mode = app.load_any(o["id"])
        except Exception:
            continue
        if mode == "cpc":
            out.append(o["id"])
        if len(out) >= limit:
            break
    return out


def main() -> None:
    # register the full prompt + one variant per dropped perspective
    variants = [("full", "LLM-Roles (all)", BASE_SYS)]
    for n, name in PERSPECTIVES.items():
        variants.append(("abl-" + name, "−" + name, ablate(BASE_SYS, n)))
    for key, label, sys in variants:
        app.DESIGNS[key] = (label, sys, KIND)

    cases = cpc_cases(LIMIT)
    out = record._load(OUT)
    out.setdefault(MODEL, {})
    total = len(variants) * len(cases)
    n = 0
    print(f"ablation: {MODEL} · {len(variants)} variants × {len(cases)} cases = {total} runs")
    for key, label, _ in variants:
        v = out[MODEL].setdefault(key, {"label": label, "runs": {}})
        v["label"] = label
        for cid in cases:
            n += 1
            if record._is_done(v["runs"].get(cid)):
                print(f"[{n}/{total}] skip {key}/{cid}")
                continue
            print(f"[{n}/{total}] {key}/{cid} ...", flush=True)
            v["runs"][cid] = record.record_one(cid, key, MODEL)
            with open(OUT, "w") as f:
                json.dump(out, f, indent=1)
    print(f"ablation done -> {OUT}")


if __name__ == "__main__":
    main()
