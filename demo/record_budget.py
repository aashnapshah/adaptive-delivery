"""Cost-budget sweep (SDBench Figure: accuracy vs. spending cap).

Reproduces the paper's cost-control experiment: run the SAME agent at several hard spending
caps and trace how diagnostic accuracy rises as the budget grows. Uses record_one's `budget`
knob (once spend hits the cap, the agent is forced to commit a diagnosis).

Output: data/budget_sweep.json  ->  {model: {budget: {design: {case: run}}}}
Resumable (skips finished combos). Local-model friendly.

Run:  OLLAMA_MODEL=qwen2.5:14b BUDGET_CASES=15 python3 record_budget.py
"""

from __future__ import annotations

import json
import os

import app
import record

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "budget_sweep.json")

BUDGETS = [400, 800, 1500, 3000, None]                 # $ caps; None = unlimited (full workup)
DESIGNS = [x.strip() for x in os.environ.get("BUDGET_DESIGNS", "single,maidxo").split(",") if x.strip()]
MODEL = os.environ.get("BUDGET_MODEL") or os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
LIMIT = int(os.environ.get("BUDGET_CASES", "15"))


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
    cases = cpc_cases(LIMIT)
    out = record._load(OUT)
    out.setdefault(MODEL, {})
    total = len(BUDGETS) * len(DESIGNS) * len(cases)
    n = 0
    print(f"budget sweep: {MODEL} · {len(BUDGETS)} caps × {len(DESIGNS)} designs × {len(cases)} cases = {total} runs")
    for b in BUDGETS:
        bk = str(int(b)) if b else "inf"
        out[MODEL].setdefault(bk, {})
        for d in DESIGNS:
            out[MODEL][bk].setdefault(d, {})
            for cid in cases:
                n += 1
                prev = out[MODEL][bk][d].get(cid)
                if record._is_done(prev):
                    print(f"[{n}/{total}] skip {bk}/{d}/{cid}")
                    continue
                print(f"[{n}/{total}] {bk}/{d}/{cid} ...", flush=True)
                out[MODEL][bk][d][cid] = record.record_one(cid, d, MODEL, budget=b)
                with open(OUT, "w") as f:
                    json.dump(out, f, indent=1)
    print(f"budget sweep done -> {OUT}")


if __name__ == "__main__":
    main()
