"""Aggregate the recorded benchmark and compare designs the way the SDBench / MAI-DxO
(Nori et al., Microsoft 2025) paper does: diagnostic accuracy vs cost, per agent design.

Reads a per-model recordings file (data/recordings/<slug>.json) and reports, per design:
accuracy (Judge >= 4), mean Judge score, mean cost, mean turns / tests / questions.

Usage:  python3 eval_benchmark.py [model-slug] [--cpc|--ehr|--all]
"""

from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.join(HERE, "data", "recordings")
DESIGN_LABEL = {"single": "LLM", "maidxo": "LLM-Roles", "debate": "LLM-Multi"}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(recs: dict, mode_filter: str):
    """recs = {case_id: {design: run}}. Returns per-design aggregate stats."""
    by_design = {}
    for cid, designs in recs.items():
        for d, run in designs.items():
            if mode_filter != "all" and run.get("mode") != mode_filter:
                continue
            turns = run.get("turns") or []
            dx = next((t for t in turns if t.get("action") == "diagnose"), None)
            if not dx:
                continue
            last = turns[-1]
            meta = run.get("meta") or {}
            by_design.setdefault(d, []).append({
                "correct": bool(dx.get("correct")),
                "score": dx.get("judge_score"),
                "cost": last.get("total_cost"),
                "turns": len(turns),
                "tests": last.get("n_tests"),
                "questions": last.get("n_questions"),
                "tokens": meta.get("total_tokens"),
                "secs": meta.get("duration_s"),
            })
    return by_design


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    mode = "cpc" if "--cpc" in flags else "ehr" if "--ehr" in flags else "all" if "--all" in flags else "cpc"

    if args:
        path = os.path.join(REC_DIR, args[0] + ".json") if not args[0].endswith(".json") else args[0]
    else:
        files = sorted(glob.glob(os.path.join(REC_DIR, "*.json")))
        if not files:
            print("no per-model recordings found in", REC_DIR); return
        path = files[0]

    recs = json.load(open(path))
    model = os.path.splitext(os.path.basename(path))[0]
    by_design = summarize(recs, mode)
    if not by_design:
        print(f"no {mode} runs in {os.path.basename(path)} yet."); return

    def _tok(rows):
        v = _mean([r["tokens"] for r in rows])
        return "   n/a" if v != v else (f"{v/1000:>5.1f}k" if v >= 1000 else f"{v:>6.0f}")

    print(f"\nmodel: {model}   |   cases: {mode}   |   n per design varies\n")
    print(f"{'design':17} {'n':>3} {'accuracy':>9} {'judge':>6} {'tokens':>7} {'secs':>6} "
          f"{'cost':>7} {'turns':>6} {'tests':>6}")
    print("-" * 74)
    order = [d for d in ("single", "maidxo", "debate") if d in by_design]
    for d in order:
        rows = by_design[d]
        n = len(rows)
        acc = _mean([1.0 if r["correct"] else 0.0 for r in rows]) * 100
        secs = _mean([r["secs"] for r in rows])
        print(f"{DESIGN_LABEL.get(d, d):17} {n:>3} {acc:>8.0f}% {_mean([r['score'] for r in rows]):>6.2f} "
              f"{_tok(rows):>7} {'   n/a' if secs != secs else f'{secs:>5.1f}s'} "
              f"${_mean([r['cost'] for r in rows]):>6.0f} {_mean([r['turns'] for r in rows]):>6.1f} "
              f"{_mean([r['tests'] for r in rows]):>6.1f}")
    print()
    # The core question: does genuine multi-agent buy accuracy worth its extra token/latency cost?
    if "single" in by_design and "debate" in by_design:
        a_s = _mean([1.0 if r["correct"] else 0.0 for r in by_design["single"]]) * 100
        a_o = _mean([1.0 if r["correct"] else 0.0 for r in by_design["debate"]]) * 100
        t_s = _mean([r["tokens"] for r in by_design["single"]])
        t_o = _mean([r["tokens"] for r in by_design["debate"]])
        print(f"LLM-Multi vs LLM:  accuracy {a_o-a_s:+.0f} pts", end="")
        if t_s == t_s and t_o == t_o and t_s:
            print(f",  tokens {t_o/t_s:.1f}x  ({t_s:.0f} -> {t_o:.0f})")
        else:
            print("   (token counts not recorded — re-run record.py to populate)")
        print("(The question: is the accuracy delta worth the token/latency multiple?)")


if __name__ == "__main__":
    main()
