"""Compute the deterministic metrics for every generated run and write them to the runs.csv score columns.

    python -m harness.evaluation.eval [--source cpc|mimic] [--designs D,D] [--force]

Rule-based, no LLM: currently the work-up cost (harness/evaluation/cost.py). A metric already filled for
a run is skipped unless --force. The LLM metrics are a separate script: python -m harness.judge.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from harness.evaluation.cost import workup_cost                           # noqa: E402
from harness.store.transcripts import load_runs, load_scores, select_runs, update_scores   # noqa: E402


def eval_run(run, have, force):
    """The deterministic score columns to fill for one run - only what's missing (unless force)."""
    cols = {}
    if force or "cost_total" not in have:
        cols["cost_total"] = workup_cost(run["ordered"])["cost_total"]
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="")            # cpc | mimic; empty = all
    ap.add_argument("--designs", default="")           # comma-separated; empty = every design
    ap.add_argument("--force", action="store_true")    # recompute even if already filled
    ap.add_argument("--limit", type=int, default=None)  # cap to the first N distinct cases
    args = ap.parse_args()

    runs = load_runs()
    scores = load_scores()
    items = select_runs(runs, args.source, args.designs, args.limit)

    updates = {rid: cols for rid, run in items
               if (cols := eval_run(run, scores.get(rid, {}), args.force))}
    update_scores(updates)                              # deterministic + cheap: one write at the end
    print(f"eval: filled {len(updates)} of {len(items)} runs (deterministic). "
          f"LLM metrics: python -m harness.judge", flush=True)


if __name__ == "__main__":
    main()
