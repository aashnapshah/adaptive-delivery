"""Run the LLM judges over every generated run and write their scores to the runs.csv score columns.

    python -m harness.judge [--source cpc|mimic] [--designs D,D] [--force] [--limit N]

For each run: score the diagnosis (CPC only), and - where a human work-up exists to compare against -
the blinded head-to-head and the ordering concordance. A metric already filled for a run is SKIPPED, so
an interrupted pass resumes cheaply and re-running is nearly free; --force re-scores. The deterministic
metrics (cost) are a separate script: python -m harness.evaluation.eval.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from harness.cases import registry                                        # noqa: E402
from harness.judge.judge import judge_concordance, judge_diagnosis, judge_workup   # noqa: E402
from harness.store.transcripts import load_runs, load_scores, select_runs, update_scores   # noqa: E402


def judge_run(run, case, have, force):
    """The LLM score columns to fill for one run: only metrics not already present (unless force), and
    only where the inputs exist. A judge that comes back unavailable is left unscored, so it retries."""
    cols = {}
    ordered = run["ordered"]

    def need(key):
        return force or key not in have

    if case.source == "cpc" and run.get("diagnosis") and need("judge_score"):
        v = judge_diagnosis(case.true_diagnosis, run["diagnosis"], context=case.abstract)
        if v["score"]:                                     # 0 = judge unavailable -> leave unscored
            cols.update(judge_score=v["score"], judge_correct=v["correct"], judge_reason=v["reason"])

    ref = case.human_seq
    if ref and ordered:                                    # both benchmarks need a work-up to compare
        if need("workup_better"):
            v = judge_workup(case.presentation or case.abstract, ordered, ref)
            if v["better"] is not None:                    # None = judge unavailable
                cols.update(workup_better=v["better"], workup_reason=v["reason"])
        if need("f1"):
            try:
                v = judge_concordance(ordered, ref)
                if v["f1"] is not None:
                    cols.update(overlap=v["overlap"], missed=v["missed"], extra=v["extra"],
                                f1=v["f1"], jaccard=v["jaccard"])
            except Exception as exc:                        # transient LLM error -> skip, retry next run
                print(f"  overlap judge failed for {run['run_id']}: {str(exc)[:70]}", flush=True)
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="")            # cpc | mimic; empty = all
    ap.add_argument("--designs", default="")           # comma-separated; empty = every design
    ap.add_argument("--force", action="store_true")    # re-score even metrics already filled
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    runs = load_runs()
    cases = registry()
    scores = load_scores()
    items = select_runs(runs, args.source, args.designs, args.limit)

    scored = skipped = 0
    for i, (rid, run) in enumerate(items, 1):
        case = cases.get(run["case_id"])
        if case is None:
            continue
        cols = judge_run(run, case, scores.get(rid, {}), args.force)
        if cols:
            update_scores({rid: cols})                     # save per run, so an interrupted pass resumes
            scores.setdefault(rid, {}).update(cols)        # keep the in-memory view current for skip checks
            scored += 1
            print(f"[{i:>3}/{len(items)}] {rid:34} {', '.join(cols)}", flush=True)
        else:
            skipped += 1
    print(f"\njudge: scored {scored}, skipped {skipped} (already done). "
          f"deterministic metrics: python -m harness.evaluation.eval", flush=True)


if __name__ == "__main__":
    main()
