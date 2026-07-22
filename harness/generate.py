"""Generate recommender runs: talk to the gatekeeper and save the transcript. No scoring - that's evaluate.py.

    python harness/generate.py [--designs single,roles] [--models ID,ID] [--source cpc|mimic]
                              [--limit N] [--force] [--show]

`--show` prints each recommender <-> gatekeeper conversation as it runs; `--force` re-generates cells
already saved (handy for a fresh small test batch).

Sweeps designs x models x cases, skips (config, case) cells already saved (so an interrupted sweep
resumes), and writes each run to results/raw/. The CPC diagnosis readout is saved too - it's the model's
answer, not a metric - but the judge / concordance / work-up metrics are left for evaluate.py.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.cases import registry                        # noqa: E402
from harness.recommender import diagnose, recommend       # noqa: E402
from harness.store.transcripts import config, done, save_run    # noqa: E402


def _print_conversation(case, out):
    """The full recommender <-> gatekeeper chat for one run."""
    print(f"\n{'=' * 78}\nCONVERSATION  {case.case_id} ({case.source})\n{'=' * 78}")
    for speaker, msg in out["conversation"]:
        print(f"[{'GATEKEEPER' if speaker == 'gatekeeper' else 'RECOMMENDER'}]")
        print(msg.strip() + "\n")


def generate(designs, models, cases, force=False, show=False, params=None):
    """Sweep designs x models x cases, skipping cells already generated, saving each. Resumable.
    `force` re-generates even cells already in transcripts (e.g. to make a fresh small test batch).
    `show` prints each recommender <-> gatekeeper conversation as it runs.
    `params` (e.g. {"tag": "gk700", "note": "..."}) forks a NEW config_id, so a re-run after a code
    change lands as its own experiment beside the old one instead of colliding with it."""
    already = set() if force else done()
    combos = [(d, m, c) for d in designs for m in models for c in cases]
    total = len(combos)
    failures = []
    for i, (design, model, case) in enumerate(combos, 1):
        tag = f"[{i:>3}/{total}] {case.case_id:22} {design:7}"
        if (config(design, model, params)["config_id"], case.case_id) in already:
            print(f"{tag} skip (already done)", flush=True)
            continue
        try:
            out = recommend(case, design=design, model=model)
            dx = diagnose(case, out["transcript"], model=model) if case.source == "cpc" else None
        except Exception as exc:
            # Drop the whole run rather than save a partial//corrupted one. It stays un-done, so
            # re-running the sweep retries it.
            failures.append((case.case_id, design, str(exc)[:120]))
            print(f"{tag} FAILED  {str(exc)[:100]}", flush=True)
            continue
        save_run(case, design, model, out, diagnosis=dx, params=params)
        if show:
            _print_conversation(case, out)
        print(f"{tag} done  tests={len(out['ordered'])} turns={out['meta']['turns']} "
              f"time={out['meta']['deliberate_s']}s" + (f"  dx={dx}" if dx else ""), flush=True)
    if failures:
        print(f"\n{len(failures)} run(s) failed and were NOT saved (re-run to retry):", flush=True)
        for cid, d, err in failures:
            print(f"  {cid} {d}: {err}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default="single,roles")
    ap.add_argument("--models", default="")      # comma-separated model ids; empty = the backend default
    ap.add_argument("--source", default="")      # cpc | mimic | toy; empty = all sources
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")   # re-generate cells already saved
    ap.add_argument("--show", action="store_true")    # print each recommender <-> gatekeeper chat
    ap.add_argument("--tag", default="")              # names this experiment; forks a new config_id
    ap.add_argument("--note", default="")             # what was different about this run
    args = ap.parse_args()

    cases = list(registry().values())
    if args.source:
        cases = [c for c in cases if c.source == args.source]
    if args.limit:
        cases = cases[:args.limit]
    designs = [d.strip() for d in args.designs.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",")] if args.models else [None]
    params = {k: v for k, v in (("tag", args.tag), ("note", args.note)) if v}
    print(f"generate: {len(designs)} designs x {len(models)} models x {len(cases)} cases"
          + (f"  [{args.tag}]" if args.tag else ""))
    generate(designs, models, cases, force=args.force, show=args.show, params=params or None)


if __name__ == "__main__":
    main()
