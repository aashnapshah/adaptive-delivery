"""Extract each CPC's real work-up - the tests actually performed in the narrative, in order - and cache
it as the case's `orders`. This gives CPC a human ordering SEQUENCE (like MIMIC's real order set), so the
ordering-concordance and head-to-head benchmarks apply symmetrically to both sources.

    python -m harness.cases.build_cpc_workup [--limit N] [--force]

Writes data/cpc_workup.json = { "<case_id>": ["<test>", ...] } in performed order, consumed by make_cpc.
Saves as it goes, so an interrupted run resumes (already-cached cases are skipped unless --force).
"""

from __future__ import annotations

import argparse
import json
import os

from ..shared import llm
from ..shared.parsing import extract_json
from .cpc import CPCS, DATA_DIR

WORKUP_PATH = os.path.join(DATA_DIR, "cpc_workup.json")

EXTRACT_SYSTEM = (
    "You read a clinical case narrative and list the diagnostic tests and investigations that were "
    "ACTUALLY PERFORMED on this patient, in the order they appear. Include laboratory tests, imaging, "
    "microbiology/cultures, pathology/biopsy, and procedures. Use concise canonical test names "
    "(e.g. 'CT abdomen and pelvis', 'Blood cultures', 'Lumbar puncture', 'Complete blood count'). "
    "Exclude the final diagnosis, treatments given, and physical-exam maneuvers. "
    "Reply ONLY JSON: {\"tests\": [\"<test>\", ...]} in the order performed."
)


def extract_workup(case, model=None, backend=None):
    """The ordered list of tests performed in a CPC narrative (the human work-up)."""
    narrative = case.case_file.split("\n\nFINAL DIAGNOSIS")[0]      # never show the answer to the extractor
    messages = [{"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": narrative}]
    # A long work-up runs to ~40 tests; at 500 the reply was truncated mid-list, leaving no closing
    # brace for extract_json. Budget for the longest case rather than silently dropping it.
    rec = extract_json(llm.chat(messages, model=model or llm.light_model(backend), backend=backend,
                                max_tokens=2000, think=False))
    tests = [str(t).strip() for t in rec.get("tests", []) if str(t).strip()]
    return list(dict.fromkeys(tests))                              # dedupe, keep order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")               # re-extract even if cached
    args = ap.parse_args()

    cache = json.load(open(WORKUP_PATH)) if (os.path.exists(WORKUP_PATH) and not args.force) else {}
    cases = CPCS[:args.limit] if args.limit else CPCS
    for i, c in enumerate(cases, 1):
        if c.case_id in cache and not args.force:
            print(f"[{i:>3}/{len(cases)}] {c.case_id} cached ({len(cache[c.case_id])} tests)", flush=True)
            continue
        try:
            tests = extract_workup(c)
        except Exception as exc:
            print(f"[{i:>3}/{len(cases)}] {c.case_id} FAILED: {exc}", flush=True)
            continue
        cache[c.case_id] = tests
        json.dump(cache, open(WORKUP_PATH, "w"), indent=1)         # save as we go (resumable)
        print(f"[{i:>3}/{len(cases)}] {c.case_id}: {len(tests)} tests -> {tests[:5]}", flush=True)
    print(f"done: {len(cache)} cases cached at {WORKUP_PATH}")


if __name__ == "__main__":
    main()
