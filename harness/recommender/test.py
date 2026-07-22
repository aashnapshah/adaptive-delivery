"""Run the recommender on a case and print its work-up (ordering trajectory).

    python harness/recommender/test.py                    # random case, single design
    python harness/recommender/test.py <case_id> [design]
"""

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from harness.cases import get_case, registry               # noqa: E402
from harness.recommender.recommender import diagnose, recommend   # noqa: E402
from harness.judge import judge_diagnosis                  # noqa: E402
from harness.store.transcripts import save_run             # noqa: E402
from harness.shared import llm                             # noqa: E402


def main():
    args = sys.argv[1:]
    case = get_case(args[0]) if args else random.choice(list(registry().values()))
    design = args[1] if len(args) > 1 else "single"
    print(f"backend: {llm.active_model()}  |  case: {case.case_id} ({case.source})  |  design: {design}\n")

    out = recommend(case, design=design)

    print("CONVERSATION  (recommender <-> gatekeeper):\n")
    for speaker, msg in out["conversation"]:
        label = "GATEKEEPER" if speaker == "gatekeeper" else "RECOMMENDER"
        print(f"[{label}]")
        print(msg.strip())
        print()
    m = out["meta"]
    print(f"METRICS: turns={m['turns']}  ordered={len(out['ordered'])}  stopped={out['stopped']}  "
          f"recommender_calls={m['calls']}  recommender_tokens={m['tokens']}  deliberate={m['deliberate_s']}s")

    # diagnosis is a SEPARATE readout (scored for CPC), not part of the ordering loop
    dx = diagnose(case, out["transcript"])
    verdict = judge_diagnosis(case.true_diagnosis, dx, context=case.abstract)
    print(f"\ndiagnosis readout (separate): {dx}")
    print(f"(true:                        {case.true_diagnosis})")
    print(f"judge: {verdict['score']}/5 {'CORRECT' if verdict['correct'] else 'incorrect'}")

    # generation saves the transcript only; the judge above is just shown here, scored by the eval drivers
    run_id = save_run(case, design, None, out, diagnosis=dx)
    print(f"\nsaved: results/raw/runs.csv + runs.jsonl  (run_id {run_id})")


if __name__ == "__main__":
    main()
