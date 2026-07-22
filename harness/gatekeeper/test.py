"""Chat with the gatekeeper, to sanity-check and tune its prompt (prompts/gatekeeper.txt).

Picks a random case, shows the presentation the recommender would start from, then you type freely and
the gatekeeper answers - order a test or ask a question, in plain language.

Run it directly (from anywhere):
    python harness/gatekeeper/test.py            # random case
    python harness/gatekeeper/test.py <case_id>  # a specific case
Type ':file' to see the full case file the gatekeeper holds; Ctrl-D to quit.
"""

import os
import random
import sys

# entry-point bootstrap: put the repo root on the path so this file runs directly (python test.py),
# then use absolute imports (no package context needed).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from harness.cases import get_case, registry          # noqa: E402
from harness.gatekeeper.gatekeeper import Gatekeeper   # noqa: E402
from harness.shared import llm                         # noqa: E402


def main() -> None:
    backend = llm.detect_backend()
    case = get_case(sys.argv[1]) if len(sys.argv) > 1 else random.choice(list(registry().values()))

    gk = Gatekeeper(case, backend=backend)
    print(f"backend: {llm.active_model(backend)}")
    print(f"case:    {case.case_id}  (source={case.source})\n")
    print("PRESENTATION (from the gatekeeper):")
    print(gk.presentation)
    print("\n(chat with the gatekeeper - order a test or ask a question; ':file' shows its case file, Ctrl-D quits)\n")

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line == ":file":
            print(gk.case_file + "\n")
            continue
        print(gk.ask(line) + "\n")


if __name__ == "__main__":
    main()
