"""Build data/cpc_presentations.json: the intake PRESENTATION for each CPC case (stage 01 tool).

This is a sequential-diagnosis benchmark - the agent starts from what is known at intake and must
ORDER tests (through the gatekeeper) to obtain any result. So the presentation is the PRESENTING
HISTORY + vital signs only: demographics, chief complaint, HPI, past/social/family/exposure history,
ROS, and vitals. The physical/neuro exam and every test result are withheld (the agent asks for the
exam and orders tests; the gatekeeper serves them).

An LLM classifies the case's numbered sentences into keep (intake history) vs withhold; we rebuild
the presentation from the kept sentences verbatim (no rewriting). The result is cached so the
benchmark input is fixed. `cpc.py` reads this cache into each Case's `presentation`.

Run:  DEMO_LLM_BACKEND=gemini python3 -m harness.cases.build_cpc_presentations [--limit N] [--only ID,ID]
"""

from __future__ import annotations

import json
import os
import re
import sys

from ..shared import llm
from ..shared.prompts import load
from .cpc import DATA_DIR, load_cpc_bench

OUT = os.path.join(DATA_DIR, "cpc_presentations.json")

SPLIT_SYSTEM = load(__file__, "cpc_split")


# Deterministic guardrail: drop any kept sentence that REPORTS a test/procedure result, even if the
# model left it in. Belt-and-suspenders over the LLM split, which leaks results on complex cases.
_RESULT = re.compile(
    r"reference range|\bmg per (?:deci|milli)liter\b|\bmmol per liter\b|per cubic millimeter|"
    r"\bU per (?:liter|milliliter)\b|\bng per\b|\b(?:biopsy|autopsy|aspirate|specimen)\b|"
    r"culture (?:grew|was|were|yielded|showed|revealed|disclosed)|\bIgG\b|\bIgM\b|positron|\bFDG\b|"
    r"(?:computed tomograph|\bCT\b|MRI|magnetic resonance|ultrasonograph|echocardiograph|radiograph|"
    r"chest film|laryngoscop|bronchoscop|endoscop|electroencephalograph|angiograph)"
    r"[^.]{0,70}(?:reveal|showed|show |disclos|demonstrat|was (?:normal|abnormal|negative|positive)|"
    r"no evidence|nonobstruct)", re.I)


def intake_presentation(sents: dict, model: str | None) -> str | None:
    """Rebuild the intake presentation from the history/vitals sentences the model keeps, with two
    hard guarantees: always keep the opening (chief complaint), and drop any sentence that reports a
    result. Then strip the discussant attribution ('Dr. X (Specialty):')."""
    keys = sorted(sents, key=lambda k: int(k))
    numbered = "\n".join(f"{k}. {sents[k]}" for k in keys)
    out = llm.chat([{"role": "system", "content": SPLIT_SYSTEM},
                    {"role": "user", "content": numbered + "\n\nSentence numbers to KEEP:"}],
                   model=model, max_tokens=250, think=False).strip()
    keep = {int(x) for x in re.findall(r"\d+", out)}
    first = int(keys[0])
    keep.add(first)                            # ALWAYS keep the opening (demographics + chief complaint)
    kept = [sents[k] for k in keys
            if int(k) in keep and (int(k) == first or not _RESULT.search(sents[k]))]
    text = " ".join(kept).strip()
    text = re.sub(r"^Dr\.[^:]{0,70}:\s*", "", text)        # strip discussant attribution
    return text or None


def _sentences_by_id() -> dict:
    """case_id -> {sentence_number: text} from the CPC-Bench source."""
    path = os.path.join(DATA_DIR, "cpc_bench.json")
    if not os.path.exists(path):
        return {}
    return {str(c.get("id")): c["presentation_of_case_sent"]
            for c in json.load(open(path)) if c.get("presentation_of_case_sent")}


def _case_len(case) -> int:
    return len((getattr(case, "case_file", "") or "").split("\n\nFINAL DIAGNOSIS")[0])


def main() -> None:
    args = sys.argv[1:]
    limit = next((int(args[i + 1]) for i, a in enumerate(args) if a == "--limit"), None)
    only = next(({x.strip() for x in args[i + 1].split(",")} for i, a in enumerate(args) if a == "--only"), None)

    print(f"backend: {llm.active_model()}")
    try:
        cache = json.load(open(OUT))
    except Exception:
        cache = {}
    sents_by_id = _sentences_by_id()

    cases = {c.case_id: c for c in load_cpc_bench()}
    ids = list(cases)
    if only:
        ids = [i for i in ids if i in only]
    if limit:
        ids = ids[:limit]

    model = os.environ.get("OLLAMA_MODEL") or None
    print(f"splitting {len(ids)} CPC cases -> {OUT}")
    for n, cid in enumerate(ids, 1):
        sents = sents_by_id.get(cid)
        pres = intake_presentation(sents, model) if sents else None
        if not pres:
            print(f"[{n}/{len(ids)}] {cid}: SKIP (no sentence data / empty result)")
            continue
        cache[cid] = pres
        frac = len(pres) / max(1, _case_len(cases[cid]))
        print(f"[{n}/{len(ids)}] {cid}: presentation {len(pres)} chars ({frac:.0%} of case) | "
              f"ends: …{pres[-70:].strip()!r}")
        json.dump(cache, open(OUT, "w"), indent=1)    # checkpoint after each
    print(f"done — {len(cache)} presentations cached in {OUT}")


if __name__ == "__main__":
    main()
