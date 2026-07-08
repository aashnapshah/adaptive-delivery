"""Build data/cpc_presentations.json: the intake PRESENTATION for each CPC case.

This is a sequential-diagnosis benchmark — the agent starts from what is known at intake and
must ORDER tests (through the gatekeeper) to obtain any result. So the presentation is the
PRESENTING HISTORY + vital signs only: demographics, chief complaint, HPI, past/social/family/
exposure history, ROS, and vitals. The physical/neuro exam and every test result are withheld
(the agent asks for the exam and orders tests; the gatekeeper serves them).

An LLM classifies the case's numbered sentences into keep (intake history) vs withhold; we
rebuild the presentation from the kept sentences verbatim (no rewriting). The result is cached
so the benchmark input is fixed.

Run:  OLLAMA_HOST=127.0.0.1:11434 DEMO_LLM_BACKEND=ollama OLLAMA_MODEL=qwen3:32b \
      python3 split_presentations.py [--limit N] [--only CASE_ID,CASE_ID]
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared import llm  # noqa: E402
import app  # reuse the case registry / loaders  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cpc_presentations.json")

SPLIT_SYSTEM = (
    "You are given an NEJM case as NUMBERED sentences. This is a SEQUENTIAL-DIAGNOSIS benchmark: the "
    "agent starts from the bedside presentation and then ORDERS tests to obtain results. Identify the "
    "sentences that make up the PRESENTING HISTORY — what is known about the patient at intake, "
    "BEFORE the clinician examines them or orders any test.\n"
    "KEEP (output these): demographics; chief complaint; history of present illness (the symptom story "
    "and its timeline); past medical / medication / surgical history; social / family / occupational / "
    "exposure / travel history; review of systems; and VITAL SIGNS at presentation. A statement that a "
    "treatment was given, or that the patient was transferred, is part of the story — keep it.\n"
    "EXCLUDE (do NOT output): (a) the PHYSICAL EXAMINATION and NEUROLOGIC EXAMINATION findings — "
    "anything the clinician elicits by examining the patient (general appearance, HEENT, heart/lung/"
    "abdomen, skin, and mental-status / cranial-nerve / motor / sensory / reflex / gait findings, "
    "Babinski, etc.): the agent must ASK for these, so they are NOT part of the intake history; "
    "(b) any sentence reporting the RESULT of a laboratory test, blood count/chemistry, imaging (CT, "
    "MRI, x-ray, ultrasound, angiogram, PET), ECG/EEG, lumbar puncture/CSF, microbiology/culture/"
    "serology, or pathology/biopsy — INCLUDING outside-hospital results; and (c) the later diagnostic "
    "work-up and hospital-course sentences.\n"
    "Vital signs (temperature, pulse, blood pressure, respiratory rate, oxygen saturation) are intake "
    "measurements — KEEP them; but narrative EXAM findings are EXCLUDED. Output ONLY the sentence "
    "numbers to KEEP, comma-separated (e.g. '1, 2, 3, 8')."
)


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
    path = os.path.join(os.path.dirname(OUT), "cpc_bench.json")
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

    opts = [o for o in app.all_case_options() if o.get("mode") == "cpc"]
    if only:
        opts = [o for o in opts if o["id"] in only]
    if limit:
        opts = opts[:limit]

    model = os.environ.get("OLLAMA_MODEL") or None
    print(f"splitting {len(opts)} CPC cases -> {OUT}")
    for n, o in enumerate(opts, 1):
        cid = o["id"]
        sents = sents_by_id.get(cid)
        pres = intake_presentation(sents, model) if sents else None
        if not pres:
            print(f"[{n}/{len(opts)}] {cid}: SKIP (no sentence data / empty result)")
            continue
        cache[cid] = pres
        frac = len(pres) / max(1, _case_len(app.load_any(cid)[0]))
        print(f"[{n}/{len(opts)}] {cid}: presentation {len(pres)} chars ({frac:.0%} of case) | "
              f"ends: …{pres[-70:].strip()!r}")
        json.dump(cache, open(OUT, "w"), indent=1)    # checkpoint after each
    print(f"done — {len(cache)} presentations cached in {OUT}")


if __name__ == "__main__":
    main()
