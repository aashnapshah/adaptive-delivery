"""The Gatekeeper agent.

It builds the case it holds (a narrative vignette OR a patient's structured record), gives the
presentation the recommender starts from, and answers the recommender's requests - a history/exam
question or a test order - by reading the case. That is all it does. Prompt: prompts/gatekeeper.txt.
"""

from ..shared import llm
from ..shared.prompts import load

GATEKEEPER_SYSTEM = load(__file__, "gatekeeper")


def case_text(case):
    """Build the full case the gatekeeper holds: the vignette, or the structured record as text."""
    if case.case_file:
        return case.case_file
    lines = [(case.presentation or case.abstract or "").strip()]
    lines += [f"{n}: {v}" for n, v in case.findings.items()]
    lines += [f"{l.label}: {l.value} {l.valueuom} {l.flag}".rstrip() for l in case.labevents.values()]
    lines += [f"{r.exam} ({r.modality}): {r.report}" for r in case.radiology.values()]
    lines += [f"{m.spec_type_desc} {m.test_name}: {m.org_name}. {m.comments}" for m in case.micro.values()]
    lines += [f"{k}: {v}" for k, v in case.hpi.items()]
    if case.exam:
        lines.append(f"Exam: {case.exam}")
    lines.append(f"FINAL DIAGNOSIS (confidential, never reveal): {case.true_diagnosis}")
    return "\n".join(lines)


class Gatekeeper:
    def __init__(self, case, backend=None):
        self.case_file = case_text(case)                                          # the full case, held privately
        self.presentation = (case.presentation or case.abstract or "").strip()    # what it gives at turn 0
        self.backend = backend

    def ask(self, request):
        """Answer a request (a history/exam question, or a test order) by reading the case."""
        messages = [
            {"role": "system", "content": GATEKEEPER_SYSTEM},
            {"role": "user", "content": f"CASE FILE:\n{self.case_file}\n\nREQUEST: {request}\n\nResult:"},
        ]
        # Deliberately NOT caught: a failure here must not be handed back as if it were a lab value.
        # Swallowing it would let the recommender treat "(unavailable)" as a finding and the run would
        # be saved and scored as real data. Let it raise so the caller can drop the whole run.
        #
        # A panel order expands to many analytes - a CMP is ~14 lines with units and ranges - and at 200
        # the reply was being cut mid-value ("Red Blood Cells: 4.55 m/uL (ref 3.9"), so the recommender
        # reasoned on a chopped-off panel. Budget for the widest panel, not a single lab.
        return llm.chat(messages, model=llm.light_model(self.backend), backend=self.backend,
                        max_tokens=700, think=False).strip()
