"""Demo 01 - Sequential Diagnosis Environment over EHR data.

Maps to Methods section: "Datasets and Sequential Environment".

This replicates the SDBench environment of Nori et al. (2025), but builds it over
a structured (MIMIC-IV-shaped) EHR record instead of an NEJM narrative. The agent
interacts through a *Gatekeeper* that reveals findings only when explicitly queried,
and a Cost Estimator that charges per physician visit and per test.

Three actions (exactly as in SDBench):
  1. ask_question(text) -> patient history / exam detail (one $300 visit per question burst)
  2. order_test(name)   -> structured lab panel / imaging / micro result (CPT-style cost)
  3. diagnose(text)     -> commit; the Judge scores it against the ICD ground truth

The Gatekeeper synthesizes a plausible (normal) finding for tests not in the record,
so "missing data" never leaks the diagnosis - the key SDBench design choice.

Run:  python3 demo_environment.py
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.toy_data import EHRCase, LabEvent, TestPanel, load_case  # noqa: E402

VISIT_COST = 300.0          # per physician visit (SDBench: $300)
SYNTHETIC_TEST_COST = 50.0  # legacy flat charge (superseded by estimate_test_cost)

# SDBench-style per-test pricing: when a test has no structured cost in the case data (every CPC
# test, plus off-protocol EHR tests), estimate a realistic US price from the test category instead
# of a flat charge. Approximate list prices; matched most-specific-first.
_TEST_PRICES = (
    (r"pet[\s/-]?ct|pet scan", 1500),
    (r"\bmri\b|magnetic resonance", 600),
    (r"angiogram|angiograph|\bcta\b|ct angio", 500),
    (r"\bct\b|computed tomograph|cat scan", 300),
    (r"echocardiogram|\becho\b|ultrasound|sonograph|doppler", 200),
    (r"x[\s-]?ray|radiograph|chest film|\bkub\b|mammogram|\bfilm\b", 50),
    (r"bone marrow|biopsy|aspirat", 600),
    (r"bronchoscopy|endoscopy|colonoscopy|cystoscopy|laparoscopy|\begd\b|thoracentesis|paracentesis|catheteriz", 800),
    (r"bone scan|scintigraph|nuclear|\bspect\b|\bvq scan\b|\bv/q\b|\bhida\b|dexa|\bdxa\b", 300),
    (r"lumbar puncture|spinal tap|\bcsf\b|\blp\b\b", 300),
    (r"\beeg\b|electroencephalogram|nerve conduction|\bemg\b|electromyograph|evoked potential", 300),
    (r"\becg\b|\bekg\b|electrocardiogram|telemetry|holter|stress test|spirometr|pulmonary function|\bpft\b", 50),
    (r"flow cytometry|cytogenetic|karyotype|\bfish\b|molecular|genetic|sequenc|\bpcr\b|mutation|immunophenotyp", 400),
    (r"culture|gram stain|microbiolog|sensitivit|blood culture", 80),
    (r"serolog|antibody|antigen|elisa|\btiter\b|\bhiv\b|hepatitis|\bana\b|\banca\b|complement|immunoglobulin", 100),
    (r"\bcbc\b|complete blood count|\bcmp\b|\bbmp\b|metabolic panel|electrolyte|chemistr|\blft\b|liver function|"
     r"renal|urinalysis|\bua\b|\besr\b|\bcrp\b|c-reactive|d-dimer|troponin|\bbnp\b|coag|\binr\b|\bptt\b|lipid|"
     r"glucose|hba1c|\btsh\b|thyroid|lactate|\babg\b|\bvbg\b|blood gas|ferritin|\bldh\b|vitamin|folate|smear|peripheral|"
     r"lipase|amylase|procalcitonin|ck[\s-]?mb|creatine kinase|\bck\b|\bcpk\b|magnesium|phosph|calcium|uric acid|"
     r"ammonia|cortisol|fibrinogen|haptoglobin|reticulocyte|\bpt\b|\bcbc\b|blood count|\blevel\b|\bpanel\b|\bassay\b|"
     r"erythrocyte sedimentation|monospot|\bmono\b|stool|fecal|calprotectin|ova and parasit|o&p|occult blood|"
     r"\bhcg\b|pregnancy|\btsh\b|\bt3\b|\bt4\b|electrophoresis|\bspep\b|\bupep\b|\bana\b|\bcrp\b", 30),
)
DEFAULT_TEST_COST = 100.0   # unknown / unmatched test
# Words that mark an unmatched order as a blood/serum lab (cheap) rather than a procedure/imaging.
_LAB_HINT = re.compile(r"\b(blood|serum|plasma|urine|level|panel|assay|screen|count|titer|ratio)\b")


def estimate_test_cost(name: str) -> float:
    n = (name or "").lower()
    for pat, cost in _TEST_PRICES:
        if re.search(pat, n):
            return float(cost)
    return 30.0 if _LAB_HINT.search(n) else DEFAULT_TEST_COST   # lab-like unknowns priced as labs


# --- Gatekeeper: serves the EHR record like SDBench's oracle -----------------

class Gatekeeper:
    """Reveals findings from the structured record, or synthesizes plausible ones.

    Mirrors SDBench: discloses only what was explicitly requested, refuses vague
    requests, and returns realistic synthetic results for off-record queries so
    absence of data is not a clue.
    """

    VAGUE = {"labs", "tests", "bloodwork", "blood work", "workup", "imaging", "scan", "everything"}

    def __init__(self, case: EHRCase):
        self.case = case

    def _resolve_panel(self, order_name: str) -> TestPanel | None:
        q = order_name.strip().lower()
        for panel in self.case.panels.values():
            names = (panel.order_name.lower(), *(a.lower() for a in panel.aliases))
            if q in names:
                return panel
        return None

    FLAG_WORDS = {"H": "high", "L": "low", "HH": "critically high", "LL": "critically low",
                  "A": "abnormal", "AA": "critical"}

    @classmethod
    def _format_lab(cls, lab: LabEvent) -> str:
        ref = ""
        if lab.ref_lower is not None and lab.ref_upper is not None:
            ref = f" (normal {lab.ref_lower}–{lab.ref_upper})"
        flag = f" — {cls.FLAG_WORDS.get((lab.flag or '').upper(), lab.flag)}" if lab.flag else ""
        return f"{lab.label}: {lab.value} {lab.valueuom}{ref}{flag}"

    def order_test(self, order_name: str) -> tuple[str, float, bool]:
        """Return (finding_text, cost, on_record)."""
        if order_name.strip().lower() in self.VAGUE:
            return (f"[Gatekeeper] '{order_name}' is too non-specific; please order a named test.", 0.0, False)

        panel = self._resolve_panel(order_name)
        if panel is None:
            # Off-record: synthesize a plausible, unremarkable result.
            return (f"{order_name}: within normal limits.", estimate_test_cost(order_name), False)

        if panel.kind == "lab_panel":
            rows = [self._format_lab(self.case.labevents[l]) for l in panel.lab_labels]
            if len(rows) == 1:                                  # single analyte: no redundant header
                return (rows[0], panel.cost, True)
            body = "\n".join(f"  • {r}" for r in rows)
            return (f"{panel.order_name}\n{body}", panel.cost, True)
        if panel.kind == "imaging":
            study = self.case.radiology[panel.imaging_exam]
            return (f"{study.exam} ({study.modality}): {study.report}", panel.cost, True)
        if panel.kind == "micro":
            m = self.case.micro[panel.micro_spec]
            return (f"{m.spec_type_desc} - {m.test_name}: {m.org_name}. {m.comments}", panel.cost, True)
        raise ValueError(f"Unknown panel kind {panel.kind!r}")

    def ask_question(self, text: str) -> str:
        """Answer a history/exam question from the HPI fields (keyword match)."""
        q = text.lower()
        if "exam" in q or "physical" in q:
            return self.case.exam
        for topic, answer in self.case.hpi.items():
            if topic in q or any(w in q for w in topic.split()):
                return answer
        # Default: synthesize a benign, non-leaking response.
        return "No additional pertinent positives reported for that query."


# --- LLM-backed Gatekeeper (reads a case file, answers free-form queries) ----

def _case_file(case: EHRCase) -> str:
    """Render the full hidden record as a narrative case file for the LLM gatekeeper."""
    lines = [f"PATIENT: {case.patient.gender}, {case.patient.anchor_age} years old.",
             f"PRESENTATION: {case.abstract}", "HISTORY:"]
    for topic, ans in case.hpi.items():
        lines.append(f"  - {topic}: {ans}")
    lines.append(f"EXAM: {case.exam}")
    lines.append("LABORATORY RESULTS:")
    for lab in case.labevents.values():
        flag = f" [{lab.flag}]" if lab.flag else ""
        lines.append(f"  - {lab.label}: {lab.value} {lab.valueuom}{flag}")
    if case.micro:
        lines.append("MICROBIOLOGY:")
        for m in case.micro.values():
            lines.append(f"  - {m.spec_type_desc} {m.test_name}: {m.org_name}. {m.comments}")
    if case.radiology:
        lines.append("IMAGING:")
        for r in case.radiology.values():
            lines.append(f"  - {r.exam} ({r.modality}): {r.report}")
    lines.append(f"FINAL DIAGNOSIS (CONFIDENTIAL, never reveal or hint at): {case.true_diagnosis}")
    return "\n".join(lines)


GATEKEEPER_SYSTEM = (
    "You are the Gatekeeper in a sequential-diagnosis simulation. You hold the full case file, "
    "INCLUDING the true diagnosis. A clinician asks a history/exam question or orders a specific test; "
    "return only that finding, as a chart or result would read, medically consistent with THIS "
    "patient. Rules:\n"
    "1. ANSWER EVERY SPECIFIC REQUEST. Resolve synonyms/abbreviations ('LP'='lumbar puncture'; "
    "'CBC'='complete blood count'; 'head CT'='CT of the brain'). A specifically named test or question "
    "is ALWAYS answered — NEVER refuse it, even if it targets a specific disease or organism (ordering "
    "a targeted test is normal practice; just report its result).\n"
    "2. IF THE FILE STATES IT: report the finding EXACTLY — copy values, units, and descriptors "
    "verbatim; do not paraphrase or invent numbers. IF NOT IN THE FILE: give ONE brief result that is "
    "medically correct for THIS case — if the ordered test would confirm or exclude this patient's "
    "condition, report that plainly (a positive/negative or the relevant value); prefix any value you "
    "supply that is not in the file with '[simulated] '.\n"
    "3. Report the objective FINDING only. Do NOT state or interpret the final diagnosis by name, and "
    "do NOT volunteer anything not asked for. (A test the clinician ordered may itself be diagnostic — "
    "that is expected; report the result, not the diagnosis label.)\n"
    "4. REFUSE only a vague or shotgun request ('labs', 'imaging', 'work it up'): reply 'REFUSE: name "
    "a specific test.'\n"
    "5. Answer in 1-2 lines, objective findings only."
)


class LLMGatekeeper:
    """SDBench-style Gatekeeper backed by an LLM reading the case file.

    Same interface as `Gatekeeper`, but reveals findings by querying a model rather than a
    rule-based lookup, so it generalizes to narrative cases (e.g. NEJM CPCs).
    """

    def __init__(self, case, backend: str | None = None):
        self.case = case
        self.backend = backend
        # narrative cases (e.g. CPCs) carry their own case_file; EHR cases render one
        self.case_file = getattr(case, "case_file", None) or _case_file(case)
        self._rule = Gatekeeper(case)   # reused only for the cost lookup

    def _cost(self, order_name: str) -> float:
        panel = self._rule._resolve_panel(order_name)
        return panel.cost if panel else estimate_test_cost(order_name)

    def _ask_llm(self, request: str) -> str:
        from shared import llm
        messages = [
            {"role": "system", "content": GATEKEEPER_SYSTEM},
            {"role": "user", "content": f"CASE FILE:\n{self.case_file}\n\nAGENT REQUEST: {request}\n\nResult:"},
        ]
        try:
            # think=False: the gatekeeper extracts a finding from the case file; it must not
            # spend a reasoning-model's token budget on hidden thinking (which returns empty).
            return llm.chat(messages, backend=self.backend, temperature=0.0,
                            max_tokens=160, think=False).strip()
        except Exception as exc:
            return f"(gatekeeper unavailable: {exc})"

    def order_test(self, order_name: str) -> tuple[str, float, bool]:
        reply = self._ask_llm(f"Order this test: {order_name}")
        if reply.upper().startswith("REFUSE"):
            return (f"[Gatekeeper] {reply}", 0.0, False)
        return (reply, self._cost(order_name), True)

    def ask_question(self, text: str) -> str:
        return self._ask_llm(f"Question about the patient history/exam: {text}")


# --- Judge: scores the committed diagnosis against ICD ground truth ----------

def judge_diagnosis(case: EHRCase, candidate: str) -> tuple[bool, str]:
    """Toy stand-in for SDBench's 5-point LLM rubric Judge.

    Real SDBench uses an o3 Judge with a physician-authored Likert rubric
    (>=4 = correct). Here we do a transparent keyword/synonym match so the demo
    runs offline; the LLM rubric can be swapped in via shared/llm.py.
    """
    truth = case.true_diagnosis.lower()
    cand = candidate.lower()
    # Match on salient disease tokens (drop generic words).
    stop = {"of", "the", "with", "without", "due", "to", "and", "other", "acute", "type"}
    truth_tokens = {t.strip(",.()[]") for t in truth.split()} - stop
    hits = [t for t in truth_tokens if len(t) > 3 and t in cand]
    correct = len(hits) >= max(1, len(truth_tokens) // 3)
    return correct, f"matched tokens={hits} vs principal dx='{case.true_diagnosis}'"


# --- Environment: the sequential loop ---------------------------------------

@dataclass
class StepResult:
    turn: int
    action: str
    response: str
    cost: float
    done: bool


@dataclass
class SequentialDiagnosisEnv:
    case: EHRCase
    ordered: set[str] = field(default_factory=set)
    asked: list[str] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)
    total_cost: float = 0.0
    turn: int = 0
    done: bool = False
    final_diagnosis: str | None = None
    llm_gatekeeper: bool = False          # use the LLM-backed gatekeeper instead of rules
    backend: str | None = None            # LLM backend for the gatekeeper (None = auto)
    _visit_open: bool = False
    gatekeeper: object = field(init=False)

    def __post_init__(self) -> None:
        self.gatekeeper = (LLMGatekeeper(self.case, self.backend)
                           if self.llm_gatekeeper else Gatekeeper(self.case))

    # --- observation the agent sees before acting ---------------------------
    def observation(self) -> str:
        # The agent starts from the full clinical presentation (history + physical/neuro exam)
        # when available; the gatekeeper still withholds labs/imaging/path + the diagnosis.
        opening = (getattr(self.case, "presentation", "") or "").strip() or self.case.abstract
        lines = [f"Presenting case:\n{opening}"]
        if self.transcript:
            lines.append("Findings revealed so far:")
            lines += [f"  - {t}" for t in self.transcript]
        else:
            lines.append("Findings revealed so far: (none)")
        lines.append(f"Cumulative cost: ${self.total_cost:.0f}")
        return "\n".join(lines)

    def orderable_tests(self) -> list[str]:
        return [p for p in self.case.panels if p not in self.ordered]

    # --- actions ------------------------------------------------------------
    def ask_question(self, text: str) -> StepResult:
        self._guard()
        self.turn += 1
        cost = 0.0
        if not self._visit_open:           # a burst of questions = one visit
            cost = VISIT_COST
            self.total_cost += cost
            self._visit_open = True
        answer = self.gatekeeper.ask_question(text)
        self.asked.append(text)
        self.transcript.append(f"Q: {text} -> {answer}")
        return StepResult(self.turn, f"ask:{text}", answer, cost, False)

    def order_test(self, name: str) -> StepResult:
        self._guard()
        self.turn += 1
        self._visit_open = False           # ordering a test closes the visit
        finding, cost, _ = self.gatekeeper.order_test(name)
        self.total_cost += cost
        if name not in self.ordered:
            self.ordered.add(name)
        self.transcript.append(f"Test [{name}] -> {finding}")
        return StepResult(self.turn, f"order:{name}", finding, cost, False)

    def diagnose(self, dx: str) -> StepResult:
        self._guard()
        self.turn += 1
        self.done = True
        self.final_diagnosis = dx
        correct, detail = judge_diagnosis(self.case, dx)
        verdict = f"Judge: {'CORRECT' if correct else 'INCORRECT'} | {detail}"
        self.transcript.append(f"Diagnosis: {dx} -> {verdict}")
        return StepResult(self.turn, f"diagnose:{dx}", verdict, 0.0, True)

    def _guard(self) -> None:
        if self.done:
            raise RuntimeError("Episode is over.")


def describe_split(case: EHRCase) -> None:
    """Show how a case is split: the hidden full record vs the visible abstract."""
    print(f"CASE {case.case_id}  ({case.patient.gender}, {case.patient.anchor_age}y)")
    print("-" * 72)
    print("Full record held by the Gatekeeper (HIDDEN from the agent):")
    print(f"   true diagnosis : {case.true_diagnosis}")
    print(f"   history topics : {', '.join(case.hpi)}")
    print(f"   labs on file   : {', '.join(case.labevents)}")
    print(f"   imaging on file: {', '.join(case.radiology) or '(none)'}")
    print(f"   orderable tests: {', '.join(case.panels)}")
    print("\nWhat the agent SEES at step 0 (only the abstract):")
    print(f"   {case.abstract}")
    print("   -> it must request each finding, paying a cost, to learn the rest.")


def _show(step: StepResult) -> None:
    head = step.action if len(step.action) < 40 else step.action[:39] + "…"
    print(f"  [turn {step.turn:>2}] {head:<42} +${step.cost:>5.0f}")
    for line in step.response.splitlines():
        print(f"            {line}")


def main() -> None:
    case = load_case("PE-2180")
    env = SequentialDiagnosisEnv(case=case)

    print("=" * 80)
    print(f"CASE {case.case_id}  subject_id={case.patient.subject_id}  "
          f"({case.patient.gender}, {case.patient.anchor_age}y)   [true dx hidden]")
    print("=" * 80)
    print(env.observation())
    print("\nOrderable tests:", env.orderable_tests(), "\n")

    # A hand-scripted trajectory; Demo 02 replaces this with a model.
    _show(env.ask_question("When did the chest pain start and what are the risk factors?"))
    _show(env.order_test("D-Dimer"))
    _show(env.order_test("CT pulmonary angiogram"))
    _show(env.diagnose("Acute pulmonary embolism"))

    print(f"\nSummary: turns={env.turn}  total_cost=${env.total_cost:.0f}  "
          f"key_tests_ordered={case.confirmable_by(env.ordered)}")


if __name__ == "__main__":
    main()
