"""Cost estimator: what a test costs (a separate component from the gatekeeper).

The gatekeeper reveals findings; it does not price them. Cost is a distinct concern used to evaluate
a workup - the environment tallies running spend during a run, and the scoring/evaluation stages price
the recommended tests. All of that goes through this one module so the prices live in a single place.

`test_cost(case, name)` prefers a case's authored panel cost (toy EHR) and falls back to a category
estimate (CPC narratives, MIMIC labs, any off-protocol order).
"""

from __future__ import annotations

import re

from ..cases.schema import Case

DEFAULT_TEST_COST = 100.0   # unknown / unmatched test

# SDBench-style category list prices, matched most-specific-first. Approximate US list prices.
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
# Words that mark an unmatched order as a blood/serum lab (cheap) rather than a procedure/imaging.
_LAB_HINT = re.compile(r"\b(blood|serum|plasma|urine|level|panel|assay|screen|count|titer|ratio)\b")


def estimate_test_cost(name: str) -> float:
    """Category estimate for a test with no authored cost (CPC / MIMIC / off-protocol)."""
    n = (name or "").lower()
    for pat, cost in _TEST_PRICES:
        if re.search(pat, n):
            return float(cost)
    return 30.0 if _LAB_HINT.search(n) else DEFAULT_TEST_COST   # lab-like unknowns priced as labs


def test_cost(case: Case, name: str) -> float:
    """Cost of ordering `name` in `case`: the authored panel price if the case has one, else an estimate."""
    panel = case.panels.get(name)
    if panel is not None:
        return panel.cost
    return estimate_test_cost(name)


def workup_cost(ordered) -> dict:
    """Total list cost of a whole work-up (deterministic, no LLM). The cost benchmark."""
    return {"n_tests": len(ordered), "cost_total": round(sum(estimate_test_cost(t) for t in ordered))}
