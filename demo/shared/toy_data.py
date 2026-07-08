"""Synthetic, MIMIC-IV-shaped patient records shared across the demos.

These are hand-authored toy cases (NOT real patients) used to build an
SDBench-style sequential diagnosis environment over *structured EHR data*,
mirroring Nori et al. (2025, "Sequential Diagnosis with Language Models") but
sourcing findings from EHR tables instead of NEJM narratives.

Real MIMIC-IV requires PhysioNet credentialing + a data use agreement, so these
toy records imitate the relevant MIMIC-IV table structure
(`patients`, `labevents` + `d_labitems`, `microbiologyevents`, `radiology`,
`diagnoses_icd` + `d_icd_diagnoses`). Swapping in real MIMIC later is a
data-loader change, not a redesign: the environment only depends on the
dataclasses below, not on how they are populated.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- MIMIC-IV-shaped record elements ----------------------------------------

@dataclass(frozen=True)
class Patient:
    subject_id: int
    gender: str          # "M" / "F"  (MIMIC patients.gender)
    anchor_age: int      # MIMIC patients.anchor_age


@dataclass(frozen=True)
class LabEvent:
    """One row of `labevents` joined to `d_labitems`."""
    label: str           # d_labitems.label, e.g. "Hematocrit"
    category: str        # d_labitems.category, e.g. "Hematology"
    value: str           # labevents.value (display string)
    valueuom: str        # units
    ref_lower: float | None
    ref_upper: float | None
    flag: str = ""       # "H" / "L" / "" (derived; MIMIC stores 'abnormal')


@dataclass(frozen=True)
class MicroEvent:
    """One `microbiologyevents` result."""
    spec_type_desc: str  # e.g. "BLOOD CULTURE"
    test_name: str       # e.g. "Aerobic bottle"
    org_name: str        # organism or "NO GROWTH"
    comments: str = ""


@dataclass(frozen=True)
class RadiologyStudy:
    """A `radiology` note (free-text report)."""
    modality: str        # "CT", "CXR", "US"
    exam: str            # "CT pulmonary angiogram"
    report: str


@dataclass(frozen=True)
class IcdDiagnosis:
    """A `diagnoses_icd` row joined to `d_icd_diagnoses` (ground truth)."""
    icd_code: str
    icd_version: int
    long_title: str
    seq_num: int         # 1 = principal diagnosis


@dataclass(frozen=True)
class TestPanel:
    """An orderable test the agent can request, and what it returns.

    Groups the structured record into clinically-orderable units (a CBC returns
    several labevents; an imaging order returns one radiology report).
    """
    order_name: str                 # canonical name the agent orders
    kind: str                       # "lab_panel" | "imaging" | "micro"
    cost: float                     # USD (CPT-style); a physician visit is separate
    informative: bool               # does it help reach the true dx? (for analysis)
    lab_labels: tuple[str, ...] = ()      # which labevents (kind == lab_panel)
    imaging_exam: str | None = None       # which radiology study (kind == imaging)
    micro_spec: str | None = None         # which microbiologyevents (kind == micro)
    aliases: tuple[str, ...] = ()         # accepted synonyms in free-text orders


@dataclass(frozen=True)
class EHRCase:
    """A full toy case: the structured record + the gateable history."""
    case_id: str
    patient: Patient
    abstract: str                       # SDBench-style opening vignette
    hpi: dict[str, str]                 # question-topic -> answer (revealed by asking)
    exam: str                           # physical exam (revealed by asking)
    labevents: dict[str, LabEvent]      # label -> lab row
    micro: dict[str, MicroEvent]        # spec_type_desc -> result
    radiology: dict[str, RadiologyStudy]  # exam -> report
    panels: dict[str, TestPanel]        # order_name -> orderable test
    diagnoses_icd: tuple[IcdDiagnosis, ...]
    key_tests: tuple[str, ...]          # panels that confirm the principal dx

    @property
    def true_diagnosis(self) -> str:
        principal = min(self.diagnoses_icd, key=lambda d: d.seq_num)
        return principal.long_title

    def confirmable_by(self, ordered: set[str]) -> bool:
        return set(self.key_tests).issubset(ordered)


def _flagged(value_lower: float, lo: float, hi: float) -> str:
    if value_lower > hi:
        return "H"
    if value_lower < lo:
        return "L"
    return ""


# --- Case 1: pulmonary embolism ---------------------------------------------

def _case_pe() -> EHRCase:
    labs = {
        "D-Dimer": LabEvent("D-Dimer", "Chemistry", "3.42", "ug/mL FEU", 0.0, 0.5, "H"),
        "Hematocrit": LabEvent("Hematocrit", "Hematology", "44.1", "%", 41.0, 53.0, ""),
        "White Blood Cells": LabEvent("White Blood Cells", "Hematology", "9.8", "K/uL", 4.0, 11.0, ""),
        "Platelet Count": LabEvent("Platelet Count", "Hematology", "230", "K/uL", 150, 440, ""),
        "Troponin T": LabEvent("Troponin T", "Chemistry", "0.04", "ng/mL", 0.0, 0.01, "H"),
        "Creatinine": LabEvent("Creatinine", "Chemistry", "1.0", "mg/dL", 0.6, 1.2, ""),
    }
    micro = {}
    rad = {
        "CT pulmonary angiogram": RadiologyStudy(
            "CT", "CT pulmonary angiogram",
            "Filling defect in the right lower lobe segmental pulmonary artery consistent "
            "with acute pulmonary embolism. Right heart strain not present.",
        ),
        "Chest radiograph": RadiologyStudy(
            "CXR", "Chest radiograph",
            "No focal consolidation, effusion, or pneumothorax. Heart size normal.",
        ),
    }
    panels = {
        "CBC": TestPanel("CBC", "lab_panel", 30.0, False,
                         lab_labels=("Hematocrit", "White Blood Cells", "Platelet Count"),
                         aliases=("complete blood count",)),
        "D-Dimer": TestPanel("D-Dimer", "lab_panel", 40.0, True, lab_labels=("D-Dimer",),
                             aliases=("d dimer", "ddimer")),
        "Troponin": TestPanel("Troponin", "lab_panel", 40.0, False, lab_labels=("Troponin T",),
                              aliases=("troponin t",)),
        "BMP": TestPanel("BMP", "lab_panel", 30.0, False, lab_labels=("Creatinine",),
                         aliases=("basic metabolic panel",)),
        "CT pulmonary angiogram": TestPanel("CT pulmonary angiogram", "imaging", 500.0, True,
                                            imaging_exam="CT pulmonary angiogram",
                                            aliases=("ctpa", "ct pa", "ct chest with contrast", "cta chest")),
        "Chest radiograph": TestPanel("Chest radiograph", "imaging", 60.0, False,
                                      imaging_exam="Chest radiograph",
                                      aliases=("cxr", "chest x-ray", "chest xray")),
    }
    return EHRCase(
        case_id="PE-2180",
        patient=Patient(subject_id=10002180, gender="F", anchor_age=52),
        abstract=("A 52-year-old woman presents with sudden pleuritic chest pain and dyspnea "
                  "three days after a long-haul flight."),
        hpi={
            "onset": "Pain began abruptly this morning and worsens with deep inspiration.",
            "risk factors": "Recent 11-hour flight; on combined oral contraceptives; no prior clots.",
            "associated symptoms": "Mild calf discomfort on the right; no fever, no hemoptysis.",
            "vitals": "HR 110, BP 128/82, RR 22, SpO2 91% on room air, T 37.1 C.",
        },
        exam="Tachycardic, clear lungs, right calf mildly tender without swelling.",
        labevents=labs,
        micro=micro,
        radiology=rad,
        panels=panels,
        diagnoses_icd=(
            IcdDiagnosis("I2699", 10, "Other pulmonary embolism without acute cor pulmonale", 1),
        ),
        key_tests=("CT pulmonary angiogram",),
    )


# --- Case 2: diabetic ketoacidosis ------------------------------------------

def _case_dka() -> EHRCase:
    labs = {
        "Glucose": LabEvent("Glucose", "Chemistry", "486", "mg/dL", 70, 100, "H"),
        "Bicarbonate": LabEvent("Bicarbonate", "Chemistry", "12", "mEq/L", 22, 32, "L"),
        "Anion Gap": LabEvent("Anion Gap", "Chemistry", "24", "mEq/L", 8, 16, "H"),
        "Potassium": LabEvent("Potassium", "Chemistry", "5.4", "mEq/L", 3.5, 5.1, "H"),
        "pH": LabEvent("pH", "Blood Gas", "7.21", "units", 7.35, 7.45, "L"),
        "Beta-Hydroxybutyrate": LabEvent("Beta-Hydroxybutyrate", "Chemistry", "5.8", "mmol/L", 0.0, 0.4, "H"),
        "White Blood Cells": LabEvent("White Blood Cells", "Hematology", "13.1", "K/uL", 4.0, 11.0, "H"),
    }
    rad = {
        "Chest radiograph": RadiologyStudy(
            "CXR", "Chest radiograph", "No acute cardiopulmonary process."),
    }
    panels = {
        "BMP": TestPanel("BMP", "lab_panel", 30.0, True,
                         lab_labels=("Glucose", "Bicarbonate", "Anion Gap", "Potassium"),
                         aliases=("basic metabolic panel", "chem 7", "electrolytes")),
        "ABG": TestPanel("ABG", "lab_panel", 60.0, True, lab_labels=("pH",),
                         aliases=("arterial blood gas", "blood gas")),
        "Ketones": TestPanel("Ketones", "lab_panel", 20.0, True, lab_labels=("Beta-Hydroxybutyrate",),
                             aliases=("serum ketones", "beta-hydroxybutyrate", "bhb")),
        "CBC": TestPanel("CBC", "lab_panel", 30.0, False, lab_labels=("White Blood Cells",),
                         aliases=("complete blood count",)),
        "Chest radiograph": TestPanel("Chest radiograph", "imaging", 60.0, False,
                                      imaging_exam="Chest radiograph", aliases=("cxr", "chest x-ray")),
    }
    return EHRCase(
        case_id="DKA-3391",
        patient=Patient(subject_id=10003391, gender="M", anchor_age=19),
        abstract=("A 19-year-old man with type 1 diabetes presents with one day of nausea, "
                  "vomiting, abdominal pain, and deep rapid breathing."),
        hpi={
            "onset": "Symptoms began yesterday; he ran out of insulin four days ago.",
            "risk factors": "Type 1 diabetes since age 11; recent viral illness.",
            "associated symptoms": "Polyuria, polydipsia, fruity breath odor.",
            "vitals": "HR 118, BP 104/68, RR 28 deep, SpO2 99%, T 37.4 C.",
        },
        exam="Dry mucous membranes, Kussmaul respirations, diffuse mild abdominal tenderness.",
        labevents=labs,
        micro={},
        radiology=rad,
        panels=panels,
        diagnoses_icd=(
            IcdDiagnosis("E1010", 10, "Type 1 diabetes mellitus with ketoacidosis without coma", 1),
        ),
        key_tests=("BMP", "Ketones"),
    )


# --- Case 3: urosepsis (exercises microbiology) -----------------------------

def _case_urosepsis() -> EHRCase:
    labs = {
        "White Blood Cells": LabEvent("White Blood Cells", "Hematology", "17.6", "K/uL", 4.0, 11.0, "H"),
        "Lactate": LabEvent("Lactate", "Blood Gas", "3.1", "mmol/L", 0.5, 2.0, "H"),
        "Creatinine": LabEvent("Creatinine", "Chemistry", "1.8", "mg/dL", 0.6, 1.2, "H"),
    }
    micro = {
        "URINE": MicroEvent("URINE", "Urine Culture", "ESCHERICHIA COLI > 100,000 CFU/mL",
                            "Pan-sensitive on susceptibility panel."),
        "BLOOD CULTURE": MicroEvent("BLOOD CULTURE", "Aerobic bottle", "ESCHERICHIA COLI",
                                    "Positive at 14 hours; same organism as urine."),
    }
    rad = {
        "CT abdomen and pelvis": RadiologyStudy(
            "CT", "CT abdomen and pelvis",
            "Right-sided hydronephrosis with an obstructing 7 mm distal ureteral calculus; "
            "perinephric stranding consistent with pyelonephritis."),
    }
    panels = {
        "CBC": TestPanel("CBC", "lab_panel", 30.0, True, lab_labels=("White Blood Cells",),
                         aliases=("complete blood count",)),
        "Lactate": TestPanel("Lactate", "lab_panel", 25.0, True, lab_labels=("Lactate",)),
        "BMP": TestPanel("BMP", "lab_panel", 30.0, False, lab_labels=("Creatinine",),
                         aliases=("basic metabolic panel",)),
        "Urine culture": TestPanel("Urine culture", "micro", 80.0, True, micro_spec="URINE",
                                   aliases=("urine cx", "urine culture and sensitivity")),
        "Blood culture": TestPanel("Blood culture", "micro", 90.0, True, micro_spec="BLOOD CULTURE",
                                   aliases=("blood cx",)),
        "CT abdomen and pelvis": TestPanel("CT abdomen and pelvis", "imaging", 350.0, True,
                                           imaging_exam="CT abdomen and pelvis",
                                           aliases=("ct abdomen", "ct ap", "ct abd/pelvis")),
    }
    return EHRCase(
        case_id="URO-4477",
        patient=Patient(subject_id=10004477, gender="F", anchor_age=64),
        abstract=("A 64-year-old woman presents with fever, right flank pain, and confusion. "
                  "She appears acutely ill."),
        hpi={
            "onset": "Two days of dysuria and frequency, now with rigors and flank pain.",
            "risk factors": "Recurrent urinary tract infections; type 2 diabetes.",
            "associated symptoms": "Nausea; no vaginal symptoms; reduced urine output.",
            "vitals": "HR 122, BP 88/54, RR 24, SpO2 95%, T 39.3 C.",
        },
        exam="Ill-appearing, right costovertebral angle tenderness, mild generalized confusion.",
        labevents=labs,
        micro=micro,
        radiology=rad,
        panels=panels,
        diagnoses_icd=(
            IcdDiagnosis("A4151", 10, "Sepsis due to Escherichia coli [E. coli]", 1),
            IcdDiagnosis("N136", 10, "Pyonephrosis", 2),
        ),
        key_tests=("Urine culture", "Blood culture"),
    )


CASES: dict[str, EHRCase] = {
    c.case_id: c for c in (_case_pe(), _case_dka(), _case_urosepsis())
}


def load_case(case_id: str = "PE-2180") -> EHRCase:
    return CASES[case_id]


def all_cases() -> list[EHRCase]:
    return list(CASES.values())
