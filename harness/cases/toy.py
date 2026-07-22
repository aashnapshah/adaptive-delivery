"""Synthetic, MIMIC-IV-shaped patient records (stage 01, source="toy").

Hand-authored toy cases (NOT real patients) that build an SDBench-style sequential-diagnosis
environment over *structured EHR data*, mirroring Nori et al. (2025) but sourcing findings from
EHR-shaped tables instead of NEJM narratives. They carry the ground-truth `informative` /
`key_tests` flags the scoring and policy stages rely on, so they double as the offline fixtures
for the whole pipeline.

Real MIMIC-IV requires PhysioNet credentialing + a DUA; see `mimic.py` for the real loader. These
records imitate the relevant table structure (`patients`, `labevents`, `microbiologyevents`,
`radiology`) so swapping in real data is a loader change, not a redesign.
"""

from __future__ import annotations

from .schema import Case, LabEvent, MicroEvent, Patient, RadiologyStudy, TestPanel


# --- Case 1: pulmonary embolism ---------------------------------------------

def _case_pe() -> Case:
    labs = {
        "D-Dimer": LabEvent("D-Dimer", "Chemistry", "3.42", "ug/mL FEU", 0.0, 0.5, "H"),
        "Hematocrit": LabEvent("Hematocrit", "Hematology", "44.1", "%", 41.0, 53.0, ""),
        "White Blood Cells": LabEvent("White Blood Cells", "Hematology", "9.8", "K/uL", 4.0, 11.0, ""),
        "Platelet Count": LabEvent("Platelet Count", "Hematology", "230", "K/uL", 150, 440, ""),
        "Troponin T": LabEvent("Troponin T", "Chemistry", "0.04", "ng/mL", 0.0, 0.01, "H"),
        "Creatinine": LabEvent("Creatinine", "Chemistry", "1.0", "mg/dL", 0.6, 1.2, ""),
    }
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
    return Case(
        case_id="PE-2180",
        source="toy",
        true_diagnosis="Other pulmonary embolism without acute cor pulmonale",
        abstract=("A 52-year-old woman presents with sudden pleuritic chest pain and dyspnea "
                  "three days after a long-haul flight."),
        patient=Patient(subject_id=10002180, gender="F", anchor_age=52),
        hpi={
            "onset": "Pain began abruptly this morning and worsens with deep inspiration.",
            "risk factors": "Recent 11-hour flight; on combined oral contraceptives; no prior clots.",
            "associated symptoms": "Mild calf discomfort on the right; no fever, no hemoptysis.",
            "vitals": "HR 110, BP 128/82, RR 22, SpO2 91% on room air, T 37.1 C.",
        },
        exam="Tachycardic, clear lungs, right calf mildly tender without swelling.",
        labevents=labs,
        radiology=rad,
        panels=panels,
        key_tests=("CT pulmonary angiogram",),
    )


# --- Case 2: diabetic ketoacidosis ------------------------------------------

def _case_dka() -> Case:
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
    return Case(
        case_id="DKA-3391",
        source="toy",
        true_diagnosis="Type 1 diabetes mellitus with ketoacidosis without coma",
        abstract=("A 19-year-old man with type 1 diabetes presents with one day of nausea, "
                  "vomiting, abdominal pain, and deep rapid breathing."),
        patient=Patient(subject_id=10003391, gender="M", anchor_age=19),
        hpi={
            "onset": "Symptoms began yesterday; he ran out of insulin four days ago.",
            "risk factors": "Type 1 diabetes since age 11; recent viral illness.",
            "associated symptoms": "Polyuria, polydipsia, fruity breath odor.",
            "vitals": "HR 118, BP 104/68, RR 28 deep, SpO2 99%, T 37.4 C.",
        },
        exam="Dry mucous membranes, Kussmaul respirations, diffuse mild abdominal tenderness.",
        labevents=labs,
        radiology=rad,
        panels=panels,
        key_tests=("BMP", "Ketones"),
    )


# --- Case 3: urosepsis (exercises microbiology) -----------------------------

def _case_urosepsis() -> Case:
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
    return Case(
        case_id="URO-4477",
        source="toy",
        true_diagnosis="Sepsis due to Escherichia coli [E. coli]",
        abstract=("A 64-year-old woman presents with fever, right flank pain, and confusion. "
                  "She appears acutely ill."),
        patient=Patient(subject_id=10004477, gender="F", anchor_age=64),
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
        key_tests=("Urine culture", "Blood culture"),
    )


TOY_CASES: dict[str, Case] = {
    c.case_id: c for c in (_case_pe(), _case_dka(), _case_urosepsis())
}


def load_case(case_id: str = "PE-2180") -> Case:
    return TOY_CASES[case_id]


def all_cases() -> list[Case]:
    return list(TOY_CASES.values())


if __name__ == "__main__":
    for c in all_cases():
        print(f"{c.case_id}  [{c.source}]  dx={c.true_diagnosis!r}  "
              f"panels={len(c.panels)}  key_tests={c.key_tests}")
