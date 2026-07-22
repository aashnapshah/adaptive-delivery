"""Unified case schema shared across the pipeline (stage 01).

Every case source - a real MIMIC-IV admission, an NEJM CPC narrative, or a hand-authored
toy EHR record - is loaded into ONE `Case` dataclass, tagged by `source`. The two things that
differ between sources are how the presentation is built and how findings are withheld, so the
`Case` carries both representations and the gatekeeper (stage 02) picks the right one:

  - CPC        source="cpc"   : a static vignette. The whole case is prose in `case_file`; the
                                LLM gatekeeper reads it and discloses findings on request.
  - MIMIC      source="mimic" : a hospital admission. Thin intake `presentation`; the real
                                structured lab results live in `findings` and are revealed only
                                when the matching test is ordered (labs-on-order). `orders` is the
                                real order set, for the ordering-concordance metric.
  - toy        source="toy"   : synthetic MIMIC-shaped record with rich `panels`/`labevents`/
                                `micro`/`radiology` and ground-truth `informative`/`key_tests`
                                flags used by the scoring/policy stages.

This replaces the old split between `EHRCase` (toy_data) and `CPCCase` (cpc_data).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- MIMIC-IV-shaped structured record elements (toy source) -----------------

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
class TestPanel:
    """An orderable test the agent can request, and what it returns.

    Groups the structured record into clinically-orderable units (a CBC returns several
    labevents; an imaging order returns one radiology report).
    """
    order_name: str                 # canonical name the agent orders
    kind: str                       # "lab_panel" | "imaging" | "micro"
    cost: float                     # USD (CPT-style); a physician visit is separate
    informative: bool               # does it help reach the true dx? (for analysis)
    lab_labels: tuple[str, ...] = ()      # which labevents (kind == lab_panel)
    imaging_exam: str | None = None       # which radiology study (kind == imaging)
    micro_spec: str | None = None         # which microbiologyevents (kind == micro)
    aliases: tuple[str, ...] = ()         # accepted synonyms in free-text orders


# --- the unified case --------------------------------------------------------

@dataclass
class Case:
    """One diagnostic case, from any source. See module docstring for the source split."""

    case_id: str
    source: str                              # "mimic" | "cpc" | "toy"
    true_diagnosis: str                      # ground truth for the Judge
    presentation: str = ""                   # turn-0 intake: history + vitals the agent starts from
    abstract: str = ""                       # one-line vignette (headers / labels)

    # --- narrative path (cpc): the vignette the LLM gatekeeper reads ---
    case_file: str = ""                      # full narrative incl. the confidential dx line

    # --- structured labs-on-order path (mimic): order_name -> result display string ---
    findings: dict[str, str] = field(default_factory=dict)
    orders: tuple[str, ...] = ()             # real order set (concordance metric)
    finding_aliases: dict[str, str] = field(default_factory=dict)   # alias -> canonical order name

    # --- rich structured record (toy): panels + underlying tables ---
    panels: dict[str, TestPanel] = field(default_factory=dict)
    labevents: dict[str, LabEvent] = field(default_factory=dict)
    micro: dict[str, MicroEvent] = field(default_factory=dict)
    radiology: dict[str, RadiologyStudy] = field(default_factory=dict)
    hpi: dict[str, str] = field(default_factory=dict)   # ask-topic -> answer (revealed by asking)
    exam: str = ""                                       # physical exam (revealed by asking)
    patient: Patient | None = None
    key_tests: tuple[str, ...] = ()          # panels that confirm the principal dx (toy analysis)

    @property
    def is_narrative(self) -> bool:
        """CPC vignettes are narrative; MIMIC/toy are structured (served without narrative LLM disclosure)."""
        return self.source == "cpc"

    @property
    def human_seq(self) -> list:
        """The human ordering the metrics compare against: MIMIC's measured labs, else the narrative
        work-up (CPC `orders`). The single definition of the reference set for concordance + head-to-head."""
        return list(self.findings) if (self.source == "mimic" and self.findings) else list(self.orders)

    def confirmable_by(self, ordered) -> bool:
        """True once every key test has been ordered (toy ground-truth analysis; else False)."""
        return bool(self.key_tests) and set(self.key_tests).issubset(set(ordered))
