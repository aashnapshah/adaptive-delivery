"""Persist recommender runs, and the metrics computed on them, so we can track many iterations.

  results/raw/runs.jsonl    - one JSON per run: the heavy CONTENT only (ordered, transcript, conversation), keyed by run_id.
  results/raw/runs.csv      - one flat row per run: generation METADATA + the eval SCORE columns, keyed by run_id.
  results/raw/prompts.json  - prompt fingerprint -> the prompt text behind it, stored once per config.

runs.jsonl and runs.csv are joined by `run_id` - every field lives in exactly one of them, no duplication.
Metadata is the flat, tabular half (read directly by tables.py / plots.py); the content is the nested half
(transcript + conversation). `load_runs()` merges the two back into a full record for the code that needs both.
The prompts that produced a run are kept once in prompts.json, keyed by `prompt_fp`, so an old run stays
readable (and reproducible) after its prompts have been edited.

Generation writes runs (metadata + content); the eval drivers score them into the runs.csv SCORE columns:
`python -m harness.judge` fills the LLM metrics, `python -m harness.evaluation.eval` the deterministic
ones. A metric already filled for a run is skipped (unless --force), so scoring resumes cheaply.

Every run records a CONFIG - design, model, a fingerprint of the prompt files it used, and any
hyperparameters - hashed into a short `config_id`. Editing a prompt changes the fingerprint, so it becomes
a NEW config_id: old runs stay attributable and new iterations don't collide with them. `done()` lists
the generated cells and `load_scores()` the metrics already filled, so an interrupted generate or score run resumes.
"""

import csv
import datetime
import glob
import hashlib
import json
import os
import re

from ..shared import llm

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # harness/ (this file is in store/)
ROOT = os.path.dirname(HARNESS)                          # repo root
RAW = os.path.join(ROOT, "results", "raw")              # runs.jsonl + runs.csv + prompts.json
JSONL = os.path.join(RAW, "runs.jsonl")                 # run content (ordered/transcript/conversation)
CSV = os.path.join(RAW, "runs.csv")                     # run metadata + eval score columns
PROMPTS = os.path.join(RAW, "prompts.json")             # prompt text behind each prompt_fp

# runs.csv is the flat per-run table: generation METADATA plus the eval SCORE columns (filled later by
# the drivers), keyed by run_id (the join key to runs.jsonl content). Each metric owns a score group.
METADATA_COLUMNS = ["ts", "run_id", "config_id", "case_id", "source", "design", "model", "prompt_fp",
                    "tag", "note", "specialty", "turns", "calls", "n_tests", "tokens", "deliberate_s",
                    "stopped", "diagnosis", "true_dx",
                    "human_tests", "llm_tests"]          # the two work-ups, "; "-joined, for side-by-side
DIAGNOSIS_COLS = ["judge_score", "judge_correct", "judge_reason"]         # judge.py (LLM), CPC only
WORKUP_COLS = ["workup_better", "workup_reason"]                          # judge.py (LLM), needs a reference
AGREEMENT_COLS = ["overlap", "missed", "extra", "f1", "jaccard"]          # judge.py (LLM): work-up vs human set overlap
COST_COLS = ["cost_total"]                                                # eval.py (deterministic)
SCORE_FIELDS = DIAGNOSIS_COLS + WORKUP_COLS + AGREEMENT_COLS + COST_COLS
COLUMNS = METADATA_COLUMNS + SCORE_FIELDS

# map a diagnosis to a broad specialty, for breakdowns
_SPECIALTY = [
    ("Infectious", r"infect|sepsis|abscess|meningitis|encephalitis|pneumon|endocarditis|viral|bacter|fungal|hepatitis|cellulitis|tuberculos"),
    ("Onc/Heme", r"lymphoma|leukemia|carcinoma|sarcoma|cancer|tumor|myeloma|metasta|neoplas|melanoma|malignan"),
    ("Cardiovascular", r"cardi|myocard|heart|aortic|vascul|embol|infarct|arrhythm|valv|pericard"),
    ("Neurology", r"encephal|neuro|stroke|seizure|sclerosis|myelo|cerebr|spinal|parkinson|dementia"),
    ("Renal", r"renal|nephr|kidney|glomerul"),
    ("GI/Hepatic", r"hepat|liver|pancreat|colitis|bowel|gastr|crohn|biliary|cirrhos"),
    ("Pulmonary", r"pulmonar|lung|pleural|respiratory|asthma|copd|sarcoidosis"),
    ("Endocrine", r"diabet|thyroid|adrenal|pituitary|endocr|ketoacidosis"),
    ("Rheum/Immune", r"vasculitis|lupus|arthritis|autoimmune|amyloid|rheum|immune|granulomato"),
]


def specialty(dx):
    """The broad specialty a diagnosis falls under (for breakdowns)."""
    d = (dx or "").lower()
    for name, pattern in _SPECIALTY:
        if re.search(pattern, d):
            return name
    return "Other"


def prompt_files(design):
    """The prompt files that shape a run: gatekeeper + judge + the recommender design."""
    files = [os.path.join(HARNESS, "gatekeeper/prompts/gatekeeper.txt"),
             os.path.join(HARNESS, "judge/prompts/diagnosis.txt")]
    prompts = os.path.join(HARNESS, "recommender/prompts")
    files += (sorted(glob.glob(os.path.join(prompts, "debate", "*.txt"))) if design == "debate"
              else [os.path.join(prompts, f"{design}.txt")])
    return files


def prompt_fp(design):
    """8-char hash of the prompt files that shape a run. Changes whenever any of them is edited."""
    h = hashlib.sha256()
    for path in prompt_files(design):
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:8]


def load_prompts():
    """fingerprint -> {prompt file: its text}, for every config ever run."""
    if os.path.exists(PROMPTS):
        with open(PROMPTS) as f:
            return json.load(f)
    return {}


def save_prompts(design, fp):
    """Store the actual prompt TEXT behind a fingerprint, ONCE, in results/raw/prompts.json.

    A run records only its `prompt_fp`, so the log stays data-efficient - but the text is kept here,
    which means an old run is still readable (and reproducible) after the prompts have been edited.
    """
    store = load_prompts()
    if fp in store:
        return
    os.makedirs(RAW, exist_ok=True)
    store[fp] = {os.path.relpath(p, HARNESS): open(p).read() for p in prompt_files(design)}
    with open(PROMPTS, "w") as f:
        json.dump(store, f, indent=1)


def resolve_model(model):
    """The actual model id: `model` if given, else the backend's default (e.g. gemini-3.5-flash)."""
    if model:
        return model
    name = llm.active_model()                       # "gemini:gemini-3.5-flash" | "stub (offline)"
    return name.split(":", 1)[1].strip() if ":" in name else name


def config(design, model, params=None):
    """The full spec that produced a run: design, model, prompt fingerprint, and any hyperparameters.
    `config_id` is a short hash of all of it - a stable key for grouping runs and for resume."""
    cfg = {"design": design, "model": resolve_model(model), "prompt_fp": prompt_fp(design),
           "params": params or {}}
    cfg["config_id"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    return cfg


def _row(run_id, ts, cfg, case_id, source, true_dx, human_tests, meta, ordered, stopped, diagnosis):
    """The flat metadata row (runs.csv) for one run - the tabular half of a run, keyed by run_id; the
    content (transcript/conversation) lives in runs.jsonl. `tag`/`note` record WHICH experiment a run
    belongs to and what was different about it - code-level changes (a token budget, a bug fix) don't
    move prompt_fp, so without a tag a re-run would collide with the old runs instead of forking."""
    params = cfg.get("params") or {}
    return {"ts": ts, "run_id": run_id, "config_id": cfg["config_id"], "case_id": case_id,
            "source": source, "design": cfg["design"], "model": cfg["model"], "prompt_fp": cfg["prompt_fp"],
            "tag": params.get("tag", ""), "note": params.get("note", ""),
            "specialty": specialty(diagnosis), "turns": meta["turns"], "calls": meta.get("calls", ""),
            "n_tests": len(ordered), "tokens": meta["tokens"], "deliberate_s": meta["deliberate_s"],
            "stopped": stopped, "diagnosis": diagnosis or "", "true_dx": true_dx or "",
            "human_tests": human_tests or "", "llm_tests": "; ".join(ordered)}


def _append_csv(row):
    new_file = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def save_run(case, design, model, result, diagnosis=None, params=None):
    """Append a generated run: its content to runs.jsonl, its metadata to runs.csv, joined by run_id.
    Metrics are NOT written here - evaluate.py computes them later. Returns the run_id."""
    os.makedirs(RAW, exist_ok=True)
    cfg = config(design, model, params)
    save_prompts(design, cfg["prompt_fp"])                 # keep the prompt text behind this run readable
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    run_id = f"{cfg['config_id']}-{case.case_id}"          # the (config, case) cell
    meta = result["meta"]

    content = {"run_id": run_id, "ordered": result["ordered"], "transcript": result["transcript"],
               "conversation": result["conversation"]}    # the heavy/nested half; metadata goes to runs.csv
    with open(JSONL, "a") as f:
        f.write(json.dumps(content) + "\n")

    _append_csv(_row(run_id, ts, cfg, case.case_id, case.source, case.true_diagnosis,
                     "; ".join(case.human_seq), meta, result["ordered"], result["stopped"], diagnosis))
    return run_id


def done():
    """The (config_id, case_id) pairs already generated - so an interrupted sweep skips them."""
    if not os.path.exists(CSV):
        return set()
    with open(CSV) as f:
        return {(r["config_id"], r["case_id"]) for r in csv.DictReader(f)}


def _num(v, cast):
    """Cast a CSV string back to int/float; leave it as-is if blank/unparseable."""
    try:
        return cast(v)
    except (TypeError, ValueError):
        return v


def _record(m, content):
    """Rebuild a full run record by joining a metadata row (runs.csv) with its content (runs.jsonl).
    Reconstructs the nested `config`/`meta` shape the rest of the code expects, so consumers are unchanged."""
    return {
        "run_id": m["run_id"], "ts": m["ts"], "case_id": m["case_id"], "source": m["source"],
        "true_dx": m["true_dx"], "stopped": m["stopped"], "diagnosis": m["diagnosis"] or None,
        "config": {"config_id": m["config_id"], "design": m["design"], "model": m["model"],
                   "prompt_fp": m["prompt_fp"], "params": {"tag": m["tag"], "note": m["note"]}},
        "meta": {"turns": _num(m["turns"], int), "calls": _num(m["calls"], int),
                 "tokens": _num(m["tokens"], int), "deliberate_s": _num(m["deliberate_s"], float)},
        "ordered": content.get("ordered", []), "transcript": content.get("transcript", []),
        "conversation": content.get("conversation", []),
    }


def _load_meta():
    """runs.csv -> {run_id: metadata row} (latest wins if a cell was re-run)."""
    if not os.path.exists(CSV):
        return {}
    with open(CSV) as f:
        return {r["run_id"]: r for r in csv.DictReader(f)}


def load_runs():
    """Every generated run, keyed by run_id: its metadata (runs.csv) joined to its content (runs.jsonl)
    on run_id. Latest wins if a cell was re-run."""
    meta = _load_meta()
    runs = {}
    if os.path.exists(JSONL):
        with open(JSONL) as f:
            for line in f:
                c = json.loads(line)
                m = meta.get(c["run_id"])
                if m is not None:                          # a run needs both halves; skip a stray content line
                    runs[c["run_id"]] = _record(m, c)
    return runs


def select_runs(runs, source="", designs="", limit=None):
    """The runs to score, from a {run_id: run} map: filtered by `source` and a comma-separated `designs`
    list (empty = no filter), and capped to the first `limit` distinct CASES (all of each case's runs) -
    so `--limit 5` scores 5 cases, not 5 runs. Returns a list of (run_id, run). Shared by the eval drivers."""
    want = {d.strip() for d in designs.split(",") if d.strip()}
    items = [(rid, r) for rid, r in runs.items()
             if (not source or r["source"] == source)
             and (not want or r["config"]["design"] in want)]
    if limit:
        keep = set(list(dict.fromkeys(r["case_id"] for _, r in items))[:limit])
        items = [(rid, r) for rid, r in items if r["case_id"] in keep]
    return items


def load_scores():
    """{run_id: {score column: value}} - the metric columns already filled for each run, read from the
    SCORE columns of runs.csv. A missing key means that metric hasn't been scored yet, so a driver can
    skip it or fill it."""
    scores = {}
    if os.path.exists(CSV):
        with open(CSV) as f:
            for r in csv.DictReader(f):
                filled = {c: r[c] for c in SCORE_FIELDS if r.get(c, "") != ""}
                if filled:
                    scores[r["run_id"]] = filled
    return scores


def update_scores(updates):
    """Fill score columns straight into runs.csv. `updates` is {run_id: {column: value}}: the given
    columns overwrite, the run's metadata and other scores are kept. Reads runs.csv, merges, rewrites -
    per-run persistence so an interrupted scoring pass resumes."""
    if not updates:
        return
    with open(CSV) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        cols = updates.get(r["run_id"])
        if cols:
            r.update({k: v for k, v in cols.items() if v is not None})
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def as_bool(v):
    """A CSV/DataFrame boolean (round-tripped as 'True'/'False') parsed back to bool; blank/NaN/None -> None."""
    s = str(v).strip().lower()
    return None if s in ("", "nan", "none") else s == "true"
