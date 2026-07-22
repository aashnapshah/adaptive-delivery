"""Summary statistics for the harness: one script that regenerates every table AND figure from runs.csv.

    python -m harness.report.summ_stats               # tables + figures, baseline tag
    python -m harness.report.summ_stats --tag gk700   # a different single experiment tag
    python -m harness.report.summ_stats --paired      # only cases every design ran

Reads results/raw/runs.csv (metadata + scores, one row per run), writes:
    results/processed/tables/results.csv, by-specialty.csv   (mean +- 95% CI per group)
    results/processed/figures/pdf|png/*.{pdf,png}            (the figure set below)

Every number is aggregated to one point/bar per (source, design) with a 95% CI - never a bare mean - and
scored on only the runs where the metric exists. Figures are ALWAYS restricted to a single experiment
`tag` (default baseline); pooling tags would double n and average two experiments into one bar.

Ordering agreement is the set overlap of the recommender's tests vs the human's: F1 (= Dice) and Jaccard
as the summaries, with the confusion counts overlap / missed (under-ordering) / extra (over-ordering).
"""

import argparse
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.font_manager as fm          # noqa: E402
import matplotlib.pyplot as plt               # noqa: E402
import numpy as np                            # noqa: E402
import pandas as pd                           # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from harness.cases import registry                       # noqa: E402
from harness.evaluation.cost import estimate_test_cost   # noqa: E402
from harness.store.transcripts import CSV, as_bool        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGDIR = os.path.join(ROOT, "results", "processed", "figures")
TABLEDIR = os.path.join(ROOT, "results", "processed", "tables")

# --- style: slide design language (Avenir Next + warm cream canvas + muted tokens, CVD-safe tokens) ---
INK, MUTED, LINE, GRID, ERR = "#1A1A1A", "#69757F", "#E4DFD8", "#E7E1D9", "#cbc3b8"
_FONT = "/System/Library/Fonts/Avenir Next.ttc"


def _font_family():
    """The report's typeface if present, else a clean installed sans. Avenir Next.ttc only exposes bold
    faces to matplotlib, so we name "Avenir" (weight 400) to avoid rendering every label heavy."""
    try:
        if os.path.exists(_FONT):
            fm.fontManager.addfont(_FONT)
    except Exception:
        pass
    normal = {f.name for f in fm.fontManager.ttflist if f.weight == 400}
    return next((n for n in ("Avenir", "Avenir Next", "Work Sans", "Helvetica Neue", "Arial") if n in normal),
                "DejaVu Sans")


plt.rcParams.update({
    "font.family": _font_family(), "font.size": 10.5,
    "text.color": INK, "axes.titlecolor": INK, "axes.labelcolor": MUTED,
    "axes.edgecolor": LINE, "axes.linewidth": 1.0,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

# The design ladder: CVD-safe (Okabe-Ito), ordered simple -> orchestrated.
DCOLOR = {"single": "#6e7681", "roles": "#e69f00", "strict-roles": "#009e73", "debate": "#0072b2"}
DNAME = {"single": "Single LLM", "roles": "Roles prompt", "strict-roles": "Strict roles", "debate": "Multi-agent"}
DORDER = {"single": 0, "roles": 1, "strict-roles": 2, "debate": 3}


# --- statistics ---------------------------------------------------------------

def mean_ci(vals):
    """Mean and 95% CI half-width of a numeric series (normal approx). (None, 0) if empty."""
    v = pd.Series(list(vals), dtype="float64").dropna()
    if v.empty:
        return None, 0.0
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), (0.0 if np.isnan(ci) else ci)


def prop_ci(flags):
    """Proportion as a fraction and its 95% CI half-width, for a boolean series. (None, 0) if empty."""
    v = pd.Series(list(flags)).dropna()
    if v.empty:
        return None, 0.0
    n, p = len(v), v.astype(bool).mean()
    return p, 1.96 * np.sqrt(p * (1 - p) / n)


# --- data ---------------------------------------------------------------------

def load_df():
    """One tidy row per run: runs.csv (metadata + score columns) + the human-workup baselines."""
    df = pd.read_csv(CSV)
    for c in ("f1", "jaccard", "overlap", "missed", "extra", "judge_score"):
        if c not in df.columns:
            df[c] = np.nan
    df["cost"] = df["cost_total"] if "cost_total" in df.columns else np.nan
    df["better"] = df["workup_better"] if "workup_better" in df.columns else np.nan
    df["correct"] = df["judge_correct"].map(as_bool) if "judge_correct" in df.columns else np.nan
    df["stopped_ok"] = df["stopped"].map(as_bool)
    df["tag"] = df["tag"].fillna("(baseline)").replace("", "(baseline)")

    reg = registry()

    def human_seq(case_id):
        c = reg.get(case_id)
        return c.human_seq if c else []

    df["human_n_tests"] = df.case_id.map(lambda cid: len(human_seq(cid)) or np.nan)
    df["human_cost"] = df.case_id.map(lambda cid: sum(estimate_test_cost(t) for t in human_seq(cid)) or np.nan)
    return df


def paired_only(df):
    """Keep only cases EVERY design ran (within each source), so designs are compared on the same set."""
    parts = []
    for source, g in df.groupby("source"):
        designs = set(g.design.unique())
        keep = {cid for cid, gc in g.groupby("case_id") if set(gc.design.unique()) == designs}
        parts.append(g[g.case_id.isin(keep)])
    return pd.concat(parts) if parts else df


def _designs(df):
    return sorted(df.design.unique(), key=lambda d: DORDER.get(d, 9))


# --- figures ------------------------------------------------------------------

def _save(fig, name):
    """Write both formats: figures/pdf/ (vector, for LaTeX) and figures/png/ (300-dpi, for slides)."""
    for ext in ("pdf", "png"):
        d = os.path.join(FIGDIR, ext)
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  figures/…/{name}", flush=True)


def _empty(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color=MUTED, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def _suptitle(fig, text):
    fig.suptitle(text, fontsize=10, x=.01, ha="left", color=MUTED)


def _legend(fig, designs, counts=None):
    """Design-ladder legend, top-right, with the n behind each design."""
    def label(d):
        base = DNAME.get(d, d)
        return f"{base}  (n={counts[d]})" if counts and d in counts else base
    handles = [plt.Line2D([], [], marker="o", ls="", color=DCOLOR.get(d, "#777"), label=label(d)) for d in designs]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower center",
               ncol=len(designs), bbox_to_anchor=(0.5, -0.02))


def _grouped(ax, df, metric, scale=1.0, is_prop=False):
    """Grouped bars: one bar per design, clustered by source, with 95% CI. Returns whether any data."""
    designs, sources = _designs(df), sorted(df.source.unique())
    x, w, any_data = np.arange(len(sources)), 0.8 / max(len(designs), 1), False
    for i, d in enumerate(designs):
        vals, errs = [], []
        for s in sources:
            g = df[(df.design == d) & (df.source == s)][metric]
            m, ci = (prop_ci(g) if is_prop else mean_ci(g))
            vals.append((m or 0) * scale); errs.append(ci * scale); any_data = any_data or m is not None
        ax.bar(x + i * w, vals, w, yerr=errs, color=DCOLOR.get(d, "#777"), capsize=3,
               error_kw={"elinewidth": 1.2, "ecolor": ERR})
    ax.set_xticks(x + w * (len(designs) - 1) / 2)
    ax.set_xticklabels([s.upper() for s in sources], fontsize=9)
    return any_data


def fig_ordering(df):
    """Ordering agreement (recommender's tests vs the human's): F1 and Jaccard by design and source."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (k, lab) in zip(axes, [("f1", "Ordering F1"), ("jaccard", "Ordering Jaccard")]):
        if not _grouped(ax, df, k):
            _empty(ax, f"{lab}\n(not scored yet)")
        ax.set_ylim(0, 1); ax.set_title(lab, fontsize=10.5, loc="left")
    _legend(fig, _designs(df), df.groupby("design").size().to_dict())
    _suptitle(fig, "Test-ordering agreement vs the human work-up; error bars = 95% CI")
    fig.tight_layout(rect=[0, 0.08, 1, .93]); _save(fig, "ordering_agreement")


def fig_over_under(df):
    """The differences: mean tests MISSED (human ordered, LLM did not) vs EXTRA (LLM ordered, human did not)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (k, lab) in zip(axes, [("missed", "Under-ordering (tests missed)"),
                                    ("extra", "Over-ordering (extra tests)")]):
        if not _grouped(ax, df, k):
            _empty(ax, f"{lab}\n(not scored yet)")
        ax.set_title(lab, fontsize=10.5, loc="left")
    axes[0].set_ylabel("Tests per case")
    _legend(fig, _designs(df), df.groupby("design").size().to_dict())
    _suptitle(fig, "How the work-ups differ from the human's: missed vs extra tests; error bars = 95% CI")
    fig.tight_layout(rect=[0, 0.08, 1, .93]); _save(fig, "over_under_ordering")


def fig_accuracy(df):
    """Diagnosis quality (CPC only): accuracy (judge >= 4) and the graded 1-5 judge score, per design."""
    sub = df[df.source == "cpc"]
    designs = _designs(sub)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (k, lab, is_prop, ylim) in zip(axes, [("correct", "Diagnosis accuracy (%)", True, 100),
                                                  ("judge_score", "Judge score (1-5)", False, 5)]):
        vals, errs = [], []
        for d in designs:
            m, ci = (prop_ci(sub[sub.design == d][k]) if is_prop else mean_ci(sub[sub.design == d][k]))
            vals.append((m or 0) * (100 if is_prop else 1)); errs.append(ci * (100 if is_prop else 1))
        if not any(vals):
            _empty(ax, f"{lab}\n(CPC not scored yet)")
        else:
            ax.bar([DNAME.get(d, d) for d in designs], vals, yerr=errs, width=.62,
                   color=[DCOLOR.get(d, "#777") for d in designs], capsize=4,
                   error_kw={"elinewidth": 1.2, "ecolor": ERR})
            ax.tick_params(axis="x", labelsize=8.5)
        ax.set_ylim(0, ylim); ax.set_title(lab, fontsize=10.5, loc="left")
    _suptitle(fig, "CPC diagnosis quality by design; error bars = 95% CI")
    fig.tight_layout(rect=[0, 0, 1, .92]); _save(fig, "diagnosis_accuracy")


def fig_head_to_head(df):
    """Blinded LLM-vs-human work-up: win / tie / loss share per source and design."""
    sub = df.dropna(subset=["better"])
    fig, ax = plt.subplots(figsize=(8, 4))
    if sub.empty:
        _empty(ax, "Head-to-head not scored yet\n(run: python -m harness.judge)")
        ax.set_title("LLM vs human work-up", fontsize=10.5, loc="left"); _save(fig, "head_to_head"); return
    groups = [(s, d) for s in sorted(sub.source.unique()) for d in _designs(sub[sub.source == s])]
    labels = [f"{s.upper()}\n{DNAME.get(d, d)}" for s, d in groups]
    bottoms = np.zeros(len(groups))
    for key, lab, colr in [("llm", "LLM wins", "#009e73"), ("tie", "Tie", "#BFC7CE"),
                           ("human", "Human wins", "#d55e00")]:
        fracs = np.array([100 * (sub[(sub.source == s) & (sub.design == d)].better == key).mean()
                          for s, d in groups])
        ax.bar(np.arange(len(groups)), fracs, .6, bottom=bottoms, label=lab, color=colr)
        bottoms += fracs
    ax.axhline(50, color=MUTED, lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(groups))); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Share of cases (%)"); ax.set_ylim(0, 100)
    ax.set_title("LLM vs human work-up (blinded)", fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    _suptitle(fig, "Which work-up is the better diagnostic pathway; dashed line = parity")
    fig.tight_layout(rect=[0, 0, 1, .93]); _save(fig, "head_to_head")


def fig_llm_vs_human(df):
    """Work-up size and cost: what the recommender ordered vs the real / expert work-up, per source."""
    panels = [("n_tests", "human_n_tests", "Tests ordered"), ("cost", "human_cost", "Work-up cost ($)")]
    sources = sorted(df.source.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (lk, hk, lab) in zip(axes, panels):
        x = np.arange(len(sources))
        lv, le, hv, he = [], [], [], []
        for s in sources:
            m, ci = mean_ci(df[df.source == s][lk]); lv.append(m or 0); le.append(ci)
            m, ci = mean_ci(df[df.source == s][hk]); hv.append(m or 0); he.append(ci)
        ax.bar(x - .2, lv, .4, yerr=le, label="LLM", color="#009e73", capsize=4,
               error_kw={"elinewidth": 1.2, "ecolor": ERR})
        ax.bar(x + .2, hv, .4, yerr=he, label="Human (actual)", color="#6e7681", capsize=4,
               error_kw={"elinewidth": 1.2, "ecolor": ERR})
        ax.set_xticks(x); ax.set_xticklabels([s.upper() for s in sources]); ax.set_title(lab, fontsize=10.5, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5)
    _suptitle(fig, "Work-up size and cost: LLM vs the real / expert work-up; error bars = 95% CI")
    fig.tight_layout(rect=[0, 0, 1, .92]); _save(fig, "llm_vs_human")


def fig_by_specialty(df):
    """Per-specialty breakdown: ordering F1 and (CPC) diagnosis accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (k, lab, is_prop) in zip(axes, [("f1", "Ordering F1", False), ("correct", "Diagnosis accuracy (%)", True)]):
        sub = df.dropna(subset=[k])
        if sub.empty:
            _empty(ax, f"{lab}\n(not scored yet)")
            ax.set_title(f"{lab.split(' (')[0]} by specialty", fontsize=10.5, loc="left"); continue
        specs, vals, errs = [], [], []
        for spec, g in sub.groupby("specialty"):
            v, ci = (prop_ci(g[k]) if is_prop else mean_ci(g[k]))
            if v is None:
                continue
            specs.append(spec); vals.append(v * (100 if is_prop else 1)); errs.append(ci * (100 if is_prop else 1))
        order = np.argsort(vals)
        ax.barh([specs[i] for i in order], [vals[i] for i in order], xerr=[errs[i] for i in order],
                color="#6e7681", height=.7, error_kw={"elinewidth": 1.2, "ecolor": ERR})
        ax.set_xlim(0, 100 if is_prop else 1); ax.set_xlabel(lab)
        ax.set_title(f"{lab.split(' (')[0]} by specialty", fontsize=10.5, loc="left")
    _suptitle(fig, "Per-specialty breakdown; error bars = 95% CI")
    fig.tight_layout(rect=[0, 0, 1, .93]); _save(fig, "by_specialty")


FIGURES = [fig_ordering, fig_over_under, fig_accuracy, fig_head_to_head, fig_llm_vs_human, fig_by_specialty]


# --- tables -------------------------------------------------------------------

# column -> (dataframe field, label, is_proportion | None for win-rate, scale, decimals)
TABLE_COLUMNS = [
    ("f1", "F1", False, 1, 2), ("jaccard", "Jaccard", False, 1, 2),
    ("overlap", "Overlap", False, 1, 1), ("missed", "Missed", False, 1, 1), ("extra", "Extra", False, 1, 1),
    ("correct", "Accuracy (%)", True, 1, 0), ("judge_score", "Judge (1-5)", False, 1, 1),
    ("better", "LLM > human (%)", None, 1, 0),
    ("cost", "Cost ($)", False, 1, 0), ("n_tests", "Tests", False, 1, 1), ("turns", "Turns", False, 1, 1),
    ("stopped_ok", "Stopped (%)", True, 1, 0), ("deliberate_s", "Time (s)", False, 1, 1),
    ("tokens", "Tokens", False, 1, 0),
]


def _cell(g, field, is_prop, scale, dp):
    """mean +- 95% CI (n), over only the runs where this metric exists. is_prop=None => head-to-head win rate."""
    if is_prop is None:
        v = g["better"].dropna()
        if v.empty:
            return "-", None, None, 0
        n = len(v); p = (v == "llm").mean()
        m, ci = 100 * p, 100 * 1.96 * (p * (1 - p) / n) ** 0.5
    else:
        v = g[field].dropna()
        if v.empty:
            return "-", None, None, 0
        n = len(v)
        m, ci = (prop_ci(v) if is_prop else mean_ci(v))
        m, ci = (m * 100, ci * 100) if is_prop else (m * scale, ci * scale)
    return f"{m:.{dp}f} ± {ci:.{dp}f} (n={n})", round(m, 3), round(ci, 3), n


def build(df, group):
    """One row per group: every measure as mean +- CI (per-metric n)."""
    rows = []
    keys = sorted({tuple(getattr(r, c) for c in group) for r in df.itertuples()},
                  key=lambda k: tuple(DORDER.get(v, 9) if c == "design" else str(v) for c, v in zip(group, k)))
    for key in keys:
        mask = pd.Series(True, index=df.index)
        for col, val in zip(group, key):
            mask &= df[col] == val
        g = df[mask]
        row = dict(zip(group, key))
        if "design" in row:
            row["design"] = DNAME.get(row["design"], row["design"])
        row["runs"] = len(g)
        flat = dict(row)
        for field, label, is_prop, scale, dp in TABLE_COLUMNS:
            cell, m, ci, n = _cell(g, field, is_prop, scale, dp)
            row[label] = cell
            flat[f"{field}_mean"], flat[f"{field}_ci95"], flat[f"{field}_n"] = m, ci, n
        rows.append((row, flat))
    return rows


def to_markdown(rows, note):
    head = list(rows[0][0])
    out = [note, "", "| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(r[0][h]) for h in head) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write_tables(df, note):
    """results.csv (design ladder) + by-specialty.csv, and print the main table."""
    os.makedirs(TABLEDIR, exist_ok=True)
    import csv
    for name, group in [("results", ("source", "tag", "design", "prompt_fp")),
                        ("by-specialty", ("source", "specialty"))]:
        rows = build(df, group)
        with open(os.path.join(TABLEDIR, f"{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0][1]))
            w.writeheader()
            for _, flat in rows:
                w.writerow(flat)
        print(f"  tables/{name}.csv", flush=True)
        if name == "results":
            print("\n" + to_markdown(rows, note))


# --- driver -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="(baseline)")     # always one experiment tag; pooling would double n
    ap.add_argument("--paired", action="store_true")   # only cases every design ran
    args = ap.parse_args()

    if not os.path.exists(CSV):
        print("no runs yet - generate + score first.")
        return
    df = load_df()
    tags = sorted(df.tag.unique())
    if args.tag not in tags:
        print(f"tag {args.tag!r} not found. available: {tags}")
        return
    df = df[df.tag == args.tag]
    print(f"tag: {args.tag}  ({len(df)} runs; other tags {[t for t in tags if t != args.tag]} excluded)")
    note = ("_Each cell is mean ± 95% CI, with the per-metric n (a metric is only scored where it can be "
            "measured, so its n can be below the run count)._")
    if args.paired:
        before = len(df)
        df = paired_only(df)
        note = f"_Paired: {len(df)} of {before} runs (only cases every design ran)._\n\n" + note

    write_tables(df, note)
    for fig in FIGURES:
        fig(df)
    print(f"\nwrote {len(FIGURES)} figures -> {FIGDIR}/  and tables -> {TABLEDIR}/", flush=True)


if __name__ == "__main__":
    main()
