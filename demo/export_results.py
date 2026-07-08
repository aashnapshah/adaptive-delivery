"""Export the collected benchmark to results/: tabular data (CSV + JSON) and publication figures.

Reads the recordings written by record.py; safe to run any time (partial or complete).
  results/per_case.csv          one row per (model, design, case) with every metric
  results/summary_by_design.csv one row per (model, design, dataset) with mean ± SE / 95% CI
  results/summary.json          the same summary as JSON
  results/figures/*.png         per-dataset metric bars (95% CI) + accuracy-vs-cost scatter

Run:  python3 export_results.py
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import app  # iter_runs, DESIGNS, refined_case_ids

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
FIGS = os.path.join(OUT, "figures")

DNAME = {"single": "Single LLM", "maidxo": "Roles prompt", "debate": "Multi-agent"}
DCOLOR = {"single": "#6e7681", "maidxo": "#e69f00", "debate": "#0072b2"}      # CVD-safe (Okabe-Ito)
DORDER = {"single": 0, "maidxo": 1, "debate": 2}
METRICS = [("tokens", "Tokens / case"), ("secs", "Time to delivery / case (s)"),
           ("cost", "Workup cost / case ($)"), ("turns", "Turns / case"),
           ("tests", "Tests ordered / case")]
CONC = [("recall", "Order concordance · recall"), ("precision", "Order concordance · precision")]


def collect():
    refined = app.refined_case_ids()
    rows = []
    for model, d, fp, cid, run in app.iter_runs():
        if cid.startswith("SAMPLE") or model.startswith("_"):
            continue
        if d not in app.DESIGNS or (refined and cid not in refined):
            continue
        turns = run.get("turns") or []
        dx = next((t for t in turns if t.get("action") == "diagnose"), None)
        if not dx:
            continue
        meta = run.get("meta") or {}
        conc = meta.get("concordance") or {}
        last = turns[-1]
        rows.append({"model": model, "design": d, "mode": run.get("mode", "?"), "case_id": cid,
                     "true_dx": run.get("true_dx", ""), "diagnosis": dx.get("query"),
                     "correct": bool(dx.get("correct")), "judge": dx.get("judge_score"),
                     "tokens": meta.get("total_tokens"), "secs": meta.get("duration_s"),
                     "cost": last.get("total_cost"), "turns": len(turns), "tests": last.get("n_tests"),
                     "recall": conc.get("recall"), "precision": conc.get("precision")})
    return rows


def _mean_se(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
    return m, sd / len(v) ** 0.5


def summarize(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r["design"], r["mode"])].append(r)
    out = []
    for (model, d, mode), rs in g.items():
        n = len(rs)
        p = sum(1 for r in rs if r["correct"]) / n
        row = {"model": model, "design": d, "design_name": DNAME.get(d, d), "mode": mode, "n": n,
               "accuracy": round(100 * p, 1), "accuracy_ci95": round(100 * 1.96 * (p * (1 - p) / n) ** 0.5, 1)}
        for k in ("judge", "tokens", "secs", "cost", "turns", "tests", "recall", "precision"):
            m, se = _mean_se([r[k] for r in rs])
            row[k] = None if m is None else round(m, 3)
            row[k + "_se"] = None if se is None else round(se, 3)
        out.append(row)
    out.sort(key=lambda r: (r["mode"], r["model"], DORDER.get(r["design"], 9)))
    return out


def _write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_figures(summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable: {e} — skipping figures, CSVs still written)")
        return
    # Slide-style design language: Avenir Next + warm cream canvas + near-black/muted tokens.
    FONT = "/System/Library/Fonts/Avenir Next.ttc"
    fam = "sans-serif"
    try:
        import matplotlib.font_manager as fm
        if os.path.exists(FONT):
            fm.fontManager.addfont(FONT)
            fam = fm.FontProperties(fname=FONT).get_name()      # -> "Avenir Next"
    except Exception:
        pass
    INK, MUTED, LINE, GRID, CREAM = "#1A1A1A", "#69757F", "#E4DFD8", "#E7E1D9", "#F4F2F0"
    plt.rcParams.update({
        "font.family": fam, "font.size": 10.5,
        "text.color": INK, "axes.titlecolor": INK, "axes.labelcolor": MUTED,
        "axes.edgecolor": LINE, "axes.linewidth": 1.0,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9, "grid.alpha": 1.0,
        "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    by_mode = defaultdict(list)
    for r in summary:
        if r["mode"] in ("cpc", "mimic"):
            by_mode[r["mode"]].append(r)

    bars = [("tokens", "Tokens"), ("secs", "Time (s)"), ("cost", "Workup cost ($)"),
            ("turns", "Turns"), ("tests", "Tests")]          # the five per-case cost axes (no "/ case")
    pareto = [("cost", "Workup cost ($)"), ("turns", "Turns"), ("tests", "Tests")]

    def _legend(fig, rows):
        seen, handles = {}, []
        for r in rows:
            if r["design"] not in seen:
                seen[r["design"]] = 1
                handles.append(plt.Line2D([], [], marker="o", ls="", color=DCOLOR.get(r["design"], "#777"),
                                          label=r["design_name"]))
        fig.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper right", ncol=len(handles))

    for mode, allrows in by_mode.items():
        # Each figure shows ONE model's design ladder — never mix models on the same axes.
        # Pick the model with the fullest ladder (most designs present, then highest total n).
        bymodel = defaultdict(list)
        for r in allrows:
            bymodel[r["model"]].append(r)
        model = max(bymodel, key=lambda m: (len(bymodel[m]), sum(x["n"] for x in bymodel[m])))
        rows = sorted(bymodel[model], key=lambda r: DORDER.get(r["design"], 9))
        labels = [r["design_name"] for r in rows]
        colors = [DCOLOR.get(r["design"], "#777") for r in rows]

        # Pareto view — accuracy (y) vs cost / turns / tests (x), shared y, one marker per model×design
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax, (k, lab) in zip(axes, pareto):
            for r in rows:
                if r.get(k) is None:
                    continue
                ax.errorbar(r[k], r["accuracy"], yerr=r.get("accuracy_ci95") or 0, fmt="o", ms=11,
                            color=DCOLOR.get(r["design"], "#777"), mec="white", mew=1.4,
                            ecolor="#cbc3b8", elinewidth=1.2, capsize=0)
            ax.set_xlabel(lab)
            ax.set_title(f"Accuracy vs {lab.split(' (')[0].lower()}", fontsize=10.5, loc="left")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_ylim(0, 100)
        _legend(fig, rows)
        fig.suptitle(f"{mode.upper()} — accuracy vs cost / turns / tests  ({model}); error bars = 95% CI",
                     fontsize=10, x=.01, ha="left", color="#69757F")
        fig.tight_layout(rect=[0, 0, 1, .95])
        p = os.path.join(FIGS, f"{mode}_pareto.png")
        fig.savefig(p, dpi=150); plt.close(fig); print(f"  figure: {p}")

        # Cost of complexity — one clean row of the five per-case cost bars
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.3))
        for ax, (k, lab) in zip(axes, bars):
            vals = [r.get(k) or 0 for r in rows]
            err = [1.96 * (r.get(k + "_se") or 0) for r in rows]
            ax.bar(labels, vals, yerr=err, color=colors, width=.62, capsize=4,
                   error_kw={"elinewidth": 1.2, "ecolor": "#cbc3b8"})
            ax.set_title(lab, fontsize=10.5, loc="left")
            ax.tick_params(axis="x", labelsize=8.5)
        fig.suptitle(f"{mode.upper()} — cost of complexity, per case  ({model}); error bars = 95% CI",
                     fontsize=10, x=.01, ha="left", color="#69757F")
        fig.tight_layout(rect=[0, 0, 1, .93])
        p = os.path.join(FIGS, f"{mode}_cost.png")
        fig.savefig(p, dpi=150); plt.close(fig); print(f"  figure: {p}")

        # MIMIC only — order-concordance bars (recall / precision)
        if mode == "mimic" and any(r.get("recall") is not None for r in rows):
            fig, axes = plt.subplots(1, 2, figsize=(7, 3.3))
            for ax, (k, lab) in zip(axes, CONC):
                vals = [100 * (r.get(k) or 0) for r in rows]
                err = [100 * 1.96 * (r.get(k + "_se") or 0) for r in rows]
                ax.bar(labels, vals, yerr=err, color=colors, width=.55, capsize=4,
                       error_kw={"elinewidth": 1.2, "ecolor": "#5b6472"})
                ax.set_title(lab, fontsize=10.5, loc="left"); ax.set_ylim(0, 100)
            fig.suptitle(f"MIMIC — order concordance, % ({model})", fontsize=11, x=.01, ha="left")
            fig.tight_layout(rect=[0, 0, 1, .93])
            p = os.path.join(FIGS, "mimic_concordance.png")
            fig.savefig(p, dpi=150); plt.close(fig); print(f"  figure: {p}")


def main():
    os.makedirs(FIGS, exist_ok=True)
    rows = collect()
    if not rows:
        print("no recordings found yet — run the benchmark first (record.py).")
        return
    summary = summarize(rows)
    _write_csv(os.path.join(OUT, "per_case.csv"), rows,
               ["model", "design", "mode", "case_id", "true_dx", "diagnosis", "correct", "judge",
                "tokens", "secs", "cost", "turns", "tests", "recall", "precision"])
    scols = ["model", "design", "design_name", "mode", "n", "accuracy", "accuracy_ci95", "judge",
             "tokens", "tokens_se", "secs", "secs_se", "cost", "cost_se", "turns", "turns_se",
             "tests", "tests_se", "recall", "recall_se", "precision", "precision_se"]
    _write_csv(os.path.join(OUT, "summary_by_design.csv"), summary, scols)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(f"wrote {len(rows)} case rows + {len(summary)} summary rows to {OUT}/")
    make_figures(summary)
    print("done.")


if __name__ == "__main__":
    main()
