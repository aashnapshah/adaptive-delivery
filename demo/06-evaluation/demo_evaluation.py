"""Demo 06 - Evaluation.

Maps to Methods section: "Evaluation".

Pulls the pieces together and measures them on the two axes that matter:

  1. Diagnosis quality vs cost  - the SDBench frame (Nori 2025): accuracy against the
     total diagnostic spend. (Optional here: needs a real model to differentiate agents;
     under the offline stub both agents are identical, so it is reported only if a model
     is configured.)
  2. Delivery value vs alert burden - the contribution of this project: how much useful,
     accepted signal each delivery policy produces, and at what alerting burden. This is
     the analogue of the accuracy/cost Pareto, applied to *when to alert*.

The plots are produced by functions the notebook calls; `main()` writes them to PNG and
prints the summary table.

Run:  python3 demo_evaluation.py
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "01_environment")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "02_generation")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "05_policy")))
import demo_generation as gen  # noqa: E402
import demo_policy as dp  # noqa: E402
from shared import llm  # noqa: E402
from shared.toy_data import all_cases  # noqa: E402


def evaluate_policies(n: int = 400, seed: int = 0) -> dict:
    """Run every delivery policy over the same stream; return their trajectories."""
    stream = dp.build_stream(n=n, seed=seed)
    return {name: dp.run_policy(pol, stream) for name, pol in dp.all_policies().items()}


def print_table(results: dict) -> None:
    print(f"{'policy':<20}{'reward':>8}{'realized':>10}{'safety-miss':>12}{'interrupt%':>11}")
    print("-" * 61)
    for name, r in results.items():
        print(f"{name:<20}{r['total_reward']:>8.1f}{r['realized_value']:>10.1f}"
              f"{r['safety_miss']:>12d}{r['interrupt_rate']*100:>10.0f}%")


def plot_learning_curves(results: dict):
    """Cumulative reward over time - the bandit's learning shows up as a rising slope."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, r in results.items():
        ax.plot(range(1, r["n"] + 1), r["cum_curve"], label=name, linewidth=2)
    ax.axhline(0, color="#B6BFC8", lw=1, ls="--")
    ax.set_xlabel("recommendations seen")
    ax.set_ylabel("cumulative reward")
    ax.set_title("Adaptive delivery: cumulative reward by policy")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_value_vs_burden(results: dict):
    """Value (accepted, useful alerts) vs alerting burden (delivery rate).

    Up-and-to-the-left is better: more accepted signal for fewer alerts. This is the
    when-to-alert analogue of the SDBench accuracy/cost Pareto frontier.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name, r in results.items():
        x = r["interrupt_rate"]
        y = r["realized_value"] / r["n"]           # latent clinical value realized per rec
        ax.scatter(x, y, s=90)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("alerting burden  (interrupt rate)")
    ax.set_ylabel("realized value per recommendation")
    ax.set_title("Realized value vs alerting burden")
    ax.margins(0.2)
    fig.tight_layout()
    return fig


def evaluate_generation(backend: str | None = None) -> dict:
    """Run both diagnostic agents over every case; return accuracy and mean cost.

    This is the SDBench axis (accuracy vs cost). It makes real LLM calls, so the
    baseline-vs-panel difference only appears with a real model - under the offline
    stub both agents follow the same scripted policy and look identical.
    """
    import contextlib
    import io

    agents = {"baseline": gen.run_baseline, "MAI-DxO panel": gen.run_panel}
    out: dict[str, dict] = {}
    for name, fn in agents.items():
        correct, cost, n = 0, 0.0, 0
        for case in all_cases():
            with contextlib.redirect_stdout(io.StringIO()):
                r = fn(case.case_id, backend=backend)
            correct += int(r["correct"])
            cost += r["cost"]
            n += 1
        out[name] = {"accuracy": correct / n, "mean_cost": cost / n, "n": n}
    return out


def plot_accuracy_vs_cost(gen_results: dict):
    """SDBench-style accuracy-vs-cost scatter for the diagnostic agents."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for name, r in gen_results.items():
        ax.scatter(r["mean_cost"], r["accuracy"] * 100, s=110)
        ax.annotate(name, (r["mean_cost"], r["accuracy"] * 100),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("mean diagnostic cost per case (USD)")
    ax.set_ylabel("diagnostic accuracy (%)")
    ax.set_title("Diagnosis quality vs cost (SDBench axes)")
    ax.set_ylim(0, 105)
    ax.margins(x=0.25)
    fig.tight_layout()
    return fig


def main() -> None:
    backend = llm.detect_backend()
    print(f"Generation eval backend: {llm.active_model(backend)}")
    if backend == "stub":
        print("(stub backend: both agents are identical; set OPENROUTER_API_KEY to differentiate)")
    gen_results = evaluate_generation(backend)
    print(f"\n{'agent':<16}{'accuracy':>10}{'mean cost':>12}")
    for name, r in gen_results.items():
        print(f"{name:<16}{r['accuracy']*100:>9.0f}%{'$'+format(r['mean_cost'],'.0f'):>12}")

    print()
    results = evaluate_policies()
    print_table(results)
    out = os.path.dirname(os.path.abspath(__file__))
    plot_accuracy_vs_cost(gen_results).savefig(os.path.join(out, "accuracy_vs_cost.png"), dpi=130, bbox_inches="tight")
    plot_learning_curves(results).savefig(os.path.join(out, "learning_curves.png"), dpi=130, bbox_inches="tight")
    plot_value_vs_burden(results).savefig(os.path.join(out, "value_vs_burden.png"), dpi=130, bbox_inches="tight")
    print(f"\nWrote accuracy_vs_cost.png, learning_curves.png, value_vs_burden.png to {out}")


if __name__ == "__main__":
    main()
