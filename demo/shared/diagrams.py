"""Reusable schematic figures for the demo notebooks.

`draw_pipeline(active=...)` renders the six-stage pipeline with one stage
highlighted, so every notebook can show "you are here". The per-stage helpers
draw a simple "how it works" schematic for that stage.

All figures are drawn with matplotlib so they are reproducible and need no
external image files.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Ordered pipeline stages: (key, label). Keys match the demo folder names.
STAGES = [
    ("environment", "Environment"),
    ("generation", "Generation"),
    ("scoring", "Scoring"),
    ("interface", "Interface"),
    ("policy", "Policy"),
    ("evaluation", "Evaluation"),
]

_ACTIVE_FACE = "#2C6FB3"
_ACTIVE_EDGE = "#1B4F82"
_IDLE_FACE = "#EEF1F4"
_IDLE_EDGE = "#B6BFC8"
_ARROW = "#8A949E"


def draw_pipeline(active: str | None = None, figsize=(13, 1.7)):
    """Draw the 6-stage pipeline left-to-right, highlighting `active`."""
    fig, ax = plt.subplots(figsize=figsize)
    w, h, gap = 1.7, 0.9, 0.55
    for i, (key, label) in enumerate(STAGES):
        x = i * (w + gap)
        on = key == active
        box = FancyBboxPatch(
            (x, 0), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=2.2,
            edgecolor=_ACTIVE_EDGE if on else _IDLE_EDGE,
            facecolor=_ACTIVE_FACE if on else _IDLE_FACE,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2, h / 2, f"{i+1}. {label}",
            ha="center", va="center", fontsize=10.5,
            color="white" if on else "#3A434C",
            fontweight="bold" if on else "normal", zorder=3,
        )
        if i < len(STAGES) - 1:
            ax.annotate(
                "", xy=(x + w + gap, h / 2), xytext=(x + w, h / 2),
                arrowprops=dict(arrowstyle="-|>", color=_ARROW, lw=1.8),
            )
    ax.set_xlim(-0.3, len(STAGES) * (w + gap) - gap + 0.3)
    ax.set_ylim(-0.25, h + 0.25)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _node(ax, x, y, w, h, text, face=_IDLE_FACE, edge=_IDLE_EDGE, tcolor="#3A434C", fontsize=10):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=2, edgecolor=edge, facecolor=face, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=tcolor, zorder=3)


def _arrow(ax, p0, p1, text=None, rad=0.0, color=_ARROW, label_dy=0.18, label_dx=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=16, lw=1.8,
        color=color, connectionstyle=f"arc3,rad={rad}", zorder=1,
    ))
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + label_dx, my + label_dy, text, ha="center", va="bottom",
                fontsize=9, color="#5A636B", style="italic")


def draw_sdbench_loop(highlight="agent", figsize=(11, 4.0)):
    """SDBench-style sequential loop: Agent <-> Gatekeeper, then Judge + cost.

    `highlight` in {"agent", "gatekeeper"} colors the box the current notebook is about.
    """
    fig, ax = plt.subplots(figsize=figsize)
    w, h = 2.6, 1.15
    y = 1.9
    ax_x, gk_x = 0.0, 6.0

    agent_on = highlight == "agent"
    gk_on = highlight == "gatekeeper"
    _node(ax, ax_x, y, w, h, "Diagnostic Agent\n(baseline / MAI-DxO panel)",
          face=_ACTIVE_FACE if agent_on else _IDLE_FACE,
          edge=_ACTIVE_EDGE if agent_on else _IDLE_EDGE,
          tcolor="white" if agent_on else "#3A434C")
    _node(ax, gk_x, y, w, h, "Gatekeeper\n(EHR record oracle)",
          face=_ACTIVE_FACE if gk_on else _IDLE_FACE,
          edge=_ACTIVE_EDGE if gk_on else _IDLE_EDGE,
          tcolor="white" if gk_on else "#3A434C")

    # sequential cycle: action out (top), finding back (bottom)
    _arrow(ax, (ax_x + w, y + h * 0.72), (gk_x, y + h * 0.72), rad=-0.28)
    ax.text((ax_x + w + gk_x) / 2, y + h + 0.35, "ask question / order test",
            ha="center", fontsize=9, color="#5A636B", style="italic")
    _arrow(ax, (gk_x, y + h * 0.28), (ax_x + w, y + h * 0.28), rad=-0.28)
    ax.text((ax_x + w + gk_x) / 2, y - 0.55, "reveal finding  (+ visit / test cost)",
            ha="center", fontsize=9, color="#5A636B", style="italic")

    # diagnose branch -> Judge -> outcome
    jx = ax_x + w / 2
    _node(ax, jx - w / 2, y - 2.0, w, h * 0.85, "Judge\n(vs ICD ground truth)")
    _arrow(ax, (jx, y), (jx, y - 2.0 + h * 0.85), text="diagnose", label_dx=0.85, label_dy=-0.1)
    _node(ax, gk_x, y - 2.0, w, h * 0.85, "Accuracy  +  total cost")
    _arrow(ax, (jx + w / 2, y - 2.0 + h * 0.42), (gk_x, y - 2.0 + h * 0.42))

    ax.set_xlim(-0.4, gk_x + w + 0.4)
    ax.set_ylim(-0.7, y + h + 0.8)
    ax.axis("off")
    fig.tight_layout()
    return fig


# Back-compat alias.
draw_generation_loop = draw_sdbench_loop


def draw_stage_io(title, inputs, center, outputs, figsize=(11, 3.4), center_color=True):
    """Generic input(s) -> processing box -> output(s) schematic for a stage."""
    fig, ax = plt.subplots(figsize=figsize)
    w, h = 2.6, 0.9
    cx = 4.0
    cy = 1.4
    # center box
    _node(ax, cx, cy, w, h + 0.3, center,
          face=_ACTIVE_FACE if center_color else _IDLE_FACE,
          edge=_ACTIVE_EDGE if center_color else _IDLE_EDGE,
          tcolor="white" if center_color else "#3A434C")
    # inputs on the left
    n_in = len(inputs)
    for i, label in enumerate(inputs):
        y = cy + (h + 0.3) / 2 - (h + 0.3) / 2 + (i - (n_in - 1) / 2) * 1.15
        _node(ax, cx - 4.0, y, w - 0.2, h, label)
        _arrow(ax, (cx - 4.0 + w - 0.2, y + h / 2), (cx, cy + (h + 0.3) / 2))
    # outputs on the right
    n_out = len(outputs)
    for i, label in enumerate(outputs):
        y = cy + (i - (n_out - 1) / 2) * 1.15
        _node(ax, cx + 4.0, y, w - 0.2, h, label)
        _arrow(ax, (cx + w, cy + (h + 0.3) / 2), (cx + 4.0, y + h / 2))
    ax.set_title(title, fontsize=11, color="#3A434C")
    ax.set_xlim(-0.4, cx + 4.0 + w)
    ax.set_ylim(cy - 1.6, cy + 1.9)
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_bandit_loop(figsize=(11, 3.7)):
    """The contextual-bandit decision loop used by the adaptive-delivery policy."""
    fig, ax = plt.subplots(figsize=figsize)
    w, h = 2.5, 1.0
    y = 1.8
    xs = [0.0, 3.4, 6.8]
    _node(ax, xs[0], y, w, h, "Context\n(est. appropriateness,\nharm, alert burden)")
    _node(ax, xs[1], y, w, h, "Delivery policy\nchoose 1 of 5 actions",
          face=_ACTIVE_FACE, edge=_ACTIVE_EDGE, tcolor="white")
    _node(ax, xs[2], y, w, h, "Clinician response\nfollow / ignore")
    _node(ax, xs[1], y - 2.0, w, h, "Reward\n(value - fatigue - omission)")

    _arrow(ax, (xs[0] + w, y + h / 2), (xs[1], y + h / 2))
    _arrow(ax, (xs[1] + w, y + h / 2), (xs[2], y + h / 2), text="if deliver", label_dy=0.12)
    _arrow(ax, (xs[2] + w / 2, y), (xs[1] + w / 2, y - 2.0 + h), rad=0.25)
    ax.text(xs[2] + w / 2 + 0.2, y - 1.0, "observe", fontsize=9, color="#5A636B", style="italic")
    _arrow(ax, (xs[1] + w / 2, y - 2.0 + h), (xs[1] + w / 2, y), text="update θ", label_dx=0.7)

    ax.set_xlim(-0.4, xs[2] + w + 0.4)
    ax.set_ylim(y - 2.2, y + h + 0.4)
    ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Quick visual check: write both schematics to PNG.
    import os
    out = os.path.dirname(os.path.abspath(__file__))
    draw_pipeline(active="generation").savefig(os.path.join(out, "_preview_pipeline.png"), dpi=130, bbox_inches="tight")
    draw_sdbench_loop(highlight="agent").savefig(os.path.join(out, "_preview_genloop.png"), dpi=130, bbox_inches="tight")
    print("wrote _preview_pipeline.png and _preview_genloop.png")
