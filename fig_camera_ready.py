"""Camera-ready versions of two paper figures, sized for the printed width.

The originals (figs_academic.py) put the legend outside the axes, which
shrinks the plot once the figure is scaled into the page. These versions
keep the same data, colours and style but place the legend inside or above
the axes and use a figure size close to the printed size.

Writes figs/fig20_profile_cr.png and figs/fig_robust_cr.png only.
figs_academic.py is NOT imported: it has no __main__ guard and would
regenerate every figure, so its style and loaders are repeated here.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 300,
})
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figs")
C = {"none": "#000000", "warmstart": "#56B4E9", "mix": "#E69F00",
     "shield": "#D55E00", "jsrl": "#009E73", "rlif": "#CC79A7",
     "residual": "#0072B2"}                                   # Okabe-Ito
L = {"none": "PPO (unguided)", "warmstart": "Warmstart", "mix": "Mix",
     "shield": "Shield", "jsrl": "JSRL-style", "rlif": "RLIF-style",
     "residual": "Residual"}
METHODS = ["mix", "warmstart", "none", "shield", "jsrl", "rlif", "residual"]
DIRECT = ["none", "warmstart", "shield", "jsrl", "rlif"]
SUCCESS_THRESHOLD = 0.30


def finals(method: str) -> list[dict]:
    """Final-run records for seeds 1-20 (same rule as figs_academic.finals)."""
    seen: dict[int, dict] = {}
    path = os.path.join(R, f"{method}.json")
    if os.path.exists(path):
        for f in json.load(open(path)).get("finals", []):
            seen[f["seed"]] = f
    for q in glob.glob(os.path.join(R, "ext", f"{method}_s*.json")):
        f = json.load(open(q))
        seen[f["seed"]] = f
    return [seen[s] for s in sorted(seen) if 1 <= s <= 20]


def profile() -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.25))
    taus = np.logspace(np.log10(0.06), np.log10(80), 300)
    for m in METHODS:
        best = np.array([f["best_mae"] for f in finals(m)])
        assert len(best) == 20, (m, len(best))
        ax.plot(taus, [(best <= t).mean() for t in taus], color=C[m], lw=1.5,
                label=L[m])
    ax.axvline(SUCCESS_THRESHOLD, color="#999999", lw=0.7, ls=":")
    ax.text(0.33, 0.03, "0.30", fontsize=7.5, color="#777777")
    ax.set_xscale("log")
    ax.set_xlabel(r"Error threshold $\tau$ (SPU)")
    ax.set_ylabel(r"Fraction of seeds with MAE $\leq \tau$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.46, 1.0), ncol=3,
              columnspacing=1.0, handlelength=1.4, handletextpad=0.5)
    fig.savefig(os.path.join(OUT, "fig20_profile_cr.png"), bbox_inches="tight")
    plt.close(fig)


def robust() -> None:
    rob = json.load(open(os.path.join(R, "robustness.json")))["runs"]

    def held(c: str, axis: str, lv: float) -> float:
        rs = [r["mae"] for r in rob if r["controller"] == c and r["axis"] == axis
              and r["level"] == lv and r["profile"] != "test"]
        return float(np.mean(rs)) if rs else float("nan")

    panels = [
        ("worth", [0.8, 0.9, 1.0], "Drum-worth scale $k_w$", "(a) Calibration error"),
        ("alpha", [0.7, 1.0, 1.3], r"Feedback scale $k_\alpha$", "(b) Feedback error"),
        ("noise", [0.0, 0.25, 0.5, 1.0], r"Sensor noise $\sigma$ (SPU)", "(c) Sensor noise"),
        ("process", [0.0, 0.05, 0.1, 0.2], r"Disturbance $\sigma_a$ ($^{\circ}$/s)", "(d) Process noise"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=True)
    for ax, (axis, levels, xlab, title) in zip(axes, panels):
        nominal_level = 0.0 if axis in ("noise", "process") else 1.0

        def val(c: str, lv: float) -> float:
            return held(c, "worth", 1.0) if lv == nominal_level else held(c, axis, lv)

        band = np.array([[val(c, lv) for lv in levels] for c in DIRECT])
        ax.fill_between(levels, band.min(0), band.max(0), color="#0072B2",
                        alpha=0.15, lw=0)
        ax.plot(levels, band.mean(0), color="#0072B2", lw=1.5, marker="o", ms=3,
                label="Learned policies (5)")
        ax.plot(levels, [val("teacher", lv) for lv in levels], "--",
                color="#000000", lw=1.4, marker="s", ms=3, label="NMPC teacher")
        ax.plot(levels, [val("residual", lv) for lv in levels], ":",
                color="#D55E00", lw=1.6, marker="^", ms=3, label="Residual")
        ax.set_yscale("log")
        ax.set_ylim(0.13, 70)
        ax.set_xticks(levels)
        ax.set_xlabel(xlab)
        ax.set_title(title)
    axes[0].set_ylabel("Held-out MAE (SPU)")
    axes[3].legend(loc="upper left", fontsize=7.5, handlelength=2.2,
                   borderaxespad=0.2)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig_robust_cr.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    profile()
    robust()
