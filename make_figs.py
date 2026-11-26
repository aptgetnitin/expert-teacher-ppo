"""Consolidation + figures for the tuned comparative study (paper 2 v4).

Reads results/{method}.json and teacher.json; prints the paper's
tables (validation + held-out columns, per-seed) and renders the figures.
Rerunnable any time; every number in the paper comes from here.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R, F = os.path.join(HERE, "results"), os.path.join(HERE, "figs")
METHODS = ["none", "warmstart", "mix", "shield", "jsrl", "rlif"]
LABELS = {"none": "PPO (unguided)", "warmstart": "Warmstart", "mix": "Mix",
          "shield": "Shield", "jsrl": "JSRL-style", "rlif": "RLIF-style"}
COLORS = {"none": "#101010", "shield": "#c23f3f", "warmstart": "#2a78d6",
          "mix": "#eb6834", "jsrl": "#1baf7a", "rlif": "#7c5cd6"}
HELD_OUT = ["lowpower", "sine", "staircase", "randpw"]
SUCCESS = 0.30           # reliability threshold: within reach of the teacher band
MUT = "#6b7573"

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7.6, "font.family": "serif",
    "mathtext.fontset": "dejavuserif", "axes.grid": True, "grid.alpha": .25,
    "grid.linewidth": .5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 400, "lines.linewidth": 1.15})


def load():
    out = {}
    for m in METHODS:
        p = os.path.join(R, f"{m}.json")
        if os.path.exists(p):
            out[m] = json.load(open(p))
    teacher = json.load(open(os.path.join(R, "teacher.json")))
    return out, teacher


def consolidate():
    data, teacher = load()
    print("=== NMPC teacher (classical baseline) ===")
    for p, v in teacher["evals"].items():
        print(f"  {p:10s} MAE {v['mae']:.3f}  effort {v['effort']:.1f} deg  "
              f"{v['ms_per_decision']:.2f} ms/dec")
    print("\n=== Per-method finals (best-checkpoint protocol) ===")
    for m in METHODS:
        if m not in data:
            print(f"  {m}: pending"); continue
        fin = data[m]["finals"]
        bests = [f["best_mae"] for f in fin]
        viol = [f["violations"] for f in fin]
        div = sum(f["diverged"] for f in fin)
        ok = sum(b < SUCCESS for b in bests)
        # held-out at best checkpoint, successful seeds only
        ho = []
        for f in fin:
            if f["best_mae"] < SUCCESS and f["best_evals"]:
                ho.append(np.mean([f["best_evals"][p][0] for p in HELD_OUT]))
        print(f"  {LABELS[m]:15s} reliable {ok}/{len(fin)}  "
              f"val bests {[round(b,3) for b in bests]}")
        print(f"  {'':15s} viol {[f'{v//1000}k' for v in viol]}  div {div}/{len(fin)}"
              + (f"  held-out mean (ok seeds) {np.mean(ho):.3f}" if ho else ""))
    return data, teacher


def fig_reliability(data, teacher, path):
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    order = ["mix", "none", "warmstart", "shield", "jsrl", "rlif"]
    for i, m in enumerate(order):
        if m not in data: continue
        bests = np.array([f["best_mae"] for f in data[m]["finals"]])
        x = np.full_like(bests, i, dtype=float) + np.linspace(-.15, .15, len(bests))
        ax.scatter(x, np.clip(bests, None, 100), s=30, color=COLORS[m],
                   edgecolors="none", alpha=.9, zorder=3)
    tmae = teacher["evals"]["test"]["mae"]
    ax.axhline(tmae, color=MUT, ls="--", lw=1.0)
    ax.text(len(order)-.45, tmae*1.25, f"NMPC teacher ({tmae:.3f})",
            fontsize=7.5, color=MUT, ha="right")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.3, 1, 10, 70])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(
        ["0.1", "0.3", "1", "10", "70"]))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axhspan(0.3, 150, color=MUT, alpha=.06)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS[m] for m in order], fontsize=8)
    ax.set_ylabel("validation MAE (SPU), best ckpt")
    fig.tight_layout(pad=.3); fig.savefig(path); print(" ", path)


def fig_safety(data, teacher, path):
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for m in METHODS:
        if m not in data: continue
        for k, f in enumerate(data[m]["finals"]):
            ax.scatter([max(f["violations"], 2000)], [min(f["best_mae"], 100)],
                       s=26, color=COLORS[m], alpha=.85, edgecolors="none",
                       label=LABELS[m] if k == 0 else None, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training violations")
    ax.set_ylabel("validation MAE (SPU), best ckpt")
    ax.set_yticks([0.1, 0.3, 1, 10, 70])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(
        ["0.1", "0.3", "1", "10", "70"]))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=6.6,
              handletextpad=.1, borderpad=.2, labelspacing=.2, markerscale=.85)
    fig.tight_layout(pad=.3); fig.savefig(path); print(" ", path)


def fig_curves(data, path):
    order = ["none", "warmstart", "mix", "shield", "jsrl", "rlif"]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.7), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, order):
        if m in data:
            for f in data[m]["finals"]:
                xs = [c[0]/1e6 for c in f["curve"]]
                ys = [c[1] for c in f["curve"]]
                if xs:
                    ax.plot(xs, np.clip(ys, 0.05, 120), color=COLORS[m],
                            alpha=.75, lw=1.0)
                    if f["diverged"]:
                        ax.scatter([xs[-1]], [np.clip(ys[-1],0.05,120)],
                                   marker="x", s=20, color=COLORS[m], lw=1.2)
        ax.set_yscale("log"); ax.set_title(LABELS[m], fontsize=8.5, pad=3)
        ax.set_ylim(0.07, 130)
        ax.set_yticks([0.15, 1, 10, 100])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(
            ["0.15", "1", "10", "100"]))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for ax in axes[1]: ax.set_xlabel("steps ($10^6$)")
    for ax in axes[:, 0]: ax.set_ylabel("val. MAE (SPU)")
    fig.tight_layout(pad=.4); fig.savefig(path); print(" ", path)


if __name__ == "__main__":
    os.makedirs(F, exist_ok=True)
    data, teacher = consolidate()
    fig_reliability(data, teacher, os.path.join(F, "fig_reliability.png"))
    fig_safety(data, teacher, os.path.join(F, "fig_safety.png"))
    fig_curves(data, os.path.join(F, "fig_curves.png"))
