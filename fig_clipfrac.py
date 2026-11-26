"""Clip-diagnostic figure and summary table (multi-seed, tuned + shared base).

Reads results/diag/multi/*.json (diag_clipfrac_multi.py) and
results/shared/*.json (run_shared_hp.py). Writes
  figs/fig_clipfrac.png
  results/clipfrac_summary.json    the numbers quoted in Section 5.2
"""
from __future__ import annotations
import json, glob, os, shutil, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same style as figs_academic.py. That module is NOT imported because it has
# no __main__ guard and regenerates every paper figure on import.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 200,
})
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
F = os.path.join(HERE, "figs")
C = {"warmstart": "#56B4E9", "mix": "#E69F00", "shield": "#D55E00"}   # Okabe-Ito

CONFIGS = [("mix", "tuned", "Mix (tuned)"), ("mix", "shared", "Mix (shared base)"),
           ("warmstart", "tuned", "Warmstart"), ("shield", "tuned", "Shield (tuned)"),
           ("shield", "shared", "Shield (shared base)")]
LS = {"tuned": "-", "shared": "--"}


def main():
    g = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(R, "diag", "multi", "*_s*.json"))):
        d = json.load(open(p)); g[(d["method"], d["base"])].append(d)

    summ = {}
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.7))
    for i, (m, b, lab) in enumerate(CONFIGS):
        runs = g[(m, b)]
        ne = len(runs[0]["by_epoch_while_injecting"])
        last = [d["by_epoch_while_injecting"][str(ne-1)]["clipfrac_inj"] for d in runs]
        first = [d["first_minibatch_clipfrac_inj"] for d in runs]
        summ[f"{m}_{b}"] = dict(
            seeds=[d["seed"] for d in runs],
            clip_range=runs[0]["params"]["clip_range"],
            learning_rate=runs[0]["params"]["learning_rate"], n_epochs=ne,
            first_step_clipfrac_injected_max=float(max(first)),
            last_epoch_clipfrac_injected=dict(mean=float(np.mean(last)),
                                              min=float(min(last)), max=float(max(last))),
            injected_fraction=float(np.mean(
                [d["injected_fraction_while_injecting"] for d in runs])),
            diverged_within_250k=[d["seed"] for d in runs if d["diverged"]])
        ax[0].bar(i, np.mean(last), color=C[m], alpha=0.55 if b == "shared" else 0.9,
                  width=0.7, hatch="//" if b == "shared" else None, edgecolor="white")
        ax[0].scatter([i]*len(last), last, s=9, color="black", zorder=3)
        # time series: last-epoch clip fraction on injected samples, per 25k bin
        # mix/warmstart inject only inside their window; shield throughout
        t_end = runs[0]["params"].get("g_window", runs[0]["steps"])
        nb = max(len(d["series"]) for d in runs)
        xs, mu, lo, hi = [], [], [], []
        for k in range(nb):
            if (k+1)*25_000 > t_end:
                break
            v = [d["series"][k]["clipfrac_inj_last"] for d in runs
                 if k < len(d["series"]) and d["series"][k]["clipfrac_inj_last"] is not None]
            if len(v) >= 3:
                xs.append((k+1)*25)
                mu.append(np.mean(v)); lo.append(min(v)); hi.append(max(v))
        ax[1].plot(xs, mu, LS[b], color=C[m], lw=1.4, label=lab)
        ax[1].fill_between(xs, lo, hi, color=C[m], alpha=0.12, lw=0)
    ax[0].set_xticks(range(len(CONFIGS)))
    ax[0].set_xticklabels([c[2].replace(" (", "\n(") for c in CONFIGS], fontsize=7)
    ax[0].set_ylabel("Clipped share of injected samples\n(last epoch of each update)")
    ax[0].set_ylim(0, 1)
    ax[1].set_xlabel("Training steps (thousands)")
    ax[1].set_ylabel("Clipped share of injected samples")
    ax[1].set_ylim(0, 1)
    ax[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    out = os.path.join(F, "fig_clipfrac.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)

    shared = {}
    for m in ("mix", "shield"):
        rs = [json.load(open(p)) for p in
              sorted(glob.glob(os.path.join(R, "shared", f"{m}_s*.json")))]
        best = [r["best_mae"] for r in rs]
        final = [None if r["diverged"] else r["final_evals"]["test"][0] for r in rs]
        shared[m] = dict(seeds=[r["seed"] for r in rs],
                         best=[round(b, 3) for b in best],
                         best_reliable=int(sum(b < 0.30 for b in best)),
                         final_reliable=int(sum(f is not None and f < 0.30 for f in final)),
                         diverged=int(sum(r["diverged"] for r in rs)))
    json.dump(dict(clip=summ, shared_hp_outcomes=shared),
              open(os.path.join(R, "clipfrac_summary.json"), "w"), indent=1)
    print(json.dumps(dict(clip={k: (round(v["last_epoch_clipfrac_injected"]["mean"], 3),
                                    round(v["last_epoch_clipfrac_injected"]["min"], 3),
                                    round(v["last_epoch_clipfrac_injected"]["max"], 3),
                                    v["first_step_clipfrac_injected_max"])
                                for k, v in summ.items()}, shared=shared), indent=1))


if __name__ == "__main__":
    main()
