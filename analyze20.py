"""20-seed analysis: merge extension runs, compute statistics, render figures.

Outputs:
  results/stats20.json               all computed statistics
  figs/fig20_reliability.png         strip plot, 20 seeds/method + success CI
  figs/fig20_safety.png              violations vs best MAE, 120 runs
  figs/fig20_curves.png              learning curves, 20 per panel
"""
from __future__ import annotations
import json, os, glob
import numpy as np
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")
METHODS = ["mix", "none", "warmstart", "shield", "jsrl", "rlif", "residual"]  # spectrum order
SEEDS = range(1, 21)          # seed 0 tuned the hyperparameters; excluded from evaluation
LABELS = {"none": "PPO (unguided)", "warmstart": "Warmstart", "mix": "Mix",
          "shield": "Shield", "jsrl": "JSRL-style", "rlif": "RLIF-style",
          "residual": "Residual"}
COLORS = {"none": "#101010", "shield": "#c23f3f", "warmstart": "#2a78d6",
          "mix": "#eb6834", "jsrl": "#1baf7a", "rlif": "#7c5cd6",
          "residual": "#8a7d1f"}
HELD_OUT = ["lowpower", "sine", "staircase", "randpw"]
SUCCESS = 0.30
MUT = "#6b7573"


def load(method):
    d = json.load(open(os.path.join(RESULTS, f"{method}.json")))
    finals = list(d["finals"])
    for p in sorted(glob.glob(os.path.join(RESULTS, "ext", f"{method}_s*.json"))):
        finals.append(json.load(open(p)))
    seen = {}
    for f in finals:
        seen[f["seed"]] = f
    return [seen[s] for s in sorted(seen) if s in SEEDS]


def iqm(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    lo, hi = int(np.floor(n/4)), int(np.ceil(3*n/4))
    return float(np.mean(x[lo:hi]))


def boot_ci(x, fn, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    s = np.array([fn(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def clopper(k, n):
    lo = st.beta.ppf(0.025, k, n-k+1) if k > 0 else 0.0
    hi = st.beta.ppf(0.975, k+1, n-k) if k < n else 1.0
    return float(lo), float(hi)


def main():
    data, stats = {}, {}
    for m in METHODS:
        fs = load(m)
        bests = np.array([f["best_mae"] for f in fs])
        ok = bests < SUCCESS
        k, n = int(ok.sum()), len(fs)
        row = dict(n=n, k=k, rate=k/n, rate_ci=clopper(k, n),
                   bests=[round(float(b), 3) for b in bests],
                   viol=[f["violations"] for f in fs],
                   div=sum(f["diverged"] for f in fs))
        if k:
            good = bests[ok]
            row["iqm"] = round(iqm(good), 3)
            row["iqm_ci"] = [round(v, 3) for v in boot_ci(good, iqm)]
            ho = [np.mean([f["best_evals"][p][0] for p in HELD_OUT])
                  for f, o in zip(fs, ok) if o and f["best_evals"]]
            row["held_out_mean"] = round(float(np.mean(ho)), 3)
            row["viol_ok_median"] = int(np.median(
                [f["violations"] for f, o in zip(fs, ok) if o]))
        data[m], stats[m] = fs, row
        print(f"{m:10s} {k:2d}/{n}  rate CI [{row['rate_ci'][0]:.2f},"
              f"{row['rate_ci'][1]:.2f}]  IQM {row.get('iqm','--')} "
              f"CI {row.get('iqm_ci','--')}  held-out {row.get('held_out_mean','--')}",
              flush=True)

    fisher = {}
    for a, b in [("jsrl","none"), ("rlif","none"), ("shield","none"),
                 ("none","mix"), ("jsrl","warmstart"), ("jsrl","shield")]:
        ka, na = stats[a]["k"], stats[a]["n"]
        kb, nb = stats[b]["k"], stats[b]["n"]
        _, p = st.fisher_exact([[ka, na-ka], [kb, nb-kb]])
        fisher[f"{a}_vs_{b}"] = round(float(p), 5)
        print(f"Fisher {a} ({ka}/{na}) vs {b} ({kb}/{nb}): p={p:.4g}", flush=True)
    json.dump(dict(success_threshold=SUCCESS, methods=stats, fisher=fisher),
              open(os.path.join(RESULTS, "stats20.json"), "w"), indent=1)

    teach = json.load(open(os.path.join(RESULTS, "teacher.json")))["evals"]
    tmae = teach["test"]["mae"]

    # -- reliability strip plot ------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    for i, m in enumerate(METHODS):
        b = np.clip(stats[m]["bests"], None, 100)
        x = i + (np.arange(len(b)) - len(b)/2) * 0.028
        ax.scatter(x, b, s=22, color=COLORS[m], alpha=.85, edgecolors="none",
                   zorder=3)
        lo, hi = stats[m]["rate_ci"]
        ax.text(i, 130, f"{stats[m]['k']}/{stats[m]['n']}", ha="center",
                fontsize=9, fontweight="bold", color=COLORS[m])
        ax.text(i, 240, f"[{lo:.2f}, {hi:.2f}]", ha="center", fontsize=6.5,
                color=MUT)
    ax.axhline(tmae, color=MUT, ls="--", lw=1.0)
    ax.text(len(METHODS)-.55, tmae*0.78, "NMPC teacher", fontsize=7.5,
            color=MUT, ha="right")
    ax.axhspan(SUCCESS, 400, color=MUT, alpha=.06)
    ax.axhline(SUCCESS, color=MUT, lw=.6, alpha=.5)
    ax.set_yscale("log"); ax.set_ylim(0.05, 400)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([LABELS[m] for m in METHODS], fontsize=8)
    ax.set_ylabel("Best-checkpoint validation MAE (SPU)")
    ax.set_title("Reliability at 20 seeds per method "
                 "(labels: successes and 95% Clopper--Pearson CI)", fontsize=9.5)
    ax.grid(axis="y", alpha=.25, lw=.5)
    fig.savefig(os.path.join(FIGS, "fig20_reliability.png"),
                bbox_inches="tight"); plt.close(fig)

    # -- safety scatter ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for m in METHODS:
        v = np.array(stats[m]["viol"], float)
        b = np.clip(stats[m]["bests"], None, 100)
        ax.scatter(v, b, s=20, color=COLORS[m], alpha=.8, edgecolors="none",
                   label=LABELS[m])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(SUCCESS, color=MUT, lw=.6, alpha=.5)
    ax.set_xlabel("Constraint violations during training (cumulative)")
    ax.set_ylabel("Best validation MAE (SPU)")
    ax.set_title("Training safety vs outcome, 120 runs", fontsize=9.5)
    ax.legend(fontsize=7, framealpha=.9)
    ax.grid(alpha=.25, lw=.5)
    fig.savefig(os.path.join(FIGS, "fig20_safety.png"),
                bbox_inches="tight"); plt.close(fig)

    # -- learning curves --------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), dpi=150,
                             sharex=True, sharey=True)
    for ax, m in zip(axes.flat, METHODS):
        for f in data[m]:
            if not f["curve"]: continue
            xs, ys = zip(*f["curve"])
            xs = np.array(xs)/1e6; ys = np.clip(ys, 0.05, 120)
            ax.plot(xs, ys, color=COLORS[m], lw=.7, alpha=.55)
            if f["diverged"]:
                ax.scatter([xs[-1]], [ys[-1]], marker="x", s=18,
                           color=COLORS[m], lw=1.1)
        ax.set_yscale("log"); ax.set_ylim(0.05, 120)
        ax.axhline(SUCCESS, color=MUT, lw=.6, alpha=.6)
        ax.set_title(f"{LABELS[m]}  ({stats[m]['k']}/{stats[m]['n']})",
                     fontsize=9)
        ax.grid(alpha=.25, lw=.5)
    for ax in axes[1]: ax.set_xlabel("Steps (millions)")
    for ax in axes[:,0]: ax.set_ylabel("Validation MAE (SPU)")
    fig.suptitle("Learning curves, 20 seeds per method "
                 "(x = numerical divergence)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig20_curves.png"),
                bbox_inches="tight"); plt.close(fig)
    print("figures written", flush=True)


if __name__ == "__main__":
    main()
