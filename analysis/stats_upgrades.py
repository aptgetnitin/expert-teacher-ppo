"""Statistics recommended by the RL evaluation-rigor literature.

Adds, on top of analyze20.py's success counts:
  - performance profile / run-score distribution (Agarwal et al. 2021):
    fraction of seeds with MAE <= tau, swept over tau, bootstrap bands.
    Our success rate is one point on this curve.
  - probability of improvement P(A > B) via Mann-Whitney U (Agarwal 2021)
  - CVaR across runs = "Risk across Runs" (Chan et al. 2020)
  - cost rate rho_c = violations / env steps (Ray, Achiam & Amodei 2019,
    Safety Gym Sec 5.1), plus normalization against unguided PPO = 1.0
  - seed-count subsampling: what a 3/5/10-seed study would have concluded
  - Fisher exact with Holm correction across the pairwise family

Inferential claims rest on Fisher exact (valid at n=20); bootstrap
intervals are reported as descriptive (Colas et al. 2019 caution).
"""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
for _d in ("core", "training", "evaluation", "analysis", "plant"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import json, os, glob, itertools
import numpy as np
from scipy import stats as st

RESULTS = os.path.join(ROOT, "results")
SUCCESS = 0.30            # fixed a priori: competence band near the teacher
FINAL_STEPS = 2_000_000


def load(method):
    d = json.load(open(os.path.join(RESULTS, f"{method}.json")))
    finals = list(d.get("finals", []))
    for p in sorted(glob.glob(os.path.join(RESULTS, "ext", f"{method}_s*.json"))):
        finals.append(json.load(open(p)))
    seen = {}
    for f in finals:
        seen[f["seed"]] = f
    return [seen[s] for s in sorted(seen) if 1 <= s <= 20]


def perf_profile(bests, taus, n_boot=2000, seed=0):
    """Fraction of runs with score <= tau (lower MAE is better), + CI band."""
    rng = np.random.default_rng(seed)
    x = np.asarray(bests, float)
    curve = np.array([(x <= t).mean() for t in taus])
    boot = np.array([[(rng.choice(x, len(x), replace=True) <= t).mean()
                      for t in taus] for _ in range(n_boot)])
    return curve, np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)


def prob_improvement(a, b):
    """P(a random run of A beats a random run of B).

    Scores are MAE, so lower is better. scipy returns U1, which counts pairs
    where A > B (i.e. A is worse); the improvement probability is its
    complement, with ties split evenly.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    u_worse = st.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(1.0 - u_worse / (len(a) * len(b)))


def cvar(x, alpha=0.25):
    """Risk across Runs (Chan et al. 2020): mean of the worst alpha fraction."""
    x = np.sort(np.asarray(x, float))[::-1]        # worst (highest MAE) first
    k = max(1, int(np.ceil(alpha * len(x))))
    return float(np.mean(x[:k]))


def subsample_study(bests, n_sub, n_draws=5000, seed=0):
    """What would a study with n_sub seeds have concluded? -> success-rate spread."""
    rng = np.random.default_rng(seed)
    x = np.asarray(bests, float)
    rates = [np.mean(rng.choice(x, n_sub, replace=False) < SUCCESS)
             for _ in range(n_draws)]
    return float(np.mean(rates)), float(np.percentile(rates, 2.5)), \
           float(np.percentile(rates, 97.5))


def holm(pvals):
    """Holm-Bonferroni corrected p-values, order preserved."""
    idx = np.argsort(pvals)
    out = np.empty(len(pvals)); run = 0.0
    for rank, i in enumerate(idx):
        adj = min(1.0, (len(pvals) - rank) * pvals[i])
        run = max(run, adj)
        out[i] = run
    return out


def main(methods):
    data = {m: load(m) for m in methods}
    data = {m: v for m, v in data.items() if v}
    rep = {}
    base = "none"

    for m, fs in data.items():
        bests = np.array([f["best_mae"] for f in fs])
        viol = np.array([f["violations"] for f in fs], float)
        ok = bests < SUCCESS
        rep[m] = dict(
            n=len(fs), k=int(ok.sum()),
            cvar25=round(cvar(bests), 3),
            cost_rate=float(np.mean(viol) / FINAL_STEPS),
            viol_mean=float(np.mean(viol)),
        )
    # normalize cost rate against unguided PPO (Safety Gym convention)
    if base in rep and rep[base]["cost_rate"] > 0:
        for m in rep:
            rep[m]["cost_rate_norm"] = round(
                rep[m]["cost_rate"] / rep[base]["cost_rate"], 3)

    # probability of improvement vs the unguided baseline
    if base in data:
        bb = [f["best_mae"] for f in data[base]]
        for m in data:
            if m != base:
                rep[m]["p_improve_vs_none"] = round(
                    prob_improvement([f["best_mae"] for f in data[m]], bb), 3)

    # pairwise Fisher with Holm correction
    pairs, ps = [], []
    for a, b in itertools.combinations(data, 2):
        ka, na = rep[a]["k"], rep[a]["n"]
        kb, nb = rep[b]["k"], rep[b]["n"]
        _, p = st.fisher_exact([[ka, na-ka], [kb, nb-kb]])
        pairs.append(f"{a}_vs_{b}"); ps.append(p)
    fisher = {k: dict(p=round(float(p), 6), p_holm=round(float(q), 6))
              for k, p, q in zip(pairs, ps, holm(np.array(ps)))}

    # seed-count subsampling: what smaller studies would have concluded
    sub = {}
    for m, fs in data.items():
        bests = [f["best_mae"] for f in fs]
        sub[m] = {n: [round(v, 3) for v in subsample_study(bests, n)]
                  for n in (3, 5, 10) if n <= len(bests)}

    out = dict(success_threshold=SUCCESS, methods=rep, fisher_holm=fisher,
               seed_subsampling=sub)
    json.dump(out, open(os.path.join(RESULTS, "stats_upgrades.json"), "w"),
              indent=1)
    for m in rep:
        r = rep[m]
        print(f"{m:10s} {r['k']:2d}/{r['n']}  CVaR25 {r['cvar25']:7.3f}  "
              f"rho_c {r['cost_rate']:.5f} (norm {r.get('cost_rate_norm','--')})  "
              f"P(imp vs none) {r.get('p_improve_vs_none','--')}")
    print("\nseed-subsampling (mean success rate [95% range] a smaller study would see):")
    for m, d in sub.items():
        print(f"  {m:10s} " + "  ".join(
            f"n={n}: {v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]" for n, v in d.items()))
    return out


if __name__ == "__main__":
    import sys
    ms = sys.argv[1:] or ["mix", "none", "warmstart", "shield", "jsrl", "rlif",
                          "residual"]
    main(ms)
