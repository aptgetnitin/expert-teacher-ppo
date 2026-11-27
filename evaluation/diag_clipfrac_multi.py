"""Multi-seed + shared-hyperparameter version of diag_clipfrac.py.

Fixes the three weaknesses of the original single run:
  * seeds: evaluation seeds (1..), never the tuning seed 0;
  * confound: --base shared runs every method on ONE base PPO configuration
    (the unguided control's tuned one), keeping only the method's own guidance
    parameters, so clip_range / learning rate no longer differ by method;
  * storage: one small JSON per (method, base, seed) instead of overwriting.

    python evaluation/diag_clipfrac_multi.py --method mix --base tuned  --seed 1
    python evaluation/diag_clipfrac_multi.py --method mix --base shared --seed 1

Output: results/diag/multi/<method>_<base>_s<seed>.json
"""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
for _d in ("core", "training", "evaluation", "analysis", "plant"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import argparse, json, os, types
import numpy as np
import diag_clipfrac as D
from run_study import make_learner, RESULTS

OUT = os.path.join(D.DIAG, "multi")
os.makedirs(OUT, exist_ok=True)
BASE_KEYS = ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
             "gae_lambda", "clip_range", "ent_coef", "vf_coef", "width")
BIN = 25_000


def get_params(method, base):
    p = json.load(open(os.path.join(RESULTS, f"{method}.json")))["params"]
    if base == "shared":
        ref = json.load(open(os.path.join(RESULTS, "none.json")))["params"]
        p = {**p, **{k: ref[k] for k in BASE_KEYS}}
    return p


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def aggregate(records, n_epochs):
    # first minibatch of every update = the only step where pi_theta == pi_old
    first = [r for i, r in enumerate(records)
             if r["epoch"] == 0 and (i == 0 or records[i-1]["epoch"] != 0)]
    inj_rows = [r for r in records if r["n_inj"] > 0]
    by_epoch = {e: dict(
        clipfrac_inj=mean([r["clipfrac_inj"] for r in inj_rows if r["epoch"] == e]),
        clipfrac_norm=mean([r["clipfrac_norm"] for r in inj_rows if r["epoch"] == e]))
        for e in range(n_epochs)}
    n_inj = sum(r["n_inj"] for r in inj_rows if r["epoch"] == 0)
    n_all = sum(r["n_inj"] + r["n_norm"] for r in inj_rows if r["epoch"] == 0)
    series = []
    tmax = max(r["t"] for r in records)
    for lo in range(0, tmax + 1, BIN):
        rows = [r for r in records if lo < r["t"] <= lo + BIN]
        if not rows:
            continue
        last = [r for r in rows if r["epoch"] == n_epochs - 1]
        series.append(dict(
            t=lo + BIN,
            inj_frac=float(sum(r["n_inj"] for r in rows) /
                           max(1, sum(r["n_inj"] + r["n_norm"] for r in rows))),
            clipfrac_inj_last=mean([r["clipfrac_inj"] for r in last]),
            clipfrac_norm_last=mean([r["clipfrac_norm"] for r in last]),
            dev_inj_p95=mean([r["ratio_inj_p95"] for r in last])))
    return dict(
        first_minibatch_clipfrac_inj=mean([r["clipfrac_inj"] for r in first]),
        first_minibatch_clipfrac_norm=mean([r["clipfrac_norm"] for r in first]),
        injected_fraction_while_injecting=(n_inj / n_all if n_all else None),
        by_epoch_while_injecting=by_epoch, series=series)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(D.INJECTION_METHODS))
    ap.add_argument("--base", required=True, choices=["tuned", "shared"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=250_000)
    a = ap.parse_args()

    params = get_params(a.method, a.base)
    model, stats, shared = make_learner(a.method, params, a.seed)
    ob = model.rollout_buffer
    model.rollout_buffer = D.DiagRolloutBuffer(
        ob.buffer_size, ob.observation_space, ob.action_space,
        device=ob.device, gae_lambda=ob.gae_lambda, gamma=ob.gamma, n_envs=ob.n_envs)
    model._diag = []
    model.collect_rollouts = types.MethodType(D._collect_rollouts, model)
    model.train = types.MethodType(D._train, model)
    diverged = False
    try:
        model.learn(total_timesteps=a.steps, progress_bar=False)
    except Exception as e:                      # numerical divergence
        diverged = True
        print(f"[diag-multi] diverged: {e}", flush=True)

    n_epochs = int(params["n_epochs"])
    agg = aggregate(model._diag, n_epochs)
    out = dict(method=a.method, base=a.base, seed=a.seed, steps=a.steps,
               diverged=diverged, params=params, **agg)
    path = os.path.join(OUT, f"{a.method}_{a.base}_s{a.seed}.json")
    json.dump(out, open(path, "w"), indent=1)
    be = agg["by_epoch_while_injecting"]
    print(f"[diag-multi] {a.method} {a.base} s{a.seed}: first-mb inj "
          f"{agg['first_minibatch_clipfrac_inj']}  inj-frac "
          f"{agg['injected_fraction_while_injecting']}  by-epoch inj "
          f"{[None if be[e]['clipfrac_inj'] is None else round(be[e]['clipfrac_inj'], 3) for e in be]}",
          flush=True)


if __name__ == "__main__":
    main()
