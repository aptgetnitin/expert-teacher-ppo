"""Residual configuration BEFORE any training (checkpoint 0), seeds 1-20.

Requested in review: the residual controller is teacher + bounded
correction, so it may already be under the 0.30 SPU success threshold at
random initialisation. This evaluates the untrained residual policy
(deterministic action, same evaluation as run_residual.eval_residual) on the
validation profile and the four held-out profiles.

    python evaluation/residual_ckpt0.py   ->  results/residual_ckpt0.json
"""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
for _d in ("core", "training", "evaluation", "analysis", "plant"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import json, os
import numpy as np
import run_study as RS
from run_residual import make_residual_learner, eval_residual

HELD_OUT = ["lowpower", "sine", "staircase", "randpw"]


def main():
    params = json.load(open(os.path.join(RS.RESULTS, "residual.json")))["params"]
    rows = []
    for seed in range(1, 21):
        model, _, scale = make_residual_learner(params, seed)
        ev = {p: eval_residual(model, p, scale)[0] for p in RS.PROFILES_EVAL}
        rows.append(dict(seed=seed, **ev))
        print(f"seed {seed:2d}: validation {ev['test']:.3f}  "
              f"held-out {np.mean([ev[p] for p in HELD_OUT]):.3f}", flush=True)
    val = np.array([r["test"] for r in rows])
    ho = np.array([np.mean([r[p] for p in HELD_OUT]) for r in rows])
    summ = dict(n=len(rows), under_0p30=int((val < 0.30).sum()),
                val_mean=float(val.mean()), val_min=float(val.min()),
                val_max=float(val.max()), heldout_mean=float(ho.mean()))
    print(summ)
    json.dump(dict(params=params, summary=summ, seeds=rows),
              open(os.path.join(RS.RESULTS, "residual_ckpt0.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
