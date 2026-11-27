"""Reproduce a specific (method, seed) run to a specific checkpoint step and
save the policy, for the robustness study. Seeds are the paper's evaluation
seeds (1-20); the run replays deterministically.

    python evaluation/robust_ckpt.py --method jsrl --seed 11 --step 1200000
"""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
for _d in ("core", "training", "evaluation", "analysis", "plant"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import argparse, json, os, sys
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import run_study as RS
import run_residual as RR


def reproduce(method, seed, stop_at):
    from stable_baselines3.common.callbacks import BaseCallback
    out = os.path.join(RS.RESULTS, f"rckpt_{method}")
    if method == "residual":
        params = json.load(open(os.path.join(RS.RESULTS, "residual.json")))["params"]
        model, stats, scale = RR.make_residual_learner(params, seed)
        evalf = lambda: RR.eval_residual(model, "test", scale)[0]
        shared = None; g_params = params
    else:
        params = json.load(open(os.path.join(RS.RESULTS, f"{method}.json")))["params"]
        model, stats, shared = RS.make_learner(method, params, seed)
        evalf = lambda: RS.eval_profile(model, "test")[0]
        g_params = params
    stages = [175, 150, 125, 100, 75, 50, 25, 0]

    class Probe(BaseCallback):
        def __init__(self): super().__init__(); self.next_at = RS.EVAL_EVERY
        def _on_step(self):
            if self.num_timesteps >= self.next_at:
                self.next_at += RS.EVAL_EVERY
                m = evalf()
                if self.num_timesteps >= stop_at:
                    print(f"[{method}] s{seed} @{self.num_timesteps}: val {m:.3f} -- saving", flush=True)
                    model.save(out)
                    return False
                if method == "jsrl" and shared is not None and shared["h"] > 0:
                    buf = model.ep_info_buffer
                    if len(buf) >= 20:
                        rate = np.mean([e["r"]/max(e["l"],1) for e in list(buf)[-20:]])
                        if rate > g_params.get("g_gate", 1.0):
                            i = stages.index(shared["h"]) if shared["h"] in stages else 0
                            shared["h"] = stages[min(i+1, len(stages)-1)]
            return True

    model.learn(total_timesteps=stop_at + RS.EVAL_EVERY, callback=Probe(),
                progress_bar=False)
    print(f"[{method}] saved {out}.zip", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--step", type=int, required=True)
    a = ap.parse_args()
    reproduce(a.method, a.seed, a.step)
