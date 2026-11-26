"""Shared-hyperparameter outcome control (full 2M-step runs).

Every method in the study was tuned separately, so mix's collapse could in
principle be a property of mix's own tuned base PPO settings (highest learning
rate, tighter clip) rather than of action injection. This runs an injection
method on the UNGUIDED control's tuned base PPO configuration, keeping only the
method's own guidance parameters, with the same protocol as the study finals.

    python run_shared_hp.py --method mix --seed 1
    ->  results/shared/<method>_s<seed>.json
"""
from __future__ import annotations
import argparse, json, os
import run_study as RS
from diag_clipfrac_multi import get_params

OUT = os.path.join(RS.RESULTS, "shared")
os.makedirs(OUT, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["mix", "shield", "warmstart"])
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    params = get_params(a.method, "shared")
    r = RS.train_once(a.method, params, seed=a.seed, steps=RS.FINAL_STEPS)
    json.dump(dict(seed=a.seed, params=params, **r),
              open(os.path.join(OUT, f"{a.method}_s{a.seed}.json"), "w"), indent=1)
    final = None if r["diverged"] else r["final_evals"]["test"][0]
    print(f"[shared] {a.method} s{a.seed}: best {r['best_mae']:.3f} "
          f"@{r['best_step']/1e3:.0f}k  final {final}  viol {r['violations']} "
          f"div={r['diverged']} [{r['wall']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
