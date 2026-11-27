"""Final-policy reliability (seeds 1-20), next to the best-checkpoint numbers.

Requested in review: best-checkpoint selection hides post-peak
divergence. For every run this reports whether the policy at the END of
training (2M steps) is under the same 0.30 SPU validation threshold. A run
that diverged numerically counts as a failure. No training is run; the final
evaluations were stored by run_study.py / run_residual.py.

    python evaluation/final_policy.py   ->  results/final_policy_1_20.json
"""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
for _d in ("core", "training", "evaluation", "analysis", "plant"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import json, os
import numpy as np
from analyze20 import load, clopper, METHODS, RESULTS, SUCCESS, HELD_OUT


def main():
    out = {}
    for m in METHODS:
        fs = load(m)
        best = np.array([f["best_mae"] for f in fs])
        final = np.array([np.inf if (f["diverged"] or not f["final_evals"])
                          else f["final_evals"]["test"][0] for f in fs])
        kb, kf, n = int((best < SUCCESS).sum()), int((final < SUCCESS).sum()), len(fs)
        # reliable at best checkpoint but not at the end of training
        lost = int(((best < SUCCESS) & ~(final < SUCCESS)).sum())
        ok = final < SUCCESS
        row = dict(n=n, best_reliable=kb, final_reliable=kf,
                   final_ci=clopper(kf, n), lost_after_peak=lost,
                   diverged=int(sum(f["diverged"] for f in fs)),
                   final_median=float(np.median(final)) if np.isfinite(np.median(final)) else None,
                   final_ok_mean=float(final[ok].mean()) if kf else None,
                   final_ok_heldout=float(np.mean(
                       [np.mean([f["final_evals"][p][0] for p in HELD_OUT])
                        for f, o in zip(fs, ok) if o])) if kf else None,
                   final=[None if not np.isfinite(v) else round(float(v), 3) for v in final])
        out[m] = row
        print(f"{m:10s} best {kb:2d}/{n}  final {kf:2d}/{n} "
              f"CI [{row['final_ci'][0]:.2f},{row['final_ci'][1]:.2f}]  "
              f"lost after peak {lost:2d}  diverged {row['diverged']:2d}  "
              f"final-ok mean {row['final_ok_mean']}")
    json.dump(dict(success_threshold=SUCCESS, methods=out),
              open(os.path.join(RESULTS, "final_policy_1_20.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
