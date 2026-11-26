"""Seed extension: run one additional final seed at a method's tuned config.

    python run_more_seeds.py --method shield --seed 7

Writes results/ext/{method}_s{seed}.json. Runs are independent so any number
can execute in parallel; merge_ext.py folds them into the main JSONs.
"""
from __future__ import annotations
import argparse, json, os, sys
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "plant"))
import run_study as RS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    params = json.load(open(os.path.join(RS.RESULTS, f"{a.method}.json")))["params"]
    r = RS.train_once(a.method, params, seed=a.seed, steps=RS.FINAL_STEPS)
    out = os.path.join(RS.RESULTS, "ext", f"{a.method}_s{a.seed}.json")
    json.dump(dict(seed=a.seed, **r), open(out, "w"), indent=1)
    print(f"[{a.method}] s{a.seed}: best {r['best_mae']:.3f} "
          f"@{r['best_step']/1e3:.0f}k viol {r['violations']} "
          f"div={r['diverged']} [{r['wall']:.0f}s]", flush=True)
