"""Typed-violation re-run: deterministically replay one tuned final and
record the per-type violation decomposition (overpower / drum-limit / error).

    python evaluation/run_typed.py --method none --seed 7

Writes results/typed/{method}_s{seed}.json and cross-checks the replayed
total and best MAE against the originally recorded run.
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
import run_study as RS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    d = json.load(open(os.path.join(RS.RESULTS, f"{a.method}.json")))
    if a.seed < len(d["finals"]):
        rec = d["finals"][a.seed]
    else:
        rec = json.load(open(os.path.join(RS.RESULTS, "ext",
                                          f"{a.method}_s{a.seed}.json")))
    r = RS.train_once(a.method, d["params"], seed=a.seed, steps=RS.FINAL_STEPS)
    match = (r["violations"] == rec["violations"]
             and abs(r["best_mae"] - rec["best_mae"]) < 1e-9)
    out = dict(seed=a.seed, viol_pow=r["viol_pow"], viol_drum=r["viol_drum"],
               viol_err=r["viol_err"], violations=r["violations"],
               episodes=r["episodes"], best_mae=r["best_mae"],
               matches_original=match)
    json.dump(out, open(os.path.join(RS.RESULTS, "typed",
                                     f"{a.method}_s{a.seed}.json"), "w"),
              indent=1)
    print(f"[{a.method}] s{a.seed}: pow={r['viol_pow']} drum={r['viol_drum']} "
          f"err={r['viol_err']} total={r['violations']} "
          f"{'MATCH' if match else 'MISMATCH vs recorded '+str(rec['violations'])}",
          flush=True)
