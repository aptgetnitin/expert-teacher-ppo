"""The NMPC teacher: a one-step model predictive controller.

At each control step it searches the admissible drum speeds, using the same
plant model the simulator uses, for the speed that brings the next second's
power closest to the demanded value (coarse grid, one refinement).  This is
the classical controller of the study: it teaches during PPO training, and
it is also evaluated standalone as the model-based baseline.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "plant"))
from nmpc import ReactorModel
from env200 import Holos200s, DTH_MAX


class NMPCTeacher:
    def __init__(self):
        self.m = ReactorModel(num_drums=1, dt=1)

    def delta(self, env) -> float:
        """Best drum-speed for the next step, in agent action units [-1, 1]."""
        target = env.profile(env.t + 1) / 100.0
        best_d, best_e = 0.0, 1e9
        lo, hi = -DTH_MAX, DTH_MAX
        for _ in range(2):
            for d in np.linspace(lo, hi, 7):
                ang = float(np.clip(env.ang + d, 0, 180))
                z = self.m.discrete_dynamics(env.z, [ang], 1.0)
                e = abs(float(z[0]) - target)
                if e < best_e:
                    best_e, best_d = e, d
            span = (hi - lo) / 6
            lo, hi = best_d - span, best_d + span
        return float(np.clip(best_d / DTH_MAX, -1, 1))


def evaluate_teacher(profiles):
    """The teacher deployed as the controller: MAE, effort, ms/decision."""
    t = NMPCTeacher()
    out = {}
    for name in profiles:
        env = Holos200s(name, train_mode=False)
        env.reset()
        errs, travel, prev = [], 0.0, None
        tic = time.perf_counter()
        n = 0
        done = False
        while not done:
            d = t.delta(env)
            n += 1
            _, _, te, tr, info = env.step([d])
            errs.append(info["err_spu"])
            if prev is not None:
                travel += abs(info["ang"] - prev)
            prev = info["ang"]
            done = te or tr
        out[name] = dict(mae=float(np.mean(errs)), effort=float(travel),
                         ms_per_decision=1e3*(time.perf_counter()-tic)/n)
    return out


if __name__ == "__main__":
    import json
    profs = ["test", "lowpower", "sine", "staircase", "randpw"]
    r = evaluate_teacher(profs)
    json.dump(dict(method="nmpc_teacher", evals=r),
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results", "teacher.json"), "w"), indent=1)
    for p, v in r.items():
        print(f"  {p:10s} MAE {v['mae']:6.3f}  effort {v['effort']:7.1f} deg  "
              f"{v['ms_per_decision']:.2f} ms/decision")
