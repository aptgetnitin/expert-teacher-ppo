"""Robustness of saved best checkpoints under plant-model mismatch and
sensor noise.

The environment's plant is perturbed; the NMPC teacher keeps planning with
its NOMINAL internal model (that is the point: the teacher assumes an exact
model, a learned policy does not). Sensor noise corrupts the measured power
seen by BOTH sides: the policies' observations and the state the teacher
plans from.

Axes (one at a time, nominal elsewhere):
  worth: drum-reactivity-worth calibration scale on rho_max
  alpha: temperature-feedback coefficient scale on alpha_f/m/c
  noise: N(0, sigma) on measured power, sigma in SPU; 5 repeats

Output: results/robustness.json + figs/fig_robust.png
"""
from __future__ import annotations
import json, os, sys
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "plant"))
from env200 import Holos200s, DTH_MAX
from teacher import NMPCTeacher

R = os.path.join(HERE, "results")
METHODS = ["none", "warmstart", "shield", "jsrl", "rlif", "residual"]
PROFILES = ["test", "lowpower", "sine", "staircase", "randpw"]
HELD = PROFILES[1:]
G_SCALE = json.load(open(os.path.join(R, "residual.json")))["params"]["g_scale"]

WORTH = [0.8, 0.9, 1.0, 1.1, 1.2]
ALPHA = [0.7, 1.0, 1.3]
NOISE = [0.0, 0.25, 0.5, 1.0]
PROCESS = [0.05, 0.1, 0.2]     # deg/s actuator disturbance, vs 0.5 deg/s authority
N_REP = 5

_teacher = NMPCTeacher()   # nominal internal model, never perturbed


def teacher_delta(env, z_meas):
    """teacher.delta but planning from a (possibly noisy) measured state."""
    target = env.profile(env.t + 1) / 100.0
    best_d, best_e = 0.0, 1e9
    lo, hi = -DTH_MAX, DTH_MAX
    for _ in range(2):
        for d in np.linspace(lo, hi, 7):
            ang = float(np.clip(env.ang + d, 0, 180))
            z = _teacher.m.discrete_dynamics(z_meas, [ang], 1.0)
            e = abs(float(z[0]) - target)
            if e < best_e:
                best_e, best_d = e, d
        span = (hi - lo) / 6
        lo, hi = best_d - span, best_d + span
    return float(np.clip(best_d / DTH_MAX, -1, 1))


def make_env(profile, worth=1.0, alpha=1.0):
    env = Holos200s(profile, train_mode=False)
    env.plant.rho_max *= worth
    env.plant.alpha_f *= alpha
    env.plant.alpha_m *= alpha
    env.plant.alpha_c *= alpha
    env.reset()
    return env


def rollout(controller, model, profile, worth=1.0, alpha=1.0, sigma=0.0,
            proc=0.0, rep=0):
    env = make_env(profile, worth, alpha)
    rng = np.random.default_rng(10_000 + rep)
    errs, prev_meas, done, scram = [], None, False, False
    while not done:
        p_meas = env.p + rng.normal(0, sigma / 100.0) if sigma else env.p
        if prev_meas is None:
            prev_meas = p_meas
        if controller == "teacher" or controller == "residual":
            z_meas = env.z.copy(); z_meas[0] = p_meas
            d_t = teacher_delta(env, z_meas)
        if controller == "teacher":
            u = d_t
        else:
            nxt = env.profile(env.t + 1) / 100.0
            obs = np.array([p_meas, nxt, prev_meas, env.ang / 180.0],
                           dtype=np.float32)
            act, _ = model.predict(obs, deterministic=True)
            a = float(np.clip(act[0], -1, 1))
            u = float(np.clip(d_t + G_SCALE * a, -1, 1)) if controller == "residual" else a
        prev_meas = p_meas
        if proc:
            u = float(np.clip(u + rng.normal(0, proc / DTH_MAX), -1, 1))
        _, _, te, tr, info = env.step([u])
        errs.append(info["err_spu"])
        if te:
            scram = True
        done = te or tr
    return float(np.mean(errs)), scram, len(errs)


def main():
    from stable_baselines3 import PPO
    models = {m: PPO.load(os.path.join(R, f"rckpt_{m}.zip"), device="cpu")
              for m in METHODS}
    out = {"axes": dict(worth=WORTH, alpha=ALPHA, noise=NOISE), "runs": []}
    controllers = ["teacher"] + METHODS

    def record(ctrl, axis, level, profile, mae, scram, n):
        out["runs"].append(dict(controller=ctrl, axis=axis, level=level,
                                profile=profile, mae=round(mae, 4),
                                scram=scram, steps=n))

    for w in WORTH:
        for prof in PROFILES:
            for c in controllers:
                mae, sc, n = rollout(c, models.get(c), prof, worth=w)
                record(c, "worth", w, prof, mae, sc, n)
        print(f"worth {w} done", flush=True)
    for al in ALPHA:
        if al == 1.0: continue
        for prof in PROFILES:
            for c in controllers:
                mae, sc, n = rollout(c, models.get(c), prof, alpha=al)
                record(c, "alpha", al, prof, mae, sc, n)
        print(f"alpha {al} done", flush=True)
    for sg in NOISE:
        reps = 1 if sg == 0 else N_REP
        for rep in range(reps):
            for prof in PROFILES:
                for c in controllers:
                    mae, sc, n = rollout(c, models.get(c), prof, sigma=sg, rep=rep)
                    record(c, "noise", sg, prof, mae, sc, n)
        print(f"noise {sg} done", flush=True)

    for pr in PROCESS:
        for rep in range(N_REP):
            for prof in PROFILES:
                for c in controllers:
                    mae, sc, n = rollout(c, models.get(c), prof, proc=pr, rep=rep)
                    record(c, "process", pr, prof, mae, sc, n)
        print(f"process {pr} done", flush=True)

    json.dump(out, open(os.path.join(R, "robustness.json"), "w"), indent=1)
    print("wrote robustness.json,", len(out["runs"]), "rollouts")


if __name__ == "__main__":
    main()
