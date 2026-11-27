"""Residual RL arm: the NMPC teacher acts on every step, PPO learns an
additive correction. u = clip(teacher + g_scale * agent, -1, 1).

Zero teacher actions enter the training data: the buffer stores the agent's
own correction, which is exactly the action credited by the on-policy update.
Observations are unchanged (the teacher's action is a deterministic function
of the Markov state, so the policy can infer it).

    python training/run_residual.py --search          # 14-trial Optuna, writes results/residual.json
    python training/run_residual.py --seed 3          # one tuned final -> results/ext/residual_s3.json
"""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
for _d in ("core", "training", "evaluation", "analysis", "plant"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import argparse, json, os, sys, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from env200 import Holos200s
from teacher import NMPCTeacher
import run_study as RS


def eval_residual(model, profile_name, scale):
    t = NMPCTeacher()
    env = Holos200s(profile_name, train_mode=False)
    obs, _ = env.reset()
    errs, done = [], False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        u = float(np.clip(t.delta(env) + scale*np.clip(act[0], -1, 1), -1, 1))
        obs, _, te, tr, info = env.step([u])
        errs.append(info["err_spu"])
        done = te or tr
    return float(np.mean(errs)), 0.0


def make_residual_learner(params, seed):
    import torch; torch.set_num_threads(1)
    import stable_baselines3 as sb3
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    import gymnasium as gym

    scale = params["g_scale"]
    base = {k: params[k] for k in ("learning_rate","n_steps","batch_size",
            "n_epochs","gamma","gae_lambda","clip_range","ent_coef","vf_coef")}
    teacher = NMPCTeacher()
    stats = dict(teacher=0)

    class ResidualWrap(gym.Wrapper):
        def step(self, action):
            stats["teacher"] += 1
            d = teacher.delta(self.env)
            u = float(np.clip(d + scale*np.clip(action[0], -1, 1), -1, 1))
            obs, rew, term, trunc, info = self.env.step([u])
            info["violation"] = info.get("violation", 0)
            return obs, rew, term, trunc, info

    def make(rank):
        def _f():
            return ResidualWrap(Holos200s("train", train_mode=True,
                                          seed=1000*seed+rank))
        return _f

    venv = VecMonitor(DummyVecEnv([make(i) for i in range(RS.N_ENVS)]))
    model = sb3.PPO("MlpPolicy", venv, verbose=0, device="cpu", seed=seed,
                    policy_kwargs=dict(net_arch=[params["width"]]*2), **base)
    return model, stats, scale


def train_once(params, seed, steps):
    from stable_baselines3.common.callbacks import BaseCallback
    model, stats, scale = make_residual_learner(params, seed)
    curve = []; safety = dict(violations=0, viol_pow=0, viol_drum=0, viol_err=0, episodes=0)
    best = dict(mae=float("inf"), step=0, evals=None)

    def full_eval():
        return {p: eval_residual(model, p, scale) for p in RS.PROFILES_EVAL}

    class Probe(BaseCallback):
        def __init__(self): super().__init__(); self.next_at = RS.EVAL_EVERY
        def _on_step(self):
            safety["episodes"] += int(np.sum(self.locals["dones"]))
            for info in self.locals["infos"]:
                safety["violations"] += info.get("violation", 0)
                for k in ("viol_pow", "viol_drum", "viol_err"):
                    safety[k] += info.get(k, 0)
            if self.num_timesteps >= self.next_at:
                self.next_at += RS.EVAL_EVERY
                m, _ = eval_residual(model, "test", scale)
                curve.append((int(self.num_timesteps), m))
                if m < best["mae"]:
                    best.update(mae=m, step=int(self.num_timesteps),
                                evals=full_eval())
            return True

    t0 = time.perf_counter(); diverged = False
    try:
        model.learn(total_timesteps=steps, callback=Probe(), progress_bar=False)
    except Exception:
        diverged = True
    final = None if diverged else full_eval()
    return dict(best_mae=best["mae"], best_step=best["step"],
                best_evals=best["evals"], final_evals=final, curve=curve,
                diverged=diverged, violations=int(safety["violations"]),
                viol_pow=int(safety["viol_pow"]),
                viol_drum=int(safety["viol_drum"]),
                viol_err=int(safety["viol_err"]),
                episodes=int(safety["episodes"]),
                teacher_steps=int(stats["teacher"]),
                wall=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--trials", type=int, default=14)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    out_json = os.path.join(RS.RESULTS, "residual.json")

    if a.search:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        def objective(trial):
            p = RS.sample(trial, "none")
            p["g_scale"] = trial.suggest_categorical("g_scale", [0.25, 0.5, 1.0])
            r = train_once(p, seed=0, steps=RS.SEARCH_STEPS)
            print(f"[residual] trial {trial.number:2d}: best {r['best_mae']:8.3f} "
                  f"@{r['best_step']/1e3:.0f}k viol {r['violations']} "
                  f"[{r['wall']:.0f}s]", flush=True)
            return r["best_mae"]
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=a.trials)
        json.dump(dict(method="residual", params=dict(study.best_params),
                       search_best=study.best_value, finals=[]),
                  open(out_json, "w"), indent=1)
        print(f"[residual] search best {study.best_value:.3f}: "
              f"{study.best_params}", flush=True)

    if a.seed is not None:
        params = json.load(open(out_json))["params"]
        r = train_once(params, seed=a.seed, steps=RS.FINAL_STEPS)
        p = os.path.join(RS.RESULTS, "ext", f"residual_s{a.seed}.json")
        json.dump(dict(seed=a.seed, **r), open(p, "w"), indent=1)
        print(f"[residual] s{a.seed}: best {r['best_mae']:.3f} "
              f"@{r['best_step']/1e3:.0f}k viol {r['violations']} "
              f"div={r['diverged']} [{r['wall']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
