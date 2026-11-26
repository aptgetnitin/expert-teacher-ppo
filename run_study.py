"""Tuned comparative study driver -- one method per invocation.

Methods: none (unguided PPO), shield, warmstart, mix, jsrl, rlif.
Each method: Optuna search (base PPO hyperparameters jointly with the
method's guidance parameters; 14 trials x 800k steps, scored by
best-checkpoint test MAE), then the winning configuration for 6 seeds x
2M steps with evaluation every 100k and best-checkpoint selection.

    python run_study.py --method shield
"""
from __future__ import annotations
import argparse, json, os, sys, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "plant"))
from env200 import Holos200s, DTH_MAX
from teacher import NMPCTeacher

RESULTS = os.path.join(HERE, "results")
N_ENVS = 10
SEARCH_STEPS, FINAL_STEPS, FINAL_SEEDS = 800_000, 2_000_000, 6
EVAL_EVERY = 100_000
PROFILES_EVAL = ["test", "lowpower", "sine", "staircase", "randpw"]


def eval_profile(model, profile_name):
    env = Holos200s(profile_name, train_mode=False)
    obs, _ = env.reset()
    errs, travel, prev = [], 0.0, None
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, te, tr, info = env.step(act)
        errs.append(info["err_spu"])
        if prev is not None:
            travel += abs(info["ang"] - prev)
        prev = info["ang"]
        done = te or tr
    return float(np.mean(errs)), float(travel)


def make_learner(method, params, seed):
    import torch; torch.set_num_threads(1)
    import torch as th
    import stable_baselines3 as sb3
    from stable_baselines3.common.utils import obs_as_tensor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    import gymnasium as gym

    g = {k: params[k] for k in params if k.startswith("g_")}
    base = {k: params[k] for k in ("learning_rate","n_steps","batch_size",
            "n_epochs","gamma","gae_lambda","clip_range","ent_coef","vf_coef")}
    width = params["width"]
    teacher = NMPCTeacher()
    stats = dict(teacher=0)
    shared = dict(h=int(g.get("g_h0", 0)))

    class Counting(gym.Wrapper):
        def step(self, action):
            obs, rew, term, trunc, info = self.env.step(action)
            info["violation"] = info.get("violation", 0)
            return obs, rew, term, trunc, info

    class JSRLWrap(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env); self._cache = {}
        def reset(self, **kw):
            obs, info = self.env.reset(**kw)
            h = int(shared["h"])
            if h == 0:
                return obs, info
            if h not in self._cache:
                for _ in range(h):
                    d = teacher.delta(self.env)
                    obs, _, te, tr, _ = self.env.step([d])
                    if te or tr:
                        obs, info = self.env.reset(**kw)
                e = self.env
                self._cache[h] = (e.z.copy(), e.ang, e.t, e.p, e.p_prev, obs.copy())
            z, ang, t2, pp, ppv, obs = self._cache[h]
            e = self.env
            e.z = z.copy(); e.ang = ang; e.t = t2; e.p = pp; e.p_prev = ppv
            e.ep_reward = 0.0
            return obs.copy(), info

    class RLIFWrap(gym.Wrapper):
        def step(self, action):
            obs, rew, term, trunc, info = self.env.step(action)
            b = self.env
            tgt = b.profile(b.t) / 100.0
            if not (term or trunc) and (100*abs(b.p-tgt) > g["g_trig"]
                                        or b.ang < 4 or b.ang > 176):
                stats["teacher"] += 1
                rew -= g["g_penalty"]
                k = 0
                while k < int(g["g_recov"]) and not (term or trunc):
                    d = teacher.delta(b)
                    obs, _, term, trunc, ri = self.env.step([d])
                    info["violation"] = info.get("violation",0)+ri.get("violation",0)
                    info["err_spu"] = ri["err_spu"]
                    if 100*abs(b.p - b.profile(b.t)/100.0) < 1.0:
                        break
                    k += 1
            return obs, rew, term, trunc, info

    def make(rank):
        def _f():
            env = Holos200s("train", train_mode=True, seed=1000*seed+rank)
            if method == "jsrl": env = JSRLWrap(env)
            elif method == "rlif": env = RLIFWrap(env)
            return Counting(env)
        return _f

    venv = VecMonitor(DummyVecEnv([make(i) for i in range(N_ENVS)]))

    class GuidedPPO(sb3.PPO):
        def _override(self, actions):
            if method not in ("warmstart", "mix", "shield"):
                return actions, False
            out = actions.copy(); changed = False
            for i, cw in enumerate(self.env.envs):
                be = cw.env
                a_agent = float(np.clip(actions[i, 0], -1, 1))
                if method == "warmstart" and self.num_timesteps < g["g_window"]:
                    out[i, 0] = teacher.delta(be); changed = True
                    stats["teacher"] += 1
                elif method == "mix" and self.num_timesteps < g["g_window"]:
                    al = 1.0 - self.num_timesteps / g["g_window"]
                    out[i, 0] = np.clip(al*teacher.delta(be)+(1-al)*a_agent, -1, 1)
                    changed = True; stats["teacher"] += 1
                elif method == "shield":
                    tgt = be.profile(be.t) / 100.0
                    if (100*abs(be.p-tgt) > g["g_trig"]
                            or be.ang < g["g_margin"]
                            or be.ang > 180 - g["g_margin"]):
                        out[i, 0] = teacher.delta(be); changed = True
                        stats["teacher"] += 1
            return out, changed

        def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
            assert self._last_obs is not None
            self.policy.set_training_mode(False)
            n_steps = 0; rollout_buffer.reset(); callback.on_rollout_start()
            while n_steps < n_rollout_steps:
                with th.no_grad():
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    actions, values, log_probs = self.policy(obs_tensor)
                actions = actions.cpu().numpy()
                actions, changed = self._override(actions)
                if changed:
                    with th.no_grad():
                        a_t = th.as_tensor(actions, dtype=th.float32,
                                           device=self.device)
                        values, log_probs, _ = self.policy.evaluate_actions(
                            obs_tensor, a_t)
                clipped = np.clip(actions, self.action_space.low,
                                  self.action_space.high)
                new_obs, rewards, dones, infos = env.step(clipped)
                self.num_timesteps += env.num_envs
                callback.update_locals(locals())
                if not callback.on_step():
                    return False
                self._update_info_buffer(infos, dones)
                n_steps += 1
                for idx, done in enumerate(dones):
                    if (done and infos[idx].get("terminal_observation") is not None
                            and infos[idx].get("TimeLimit.truncated", False)):
                        tobs = self.policy.obs_to_tensor(
                            infos[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            tval = self.policy.predict_values(tobs)[0]
                        rewards[idx] += self.gamma * tval
                rollout_buffer.add(self._last_obs, actions, rewards,
                                   self._last_episode_starts, values, log_probs)
                self._last_obs = new_obs
                self._last_episode_starts = dones
            with th.no_grad():
                values = self.policy.predict_values(
                    obs_as_tensor(new_obs, self.device))
            rollout_buffer.compute_returns_and_advantage(last_values=values,
                                                         dones=dones)
            callback.update_locals(locals()); callback.on_rollout_end()
            return True

    model = GuidedPPO("MlpPolicy", venv, verbose=0, device="cpu", seed=seed,
                      policy_kwargs=dict(net_arch=[width, width]), **base)
    model.env.envs = venv.venv.envs
    return model, stats, shared


def train_once(method, params, seed, steps):
    from stable_baselines3.common.callbacks import BaseCallback
    model, stats, shared = make_learner(method, params, seed)
    curve = []; safety = dict(violations=0, viol_pow=0, viol_drum=0, viol_err=0, episodes=0)
    best = dict(mae=float("inf"), step=0, evals=None)
    stages = [175, 150, 125, 100, 75, 50, 25, 0]

    def full_eval():
        return {p: eval_profile(model, p) for p in PROFILES_EVAL}

    class Probe(BaseCallback):
        def __init__(self): super().__init__(); self.next_at = EVAL_EVERY
        def _on_step(self):
            safety["episodes"] += int(np.sum(self.locals["dones"]))
            for info in self.locals["infos"]:
                safety["violations"] += info.get("violation", 0)
                for k in ("viol_pow", "viol_drum", "viol_err"):
                    safety[k] += info.get(k, 0)
            if self.num_timesteps >= self.next_at:
                self.next_at += EVAL_EVERY
                m, _tv = eval_profile(model, "test")
                curve.append((int(self.num_timesteps), m))
                if m < best["mae"]:
                    best.update(mae=m, step=int(self.num_timesteps),
                                evals=full_eval())
                if method == "jsrl" and shared["h"] > 0:
                    buf = model.ep_info_buffer
                    if len(buf) >= 20:
                        rate = np.mean([e["r"]/max(e["l"],1)
                                        for e in list(buf)[-20:]])
                        if rate > params.get("g_gate", 1.0):
                            i = stages.index(shared["h"]) if shared["h"] in stages else 0
                            shared["h"] = stages[min(i+1, len(stages)-1)]
            return True

    t0 = time.perf_counter()
    diverged = False
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


def sample(trial, method):
    p = dict(
        learning_rate=trial.suggest_float("learning_rate", 3e-5, 1e-3, log=True),
        n_steps=trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
        n_epochs=trial.suggest_categorical("n_epochs", [3, 5, 10]),
        gamma=trial.suggest_categorical("gamma", [0.99, 0.995, 0.999]),
        gae_lambda=trial.suggest_categorical("gae_lambda", [0.9, 0.95]),
        clip_range=trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        ent_coef=trial.suggest_float("ent_coef", 1e-8, 1e-3, log=True),
        vf_coef=trial.suggest_categorical("vf_coef", [0.5, 1.0]),
        width=trial.suggest_categorical("width", [64, 128]))
    if method in ("warmstart", "mix"):
        p["g_window"] = trial.suggest_categorical("g_window",
                                                  [100_000, 200_000, 400_000])
    if method == "shield":
        p["g_trig"] = trial.suggest_categorical("g_trig", [2.0, 3.0, 4.0])
        p["g_margin"] = trial.suggest_categorical("g_margin", [2.0, 4.0, 8.0])
    if method == "jsrl":
        p["g_h0"] = trial.suggest_categorical("g_h0", [175, 150, 100])
        p["g_gate"] = trial.suggest_categorical("g_gate", [0.5, 1.0, 1.5])
    if method == "rlif":
        p["g_trig"] = trial.suggest_categorical("g_trig", [3.0, 4.0])
        p["g_penalty"] = trial.suggest_categorical("g_penalty", [1.0, 5.0, 10.0])
        p["g_recov"] = trial.suggest_categorical("g_recov", [4, 12])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["none","shield","warmstart","mix","jsrl","rlif"])
    ap.add_argument("--trials", type=int, default=14)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        globals()["SEARCH_STEPS"] = 30_000
        globals()["FINAL_STEPS"] = 30_000
        globals()["FINAL_SEEDS"] = 1
        globals()["EVAL_EVERY"] = 10_000

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = sample(trial, a.method)
        r = train_once(a.method, p, seed=0, steps=SEARCH_STEPS)
        print(f"[{a.method}] trial {trial.number:2d}: best {r['best_mae']:8.3f} "
              f"@{r['best_step']/1e3:.0f}k  viol {r['violations']}  "
              f"[{r['wall']:.0f}s]", flush=True)
        return r["best_mae"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=a.trials)
    bestp = dict(study.best_params)
    print(f"[{a.method}] search best {study.best_value:.3f}: {bestp}", flush=True)

    finals = []
    for seed in range(FINAL_SEEDS):
        r = train_once(a.method, bestp, seed=seed, steps=FINAL_STEPS)
        finals.append(dict(seed=seed, **r))
        print(f"[{a.method}] final s{seed}: best {r['best_mae']:.3f} "
              f"@{r['best_step']/1e3:.0f}k  viol {r['violations']}  "
              f"div={r['diverged']}  [{r['wall']:.0f}s]", flush=True)
        json.dump(dict(method=a.method, params=bestp,
                       search_best=study.best_value, finals=finals),
                  open(os.path.join(RESULTS, f"{a.method}.json"), "w"), indent=1)
    print(f"[{a.method}] COMPLETE", flush=True)


if __name__ == "__main__":
    main()
