"""Trajectory figures: best tuned policy per method vs NMPC teacher, all profiles.

Phase 1 (--retrain METHOD): reproduce the best final seed of METHOD with its
recorded hyperparameters and seed (deterministic), stop at its recorded best
checkpoint step, save the policy to results/ckpt_{method}.zip.

Phase 2 (--plot): roll out every saved policy plus the teacher on all six
profiles, save figs/traj_{profile}.png.
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
from env200 import Holos200s
from teacher import NMPCTeacher
import run_study as RS

RESULTS = os.path.join(ROOT, "results")
METHODS = ["none", "warmstart", "mix", "shield", "jsrl", "rlif"]
LABELS = {"none": "PPO (unguided)", "warmstart": "Warmstart", "mix": "Mix",
          "shield": "Shield", "jsrl": "JSRL-style", "rlif": "RLIF-style"}
PROFILES = ["train", "test", "lowpower", "sine", "staircase", "randpw"]
PTITLES = {"train": "Training ramp", "test": "Validation ramp",
           "lowpower": "Low-power (held-out)", "sine": "Sinusoid (held-out)",
           "staircase": "Staircase (held-out)",
           "randpw": "Random piecewise (held-out)"}


def pick_best(method):
    d = json.load(open(os.path.join(RESULTS, f"{method}.json")))
    fs = d["finals"]
    ok = [f for f in fs if f["best_mae"] < 0.30] or fs
    b = min(ok, key=lambda f: f["best_mae"])
    return d["params"], b


def retrain(method):
    from stable_baselines3.common.callbacks import BaseCallback
    params, b = pick_best(method)
    seed, stop_at = b["seed"], b["best_step"]
    print(f"[{method}] reproducing seed {seed} to step {stop_at} "
          f"(recorded best {b['best_mae']:.3f})", flush=True)
    model, stats, shared = RS.make_learner(method, params, seed)
    stages = [175, 150, 125, 100, 75, 50, 25, 0]
    out = os.path.join(RESULTS, f"ckpt_{method}")

    # Mirrors run_study.Probe exactly (eval, then JSRL gate) so the run
    # replays identically; saves and stops at the recorded best step.
    class Probe(BaseCallback):
        def __init__(self): super().__init__(); self.next_at = RS.EVAL_EVERY
        def _on_step(self):
            if self.num_timesteps >= self.next_at:
                self.next_at += RS.EVAL_EVERY
                m, _tv = RS.eval_profile(model, "test")
                if self.num_timesteps >= stop_at:
                    print(f"[{method}] @{self.num_timesteps}: test MAE {m:.3f} "
                          f"(recorded {b['best_mae']:.3f}) -- saving", flush=True)
                    model.save(out)
                    return False
                if method == "jsrl" and shared["h"] > 0:
                    buf = model.ep_info_buffer
                    if len(buf) >= 20:
                        rate = np.mean([e["r"]/max(e["l"],1)
                                        for e in list(buf)[-20:]])
                        if rate > params.get("g_gate", 1.0):
                            i = stages.index(shared["h"]) if shared["h"] in stages else 0
                            shared["h"] = stages[min(i+1, len(stages)-1)]
            return True

    model.learn(total_timesteps=stop_at + RS.EVAL_EVERY, callback=Probe(),
                progress_bar=False)
    print(f"[{method}] saved {out}.zip", flush=True)


def rollout_policy(model, profile):
    env = Holos200s(profile, train_mode=False)
    obs, _ = env.reset()
    ts, ps, tg = [], [], []
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, te, tr, info = env.step(act)
        ts.append(env.t); ps.append(env.p * 100)
        tg.append(env.profile(env.t))
        done = te or tr
    return np.array(ts), np.array(ps), np.array(tg)


def rollout_teacher(profile):
    t = NMPCTeacher()
    env = Holos200s(profile, train_mode=False)
    env.reset()
    ts, ps, tg = [], [], []
    done = False
    while not done:
        d = t.delta(env)
        _, _, te, tr, _ = env.step([d])
        ts.append(env.t); ps.append(env.p * 100)
        tg.append(env.profile(env.t))
        done = te or tr
    return np.array(ts), np.array(ps), np.array(tg)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO
    COLORS = RS_COLORS
    models = {}
    for m in METHODS:
        p = os.path.join(RESULTS, f"ckpt_{m}.zip")
        if os.path.exists(p):
            models[m] = PPO.load(p, device="cpu")
        else:
            print(f"!! missing checkpoint for {m}, skipping", flush=True)
    figdir = os.path.join(ROOT, "figs")
    for prof in PROFILES:
        fig, (ax, axe) = plt.subplots(
            2, 1, figsize=(8.2, 5.4), sharex=True, dpi=150,
            gridspec_kw=dict(height_ratios=[2.6, 1.0], hspace=0.08))
        tt, pt, tg = rollout_teacher(prof)
        ax.plot(tt, tg, color="#555555", ls="--", lw=1.4, label="Demand",
                zorder=1)
        rows = []
        for m in METHODS:
            if m not in models: continue
            ts, ps, _ = rollout_policy(models[m], prof)
            mae = float(np.mean(np.abs(ps - tg[:len(ps)])))
            rows.append((m, ts, ps, mae))
        tmae = float(np.mean(np.abs(pt - tg)))
        for m, ts, ps, mae in rows:
            ax.plot(ts, ps, color=COLORS[m], lw=1.3, alpha=.95,
                    label=f"{LABELS[m]}  ({mae:.2f})", zorder=3)
            axe.plot(ts, np.abs(ps - tg[:len(ps)]), color=COLORS[m], lw=1.0,
                     alpha=.9)
        ax.plot(tt, pt, color="#8a8f8d", lw=1.6, label=f"NMPC teacher  ({tmae:.2f})",
                zorder=4)
        axe.plot(tt, np.abs(pt - tg), color="#8a8f8d", lw=1.2)
        lo, hi = tg.min(), tg.max()
        ax.set_ylim(lo - 8, hi + 8)
        ax.set_ylabel("Power (% rated)")
        ax.set_title(f"{PTITLES[prof]} -- best tuned checkpoint per method "
                     f"(legend: MAE, SPU)", fontsize=10)
        ax.legend(fontsize=7, ncol=2, framealpha=.9)
        ax.grid(alpha=.25, lw=.5)
        axe.set_yscale("log"); axe.set_ylim(1e-3, 70)
        axe.axhline(5.0, color="#c23f3f", ls=":", lw=1.0)
        axe.text(199, 5.6, "training termination threshold", fontsize=6.5,
                 color="#c23f3f", ha="right")
        axe.set_ylabel("|error| (SPU)"); axe.set_xlabel("Time (s)")
        axe.grid(alpha=.25, lw=.5)
        out = os.path.join(figdir, f"traj_{prof}.png")
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print("wrote", out, flush=True)


RS_COLORS = {"none": "#101010", "shield": "#c23f3f", "warmstart": "#2a78d6",
             "mix": "#eb6834", "jsrl": "#1baf7a", "rlif": "#7c5cd6"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", choices=METHODS)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.retrain:
        retrain(a.retrain)
    if a.plot:
        plot()
