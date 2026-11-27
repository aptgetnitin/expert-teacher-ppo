"""Regenerate every figure in a consistent academic style.

Style: STIX serif (matches LaTeX), Okabe-Ito colorblind-safe palette,
no chart junk, legends outside the axes, sentence-style axis labels.
Outputs to figs/.
"""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
for _d in ("core", "training", "evaluation", "analysis", "plant"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import json, glob, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.dpi": 200,
})

R = os.path.join(ROOT, "results")
F = os.path.join(ROOT, "figs")

# Okabe-Ito
C = {"none": "#000000", "warmstart": "#56B4E9", "mix": "#E69F00",
     "shield": "#D55E00", "jsrl": "#009E73", "rlif": "#CC79A7",
     "residual": "#0072B2", "teacher": "#8a8a8a"}
L = {"none": "PPO (unguided)", "warmstart": "Warmstart", "mix": "Mix",
     "shield": "Shield", "jsrl": "JSRL-style", "rlif": "RLIF-style",
     "residual": "Residual", "teacher": "NMPC teacher"}
METHODS = ["mix", "warmstart", "none", "shield", "jsrl", "rlif", "residual"]

def finals(m):
    seen = {}
    p = os.path.join(R, f"{m}.json")
    if os.path.exists(p):
        for f in json.load(open(p)).get("finals", []):
            seen[f["seed"]] = f
    for q in glob.glob(os.path.join(R, "ext", f"{m}_s*.json")):
        f = json.load(open(q)); seen[f["seed"]] = f
    return [seen[s] for s in sorted(seen) if 1 <= s <= 20]

def save(fig, name):
    fig.savefig(os.path.join(F, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)

teacher_val = 0.075

# ---- 1. reliability strip ------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.0))
stats = {}
for i, m in enumerate(METHODS):
    b = np.array([f["best_mae"] for f in finals(m)])
    ok = int((b < 0.30).sum()); stats[m] = ok
    x = i + (np.arange(len(b)) - len(b)/2) * 0.022
    ax.scatter(x, np.clip(b, None, 100), s=14, color=C[m], alpha=0.85,
               edgecolors="none", zorder=3)
    ax.text(i, 200, f"{ok}/20", ha="center", fontsize=8.5)
ax.axhline(0.30, color="#999999", lw=0.7)
ax.axhline(teacher_val, color="#999999", lw=0.7, ls="--")
ax.set_xlim(-0.85, 6.55)
ax.text(-0.72, 0.345, "success\nthreshold", fontsize=7, color="#777777",
        ha="left", va="bottom")
ax.text(-0.72, 0.082, "teacher", fontsize=7, color="#777777", ha="left",
        va="bottom")
ax.set_yscale("log"); ax.set_ylim(0.05, 700)
ax.set_xticks(range(len(METHODS)))
ax.set_xticklabels([L[m].replace(" (unguided)", "\n(unguided)") for m in METHODS],
                   fontsize=7.5)
ax.set_ylabel("Validation MAE at best checkpoint (SPU)")
ax.grid(axis="x", visible=False)
save(fig, "fig20_reliability.png")

# ---- 2. performance profile ---------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.0))
taus = np.logspace(np.log10(0.06), np.log10(80), 300)
for m in METHODS:
    b = np.array([f["best_mae"] for f in finals(m)])
    ax.plot(taus, [(b <= t).mean() for t in taus], color=C[m], lw=1.4,
            label=L[m])
ax.axvline(0.30, color="#999999", lw=0.7, ls=":")
ax.text(0.32, 0.04, "0.30", fontsize=7, color="#777777")
ax.set_xscale("log")
ax.set_xlabel(r"Error threshold $\tau$ (SPU)")
ax.set_ylabel(r"Fraction of seeds with MAE $\leq \tau$")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
save(fig, "fig20_profile.png")

# ---- 3. subsampling ------------------------------------------------------
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(5.2, 2.9))
sizes = [3, 5, 10, 20]
offs = np.linspace(-0.30, 0.30, len(METHODS))
for off, m in zip(offs, METHODS):
    ok = (np.array([f["best_mae"] for f in finals(m)]) < 0.30).astype(float)
    lo, hi, mean = [], [], []
    for n in sizes:
        s = (np.array([ok.mean()]) if n == 20 else
             np.array([rng.choice(ok, n, replace=False).mean()
                       for _ in range(4000)]))
        lo.append(np.percentile(s, 2.5)); hi.append(np.percentile(s, 97.5))
        mean.append(ok.mean())
    x = np.arange(len(sizes)) + off
    ax.errorbar(x, mean, yerr=[np.array(mean)-lo, np.array(hi)-np.array(mean)],
                fmt="o", ms=3, color=C[m], lw=1.0, capsize=2, label=L[m])
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels([f"{n} seeds" for n in sizes])
ax.set_ylim(-0.05, 1.06)
ax.set_ylabel("Reported success rate")
ax.grid(axis="x", visible=False)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
save(fig, "fig20_subsample.png")

# ---- 4. learning curves, 7 panels ---------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.4), sharex=True, sharey=True)
order = ["mix", "warmstart", "none", "shield", "jsrl", "rlif", "residual"]
for ax, m in zip(axes.flat, order):
    for f in finals(m):
        if not f["curve"]:
            continue
        xs, ys = zip(*f["curve"])
        ax.plot(np.array(xs)/1e6, np.clip(ys, 0.05, 120), color=C[m],
                lw=0.6, alpha=0.5)
    ax.set_yscale("log"); ax.set_ylim(0.05, 120)
    ax.axhline(0.30, color="#999999", lw=0.6)
    k = sum(1 for f in finals(m) if f["best_mae"] < 0.30)
    ax.set_title(f"{L[m]} ({k}/20)")
axes.flat[-1].axis("off")
for ax in axes[1]:
    ax.set_xlabel("Training steps (millions)")
for ax in axes[:, 0]:
    ax.set_ylabel("Validation MAE (SPU)")
fig.tight_layout()
save(fig, "fig20_curves.png")

# ---- 5. robustness, 4 panels --------------------------------------------
rob = json.load(open(os.path.join(R, "robustness.json")))["runs"]
DIRECT = ["none", "warmstart", "shield", "jsrl", "rlif"]
def held(c, axis, lv):
    rs = [r for r in rob if r["controller"] == c and r["axis"] == axis
          and r["level"] == lv and r["profile"] != "test"]
    return float(np.mean([r["mae"] for r in rs])) if rs else float("nan")
def nominal(c): return held(c, "worth", 1.0)
fig, axes = plt.subplots(1, 4, figsize=(9.8, 2.7), sharey=True)
panels = [("worth", [0.8, 0.9, 1.0], "Drum-worth scale $k_w$", "(a) Calibration error"),
          ("alpha", [0.7, 1.0, 1.3], r"Feedback scale $k_\alpha$", "(b) Feedback error"),
          ("noise", [0.0, 0.25, 0.5, 1.0], r"Sensor noise $\sigma$ (SPU)", "(c) Sensor noise"),
          ("process", [0.0, 0.05, 0.1, 0.2], r"Disturbance $\sigma_a$ ($^{\circ}$/s)", "(d) Process noise")]
for ax, (axis, levels, xlab, title) in zip(axes, panels):
    def val(c, lv):
        if (axis in ("noise", "process") and lv == 0.0) or \
           (axis in ("worth", "alpha") and lv == 1.0):
            return nominal(c)
        return held(c, axis, lv)
    band = np.array([[val(c, lv) for lv in levels] for c in DIRECT])
    ax.fill_between(levels, band.min(0), band.max(0), color="#0072B2",
                    alpha=0.15, lw=0)
    ax.plot(levels, band.mean(0), color="#0072B2", lw=1.5, marker="o", ms=3,
            label="Learned policies (5)")
    ax.plot(levels, [val("teacher", lv) for lv in levels], "--",
            color="#000000", lw=1.4, marker="s", ms=3, label="NMPC teacher")
    ax.plot(levels, [val("residual", lv) for lv in levels], ":",
            color="#D55E00", lw=1.6, marker="^", ms=3, label="Residual")
    ax.set_yscale("log"); ax.set_ylim(0.13, 70)
    ax.set_xlabel(xlab); ax.set_title(title)
axes[0].set_ylabel("Held-out MAE (SPU)")
axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout()
save(fig, "fig_robust.png")

# ---- 6. trajectory (validation ramp), clean 2-panel ----------------------
import sys
from robustness import rollout, make_env, teacher_delta, G_SCALE
from env200 import Holos200s
from stable_baselines3 import PPO
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 3.9), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1.0],
                                              hspace=0.12))
env = Holos200s("test", train_mode=False); env.reset()
ts = np.arange(1, 201); tg = [env.profile(t) for t in ts]
a1.plot(ts, tg, color="#999999", ls="--", lw=1.2, label="Demand")
show = ["none", "jsrl", "residual"]
for m in show:
    mod = PPO.load(os.path.join(R, f"rckpt_{m}.zip"), device="cpu")
    env = Holos200s("test", train_mode=False)
    obs, _ = env.reset(); ps = []; done = False
    while not done:
        if m == "residual":
            d = teacher_delta(env, env.z.copy())
            act, _ = mod.predict(obs, deterministic=True)
            u = float(np.clip(d + G_SCALE*np.clip(act[0], -1, 1), -1, 1))
        else:
            act, _ = mod.predict(obs, deterministic=True); u = act
        obs, _, te, tr, info = env.step([u] if m == "residual" else u)
        ps.append(100*env.p); done = te or tr
    a1.plot(ts[:len(ps)], ps, color=C[m], lw=1.0, label=L[m])
    a2.plot(ts[:len(ps)], np.abs(np.array(ps) - np.array(tg[:len(ps)])),
            color=C[m], lw=0.8)
env = Holos200s("test", train_mode=False); env.reset()
ps = []; done = False
while not done:
    u = teacher_delta(env, env.z.copy())
    _, _, te, tr, _ = env.step([u]); ps.append(100*env.p); done = te or tr
a1.plot(ts[:len(ps)], ps, color=C["teacher"], lw=1.0, ls="--",
        label="NMPC teacher")
a2.plot(ts[:len(ps)], np.abs(np.array(ps) - np.array(tg[:len(ps)])),
        color=C["teacher"], lw=0.8, ls="--")
a1.set_ylabel("Power (% rated)")
a1.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
a2.set_yscale("log"); a2.set_ylim(1e-3, 10)
a2.set_ylabel("|error| (SPU)"); a2.set_xlabel("Time (s)")
save(fig, "fig_traj_clean.png")
print("all figures done")
