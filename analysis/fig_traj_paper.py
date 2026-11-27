"""Two-profile trajectory figure for the paper: validation ramp (seen
family) and staircase (held-out, hardest), same academic style."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
for _d in ("core", "training", "evaluation", "analysis", "plant"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import os, sys, shutil
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral","Times New Roman"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 200})
from robustness import teacher_delta, G_SCALE
from env200 import Holos200s
from stable_baselines3 import PPO
R=os.path.join(ROOT,"results")
C={"none":"#000000","jsrl":"#009E73","residual":"#0072B2","teacher":"#8a8a8a"}
L={"none":"PPO (unguided)","jsrl":"JSRL-style","residual":"Residual",
   "teacher":"NMPC teacher"}

def run(ctrl, profile):
    env=Holos200s(profile, train_mode=False)
    obs,_=env.reset(); ps=[]; done=False
    mod=None if ctrl=="teacher" else PPO.load(os.path.join(R,f"rckpt_{ctrl}.zip"),device="cpu")
    while not done:
        if ctrl=="teacher":
            u=teacher_delta(env, env.z.copy())
        elif ctrl=="residual":
            d=teacher_delta(env, env.z.copy())
            a,_=mod.predict(obs,deterministic=True)
            u=float(np.clip(d+G_SCALE*np.clip(a[0],-1,1),-1,1))
        else:
            a,_=mod.predict(obs,deterministic=True); u=float(np.clip(a[0],-1,1))
        obs,_,te,tr,_=env.step([u]); ps.append(100*env.p); done=te or tr
    return np.array(ps)

fig,axes=plt.subplots(2,2,figsize=(9.4,4.2),sharex="col",
                      gridspec_kw=dict(height_ratios=[2.1,1.0],hspace=0.12,wspace=0.16))
for col,(prof,title) in enumerate([("test","Validation ramp"),
                                   ("staircase","Staircase (held-out)")]):
    env=Holos200s(prof,train_mode=False); env.reset()
    ts=np.arange(1,201); tg=np.array([env.profile(t) for t in ts])
    aP,aE=axes[0,col],axes[1,col]
    aP.plot(ts,tg,color="#bbbbbb",ls="--",lw=1.3,label="Demand",zorder=1)
    for ctrl in ["none","jsrl","residual","teacher"]:
        ps=run(ctrl,prof)
        ls="--" if ctrl=="teacher" else "-"
        aP.plot(ts[:len(ps)],ps,ls,color=C[ctrl],lw=1.0,label=L[ctrl],zorder=3)
        aE.plot(ts[:len(ps)],np.abs(ps-tg[:len(ps)]),ls,color=C[ctrl],lw=0.8)
    aP.set_title(title)
    aE.set_yscale("log"); aE.set_ylim(1e-3,10)
    aE.axhline(5.0,color="#cccccc",lw=0.6)
    aE.set_xlabel("Time (s)")
axes[0,0].set_ylabel("Power (% rated)")
axes[1,0].set_ylabel("|error| (SPU)")
axes[0,1].legend(loc="center left",bbox_to_anchor=(1.02,0.35))
fig.savefig(os.path.join(ROOT,"figs","fig_traj2.png"),bbox_inches="tight")
print("fig_traj2.png written")
