"""Correctness audit for the warmstart / mix / shield override logic.

Stubs the teacher with a constant action (0.37) so every override is
predictable, then checks _override outputs and the rollout buffer contents
against hand-computed expectations.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
for _d in ("core", "training", "evaluation", "analysis", "plant"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import os, sys
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import run_study as RS

T = 0.37
class StubTeacher:
    def delta(self, env): return T
RS.NMPCTeacher = StubTeacher          # patched before make_learner builds one

BASE = dict(learning_rate=3e-4, n_steps=512, batch_size=128, n_epochs=3,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=1e-5,
            vf_coef=0.5, width=64)
ok = 0

def check(name, cond):
    global ok
    assert cond, f"FAIL: {name}"
    ok += 1; print(f"  pass: {name}")

# ---------- unit tests of _override ------------------------------------
def unit(method, extra, num_timesteps, sampled):
    m, _, _ = RS.make_learner(method, {**BASE, **extra}, seed=0)
    for cw in m.env.envs:
        cw.reset()
    m.num_timesteps = num_timesteps
    a = np.full((RS.N_ENVS, 1), sampled, dtype=np.float32)
    out, changed = m._override(a.copy())
    return m, a, out, changed

print("warmstart:")
_, a, out, ch = unit("warmstart", dict(g_window=100_000), 50_000, 0.8)
check("inside window -> every action replaced by teacher's",
      ch and np.allclose(out, T))
_, a, out, ch = unit("warmstart", dict(g_window=100_000), 100_000, 0.8)
check("at/after window -> untouched", (not ch) and np.allclose(out, a))

print("mix:")
for ts in (0, 30_000, 99_000):
    _, a, out, ch = unit("mix", dict(g_window=100_000), ts, 0.8)
    al = 1.0 - ts/100_000
    expect = np.clip(al*T + (1-al)*0.8, -1, 1)
    check(f"alpha at {ts} steps = {al:.2f} -> blend {expect:.4f}",
          ch and np.allclose(out, expect, atol=1e-6))
_, a, out, ch = unit("mix", dict(g_window=100_000), 100_000, 0.8)
check("anneal finished -> untouched", (not ch) and np.allclose(out, a))
# sampled action outside [-1,1] must be clipped before blending
_, a, out, ch = unit("mix", dict(g_window=100_000), 50_000, 2.5)
check("agent action clipped to 1 before blending",
      np.allclose(out, 0.5*T + 0.5*1.0, atol=1e-6))

print("shield:")
m, a, out, ch = unit("shield", dict(g_trig=3.0, g_margin=2.0), 500_000, 0.8)
check("healthy state (err=0, mid-range drum) -> no intervention",
      (not ch) and np.allclose(out, a))
for i, cw in enumerate(m.env.envs):     # push power 4 SPU off demand
    cw.env.p = cw.env.profile(cw.env.t)/100.0 + 0.04
_, out2 = m._override(a.copy())[1], m._override(a.copy())[0]
check("error 4 SPU > trigger 3 -> teacher override", np.allclose(out2, T))
for i, cw in enumerate(m.env.envs):     # restore power, park drum at margin
    cw.env.p = cw.env.profile(cw.env.t)/100.0
    cw.env.ang = 1.0
out3 = m._override(a.copy())[0]
check("drum angle 1 deg < margin 2 -> teacher override", np.allclose(out3, T))
m2, a, out, ch = unit("shield", dict(g_trig=3.0, g_margin=2.0), 500_000, 0.8)
for cw in m2.env.envs:
    cw.env.p = cw.env.profile(cw.env.t)/100.0 + 0.02
check("error 2 SPU < trigger 3 -> no intervention",
      not m2._override(a.copy())[1])

# ---------- integration: buffer contents during warmstart window --------
print("integration (one rollout, warmstart window active):")
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
model, stats, _ = RS.make_learner("warmstart", {**BASE, "g_window": 100_000}, seed=0)

class Grab(BaseCallback):
    def _on_step(self): return True
    def _on_rollout_end(self):
        buf = self.model.rollout_buffer
        check("buffer stores the teacher's action at every step",
              np.allclose(buf.actions, T, atol=1e-6))
        obs = th.as_tensor(buf.observations.reshape(-1, 4), dtype=th.float32)
        act = th.as_tensor(buf.actions.reshape(-1, 1), dtype=th.float32)
        with th.no_grad():
            v, lp, _ = self.model.policy.evaluate_actions(obs, act)
        check("stored log-probs = log pi(teacher action | state) under current policy",
              np.allclose(lp.numpy(), buf.log_probs.flatten(), atol=1e-4))
        check("stored values = V(state) under current policy",
              np.allclose(v.numpy().flatten(), buf.values.flatten(), atol=1e-4))
        check("teacher was queried once per env-step",
              stats["teacher"] == buf.actions.shape[0]*buf.actions.shape[1])
        raise KeyboardInterrupt   # one rollout is enough

try:
    model.learn(total_timesteps=BASE["n_steps"]*RS.N_ENVS, callback=Grab())
except KeyboardInterrupt:
    pass

print(f"\nALL {ok} CHECKS PASSED")
