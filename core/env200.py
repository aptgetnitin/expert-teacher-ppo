"""The published 200-second benchmark of Tunkle et al. (2025), reproduced.

Faithful to the original environment (envs/holos_single.py) in every element
that defines the task:
    observation  [p_t, p*_{t+1}, p_{t-1}, theta/180]   (SPU-normalized powers)
    action       one shared drum speed, |dtheta| <= 0.5 deg/s, 1 s interval
    reward       r_t = 2 - |p_{t+1} - p*_{t+1}|        (powers in SPU)
    train term.  |err| > 5 SPU, or drums at 0/180 deg, or p > 110 SPU
    profiles     'train' 100->55->80 over 200 s; 'test' and 'low-power'
                 evaluation ramps as in the paper
The plant is the repository's own Heun integrator (nmpc.ReactorModel), which
under closed-loop conditions agrees with a stiff LSODA reference to 0.045 SPU
worst-case; the prompt-jump fast model used for the 28-hour study is NOT
adequate at 1-second steps (1.6 SPU) and is not used here.
"""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
for _d in ("core", "training", "evaluation", "analysis", "plant"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import os, sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from nmpc import ReactorModel

EP_LEN = 200
DTH_MAX = 0.5          # deg per second, as in the paper

# The paper's piecewise-linear profiles (SPU vs seconds).
PROFILES = {
    "train":    ([0, 40, 90, 140, 200],      [100, 100, 55, 80, 80]),
    "test":     ([0, 30, 60, 100, 140, 200], [100, 100, 60, 90, 70, 70]),
    "lowpower": ([0, 30, 80, 130, 200],      [100, 100, 30, 30, 65]),
    # generalization profiles: never used in training or tuning
    "staircase": ([0, 25, 30, 70, 75, 115, 120, 160, 165, 200],
                  [100, 100, 70, 70, 85, 85, 60, 60, 90, 90]),
    "randpw":    ([0, 20, 45, 80, 110, 150, 180, 200],
                  [100, 92, 64, 78, 51, 88, 73, 73]),
}


def sine_profile(t):
    """Sinusoidal demand, period 100 s, mean 75, amplitude 20 (SPU)."""
    x = np.asarray(t, float)
    r = np.where(x < 10.0, 100.0,
                 75.0 + 20.0 * np.cos(2 * np.pi * (x - 10.0) / 100.0))
    return float(r) if np.ndim(t) == 0 else r


def profile_fn(name):
    if name == "sine":
        return sine_profile
    xs, ys = PROFILES[name]
    return lambda t: float(np.interp(t, xs, ys))


class Holos200s(gym.Env):
    """The original benchmark; reward and termination exactly as published."""

    def __init__(self, profile: str = "train", train_mode: bool = True,
                 seed: int | None = None):
        super().__init__()
        self.plant = ReactorModel(num_drums=1, dt=1)
        self.profile = profile_fn(profile)
        self.train_mode = train_mode
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-10, 10, shape=(4,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def _obs(self):
        nxt = self.profile(self.t + 1) / 100.0
        return np.array([self.p, nxt, self.p_prev, self.ang / 180.0],
                        dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.z = self.plant.get_initial_state(1.0)
        self.ang = float(self.plant.u0)
        self.t = 0
        self.p = float(self.z[0])
        self.p_prev = self.p
        self.ep_reward = 0.0
        return self._obs(), {}

    def step(self, action):
        d = float(np.clip(np.asarray(action).reshape(-1)[0], -1, 1)) * DTH_MAX
        self.p_prev = self.p
        self.ang = float(np.clip(self.ang + d, 0.0, 180.0))
        self.z = self.plant.discrete_dynamics(self.z, [self.ang], 1.0)
        self.t += 1
        self.p = float(self.z[0])
        target = self.profile(self.t) / 100.0
        err_spu = 100.0 * abs(self.p - target)
        reward = 2.0 - err_spu
        self.ep_reward += reward

        terminated = False
        if self.train_mode and (err_spu > 5.0
                                or self.ang <= 0.0 or self.ang >= 180.0):
            terminated = True
        if self.p > 1.10:
            terminated = True
        truncated = self.t >= EP_LEN
        # exclusive violation type, severity priority: overpower > drum > error
        v_pow = v_drum = v_err = 0
        if terminated:
            if self.p > 1.10:
                v_pow = 1
            elif self.ang <= 0.0 or self.ang >= 180.0:
                v_drum = 1
            else:
                v_err = 1
        info = dict(err_spu=err_spu, ang=self.ang, power=self.p,
                    violation=int(terminated), viol_pow=v_pow,
                    viol_drum=v_drum, viol_err=v_err)
        return self._obs(), reward, terminated, truncated, info


def check_against_repo(n_steps=200):
    """Closed comparison with a stiff LSODA integration of the repository's
    own 12-state right-hand side (the same equations the envs/ package
    integrates), under an identical zero-order-hold drum sequence."""
    from scipy.integrate import solve_ivp
    from holos_fast import reference_rhs
    from envs import holos_constants as C

    rng = np.random.default_rng(0)
    fast = Holos200s("train", train_mode=False)
    fast.reset()
    x = np.array([1.0] + [1.0]*6 + [C.Xe0, C.I0, C.Tf0, C.Tm0, C.Tc0])
    worst = 0.0
    for k in range(n_steps):
        a = float(rng.uniform(-1, 1))
        _, _, _, _, inf = fast.step([a])
        sol = solve_ivp(lambda t, y: reference_rhs(y, inf["ang"]),
                        (0.0, 1.0), x, method="LSODA", rtol=1e-9, atol=1e-11)
        x = sol.y[:, -1]
        worst = max(worst, abs(float(x[0]) - inf["power"]))
    return worst


if __name__ == "__main__":
    w = check_against_repo()
    print(f"max |power| discrepancy vs repository env over 120 random steps: "
          f"{w:.5f} (SPU x0.01)")
