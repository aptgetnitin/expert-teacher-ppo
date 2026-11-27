"""
Experiment 1 — the adjudicator for the "rho=1 / the clip never engages" claim.

Question it answers
-------------------
Section 5.2 claims that faithfully recomputing an injected (teacher) action's
log-prob makes the PPO ratio rho=1 so "the clip never engages" on injected
samples -- and that this is what separates action-injection methods (mix/
warmstart/shield) from the safe ones.

A reviewer's point: rho=1 at the first minibatch step is universal (it
holds for EVERY sample, injected or not), and the clip DOES engage in later
epochs, on injected samples too. So the clip is not the discriminator.

This script settles it empirically. It tags every rollout sample as
injected/normal, then during PPO's update it records, per epoch, the fraction
of samples whose |rho-1| > clip_range (i.e. the clip is active) -- split by
injected vs normal.

Read the output
---------------
  * If clipfrac_injected ~= 0 across ALL epochs while clipfrac_normal rises
      -> the original strong claim holds; say it, with this figure.
  * If clipfrac_injected rises across epochs like clipfrac_normal (expected)
      -> the clip does engage on injected samples; the explanation must
         rest on the off-policy bias, not on an inactive clip.

It does NOT modify run_study.py or the study; it reuses make_learner() and
monkeypatches the single learner instance it builds.

Usage
-----
  python evaluation/diag_clipfrac.py --method mix        --steps 200000
  python evaluation/diag_clipfrac.py --method warmstart  --steps 200000
  python evaluation/diag_clipfrac.py --method shield     --steps 300000
  python evaluation/diag_clipfrac.py --method mix --smoke        # quick 30k sanity run

Outputs (results/diag/):
  clipfrac_<method>.json   per-batch records + per-epoch aggregates + verdict
  clipfrac_<method>.png    two-panel figure (per-epoch, and over training time)
"""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
for _d in ("core", "training", "evaluation", "analysis", "plant"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import argparse, json, os, sys, types
import numpy as np


import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor, explained_variance

from run_study import make_learner, RESULTS

DIAG = os.path.join(RESULTS, "diag")
os.makedirs(DIAG, exist_ok=True)

INJECTION_METHODS = ("warmstart", "mix", "shield")


# ---------------------------------------------------------------------------
# 1. A rollout buffer that also carries a per-sample "injected" flag, threaded
#    through SB3's shuffle so it stays aligned with each minibatch.
# ---------------------------------------------------------------------------
class DiagRolloutBuffer(RolloutBuffer):
    def reset(self) -> None:
        super().reset()
        self.injected = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self._inj_flat = None
        self.last_injected = None

    def add(self, *args, injected=None, **kwargs) -> None:
        pos = self.pos                       # super().add advances self.pos
        super().add(*args, **kwargs)
        if injected is not None:
            self.injected[pos] = np.asarray(injected, dtype=np.float32).reshape(self.n_envs)

    def _get_samples(self, batch_inds, env=None):
        data = super()._get_samples(batch_inds, env)
        if self._inj_flat is None:
            # mirror SB3's swap_and_flatten so batch_inds (into the flattened
            # n_envs*buffer_size space) select the right injected flags.
            self._inj_flat = self.swap_and_flatten(
                self.injected.reshape(self.buffer_size, self.n_envs, 1)
            ).reshape(-1)
        self.last_injected = self._inj_flat[batch_inds]
        return data


# ---------------------------------------------------------------------------
# 2. A collect_rollouts identical to GuidedPPO's, but it computes a per-env
#    injected mask and stores it in the buffer.
# ---------------------------------------------------------------------------
def _collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
    assert self._last_obs is not None
    self.policy.set_training_mode(False)
    n_steps = 0
    rollout_buffer.reset()
    callback.on_rollout_start()
    while n_steps < n_rollout_steps:
        with th.no_grad():
            obs_tensor = obs_as_tensor(self._last_obs, self.device)
            actions, values, log_probs = self.policy(obs_tensor)
        actions = actions.cpu().numpy()

        before = actions.copy()
        actions, changed = self._override(actions)              # GuidedPPO method
        inj = np.any(np.abs(actions - before) > 1e-12, axis=1)  # per-env injected

        if changed:
            with th.no_grad():
                a_t = th.as_tensor(actions, dtype=th.float32, device=self.device)
                values, log_probs, _ = self.policy.evaluate_actions(obs_tensor, a_t)

        clipped = np.clip(actions, self.action_space.low, self.action_space.high)
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
                tobs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with th.no_grad():
                    tval = self.policy.predict_values(tobs)[0]
                rewards[idx] += self.gamma * tval
        rollout_buffer.add(self._last_obs, actions, rewards,
                           self._last_episode_starts, values, log_probs,
                           injected=inj)
        self._last_obs = new_obs
        self._last_episode_starts = dones
    with th.no_grad():
        values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
    callback.update_locals(locals())
    callback.on_rollout_end()
    return True


# ---------------------------------------------------------------------------
# 3. A train() = exact copy of SB3 2.9.0 PPO.train() + per-batch diagnostic.
#    (No change to the optimisation; we only READ ratio/clip and record it.)
# ---------------------------------------------------------------------------
def _train(self):
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)
    clip_range = self.clip_range(self._current_progress_remaining)
    if self.clip_range_vf is not None:
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

    entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
    continue_training = True

    for epoch in range(self.n_epochs):
        approx_kl_divs = []
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            inj = self.rollout_buffer.last_injected            # aligned with this batch
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = rollout_data.actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations, actions)
            values = values.flatten()
            advantages = rollout_data.advantages
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = th.exp(log_prob - rollout_data.old_log_prob)

            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

            pg_losses.append(policy_loss.item())
            clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
            clip_fractions.append(clip_fraction)

            # ---- DIAGNOSTIC: split clip activity by injected vs normal --------
            with th.no_grad():
                over = (th.abs(ratio - 1) > clip_range).float().cpu().numpy()
                rvals = ratio.detach().cpu().numpy()
            m = np.asarray(inj).astype(bool)
            ni, nn = int(m.sum()), int((~m).sum())
            self._diag.append(dict(
                t=int(self.num_timesteps), epoch=int(epoch),
                clip_range=float(clip_range), n_inj=ni, n_norm=nn,
                clipfrac_inj=(float(over[m].mean()) if ni else None),
                clipfrac_norm=(float(over[~m].mean()) if nn else None),
                ratio_inj=(float(rvals[m].mean()) if ni else None),
                ratio_norm=(float(rvals[~m].mean()) if nn else None),
                ratio_inj_p95=(float(np.percentile(np.abs(rvals[m] - 1), 95)) if ni else None),
            ))
            # -------------------------------------------------------------------

            if self.clip_range_vf is None:
                values_pred = values
            else:
                values_pred = rollout_data.old_values + th.clamp(
                    values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses.append(value_loss.item())

            entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
            entropy_losses.append(entropy_loss.item())
            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            with th.no_grad():
                log_ratio = log_prob - rollout_data.old_log_prob
                approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                approx_kl_divs.append(approx_kl_div)

            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                continue_training = False
                break

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += 1
        if not continue_training:
            break

    explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                        self.rollout_buffer.returns.flatten())
    self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


# ---------------------------------------------------------------------------
# 4. Aggregation + plotting + verdict
# ---------------------------------------------------------------------------
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(records, method, n_epochs):
    # per-epoch means (averaged over all updates at that epoch index)
    by_epoch = {}
    for e in range(n_epochs):
        rows = [r for r in records if r["epoch"] == e]
        by_epoch[e] = dict(
            clipfrac_inj=_mean([r["clipfrac_inj"] for r in rows]),
            clipfrac_norm=_mean([r["clipfrac_norm"] for r in rows]),
            ratio_inj=_mean([r["ratio_inj"] for r in rows]),
            ratio_norm=_mean([r["ratio_norm"] for r in rows]),
            n_inj=int(np.sum([r["n_inj"] for r in rows])),
            n_norm=int(np.sum([r["n_norm"] for r in rows])),
        )
    # Primary test (works even when 100% of samples are injected, as in the
    # mix/warmstart window): does the clip stay OFF on injected samples across
    # epochs? Compare injected clip-fraction at epoch 0 vs epochs >= 1.
    ep0 = [r for r in records if r["epoch"] == 0]
    later = [r for r in records if r["epoch"] >= 1]
    cf_inj_ep0 = _mean([r["clipfrac_inj"] for r in ep0])
    cf_inj_later = _mean([r["clipfrac_inj"] for r in later])
    # Secondary test (only when normal samples coexist, e.g. shield): injected
    # vs normal in later epochs.
    cf_norm_later = _mean([r["clipfrac_norm"] for r in later])

    def f(x):
        return "n/a" if x is None else f"{x:.3f}"

    if cf_inj_later is None:
        verdict = ("NO INJECTED SAMPLES recorded -- for mix/warmstart make sure "
                   "--steps is inside g_window; for shield the trigger may never "
                   "have fired.")
    elif cf_inj_later < 0.02:
        verdict = (f"clip stays ~INACTIVE on injected samples across all epochs "
                   f"(injected clipfrac: epoch0={f(cf_inj_ep0)}, later={f(cf_inj_later)}) "
                   f"-> the ORIGINAL strong claim holds; report this plot.")
    else:
        extra = "" if cf_norm_later is None else f"; normal(later)={f(cf_norm_later)}"
        verdict = (f"clip ENGAGES on injected samples in later epochs "
                   f"(injected clipfrac: epoch0={f(cf_inj_ep0)} -> later={f(cf_inj_later)}{extra}) "
                   f"-> the clip engages; the explanation rests on the off-policy bias.")
    return dict(by_epoch=by_epoch, cf_inj_ep0=cf_inj_ep0, cf_inj_later=cf_inj_later,
                cf_norm_later=cf_norm_later, verdict=verdict)


def plot(records, summ, method, n_epochs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: clip fraction by epoch index
    es = list(range(n_epochs))
    ci = [summ["by_epoch"][e]["clipfrac_inj"] for e in es]
    cn = [summ["by_epoch"][e]["clipfrac_norm"] for e in es]
    w = 0.38
    ax[0].bar([e - w / 2 for e in es], [x or 0 for x in ci], w, label="injected", color="#D55E00")
    ax[0].bar([e + w / 2 for e in es], [x or 0 for x in cn], w, label="normal", color="#0072B2")
    ax[0].set_xlabel("PPO epoch index within each update")
    ax[0].set_ylabel("clip fraction  ( |rho-1| > clip_range )")
    ax[0].set_title(f"{method}: does the clip engage on injected samples?")
    ax[0].set_xticks(es)
    ax[0].legend()

    # Panel B: clip fraction over training time
    inj = [(r["t"], r["clipfrac_inj"]) for r in records if r["clipfrac_inj"] is not None]
    nor = [(r["t"], r["clipfrac_norm"]) for r in records if r["clipfrac_norm"] is not None]

    def binned(pairs, nbins=40):
        if not pairs:
            return [], []
        ts = np.array([p[0] for p in pairs]); vs = np.array([p[1] for p in pairs])
        edges = np.linspace(ts.min(), ts.max() + 1, nbins + 1)
        idx = np.clip(np.digitize(ts, edges) - 1, 0, nbins - 1)
        xs, ys = [], []
        for b in range(nbins):
            sel = idx == b
            if sel.any():
                xs.append(0.5 * (edges[b] + edges[b + 1]) / 1e3)
                ys.append(vs[sel].mean())
        return xs, ys

    xi, yi = binned(inj); xn, yn = binned(nor)
    ax[1].plot(xi, yi, label="injected", color="#D55E00")
    ax[1].plot(xn, yn, label="normal", color="#0072B2")
    ax[1].set_xlabel("training steps (thousands)")
    ax[1].set_ylabel("clip fraction")
    ax[1].set_title(f"{method}: clip activity over training")
    ax[1].legend()

    fig.suptitle(summ["verdict"], fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    print(f"[diag] figure -> {path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=list(INJECTION_METHODS))
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="quick 30k-step run")
    a = ap.parse_args()
    if a.smoke:
        a.steps = 30_000

    params = json.load(open(os.path.join(RESULTS, f"{a.method}.json")))["params"]
    print(f"[diag] method={a.method} steps={a.steps} params={params}")

    model, stats, shared = make_learner(a.method, params, a.seed)

    # swap in the injected-aware buffer (identical constructor args)
    ob = model.rollout_buffer
    model.rollout_buffer = DiagRolloutBuffer(
        ob.buffer_size, ob.observation_space, ob.action_space,
        device=ob.device, gae_lambda=ob.gae_lambda, gamma=ob.gamma, n_envs=ob.n_envs)

    # instrument this instance only
    model._diag = []
    model.collect_rollouts = types.MethodType(_collect_rollouts, model)
    model.train = types.MethodType(_train, model)

    model.learn(total_timesteps=a.steps, progress_bar=False)

    records = model._diag
    n_epochs = int(params["n_epochs"])
    summ = summarize(records, a.method, n_epochs)

    out = dict(method=a.method, steps=a.steps, seed=a.seed,
               n_epochs=n_epochs, clip_range=params["clip_range"],
               n_records=len(records), summary=summ, records=records)
    jpath = os.path.join(DIAG, f"clipfrac_{a.method}.json")
    json.dump(out, open(jpath, "w"), indent=1)
    print(f"[diag] json -> {jpath}")

    print("\n=== per-epoch clip fraction (injected vs normal) ===")
    for e in range(n_epochs):
        b = summ["by_epoch"][e]
        print(f"  epoch {e}: injected={b['clipfrac_inj']}  normal={b['clipfrac_norm']}  "
              f"(n_inj={b['n_inj']}, n_norm={b['n_norm']})")
    print("\nVERDICT:", summ["verdict"])

    try:
        plot(records, summ, a.method, n_epochs, os.path.join(DIAG, f"clipfrac_{a.method}.png"))
    except Exception as e:
        print(f"[diag] plotting skipped: {e}")


if __name__ == "__main__":
    main()
