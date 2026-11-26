# Does an Expert Teacher Help PPO?

Code and recorded results for the paper

> Nitin Jotwani. *Does an Expert Teacher Help PPO? A Controlled Comparison of
> Training-Time Guidance Mechanisms on a Reactor Control Benchmark.*
> NeurIPS 2026 Workshop on Reinforcement Learning for Experimental Sciences (RL4XS).
> https://openreview.net/forum?id=WfoBPquvlS

One on-policy learner (PPO, Stable-Baselines3) is trained on a 200-second
microreactor load-following task under six ways of keeping a one-step NMPC
teacher in the training loop, plus an unguided control. Every configuration
gets its own 14-trial hyperparameter search and twenty evaluation seeds.

`results/` holds the recorded outcome of every run reported in the paper, so
the analysis scripts reproduce all reported numbers without retraining.

## Setup

    git clone --recurse-submodules https://github.com/aptgetnitin/expert-teacher-ppo
    cd expert-teacher-ppo
    pip install -r requirements.txt

The plant model and the NMPC machinery are not copied here. They live in the
`plant/` git submodule, pinned to the commit the study used (see
Acknowledgments). If you cloned without `--recurse-submodules`, run
`git submodule update --init`.

## Layout

| File | Purpose |
|---|---|
| `env200.py` | benchmark environment (200-step load-following episode) |
| `teacher.py` | one-step NMPC teacher; standalone evaluation |
| `run_study.py` | search and final runs for unguided, shield, warmstart, mix, JSRL-style, RLIF-style |
| `run_residual.py` | search and final runs for the residual configuration |
| `run_more_seeds.py` | one extra evaluation seed at a tuned configuration |
| `run_typed.py` | deterministic replay that records violations by type |
| `robust_ckpt.py`, `robustness.py` | deployment-robustness ablations |
| `analyze20.py`, `stats_upgrades.py` | main results table, cost rate, CVaR, Holm-corrected Fisher tests, seed subsampling |
| `final_policy.py` | reliability at the final policy (appendix) |
| `residual_ckpt0.py` | residual configuration before any training |
| `diag_clipfrac.py`, `diag_clipfrac_multi.py` | PPO clip activity on injected samples |
| `run_shared_hp.py` | injection methods on the unguided control's base hyperparameters |
| `figs_academic.py`, `fig_traj_paper.py`, `fig_clipfrac.py`, `fig_camera_ready.py` | figures, written to `figs/` |
| `test_guidance.py` | checks of the guidance bookkeeping |
| `results/` | recorded outcomes: per-method searches and seeds, typed replays, robustness, diagnostics, saved checkpoints |

## Reproducing the paper

From the recorded results (seconds):

    python analyze20.py            # main results table
    python stats_upgrades.py       # cost rate, CVaR, P(improvement), Holm-corrected tests
    python final_policy.py         # final-policy reliability
    python fig_camera_ready.py     # performance-profile and robustness figures
    python fig_clipfrac.py         # clip-diagnostic figure

Retraining (seed 0 tunes; seeds 1-20 evaluate; about one CPU-hour per 2M-step run):

    python run_study.py --method jsrl                 # search + first seeds
    python run_more_seeds.py --method jsrl --seed 7   # ... through seed 20
    python run_residual.py --search
    python run_residual.py --seed 1
    python run_typed.py --method jsrl --seed 7        # typed violation replay
    python robust_ckpt.py --method jsrl --seed 11 --step 1200000
    python robustness.py

Diagnostics added for the final version:

    python diag_clipfrac_multi.py --method mix --base tuned --seed 1
    python run_shared_hp.py --method mix --seed 1
    python residual_ckpt0.py

Training is deterministic given a seed. The scripts set `OMP_NUM_THREADS=1`;
without it BLAS oversubscription dominates runtime when runs are parallel.

## Acknowledgments

The microreactor load-following benchmark is due to Leo Tunkle, Kamal
Abdulraheem, and Majdi I. Radaideh of the AIMS Lab at the University of
Michigan. The plant model and NMPC code in `plant/` come from the hybrid
NMPC-PPO repository at https://github.com/kamara3k/rl_nmpc_hybrid, used here
through a fork pinned as a submodule. The study in this repository was carried
out independently; any errors are the author's own.

Third-party libraries: Stable-Baselines3, Optuna, Gymnasium (MIT);
NumPy, SciPy, Matplotlib (BSD).
