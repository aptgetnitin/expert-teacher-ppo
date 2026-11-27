# Does an Expert Teacher Help PPO?

Code and recorded results for the paper

> *Does an Expert Teacher Help PPO? A Controlled Comparison of
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

| Folder | Contents |
|---|---|
| `core/` | `env200.py`, the benchmark environment (200-step load-following episode); `teacher.py`, the one-step NMPC teacher and its standalone evaluation |
| `training/` | `run_study.py` (search and final runs for unguided, shield, warmstart, mix, JSRL-style, RLIF-style); `run_residual.py` (the residual configuration); `run_more_seeds.py` (one extra evaluation seed); `run_shared_hp.py` (injection methods on the unguided control's base hyperparameters) |
| `evaluation/` | `run_typed.py` (replay that records violations by type); `robust_ckpt.py`, `robustness.py` (deployment-robustness ablations); `final_policy.py` (reliability at the final policy); `residual_ckpt0.py` (residual before any training); `diag_clipfrac.py`, `diag_clipfrac_multi.py` (PPO clip activity on injected samples) |
| `analysis/` | `analyze20.py`, `stats_upgrades.py` (main table, cost rate, CVaR, Holm-corrected Fisher tests, seed subsampling); `figs_academic.py`, `fig_traj_paper.py`, `fig_clipfrac.py`, `fig_camera_ready.py`, `make_figs.py`, `make_traj.py` (figures, written to `figs/`) |
| `tests/` | `test_guidance.py`, checks of the guidance bookkeeping |
| `results/` | recorded outcomes: per-method searches and seeds, typed replays, robustness, diagnostics, saved checkpoints |
| `figs/` | output folder for regenerated figures |
| `plant/` | plant model and NMPC code (git submodule) |

Run every script from the repository root.

## Reproducing the paper

From the recorded results (seconds):

    python analysis/analyze20.py            # main results table
    python analysis/stats_upgrades.py       # cost rate, CVaR, P(improvement), Holm-corrected tests
    python evaluation/final_policy.py       # final-policy reliability
    python analysis/fig_camera_ready.py     # performance-profile and robustness figures
    python analysis/fig_clipfrac.py         # clip-diagnostic figure

Retraining (seed 0 tunes; seeds 1-20 evaluate; about one CPU-hour per 2M-step run):

    python training/run_study.py --method jsrl                 # search + first seeds
    python training/run_more_seeds.py --method jsrl --seed 7   # ... through seed 20
    python training/run_residual.py --search
    python training/run_residual.py --seed 1
    python evaluation/run_typed.py --method jsrl --seed 7      # typed violation replay
    python evaluation/robust_ckpt.py --method jsrl --seed 11 --step 1200000
    python evaluation/robustness.py

Diagnostics added for the final version:

    python evaluation/diag_clipfrac_multi.py --method mix --base tuned --seed 1
    python training/run_shared_hp.py --method mix --seed 1
    python evaluation/residual_ckpt0.py

Checks:

    python tests/test_guidance.py

Training is deterministic given a seed. The scripts set `OMP_NUM_THREADS=1`;
without it BLAS oversubscription dominates runtime when runs are parallel.

## Acknowledgments

The microreactor load-following benchmark is due to Leo Tunkle, Kamal
Abdulraheem, and Majdi I. Radaideh of the AIMS Lab at the University of
Michigan. The plant model and NMPC code are referenced through the `plant/`
submodule and are not part of this repository's own code. The study in this
repository was carried out independently; any errors are the author's own.

## License

The code and recorded results in this repository are released under the MIT
License (see `LICENSE`). The `plant/` submodule is a separate repository and
is not covered by it.

Third-party libraries: Stable-Baselines3, Optuna, Gymnasium (MIT);
NumPy, SciPy, Matplotlib (BSD).
