# FiNAR

Experiments testing whether FiNAR's iterative refinement supplies an inductive
bias for dependency *inside* the forecast target, and the paper source.

| Directory | Question |
|---|---|
| `exp001_synthetic_data` | Does the gain track dependency along time / across variates? |
| `exp002_feedback_factorization` | Which half of the feedback earns it — target or covariates? |
| `exp002_..._pseudo-target` | The same split where every variate is a target |
| `exp003_regularization` | Dependency modelling, or just regularisation? |
| `latex` | ICLR 2027 manuscript |

## Usage

One script drives each experiment, run over ssh on the A100. Full options are
in each script's header.

```bash
cd exp001_synthetic_data
bash run_exp001.sh --stage all --ckpt /path/to/eo-v4-K2/best_checkpoints
```

`--stage` picks the phase, `--shallow` scores a baseline with no iteration axis
(e.g. `autogluon/chronos-2`), exp002/exp003 take `--repo <tsm-trainer>`.
Detach long runs: `nohup ... &`.
