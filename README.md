# FiNAR

Experiments testing whether FiNAR's iterative refinement supplies an inductive
bias for dependency *inside* the forecast target, plus the paper source.

| Directory | Question |
|---|---|
| `exp001_synthetic_data` | Does the gain track cross-horizon / cross-variate dependency? (3x3x5 synthetic grid) |
| `exp002_feedback_factorization` | Which half of the feedback earns it — targets or covariates? (fev-bench) |
| `exp003_regularization` | Dependency modelling, or just regularisation? (horizon sweep) |
| `latex` | ICLR 2027 manuscript |

## Usage

One script drives each experiment, run over ssh on the A100. Full options are
in each script's header.

```bash
cd exp001_synthetic_data
bash run_exp001.sh --stage all --ckpt /path/to/eo-v4-K2/best_checkpoints
```

`--stage data|eval|table|all` picks the phase, `--shallow` scores a baseline
with no iteration axis (e.g. `autogluon/chronos-2`), and exp002/exp003 also
take `--repo <tsm-trainer>`. Detach long runs: `nohup ... &`.
