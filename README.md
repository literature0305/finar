# FiNAR

Experiments testing whether FiNAR's iterative refinement supplies an inductive
bias for dependency *inside* the forecast target.

| Directory | Question |
|---|---|
| `exp001_synthetic_data` | Does the gain track dependency in time / across variates? |
| `exp002_feedback_factorization` | Which half of the feedback earns it? |
| `exp002_..._pseudo-target` | The same split where every variate is a target |
| `exp003_regularization` | Dependency modelling, or regularisation? |
| `exp004_true_value_feedback` | What does pass 2 do when handed the truth? |
| `exp005_..._known_covariate` | Do a model's own covariate forecasts help as known-future? |
| `latex` | ICLR 2027 manuscript |

## Usage

One script per experiment, run over ssh on the A100; full options are in
each script's header.

```bash
cd exp001_synthetic_data
bash run_exp001.sh --stage all --ckpt /path/to/eo-v4-K2/best_checkpoints
```

`--stage` picks the phase and exp002–exp005 take `--repo <tsm-trainer>`.
`--num-workers N` caps CPU use; the default is read from the cgroup quota
and affinity mask — see `finar_cpu.py`.
