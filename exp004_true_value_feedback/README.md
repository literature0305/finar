# Appendix: True-Value Feedback (FiNAR Experiment 4)

## A.1 The question

EO v4's second recursion pass is handed the first pass's forecast and predicts
a residual against it. This experiment asks **what the second pass is actually
limited by** — the quality of the input it is handed, or something else — by
replacing that input:

$$\text{intermediate\_pred}_{\text{new}} = (1-\alpha)\,\text{intermediate\_pred} + \alpha\,\text{target}$$

and sweeping $\alpha$. At $\alpha = 0$ this is the stock run. At $\alpha = 1$
pass 2 is handed the ground truth. If the second pass still cannot reach the
truth from a perfect input, the refinement is bounded by something other than
its input, and that is the finding.

> **Every $\alpha > 0$ result is label leakage by construction.** fev's own API
> says of the data this feeds the model: *"This data should never be provided
> to the model!"*. These numbers are a diagnostic upper bound and must never be
> reported as model performance. Only the $\alpha = 0$ row is comparable to
> anything published.

## A.2 The truth is not in the model's input

tsm-trainer already implements this exact blend —
`EOModelV4._interpolate_initial_value` does `v <- v*(1-c) + target*c` — but it
is **training-only** (`_init_options_active` is `self.training and
warmup_active()`), and the "target" it reaches for is the `context` tensor,
which carries the future at training time and does not at evaluation time. The
update region is masked at eval; the horizon truth lives in the **benchmark
adapter**, as the labels it scores against.

So the truth has to be carried from the adapter into the model, and the two
adapters hold it differently:

| benchmark | seam | note |
|---|---|---|
| fev-bench | `EvaluationWindow.get_ground_truth()`, stashed at `get_input_data` and consumed by `_window_to_task_inputs` | fev **strips the target from `future_data`** — it is the answer — so the truth is not an argument to the input builder at all |
| GIFT-Eval | `Dataset.test_data` wrapped so pair iteration, `.input` and `.label` all record | the model is handed tensors rebuilt from the inputs, so identity is lost; the truth is keyed by a digest of the context |

Both feed one pipeline-level seam, `eo_pipeline._row_budget_chunks`, which
every route into the model shares. Row counts are checked there: a mismatch
raises rather than blending the wrong rows silently. `run_eval.py` also records
`truth_hits` / `truth_misses` per arm and **refuses** an $\alpha > 0$ arm whose
truth never reached the model — otherwise a stock run would be published
wearing an alpha label.

**Why every access path records.** GIFT-Eval's multivariate adapter never walks
`test_data` as pairs before the forward — it iterates `test_data.input`, and
walks pairs only afterwards over the `to_univariate` view, whose 1-D targets
cannot match the `(V, T)` the model saw. A wrapper that overrode only
`__iter__` therefore recorded nothing on the multivariate route while the
univariate tasks still produced hits, so the global "did any truth arrive"
gate stayed green and the multivariate half ran silently at $\alpha = 0$.
Measured before the fix: 20 entries iterated, 0 recorded.

## A.3 Where the blend is applied, and why there

On `med` — pass 1's median, the **input** to `_coe_write_back` — never on its
outputs. The write-back returns three things that must agree:

| output | what it is |
|---|---|
| `value_channel` | the median written over the update region |
| `mask_channel` | the "filled at depth $i$" stamp |
| `acc` | the residual accumulator, **seeded from the value channel** |

Blending the value channel afterwards and leaving `acc` seeded from the
unblended median is exactly the failure the work order warns about: pass 2 then
adds a residual computed against one input to an accumulator holding another.
Blending the input instead makes all three derive from the blended median
through tsm-trainer's own code, so they cannot disagree.

**Spaces and scaling.** `med` is in prediction space, normalised with the
loc/scale the encoder used. The raw truth is therefore normalised with *that*
loc/scale (`self.instance_norm(x, loc_scale)`), never a fresh one — a target
normalised on its own statistics is a different scale wearing the same units —
then converted with `to_pred_space`, a no-op unless the checkpoint uses a
next-patch objective. NaN truth becomes 0 **after** normalising, i.e. the
series mean, which is what upstream does.

**Checkpoint requirements**, refused up front: `coe_residual: True` (no
accumulator otherwise, so nothing to keep consistent), `coe_bottleneck: True`
(the passes would chain in hidden space and never write a forecast back), and
`coe_train_depth_max >= 2`.

## A.4 Usage

```bash
cd exp004_true_value_feedback

bash run_exp004.sh --ckpt /path/to/eo-v4-K2/best_checkpoints \
                   --repo /group-volume/.../tsm-trainer_001/tsm-trainer

bash run_exp004.sh --ckpt <ckpt> --alpha "0.0 0.25 0.5 0.75 1.0"
bash run_exp004.sh --ckpt <ckpt> --benchmarks fev
bash run_exp004.sh --ckpt <ckpt> --task-subset "0 4"    # slice 0 of 4
bash run_exp004.sh --ckpt <ckpt> --stage table
```

Full options are in the script header (`--help`). Long runs detach with
`nohup ... &`. Re-running is safe — a `(benchmark, alpha)` whose `results.csv`
exists is skipped — and a re-run with a different checkpoint, depth, fev subset
or task slice into the same `--out` is refused rather than silently mixing
arms.

`--fev-subset` defaults to `multivariate`, where every variate is a target, so
"the truth is fed back for every variate" has no covariate exception. On a
covariate task only target rows receive truth: a known-future covariate's
future is already observed, and a past-only covariate's is not scored.

### Repo requirement

`--repo` must point at a tsm-trainer checkout at or after **`a53691ba`**
(2026-09-07). `to_pred_space`, `_pred_shift`, `_across` and `_crosses_scales`
all arrived in that one commit, and the blend is defined in the prediction
space they establish — an older checkout does not need a different import, it
needs different arithmetic, so there is no fallback to write.

`run_eval.py` checks this **before the checkpoint is loaded** and names the
commit. Without the check the failure was an `ImportError` raised from inside
the patch, one benchmark into an alpha sweep, minutes after loading — and it
fired for the $\alpha = 0$ arm too, which does no blending and looks like it
should be immune.

### fev's three input modes, and which are captured

fev-bench chooses one input-construction mode per run (`fev_bench.py`,
`mode = ...`), and the mode decides which builder the truth has to be caught in:

| mode | when | builder | captured |
|---|---|---|---|
| `covariate-aware` | `supports_covariates`, no pairwise | `_window_to_task_inputs` | ✓ |
| `independent (group-id ignored)` | `supports_covariates` is False — for EO, `config.group_attention` reduced to false | `_window_to_contexts` | ✓ |
| `aed-pairwise` | `supports_pairwise` wins the priority test | groups of `(past, future)` tuples | ✗ refused |

**Group attention is not required.** exp004 replaces a variate's own pass-1
forecast with its own true future — self-feedback along time. It needs no
cross-variate path, so a channel-independent checkpoint is a perfectly valid
subject and its `independent` mode is hooked. Verified on ETT_15T: 280 hits /
0 misses in `independent`, 40 / 0 in `covariate-aware`.

`aed-pairwise` is refused because it differs in kind, not just in seam — the
model is handed groups of `(past, future)` tuples rather than task dicts, so
this experiment's per-row truth layout does not describe the rows the write-back
sees. Blending there would feed the wrong values rather than none.

GIFT-Eval is unaffected either way: its capture patches `gift_eval.data.Dataset`.

### What comes out

| path | contents |
|---|---|
| `<out>/manifest.json` | checkpoint, depth, subset and slice the arms share |
| `<out>/<bench>/alpha<a>/results.csv` | per-task scores at every depth |
| `<out>/<bench>/alpha<a>/truth.json` | how often the truth reached the model |
| `<out>/exp004_table.csv` | iteration 1 vs 2 per alpha, improvement % |
| `<out>/exp004_improvement.png` | improvement against alpha |

## A.5 Verification

Run on `eo-v4-120M-toto_2080ti/best_checkpoints` (K=2, `coe_residual` and
`coe_bottleneck` both true), on subsets of each benchmark.

| benchmark | slice | truth hits | MASE $\alpha{=}0$ | MASE $\alpha{=}1$ |
|---|---|---|---|---|
| fev-bench multivariate | ETT_15T, ETT_1D | 80 / 0 miss | +6.1% | **+31.0%** |
| GIFT-Eval univariate | electricity ×5 | 13,319 | −0.6% | **+37.2%** |
| GIFT-Eval **multivariate** | bizitobs ×5 `[MUL]` | 343 | −6.3% | **+33.4%** |

Percentages are the iteration-1 → iteration-2 improvement; the $\alpha = 1$
column is the leakage bound, not a score.

Four things this establishes, and one it does not:

1. **The truth reaches the model on every route.** The first fev attempt read
   the target columns out of `future_data` and raised `KeyError`, which is how
   fev's deliberate stripping of the answer was found. The first GIFT attempt
   recorded nothing on the multivariate route — that adapter iterates
   `test_data.input` rather than pairs — and the univariate tasks still
   produced hits, so the global gate stayed green while the multivariate half
   ran at $\alpha = 0$. Both are fixed and both routes are now verified with
   `[MUL]` tasks in the run.
2. **Iteration 1 does not move with $\alpha$** — identical to every printed
   digit, so the blend is not leaking into pass 1. `build_table.py` asserts it.
3. **A better input helps a lot but does not saturate.** Handing pass 2 the
   exact answer still leaves MASE at 0.94 (fev) and 1.99 (GIFT multivariate)
   rather than 0. The residual chain does not pass a perfect input through; it
   pulls back towards the model's own forecast. **That ceiling is the
   experiment's substantive result** and is what a larger sweep should
   characterise.
4. **The second pass can be actively harmful without it.** At $\alpha = 0$ the
   GIFT arms are negative (−0.6%, −6.3%): for this checkpoint iteration 2 makes
   GIFT-Eval worse, and only the injected truth turns it positive.

What it does not establish: coverage. `truth_misses` and
`ambiguous_fingerprints` are recorded per arm but only a global "no truth at
all" gate refuses a run, so a partially-treated arm is reported rather than
refused. The multivariate GIFT slice above ran with 42 misses and 126 ambiguous
fingerprints out of 385 — windows whose contexts are identical but whose
futures differ, which are dropped rather than guessed. Read `truth.json`
alongside any result.
