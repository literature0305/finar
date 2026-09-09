# Appendix: Pseudo Known-Future Covariates (FiNAR Experiment 5)

## A.1 The question

Models trained without iterative refinement gain a great deal from covariates
whose future is genuinely **observed** — fev-bench's future-known subset is
where Chronos-2 reports its largest margin. This asks whether they gain
anything when that future is not observed but **predicted, by the model
itself**:

| step | what the model is given | what is scored |
|---|---|---|
| 1 | all $V$ variates as targets (one multivariate pass) | variate 0 |
| 2 | variate 0 as target; variates $1..V{-}1$ as `past_covariates` + `future_covariates`, the future filled from step 1 | variate 0 |

Only variate 0 — the designated target — is scored, in both steps, so the pair
answers one question about one series. Variate 0 is positional, so it is the
same series in both steps and across models.

The target models are one-pass by construction (Chronos-2, TiRex-2, TimesFM-3)
or made one-pass with `--depth 1` (EO v4).

## A.2 The benchmarks have no covariate slots, so the task is rebuilt

This cannot reuse the adapters' inputs. fev-bench's `multivariate` subset is
*defined* as ">1 target column **and** no dynamic covariates" (`_task_subset`),
and GIFT-Eval has no covariates at all — its own adapter records this. There is
nothing to fill, so step 2 restructures the task: variates $1..V{-}1$ move out
of `target` and into the covariate dicts, keeping the history they already had
and gaining a future half from step 1.

That restructuring is also why the *selection* of scored rows is done here
rather than by an adapter: step 1 scores $V$ targets and step 2 scores one, so
an adapter's aggregate would compare two different populations. The *metrics*
are the repository's own — `MetricRegistry.compute_chronos_metrics_fast`, the
per-item-normalised WQL and MASE every other benchmark here reports, applied to
variate 0 only. The benchmarks are still read through their own libraries,
fev's task list and sharding come from the adapter, and the GIFT-Eval
multivariate set is read from `baselines/gift_eval_hf_variate_types.csv`
rather than named here, so it cannot go stale silently.

## A.3 Step 2 must actually differ from step 1

`Chronos2Forecaster` and `EOForecaster` resolve `supports_covariates` through
the **same function** as `supports_multivariate` — covariates and extra targets
enter by one group-attention path. So "variates as targets" and "variates as
covariates" can reach the model as nearly the same thing, and the only real
difference is that `future_covariates` carries values.

If that difference does not land, step 2 returns step 1 and the experiment
reports "no gain" for a reason unrelated to the question. Every run therefore
counts items whose two forecasts agree to floating point, `run_eval.py`
**refuses** a run where all of them do, and the count is carried into the
table so a partially-identical run is visible too.

**Unscorable items are dropped and counted.** GIFT-Eval's bitbrains datasets
carry non-finite values and constant histories; either makes MASE or WQL NaN
for the whole task if averaged in, which silently removes that task from every
comparison. One `prepare()` builds the truth stack and the seasonal-naive
denominators once and both steps score through it, so the mask, the
denominator and the population are the same by construction rather than by
two hand-rolled agreements.

## A.4 Usage

```bash
cd exp005_pseudo_future_known_covariate

# every model, both benchmarks, then table + figure
bash run_exp005.sh --repo /group-volume/.../tsm-trainer_001/tsm-trainer

bash run_exp005.sh --ckpt amazon/chronos-2
bash run_exp005.sh --ckpt google/timesfm-3.0-pytorch
bash run_exp005.sh --ckpt NX-AI/TiRex-2
bash run_exp005.sh --ckpt Datadog/Toto-2.0-313m
bash run_exp005.sh --ckpt /path/to/eo-v4/best_checkpoints --depth 1

# Toto-2.0 across sizes
for S in 4m 22m 313m 1B 2.5B; do bash run_exp005.sh --ckpt "Datadog/Toto-2.0-$S"; done
bash run_exp005.sh --ckpt amazon/chronos-2 --max-tasks 2 --max-items 16
bash run_exp005.sh --stage table
```

Full options are in the script header (`--help`). Long runs detach with
`nohup ... &`; a model whose csv exists is skipped, and a model that fails does
not stop the rest.

`--max-windows` defaults to 8 for fev because its multivariate tasks carry very
few *series* — ETT is one per window — so a single window gives a two-item task
and no statistic worth reading.

### What comes out

| path | contents |
|---|---|
| `<out>/<model>.csv` | per task: step1/step2 MASE, WQL, win rate, identical count |
| `<out>/<model>.meta.json` | `supports_multivariate`, quantile grid, identical-item total |
| `<out>/exp005_table.csv` | pooled per model and benchmark |
| `<out>/exp005_improvement.png` | the same as grouped bars |

`improvement_pct` is $(\text{step1} - \text{step2}) / \text{step1}$, so
**positive means the pseudo-known-future pass helped**.

### Why Toto-2.0 is in this list

Toto-2 is the one model here that takes a **real** known future rather than
only extra past variates. `Toto2Forecaster._forecast_tasks_batch` routes
known-dynamic covariates into the model's `known_dynamic` slot, spanning
context $+$ horizon, so their future half conditions the forecast; a covariate
with no future stays an ordinary extra past variate. That was verified rather
than assumed: supplying a future changes the forecast, and *reversing* that
future changes it again, so the model reads the values and not merely the
presence of the slot.

Every size is accepted — `Datadog/Toto-2.0-{4m,22m,313m,1B,2.5B,2.5B-FT}` —
because `run_benchmark._detect_model_type` dispatches on the `toto-2`
substring, so the size varies freely. This makes exp005's question askable
*across scale*: whether a predicted future starts to help once the model is
large enough to use a real one.

Toto-2's **published** fev-bench numbers use no covariates at all (its official
adapter builds `Toto2GluonTSModelConfig` with no covariate dims), so this run
is deliberately not the leaderboard protocol — `--leaderboard-parity` in
`run_benchmark.py` is what restores it.

#### A horizon limit that must be read with every Toto-2 row

`Toto2Forecaster` reaches the model with only the **first
$\lceil H/32 \rceil - 1$ patches** of the known future (32 is Toto's
`patch_size`). The final patch of the horizon never sees its future values, and
when $H \le 32$ that is the *entire* future.

Measured, not inferred. Perturbing one patch of the future at a time and
watching the forecast:

| horizon | `future[0:32]` | `future[32:64]` | `future[64:96]` |
|---|---|---|---|
| $H=48$ | 0.060 | **0.000** | — |
| $H=96$ | 0.601 | 0.357 | **0.000** |

and sweeping $H$ with the context length and covariate count held free, the
future is ignored for $H \in \{13, 28, 32\}$ and read from $H = 33$ upward, at
every context length in $\{38, 164, 640, 4064\}$ and for both 1 and 6
covariates. Reversing the future changes nothing below the threshold either, so
this is the tensor not being consumed, not the model declining to use it.

**Consequence for this experiment.** Of the 45 tasks exp005 scores, **15 have
$H \le 32$** (10 of 26 fev, 5 of 19 GIFT-Eval), and on those a Toto-2 row is a
guaranteed null: step 2 returns step 1 exactly. `run_eval.py` logs a WARNING
naming each such task and its horizon, and the per-task `n_identical` column
carries it into the table — a Toto-2 row whose `ident` equals its `n_items` is
plumbing, not a finding. The run-level refusal does not fire here, because
Toto-2 *does* differ on the other 30 tasks.

This is a defect in `Toto2Forecaster._forecast_tasks_batch`'s `known_dynamic`
layout, in the read-only tsm-trainer checkout; it is reported, not patched from
here.

### Two environment notes

* `amazon/chronos-2` routes to tsm-trainer's official-Chronos loader, which
  needs the `chronos-forecasting` package in the interpreter being used. Where
  it is absent, `autogluon/chronos-2` reaches the same model through
  `Chronos2Forecaster`, and that is what the local verification used.
* `NX-AI/TiRex-2` builds a CUDA extension and needs `ninja` on `PATH`.
* `Datadog/Toto-2.0-*` needs the `toto2` package:
  `pip install --no-deps 'toto-2 @ git+https://github.com/DataDog/toto.git#subdirectory=toto2'`
  (without `--no-deps` it replaces the installed torch).

## A.5 Verification

Two models ran locally; TiRex-2 and TimesFM-3 were not reachable here for the
environment reasons above. Sample sizes are small — this is a smoke of the
implementation, not the measurement.

| model | benchmark | items | step1 MASE | step2 MASE | improve% | win% | identical |
|---|---|---|---|---|---|---|---|
| chronos-2 | fev | 58 | 1.1434 | 1.1425 | +0.02 | 51.7 | 0 |
| chronos-2 | gift | 126 | 0.5480 | 0.5592 | −1.73 | 38.1 | 0 |
| EO v4 (depth 1) | fev | 58 | 1.4208 | 1.5154 | −7.47 | 27.6 | 0 |
| EO v4 (depth 1) | gift | 126 | 0.9567 | 1.0257 | −9.09 | 29.4 | 0 |

WQL moves the same way (−21.2% / −0.5% for chronos-2, −7.5% / −5.7% for EO v4).

Three things this establishes:

1. **The manipulation reaches the model.** `identical = 0` on all 186 items per
   model: step 2's forecast is never step 1's, so a null result here is a
   property of the model, not of the plumbing.
2. **Both steps score the same series on the same items**, with one `prepare()`
   shared between them — and both models see the identical 184-item sample
   (2 bitbrains items dropped for a degenerate seasonal-naive denominator).
3. **On this sample the answer is negative.** Feeding a one-pass model its own
   covariate forecasts as known-future values makes the target forecast
   slightly *worse*, on both benchmarks and both models, on MASE and WQL alike.
   Win rates are at or below 50% everywhere except chronos-2 on fev, where the
   effect is indistinguishable from zero (+0.02%). That is the opposite of what
   genuinely observed known-future covariates do, and is consistent with the
   models treating a known-future covariate as exact — an assumption a forecast
   violates.

What this does **not** establish: anything about TiRex-2 or TimesFM-3, and
nothing at scale. The full sweep belongs on the A100.

**The contrast is not clean, and the sign should not be attributed yet.** Step 1
and step 2 differ in *two* ways at once — the future covariates appear, and the
task is restructured from $V$ targets to one target with $V{-}1$ covariates.
A negative number could come from either. The missing arm is the same
restructured task with `future_covariates` omitted; with it, "predicted future
hurts" separates from "demotion to covariate hurts". tsm-trainer's
`engine/covariate_ablation.py` runs exactly that contrast
(`--fev-covariate-mode none|pseudo|true`) on fev-bench's **covariate** subset,
where the futures are real rather than synthesised, and is the direct form of
this measurement.
