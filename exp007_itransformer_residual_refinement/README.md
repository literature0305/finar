# Appendix: Residual Refinement for iTransformer (FiNAR Experiment 7)

## A.1 The question

iTransformer forecasts the whole horizon in **one pass**. Each variate's
lookback becomes a token, the variates attend to one another, and a linear head
emits all $S$ steps at once. EO v4 does the opposite: one weight-shared encoder
runs $N$ times, each pass writing its forecast back into the positions it has
to predict, so a later pass starts from a better guess and only has to correct
it.

This experiment puts EO v4's chain of encoders on iTransformer and asks whether
the correction buys anything **on the paper's own benchmark** — nine datasets,
four horizons, lookback 96, Table 10 of arXiv:2310.06625v3.

The comparison is only worth reading if the baseline is the paper's model and
not a lookalike, so that is checked before anything is trained.

## A.2 The write-back has no masked channel to write into

EO v4 refreshes a *value channel* at positions the content mask excludes. An
inverted transformer has no such channel — a variate's token **is** its whole
window — so the write-back becomes a widened window:

$$\text{token}_v^{(i)} = \mathrm{Linear}\big([\,x_v\ (96)\ ;\ A^{(i-1)}_v\ (S)\,]\big),
\qquad A^{(0)} = 0,\quad A^{(i)} = \text{the forecast reported by pass } i$$

Everything else follows EO v4 exactly: the observed half is never overwritten,
the loc/scale of the non-stationary normalization is taken from the context
**once** so every pass speaks the same units, and denormalization happens only
on the final reported forecast.

With `coe_residual=true` (the default, and the point of the experiment) each
pass emits a **delta** and reports $A^{(i-1)} + \delta_i$. Two invariants fall
out of $A^{(0)}=0$, and `precheck.py` pins both:

* one pass is bit-identical with and without the residual;
* with the feedback slot's weights zeroed and no residual, the chain at **any**
  depth equals the baseline model exactly — the widened embedding is a pure
  superset, not a different architecture.

`coe_bottleneck=false` is EO v4's ablation, transcribed: embed once, chain the
encoder in hidden space ($h_i = \mathrm{enc}(h_{i-1}) + h_{i-1}$ under
`coe_residual`), project once. Nothing is fed back there.

## A.3 "No improvement" and "never ran" look identical

The recurring failure in exp002, exp004 and exp005 was a manipulation that
never reached the model, producing a plausible null. R3 exists for exactly
that — and it has to be run **without** the residual to mean anything.

With `coe_residual=true` the reported forecast is $A_{i-1} + \mathrm{raw}_i$,
so an embedding that ignores the fed-back window entirely still emits
$r, 2r, 3r$: every depth different, every pass blind. Measured, with
`value_embedding.weight[:, 96:]` zeroed, the per-pass change is 0.456 under
`coe_residual=true` and exactly 0 under `coe_residual=false`. So R3 runs the
chain with the residual off, where $\mathrm{pred}_i = f([x\,;\,\mathrm{pred}_{i-1}])$
and a change can only come from the encoder having read the previous forecast.
R3b then checks separately that the accumulation is a real second mechanism on
top of it.

This was a codex review finding, not a designed-in check; the first version
used the residual and would have passed on a dead feedback path.

## A.4 The baseline reproduces the paper

`precheck.py` refuses to report a comfortable pass when it could not check.
Against a `thuml/iTransformer` checkout it asserts, before training:

| check | what it asserts |
|---|---|
| `hparams` | `paper.py`'s per-cell settings are what the 36 official `scripts/multivariate_forecasting/*` invocations pass |
| `model` | the official `Model` and this `ITransformer` accept the same `state_dict` and return **bit-identical** forecasts |
| `data` | every split's window count, and its first/middle/last window, match the official loader — all nine datasets |
| `refine` | R1/R2 of A.2, R3/R3b of A.3, R4 the same reachability test for the hidden chain, R5 that depth *d* read out of one deeper forward equals running the chain to exactly *d* — what the whole depth sweep rests on |
| `table` | a published target exists for every dataset × horizon |

Each one refuses to pass vacuously. `hparams` compares the parsed set against
all 36 expected cells before checking any of them, so a renamed script reduces
it to a failure rather than to "the cells I found agree". `data` reports SKIP,
not PASS, when only some datasets are on disk. And without `--reference` the
model and data comparisons cannot run at all: the precheck exits non-zero, and
so does `train.sh`, unless `--no-reference` is passed explicitly.

Measured, at lookback 96 and `pred_len` 96, with the official settings and a
single seed (RTX 5070, torch 2.11):

| cell | paper MSE / MAE | measured | Δ MSE |
|---|---|---|---|
| ETTh1 | 0.386 / 0.405 | 0.3868 / 0.4051 | +0.2 % |
| ETTh2 | 0.297 / 0.349 | 0.2999 / 0.3492 | +1.0 % |
| ETTm1 | 0.334 / 0.368 | 0.3430 / 0.3765 | +2.7 % |
| ETTm2 | 0.180 / 0.264 | 0.1856 / 0.2723 | +3.1 % |
| Weather | 0.174 / 0.214 | 0.1778 / 0.2179 | +2.2 % |

The tolerance in `paper.py` (5 % relative, floor 0.005 absolute) is calibrated
from the paper's own Table 5 — five seeds, largest σ 2.7 % relative — not from
these runs; the file records both so the choice can be audited.

## A.5 Layout

| file | what it is |
|---|---|
| `itransformer.py` | the paper's model, transcribed, plus the COE chain |
| `data.py` | the official splits, scaler and timestamp features, reimplemented |
| `paper.py` | Table 10's targets and the official per-cell hyperparameters |
| `precheck.py` | A.4's checks; `--reproduce ETTh1/96` also trains one cell |
| `train.py` / `run_eval.py` | one cell, and its test score at each depth |
| `build_table.py` | every run into one csv, one comparison csv, one figure |
| `prepare_data.sh` / `verify_data.py` | fetch the nine datasets and check their shape |
| `train.sh` / `eval.sh` / `submit_job.py` | the launchers, local or via `ssub` |
| `_common.sh` | the interpreter search and `note()`, sourced by all three |

Nothing imports tsm-trainer. The launchers borrow that checkout's interpreter
when `PYTHON` is unset; `requirements.txt` is there to drop even that.

## A.6 Usage

```bash
bash prepare_data.sh --with-reference   # once: the data AND the official checkout

# scenario A — the paper's model
bash train.sh --dataset ETTh1 --pred-len 96 --refinement off

# scenario B — the same model with the EO v4 chain
bash train.sh --dataset ETTh1 --pred-len 96 --refinement on \
              --train-depth 3 --depth 3 --eval-depths "1 2 3 4"

bash eval.sh --stage table
```

`train.sh --help` carries the full option list, the remote-A100 recipe and the
`--mode job` submission. `--depth` is `coe_eval_depth`, as in every other
experiment here; `--train-depth` is `coe_train_depth_max`, which only training
has.

## A.7 Reading the output

`metrics.json` reports every swept depth, and names the **headline depth** —
the one the checkpoint is configured to run. A sweep is a measurement *around*
that setting, so `--depths 1 2 3 4` cannot quietly move the reported number to
depth 4.

`exp007_improvement.csv` carries two gains for the same reason:
`improvement_pct` at the configured depth, which is the result, and
`improvement_pct_best_depth` at whichever swept depth scored best, which is
selection on the test set and is labelled as an upper bound. When a baseline
misses its published cell, every row against it says so.

A pair is only formed when the two runs share a **protocol**: everything in
`config.json` except the `coe_*` fields (the thing under comparison), the data
root (a mount path, not an identity) and the derived parameter count. A
denylist, so a setting added later is protected by default — the allowlist this
started as had already lost `activation`. Flipping `--refinement` changes none
of it, so a legitimate pair always matches; a refinement measured against a
baseline from another session with a different seed is refused and named,
rather than reported with a footnote.

Run directories are named `<dataset>_<seq>_<pred>_<variant>` plus a digest of
whatever was overridden, so two runs differing only in a seed cannot share one
and nothing has to be refused. Each run deletes the previous `checkpoint.pt` /
`metrics.json` before training, so a rerun that fails early cannot leave the
old numbers to be scored as the new ones.

## A.8 What is deliberately not ported

`coe_repeat_encoding` and the `init_*` warm-up family write into EO v4's **mask
channel** and masked value region; an inverted embedding has neither.
`coe_early_stop` is inference instrumentation for a sample-level path this
experiment does not have. `coe_backprop`, `coe_internal_loss`,
`coe_stochastic_repeat`, `coe_residual` and `coe_bottleneck` are all present,
with EO v4's names and EO v4's semantics — including its rule that
`coe_internal_loss` requires `coe_backprop=all`, since under `last` the earlier
passes are produced in `no_grad` and their loss would be a constant.
