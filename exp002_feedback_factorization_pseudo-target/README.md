# Appendix: Feedback Factorization on Multivariate Targets (FiNAR Experiment 2, pseudo-target)

## A.1 The question

FiNAR feeds the whole intermediate forecast back into the next recursion pass.
This asks **which half earns the improvement**: the model's own trajectory
(self-regression) or its forecast of the other variates (cross-variate
regression). One checkpoint is scored under three feedback regimes,

| regime | what pass 2 is handed |
|---|---|
| `full` | every variate (stock EO v4 evaluation) |
| `self_only` | the designated target only |
| `cov_only` | every variate except the designated target |

and each is compared against **iteration 1**, which is the no-feedback baseline.
Iteration 1 is one column, not three: the regimes change only what the
write-back hands to pass 2, so pass 1 is identical in all of them by
construction — asserted at table time, not assumed.

## A.2 Why a pseudo-target, and why not the covariate subset

The covariate-subset version (`../exp002_feedback_factorization`) runs where
fev-bench names the target and covariate columns, and it has a defect this
version exists to remove. 9 of its 42 rows are **multi-target** — including 7
of the 12 `past_only` rows, which is the only stratum where `self_only`
suppresses anything — and `self_only` feeds back *every* target of such a task.
On those rows "self" already contains cross-variate feedback, so the split it
reports is not the split it names.

| subset | rows | single-target | multi-target |
|---|---|---|---|
| past_only | 12 | 5 | **7** |
| future_known | 18 | 16 | 2 |
| mixed | 12 | 12 | 0 |

This experiment therefore evaluates on suites where every variate is a target
and there are no covariates at all — fev-bench's `multivariate` subset and
GIFT-Eval's multivariate tasks — and **designates the first variate of each
group as the target**, treating the rest as pseudo-covariates. "Self" is then
exactly one variate and "cross" exactly the others, on every task.

**The designation cannot move.** It is positional: variate 0 of each group, in
the array order the data is stored in, derived from the same `inputs` dicts the
model is about to consume by the same rule the pipeline lays rows out with
(`_task_n_rows` — targets first, then one row per past covariate). Every run
writes `designation.json`, and `run_eval.py` **refuses to finish** if the three
regimes disagree about what they designated. A target that moved between arms
would make the comparison meaningless without making it look wrong.

## A.3 Implementation notes

**Two hooks, because the two benchmarks take different routes.** fev-bench's
adapter reaches `EOPipeline.predict_quantiles_tasks` → `_tasks_forward`;
GIFT-Eval's reaches `predict_main_and_repeats` → `predict` → the module-level
`prepare_batch`. The row layout is identical (both chunk with
`_row_budget_chunks(inputs, _task_n_rows, ...)`), only the entry point differs.
Hooking only the first is what would make every GIFT-Eval regime *silently*
identical to `full`: the write-back patch would fire, find no stashed roles,
and suppress nothing.

**Roles are computed per chunk, never over the whole input list.** Both routes
split under a row budget, so roles computed once would be the wrong length for
every chunk after the first and would suppress the wrong variates rather than
fail.

**Suppression restores all three write-back outputs.** Zeroing only the value
channel would leave the accumulator holding pass 1's median, so pass 2 would
add a residual computed against a zero input to a prediction the variate was
never shown. The mask channel matters for the same reason: leaving the
"filled at depth i" stamp on a suppressed variate tells the encoder a value is
present while handing it a zero.

**The write-back wrapper binds the original's signature.**
`EOModelV4._coe_write_back` takes `(med, i, value_channel, validity,
update_mask, loc_scale)`; a wrapper that spells its parameters out positionally
fails on every task with `takes 6 positional arguments but 7 were given`, after
the model is already loaded. The two parameters the patch needs are found by
name and everything else is passed through.

## A.4 Usage

Identical in shape to the covariate version. Full options are in the script's
header (`bash run_exp002_pseudo.sh --help`).

```bash
cd exp002_feedback_factorization_pseudo-target

# everything: three regimes on both benchmarks, then table + figure
bash run_exp002_pseudo.sh --ckpt /path/to/eo-v4-K2/best_checkpoints \
                          --repo /group-volume/.../tsm-trainer_001/tsm-trainer

bash run_exp002_pseudo.sh --ckpt <ckpt> --benchmarks fev_mul   # one benchmark
bash run_exp002_pseudo.sh --ckpt <ckpt> --task-subset "0 4"    # task slice 0 of 4
bash run_exp002_pseudo.sh --ckpt <ckpt> --stage table          # table only
```

Long runs detach with `nohup ... &`. Re-running is safe: a
`(benchmark, scenario)` whose `results.csv` exists is skipped. A re-run with a
different checkpoint, depth or task slice into the same `--out` is **refused**
rather than silently mixing arms — `manifest.json` records what the directory
was built from.

**Checkpoint requirement.** The trained depth must be ≥ 2, and the trained
depth is `coe_train_depth_max` **plus the grad-free warm-up**. A checkpoint at
K=1 with `init_warmup_depth_max >= 1` was trained on inputs a pass already
refined and qualifies; the same checkpoint with `init_warmup_depth_max: null`
does not, because the warm-up is inert at K=1 even though
`init_from_repeat_prediction` reads true.

### What comes out

| path | contents |
|---|---|
| `<out>/manifest.json` | checkpoint, depth and task slice the arms share |
| `<out>/<bench>/<scenario>/results.csv` | per-task scores at every depth |
| `<out>/<bench>/<scenario>/designation.json` | what was designated, per task |
| `<out>/exp002_pseudo_table.csv` | iter1 vs iter2 per regime, per benchmark |
| `<out>/exp002_pseudo_improvement.png` | the same as grouped bars |

`gap_pct` is `(iter1 - iter2) / iter1`, so **positive means the second pass
helped**. The question is which of `self_only` and `cov_only` recovers more of
`full`'s improvement.

## A.5 Two things to hold in view when reading the result

**The arms are asymmetric in volume, not only in kind.** `self_only` feeds back
one variate and `cov_only` feeds back n−1, so on wide groups they differ in how
*much* is fed back as well as in what. The table's `mv_width_mean` column
carries the mean group width of the population each row was computed on — on
fev-bench's multivariate subset that mean is 39.7 and the widest group is 100,
so the two readings ("cross-variate matters" vs "more feedback is better")
cannot be separated from this design alone.

**Univariate tasks are degenerate, and are not run.** With one variate there is
no "other": `self_only` feeds back everything (identical to `full`) and
`cov_only` feeds back nothing (identical to iteration 1). Both arms are
therefore restricted to multivariate tasks — fev-bench by `subset="multivariate"`
and GIFT-Eval by an explicit `datasets=` list built from
`baselines/gift_eval_hf_variate_types.csv`.

The GIFT-Eval half of that was missing at first, and the omission is worth
naming because the adapter's name invites it: the `_mul` in
`GiftEvalHFMulAdapter` is a HANDLING mode — feed a multivariate dataset whole
through group attention rather than the official `to_univariate=True`
flattening — not a subset. Constructed without `datasets=` it returns all 97
tasks, 54 of them univariate, so the two arms were not measuring the same
population and 82% of the run's cost bought rows that cannot answer the
question. Filtering leaves 43 tasks and cuts the benchmark's total
$\text{items} \times \text{horizon}$ from 34.1M to 6.0M.

`build_table.py` still stratifies on width. A univariate row appearing in a
fresh result is now a bug report, not a footnote.

## A.6 Verification

Run on `eo-v4-120M-toto_2080ti/best_checkpoints` (K=2, `coe_bottleneck` true).

**fev_mul, end to end** — 26 tasks × 3 regimes. The designation was identical
across all three arms (561 fed tasks, group widths 7–100, no univariate tasks),
and iteration 1 was identical across arms on all 26 rows.

| metric | iter1 | `full` | `self_only` | `cov_only` |
|---|---|---|---|---|
| MASE | 2.5227 | +27.47% | +1.16% | +27.37% |
| WQL | 0.4162 | −2.39% | −0.45% | −3.04% |

On MASE `cov_only` recovers essentially all of `full`'s improvement while
`self_only` recovers almost none. Two caveats belong with that number: the win
rate is 42.3%, so the mean is carried by a minority of tasks while most get
worse, and WQL is negative in every regime — the second pass hurts it for this
checkpoint.

**gift_mul route** — the full benchmark does not fit in 23 GB in one pass on
these hosts (`--task-subset` exists for that), so the hook was verified
directly on the route GIFT-Eval takes: three 4-variate contexts through
`predict_main_and_repeats`. The `prepare_batch` hook fired (3 tasks recorded,
`n_variates = [4, 4, 4]`), iteration 1 was identical to `full` in both
restricted regimes, and iteration 2 genuinely differed (max absolute difference
0.44 for `self_only`, 0.47 for `cov_only`) — so the suppression reaches the
model on that route rather than passing through unnoticed.
