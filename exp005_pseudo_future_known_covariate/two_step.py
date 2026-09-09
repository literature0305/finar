#!/usr/bin/env python3
"""Two-step forecasting: predict the covariates, then use them as known-future.

exp005 asks whether a model trained WITHOUT iterative refinement gains anything
from being handed its own covariate forecasts as if they were observed:

    step 1   multivariate forecast of every variate            (one pass)
    step 2   variate 0 forecast again, with variates 1..n-1 supplied as
             past_covariates + future_covariates, the future filled from step 1

Only variate 0 — the designated target — is scored, in BOTH steps, so the two
numbers answer the same question about the same series.

=============================================================================
THE BENCHMARKS HAVE NO COVARIATE SLOTS, SO THE TASK IS REBUILT
=============================================================================
This cannot reuse the stock adapters' inputs. fev-bench's `multivariate` subset
is *defined* as ">1 target column AND no dynamic covariates" (`_task_subset`),
and GIFT-Eval has no covariates at all — its own adapter says so. There is
therefore nothing to fill: step 2 needs the task RESTRUCTURED, moving variates
1..n-1 out of `target` and into the covariate dicts.

That restructuring is also why the SELECTION of scored rows is done here rather
than by an adapter: step 1 scores n targets and step 2 scores one, so an
adapter's aggregate would compare two different populations. The METRICS
themselves are tsm-trainer's — `MetricRegistry.compute_chronos_metrics_fast`,
the same per-item-normalised WQL and MASE every other benchmark in the repo
reports — applied to variate 0 only.

=============================================================================
STEP 2 MUST ACTUALLY DIFFER FROM STEP 1
=============================================================================
`Chronos2Forecaster` and `EOForecaster` resolve `supports_covariates` through
the SAME function as `supports_multivariate` — covariates and extra targets
enter by one group-attention path. So "variates 1..n-1 as targets" and
"variates 1..n-1 as covariates" can reach the model as nearly the same thing,
and the only real difference is that `future_covariates` carries values.

If that difference does not land, step 2 returns step 1 and the experiment
reports "no gain" for a reason that has nothing to do with the question. Every
run therefore records `n_identical` — tasks where the two steps' target
forecasts agree to floating point — and `run_eval.py` refuses a run where every
task is identical. This is the exp004 lesson: a result that looks ordinary is
the expensive kind of wrong.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("finar_exp005")


#: Name given to the pseudo-covariate built from variate j. Any stable name
#: works; it only has to match between the past and future dicts, which is what
#: `_window_to_task_inputs` relies on upstream too.
def cov_name(j: int) -> str:
    return f"v{j}"


def forecastable(items) -> "np.ndarray":
    """Items every model can be asked to forecast, as a mask.

    AN ALL-NaN VARIATE IS DROPPED FROM THE OUTPUT BY THE MODEL, NOT FORECAST AS
    NaN, and that silently breaks this experiment's one positional assumption.
    Measured on `autogluon/chronos-2` with a 3-variate group at levels
    100/200/300:

        all-NaN variates   returned rows   what `t[0]` actually is
        none               3               variate 0   (correct)
        variate 1          2               variate 0   (correct)
        variate 0          2               variate 1   WRONG
        variates 0, 1      1               variate 2   WRONG
        all three          0               IndexError

    Only the last row raises. The middle two return a plausible forecast of the
    wrong series, scored against variate 0's truth — no error, no NaN. So the
    crash that surfaced this was the safe case.

    Dropping the ITEM is the only sound response: the rows the model returns
    carry no labels, so a partial group cannot be realigned after the fact.
    These items are unscorable anyway — `prepare()` already rejects a target
    whose history is entirely NaN, because the seasonal-naive denominator is
    undefined for it.
    """
    return np.array([
        not np.isnan(it["context"]).all(axis=1).any()
        for it in items])


def step1_inputs(items) -> list[dict]:
    """One multivariate task per item: every variate a target."""
    return [{"target": np.asarray(it["context"], dtype=np.float32)}
            for it in items]


def oracle_future(items, cov_future, horizon: int) -> list:
    """Per item, everything the blend needs that does NOT depend on alpha.

    Computed once per task and reused by every alpha, for the same reason step 1
    and `prepare()` are: the oracle slice and the two finite masks are properties
    of the data, and recomputing them per alpha cost ~40% of `blend_future`.
    ``None`` marks a univariate item, which has no covariate to blend.
    """
    prep = []
    for it, pseudo in zip(items, cov_future):
        if not pseudo.size:
            prep.append(None)
            continue
        oracle = np.asarray(it["truth"], dtype=np.float32)[1:, :horizon]
        if oracle.shape != pseudo.shape:
            raise RuntimeError(
                f"oracle future is {oracle.shape} but step 1 predicted "
                f"{pseudo.shape} covariate rows — the two describe different "
                f"variates and blending them would mix series")
        prep.append((oracle, np.isfinite(oracle), np.isfinite(pseudo)))
    return prep


def blend_future(prep, cov_future, alpha: float):
    """``(futures, n_substituted)`` — `alpha*oracle + (1-alpha)*pseudo`.

    ORACLE is the covariate variates' TRUE future, which the benchmark holds as
    the label and which no honest forecast may see; PSEUDO is step 1's own
    forecast of them. alpha therefore sweeps from the scenario exp005 measured
    (alpha = 0, a model fed its own guess) to a known-future upper bound
    (alpha = 1, a model handed the answer for its covariates). Only the
    COVARIATES are ever oracle — variate 0, the scored target, is never touched.

    THE BLEND IS ONLY DEFINED WHERE BOTH SIDES ARE FINITE. Writing the product
    unconditionally is wrong in both directions, because IEEE makes
    `0 * nan = nan`:

      * a non-finite ORACLE would poison a cell where alpha = 0 had a number,
        so the axis would measure missingness instead of information;
      * a non-finite PSEUDO would poison a cell even at alpha = 1, so the
        "oracle" endpoint would not be the oracle — which is the bound the whole
        sweep is read against.

    So, cellwise:

      both finite            blend
      oracle finite only     pseudo at alpha = 0, oracle above it
      oracle not finite      pseudo

    The middle rule is deliberately discontinuous at 0: alpha = 0 has to
    reproduce the pure-pseudo scenario exactly, NaN included, because that is
    the published result the sweep is anchored to — while any alpha above 0 has
    real information in hand and withholding it would understate the bound.
    `n_substituted` counts those cells so the substitution is reported rather
    than hidden.
    """
    if alpha == 0.0:
        # Identity, and the default. The arithmetic below reproduces `pseudo`
        # exactly at alpha = 0, so running it buys nothing but an allocation
        # per item; the shape guard it would have run lives in `oracle_future`.
        return list(cov_future), 0
    out, n_sub = [], 0
    for pr, pseudo in zip(prep, cov_future):
        if pr is None:
            out.append(pseudo)
            continue
        oracle, o_ok, p_ok = pr
        fill = o_ok & ~p_ok
        n = int(fill.sum())
        if n:
            # Substituted INTO pseudo before the blend: alpha*x + (1-alpha)*x
            # is x, so one `where` then covers both cases and no cell is
            # written twice.
            pseudo = np.where(fill, oracle, pseudo)
            p_ok = p_ok | fill
            n_sub += n
        out.append(np.where(o_ok & p_ok,
                            alpha * oracle + (1.0 - alpha) * pseudo, pseudo))
    return out, n_sub


def step2_inputs(items, cov_future) -> list[dict]:
    """Variate 0 as the target, the rest as known-future covariates.

    `cov_future[i]` is `(n_variates - 1, H)` — the future handed to the demoted
    variates, pseudo or blended (see `blend_future`). Their PAST comes from the
    same context the model already had, so the only new information is the
    future half, which is the whole manipulation.
    """
    out = []
    for it, fut in zip(items, cov_future):
        ctx = np.asarray(it["context"], dtype=np.float32)
        d = {"target": ctx[0]}
        if ctx.shape[0] > 1:
            d["past_covariates"] = {cov_name(j): ctx[j]
                                    for j in range(1, ctx.shape[0])}
            d["future_covariates"] = {cov_name(j): fut[j - 1]
                                      for j in range(1, ctx.shape[0])}
        out.append(d)
    return out


def median_of(q_arr, quantile_levels) -> np.ndarray:
    """The q=0.5 slice of a ``(rows, Q, H)`` quantile array, as a COPY.

    Integer indexing returns a strided VIEW whose `.base` is the whole
    `(rows, Q, H)` block, so returning it keeps every quantile of every variate
    alive for as long as the caller holds the median. That was tolerable when
    the median was consumed by one step-2 pass; the alpha sweep holds it across
    all of them. Measured at `(21, 9, 720)`: 532 KiB retained where 56 KiB is
    needed, and 11.8 us vs 0.9 us to run `isfinite` over the strided slice.
    """
    return np.ascontiguousarray(
        np.asarray(q_arr)[:, quantile_levels.index(0.5), :])


def _check_rows(out, expected, label: str) -> None:
    """Every task must come back, and with one row per variate it was given.

    `expected` is the per-task row count. The OUTER length is checked first: a
    forecaster that returns fewer tasks than it was handed would otherwise pass
    on the common prefix, because `zip` stops at the shorter side — and every
    consumer downstream indexes by position.
    """
    if len(out) != len(expected):
        raise RuntimeError(
            f"{label}: handed {len(expected)} task(s) but the model returned "
            f"{len(out)}. Everything downstream is positional, so a missing "
            f"task would shift every score onto the wrong series.")
    bad = [(i, o.shape, want) for i, (o, want) in enumerate(zip(out, expected))
           if o.shape[0] != want]
    if bad:
        i, got, want = bad[0]
        raise RuntimeError(
            f"{label}: task {i} was given {want} variate(s) but the model "
            f"returned {got[0]} row(s) (shape {got}); {len(bad)} of "
            f"{len(out)} tasks disagree. The rows are unlabelled, so variate 0 "
            f"can no longer be identified and every score would be against the "
            f"wrong series. See `forecastable`.")


def _predict(forecaster, tasks, expected, label: str, horizon: int,
             quantile_levels, batch_size: int):
    """One forward, with the model's own NaN policy and the row check.

    ONE definition for both steps. The policy is what every published run in
    tsm-trainer feeds, and skipping it is not neutral: on `uci_air_quality_1D`
    an unwanted zero-fill moved Chronos-2's MASE +29% (the note at the top of
    `benchmarks/fev_bench.py`). Two independently-resolved copies is how the
    sweep's baseline and its arms would stop being fed the same thing.

    The import stays inside: `benchmarks` is only importable once `run_eval`
    has extended `sys.path`.
    """
    from benchmarks.fev_bench import _apply_missing_policy

    policy = getattr(forecaster, "missing_value_policy", "keep")
    out = [np.asarray(t) for t in forecaster.predict_quantiles_tasks(
        _apply_missing_policy(tasks, policy), prediction_length=horizon,
        quantile_levels=quantile_levels, batch_size=batch_size)]
    _check_rows(out, expected, label)
    return out


def run_step1(forecaster, items, horizon: int, quantile_levels,
              batch_size: int = 32):
    """``(target1_q, cov_future)`` — the no-covariate-future baseline.

    ALPHA-INDEPENDENT, and computed once per task for that reason: the blend
    only ever touches what step 2 is handed, so a step 1 recomputed per alpha
    would cost a forward per alpha and give the sweep a baseline that could
    drift between its own columns.

    Inputs go through the model's own NaN policy
    (`BaseForecaster.missing_value_policy`, applied by `_apply_missing_policy`),
    which is what every published run in tsm-trainer feeds. Skipping it is not
    neutral: on `uci_air_quality_1D` an unwanted zero-fill moved Chronos-2's
    MASE +29%, which is the note at the top of `benchmarks/fev_bench.py`.
    """
    tasks = step1_inputs(items)
    # LOUD, not positional. The model returns rows without labels, so a task
    # that comes back with fewer rows than it was given variates has silently
    # renumbered them — `t[0]` would then be a different series than the one
    # scored against. `forecastable()` removes the known cause (an all-NaN
    # variate); this catches any other.
    s1 = _predict(forecaster, tasks, [t["target"].shape[0] for t in tasks],
                  "step 1", horizon, quantile_levels, batch_size)
    target1 = np.stack([t[0] for t in s1])
    cov_future = [median_of(t[1:], quantile_levels) for t in s1]
    return target1, cov_future


def run_step2(forecaster, items, cov_future, prep, target1, alpha: float,
              horizon: int, quantile_levels, batch_size: int = 32):
    """``(target2_q, n_identical, n_substituted)`` for one alpha.

    `n_identical` counts items whose step-2 forecast equals step 1's to
    floating point. At alpha = 0 that is exp005's original control — a model
    that ignores `future_covariates` produces an all-identical column, which
    reads as "no gain" but is really "the manipulation never reached the
    model".
    """
    fut, n_substituted = blend_future(prep, cov_future, alpha)
    # Step 2 hands the model ONE target, so every task must come back with
    # exactly one row.
    s2 = _predict(forecaster, step2_inputs(items, fut), [1] * len(items),
                  "step 2", horizon, quantile_levels, batch_size)
    target2 = np.stack([t[0] for t in s2])
    # Per item, not one `isclose` over the whole (n, Q, H) stack: that
    # allocated a 1.6 GB boolean temporary per alpha at full scale, and was
    # slower (401 ms vs 327 ms at n = 22,500).
    identical = sum(bool(np.allclose(target1[i], target2[i],
                                     rtol=1e-6, atol=1e-8))
                    for i in range(len(target1)))
    return target2, identical, n_substituted


def prepare(items, horizon: int, seasonal_period: int, Metrics) -> dict:
    """Everything the scoring of BOTH steps shares, computed once.

    Returns ``{"idx", "y_true", "scales", "n_scored", "n_dropped"}``. The two
    steps differ only in their forecast array, so rebuilding the truth stack and
    the seasonal-naive denominators per step would be pure duplicate work — and
    a second chance for the two steps to disagree about which items they scored.

    Three ways a real corpus makes an item unscorable, all present in
    GIFT-Eval's bitbrains datasets and all of which turn MASE or WQL into NaN
    for the WHOLE task if they are averaged in:

      * a non-finite value in the target's horizon — nothing to compare against
      * a target history too short for the seasonal lag
      * a CONSTANT target history, which makes the seasonal-naive denominator
        zero and MASE infinite

    Dropping them and reporting the count is the honest form. Letting one NaN
    poison a task's mean is not a conservative choice — it silently removes the
    task from every comparison the table makes.

    The denominator is `MetricRegistry.compute_seasonal_naive_scales` on the
    target variate's OWN history, which is what MASE is defined on — not the
    whole multivariate block — and it is the same function, on the same NaN
    convention (`nanmean`), that decides which items survive and what they are
    divided by. Two hand-rolled denominators that disagreed about NaN would
    drop one population and score another.
    """
    # Variate 0 only, and indexed BEFORE any dtype conversion: `context` is a
    # (V, T) float32 block up to 69k steps wide, and converting it whole to
    # float64 per item costs gigabytes to then throw V-1 variates away.
    hist = [np.asarray(it["context"][0], dtype=np.float64) for it in items]
    truth = [np.asarray(it["truth"][0, :horizon], dtype=np.float64)
             for it in items]
    scales = Metrics.compute_seasonal_naive_scales(hist, seasonal_period)

    keep = np.array(
        [t.size == horizon and bool(np.isfinite(t).all()) for t in truth])
    keep &= np.isfinite(scales) & (scales > 0)
    idx = np.flatnonzero(keep)
    return {
        "idx": idx,
        "y_true": (np.stack([truth[i] for i in idx]) if idx.size
                   else np.zeros((0, horizon))),
        "scales": scales[idx],
        "n_scored": int(idx.size),
        "n_dropped": int(len(items) - idx.size),
    }


def score(target_q, prep: dict, quantile_levels, Metrics) -> dict:
    """MASE / WQL / per-item MASE for variate 0, over ``prep``'s survivors."""
    if not prep["n_scored"]:
        return {"MASE": float("nan"), "WQL": float("nan"),
                "mase_per_item": np.zeros(0)}
    # Mask FIRST, upcast second: the dropped rows' float64 copy is wasted.
    q = np.asarray(np.asarray(target_q)[prep["idx"]], dtype=np.float64)
    m = Metrics.compute_chronos_metrics_fast(
        q, prep["y_true"], None, list(quantile_levels),
        seasonal_period=1, scales=prep["scales"])
    return {"MASE": m["MASE"], "WQL": m["WQL"],
            "mase_per_item": m["mase_per_item"]}


def improvement_pct(step1, step2) -> float:
    """``(step1 - step2) / step1 * 100`` — positive means step 2 helped.

    One definition, used by run_eval per task and by build_table per row, so a
    zero-guard cannot drift between the csv and the table built from it.
    Works elementwise on arrays as well as on scalars.
    """
    a = np.asarray(step1, dtype=float)
    return (np.asarray(step2, dtype=float) - a) / np.where(
        (a == 0) | ~np.isfinite(a), np.nan, a) * -100.0


def win_rate(a: dict, b: dict) -> float:
    """Fraction of scored items where step 2 beat step 1, ties at 0.5.

    Per item, so the denominator is larger than one task. This is deliberately
    NOT `engine/evaluator.py::compute_win_rate`, which measures "beat the
    seasonal naive" (MASE < 1); here the comparison is between the two steps.
    """
    if not a["mase_per_item"].size:
        return float("nan")
    x, y = a["mase_per_item"], b["mase_per_item"]
    return float(((y < x) + 0.5 * (y == x)).mean())
