#!/usr/bin/env python3
"""Restrict WHICH variates are fed back into EO v4's second pass — PSEUDO-TARGET.

FiNAR feeds the whole intermediate forecast back into the next pass. This asks
which half of it earns the improvement — the model's own trajectory
(self-regression) or its forecast of the OTHER variates (cross-variate
regression) — by running the same checkpoint under three feedback regimes:

    full        every variate is fed back (stock EO v4 evaluation)
    self_only   the target is fed back; the other variates are not
    cov_only    the other variates are fed back; the target is not

=============================================================================
WHY "PSEUDO"-TARGET
=============================================================================
The covariate-subset version of this experiment runs on tasks where fev-bench
names the target and covariate columns for us. It has a defect that this
version exists to remove: 9 of its 42 rows are MULTI-TARGET (7 of the 12
past_only rows, where `self_only` is the only regime that suppresses
anything), and `self_only` feeds back EVERY target of such a task. On those
rows "self" already includes cross-variate feedback between targets, so the
self/cross split it reports is not the split it names.

fev-bench's multivariate subset and GIFT-Eval's multivariate tasks have no
target/covariate distinction at all: every variate of a group is a target.
This experiment therefore DESIGNATES one — the FIRST variate of each group —
and treats the rest as pseudo-covariates. "Self" is then exactly one variate's
own trajectory, and "cross" is exactly the others, on every task.

THE DESIGNATION MUST NOT MOVE. If the chosen variate differed between
scenarios, or between iteration 1 and iteration 2, the three arms would not be
measuring the same quantity and the comparison would be meaningless. It cannot
move here: the roles are derived positionally from the SAME `inputs` dicts the
model is about to consume, by the same rule the pipeline uses to lay rows out
(`_task_n_rows`: targets first, in array order, then one row per past
covariate). Array order on disk is fixed, so variate 0 of a group is the same
series in every scenario and at every depth. `run_eval.py` records the
designation per task so the claim is auditable rather than asserted.

WHAT IS SCORED DOES NOT CHANGE. The benchmark still scores every variate; the
designation splits only the FEEDBACK. That leaves one asymmetry worth stating
plainly: `self_only` feeds back 1 variate and `cov_only` feeds back n-1, so on
wide groups the two arms differ in how MUCH is fed back as well as in what.
`build_table.py` reports the group width alongside the improvement so the two
readings can be told apart.

tsm-trainer is READ-ONLY here. Nothing on disk is touched: the two functions
below are installed over the loaded module objects at runtime and removed on
exit, so the same checkout can run a stock evaluation in the next process.

=============================================================================
WHAT "NOT FED BACK" HAS TO MEAN
=============================================================================
Pass 2 must see, for a suppressed variate, EXACTLY what pass 1 saw. In the
bottleneck chain that is three things, and `_coe_write_back` returns all three
together:

    acc          = _seed_acc(med, update_mask)                  residual base
    value_channel= where(update_mask, med, value_channel)       the fed-back median
    mask_channel = validity.masked_fill(update_mask, label)     "depth i filled this"

Zeroing only the value channel is the trap the work order names: pass 2 would
then produce a residual against a ZERO input and add it to an accumulator still
holding pass 1's median, so the suppressed variate would carry a prediction it
was never given. All three are restored together here.

The mask channel matters for the same reason and is easier to miss: with
`coe_repeat_encoding` the write-back stamps the update region with "filled at
depth i". Leaving that stamp on a suppressed variate tells the encoder a value
is present while handing it a zero.

Training does not need this care because it cannot make the mistake: the
drop in `_drop_feedback_variables` happens BEFORE `_seed_acc` derives the
accumulator from the value channel, so the two agree by construction. At eval
the write-back happens between passes, so the agreement has to be restored by
hand — which is what this module does, in the same order.

=============================================================================
HOW A ROW'S ROLE IS KNOWN
=============================================================================
Every path lays a task out by the same rule — `eo_pipeline._task_n_rows`:
`n_targets` rows in array order, then one row per past covariate (known-future
covariates reuse their past row). So, with variate 0 designated:

    target                 the FIRST row of the task's block
    pseudo-covariate       any other variate row of that block
    known-future covariate a past-covariate row whose name is in future_covariates
    past-only covariate    a past-covariate row that is not

`self_only` keeps the target and any known-future covariate: a known-future
covariate's forecast region is observed, not predicted, so it is not feedback
at all and blanking it would remove real information the scenario is not
about. These benchmarks carry no covariates, so in practice that clause is
inert here — it is kept because the code is shared with the covariate version
and silently dropping it would change what `self_only` means if a covariate
task ever reached this file.

=============================================================================
ONE SEAM FOR EVERY BENCHMARK ROUTE
=============================================================================
Every path into the model chunks its inputs through the same generator —
`eo_pipeline._row_budget_chunks(inputs, _task_n_rows, ...)` — and `predict`,
`embed`, `predict_repeats` and `predict_quantiles_tasks` all call it verbatim.
Wrapping that one function stashes the roles for whichever route is in use.

Naming the routes instead would be a trap. fev-bench arrives through
`predict_quantiles_tasks`, but GIFT-Eval does NOT arrive through `predict`: it
goes `predict_main_and_repeats` -> `predict_repeats_batch` ->
`EOPipeline.predict_repeats`, a third chunk loop entirely. A per-route hook has
to name every route correctly and silently covers none of the ones it missed —
and a missed route does not fail, it makes that benchmark's three regimes
identical to `full`.

Roles are computed per CHUNK, never over the whole input list: the budget
splits it, so roles computed once would be the wrong length for every chunk
after the first and would suppress the wrong variates rather than fail.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import logging

import torch

logger = logging.getLogger("finar_feedback")

#: The three regimes. `full` installs nothing, so it is bit-identical to a
#: stock run rather than "the patch with an all-true mask" — that keeps the
#: baseline arm honest about the patch itself.
SCENARIOS = ("full", "self_only", "cov_only")

#: Where the per-row decision is parked between the pipeline call that can
#: compute it and the model method that needs it. An attribute on the MODEL,
#: not a global: two pipelines in one process would otherwise share it.
_ATTR = "_finar_feedback_rows"


#: Which variate of a group is the designated target. 0 = the first, which is
#: the one guarantee available without a semantic column name: array order on
#: disk is fixed, so it is the same series in every scenario and at every
#: depth. Anything derived from the DATA — the highest-variance variate, say —
#: would be a different series per task and could in principle move between
#: runs, which is the one thing this experiment cannot afford.
TARGET_VARIATE = 0


def task_row_roles(inputs):
    """``(is_target, is_known_future, counts)`` per row, for one chunk.

    Walks `inputs` in the SAME order the pipeline lays rows out — every variate
    of the task in array order, then one row per past covariate — so the labels
    line up with the rows the model actually sees. `counts` is the
    ``(n_variates, n_past_covariates)`` pair per task, returned rather than
    recomputed by a second walk: it is what the audit needs and this function
    has already derived it.

    PSEUDO-TARGET: exactly one variate per group is a target, and it is
    `TARGET_VARIATE`. The remaining variates are pseudo-covariates.

    A target is never counted as known-future even if it carried future
    content: the scenarios are defined on the target/covariate split first.

    WHAT THE LENGTH CHECK IN `patched_write` DOES AND DOES NOT CATCH. It
    compares row COUNTS, so it catches a task contributing a different number
    of rows — but not a REORDERING within the same count. The two routes do
    order covariates differently: `_tasks_forward` walks `past_covariates` in
    insertion order, while the `prepare_batch` route sorts them and puts the
    known-future ones last (`chronos2/dataset.py`). The `is_known` positions
    computed here follow the first. That is inert for these benchmarks, which
    carry no covariates at all, and the `self_only` known-future clause is
    therefore UNVALIDATED on the second route — stated here rather than left to
    be discovered, because the count check cannot see it.
    """
    is_target, is_known, counts = [], [], []
    for inp in inputs:
        tgt = inp["target"]
        n_t = 1 if getattr(tgt, "ndim", 1) == 1 else int(tgt.shape[0])
        past_cov = inp.get("past_covariates") or {}
        future_cov = inp.get("future_covariates") or {}
        # ENFORCED, not assumed. `_tasks_forward` builds one row per
        # past_covariates key and reads future values out of future_covariates
        # by that key, so a future-only covariate gets NO ROW and is never
        # shown to the model at all — while fev metadata would still report the
        # task as future-known, and all three scenarios would be read as a
        # known-future result on a task the covariate never reached.
        orphan = set(future_cov) - set(past_cov)
        if orphan:
            raise RuntimeError(
                f"future_covariates {sorted(orphan)} are not in "
                f"past_covariates, so they get no encoder row. Task target "
                f"shape {getattr(tgt, 'shape', None)}.")
        if n_t <= TARGET_VARIATE:
            # A group narrower than the designated index would silently get NO
            # target and turn `self_only` into "feed back nothing", which
            # scores as iteration 1 and looks like a null result.
            raise RuntimeError(
                f"task has {n_t} variate(s), so there is no variate "
                f"{TARGET_VARIATE} to designate as the target")
        is_target += [i == TARGET_VARIATE for i in range(n_t)] \
            + [False] * len(past_cov)
        is_known += [False] * n_t + [n in future_cov for n in past_cov]
        counts.append((n_t, len(past_cov)))
    return (torch.tensor(is_target, dtype=torch.bool),
            torch.tensor(is_known, dtype=torch.bool), counts)


class Designation:
    """What was designated over a run — bounded, and comparable across arms.

    "The target did not move between the arms" is the assumption the whole
    comparison rests on, so it is recorded and checked rather than asserted.
    What it is NOT is a per-task dump: the record is one entry per task per
    chunk, which on GIFT-Eval reaches six figures, and a list of dicts at ~192
    bytes each would be the only unbounded allocation in a process that has
    already been OOM-killed once on this host.

    An order-sensitive digest plus a histogram carries everything the check and
    the log line need: the digest distinguishes any difference in the ordered
    sequence, and the histogram gives the group widths without keeping a row
    per task. `TARGET_VARIATE` is a module constant and so carries no
    information — it is folded into the digest's prefix rather than repeated.
    """

    def __init__(self):
        self._h = hashlib.blake2b(digest_size=16)
        self._h.update(b"target_variate=%d;" % TARGET_VARIATE)
        self.n_tasks = 0
        self.widths: "collections.Counter" = collections.Counter()

    def update(self, counts) -> None:
        for n_t, n_pc in counts:
            self._h.update(b"%d,%d;" % (n_t, n_pc))
            self.n_tasks += 1
            self.widths[(n_t, n_pc)] += 1

    def signature(self) -> tuple:
        return (self.n_tasks, self._h.hexdigest())

    def to_dict(self) -> dict:
        w = [n for (n, _), c in self.widths.items() for _ in range(c)]
        return {"n_tasks_seen": self.n_tasks,
                "target_variate": TARGET_VARIATE,
                "digest": self._h.hexdigest(),
                "n_univariate": sum(1 for n in w if n < 2),
                "width_min": min(w, default=0), "width_max": max(w, default=0),
                "histogram": {f"{n}v_{c}cov": k
                              for (n, c), k in sorted(self.widths.items())}}


def feedback_rows(scenario: str, is_target, is_known_future):
    """Boolean per row: may this variate's forecast be fed into the next pass?"""
    if scenario == "full":
        return torch.ones_like(is_target)
    if scenario == "self_only":
        # The target yes; known-future covariates yes (their future is
        # observed, so it is not feedback); everything else no.
        return is_target | is_known_future
    if scenario == "cov_only":
        return ~is_target
    raise ValueError(f"unknown scenario {scenario!r}; known: {SCENARIOS}")


@contextlib.contextmanager
def feedback_scenario(pipeline, scenario: str, audit: "Designation | None" = None):
    """Run a block with `scenario`'s feedback restriction in force.

    ONE SEAM, not one per benchmark route. Every path into the model chunks its
    inputs through `eo_pipeline._row_budget_chunks(inputs, _task_n_rows, ...)`
    — `predict`, `embed`, `predict_repeats` and `predict_quantiles_tasks` all
    do, verbatim — so wrapping that generator stashes the roles for whichever
    route is in use. fev-bench arrives via `predict_quantiles_tasks`;
    GIFT-Eval, through `predict_main_and_repeats` -> `predict_repeats_batch`,
    arrives via `predict_repeats`. Hooking a per-route entry point instead
    would have to name every route correctly and would miss any new one — and
    the route GIFT-Eval actually takes is not the obvious one.

    Roles are computed per CHUNK, never over the whole input list: the budget
    splits it, so roles computed once would be the wrong length for every chunk
    after the first and would suppress the wrong variates rather than fail.

    `_row_budget_chunks` is a module global, so the patch is process-wide while
    installed — narrower than patching `prepare_batch`, which `aed.pipeline`
    and `aed.hybrid_pipeline` share, but still global. `run_eval.py` scores one
    scenario at a time in one process, the context manager removes the patch on
    exit including on failure, and `_finar_patched` refuses a second concurrent
    installation.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; known: {SCENARIOS}")
    suppress = scenario != "full"
    if not suppress and audit is None:
        # Nothing to restrict and nothing to record: install NOTHING, so the
        # baseline arm is bit-identical to a stock run rather than "the patch
        # with an all-true mask".
        yield
        return

    model = pipeline.model
    if suppress:
        eo = getattr(model, "eo_config", None)
        if not getattr(eo, "coe_bottleneck", False):
            raise SystemExit(
                "coe_bottleneck is not True on this checkpoint, so passes "
                "exchange hidden state instead of a written-back forecast and "
                "there is no per-variate feedback to restrict. This experiment "
                "is undefined for it.")
    if hasattr(pipeline, "_finar_patched"):
        raise RuntimeError(
            "feedback_scenario is already active on this pipeline; nesting two "
            "regimes would make the inner one record the outer's patch as its "
            "original and leave a wrapper installed on exit.")

    import types as _types

    import aed.eo_pipeline as _eop
    orig_chunks = _eop._row_budget_chunks
    orig_write = model._coe_write_back

    def patched_chunks(items, n_rows, max_rows):
        """Stash each chunk's roles as the chunk is handed to the caller.

        A generator, so the stash lands immediately before the consumer's
        forward rather than all at once — every call site is a bare
        ``for chunk in _row_budget_chunks(...)``, so laziness is safe here and
        is what keeps the stash aligned with the chunk being processed.
        """
        for chunk in orig_chunks(items, n_rows, max_rows):
            is_target, is_known, counts = task_row_roles(chunk)
            if audit is not None:
                audit.update(counts)
            if suppress:
                allow = feedback_rows(scenario, is_target, is_known)
                setattr(model, _ATTR, allow.to(pipeline.device))
            yield chunk

    # BOUND to the original's signature, not spelled out. `_coe_write_back`
    # grew a `loc_scale` parameter, and a wrapper that names its arguments
    # positionally fails with "takes 6 positional arguments but 7 were given"
    # — on every task, at eval time, after the model is already loaded.
    # Binding means the two parameters this needs are found by NAME and every
    # other one is passed straight through, whatever they become.
    import inspect as _inspect
    _write_sig = _inspect.signature(orig_write)   # bound: no `self`

    def patched_write(self, *args, **kwargs):
        """The stock write-back, undone for the rows this scenario suppresses.

        Restores ALL THREE of what the write-back produces — see WHAT "NOT FED
        BACK" HAS TO MEAN in the module docstring. Restoring the value channel
        alone would leave the accumulator holding pass i's median, so pass i+1
        would add a residual computed against a zero input to a prediction the
        variate was never shown.
        """
        new_value, new_mask, new_acc = orig_write(*args, **kwargs)
        allow = getattr(self, _ATTR, None)
        if allow is None:
            return new_value, new_mask, new_acc
        bound = _write_sig.bind(*args, **kwargs)
        value_channel = bound.arguments["value_channel"]
        validity = bound.arguments["validity"]
        if allow.shape[0] != new_value.shape[0]:
            # Loud, because the alternative is suppressing the WRONG variates:
            # the roles are positional, so a length mismatch means the rows
            # this saw are not the rows they were computed for.
            raise RuntimeError(
                f"feedback mask has {allow.shape[0]} rows but the write-back "
                f"has {new_value.shape[0]} — the hook is not aligned with the "
                f"forward's chunk")
        # (rows,) -> (rows, 1, ...) so it broadcasts over the time axis.
        keep = allow.reshape(-1, *([1] * (new_value.dim() - 1)))
        value = torch.where(keep, new_value, value_channel)
        # `is`, not a blanket where: with coe_repeat_encoding off the write-back
        # returns `validity` ITSELF as the mask channel, so the select would
        # allocate a full-size copy of a tensor it cannot change.
        mask = new_mask if new_mask is validity else torch.where(
            keep, new_mask, validity)
        # A 0-dim zero broadcasts; `zeros_like` would allocate a full-size
        # tensor purely to be selected against.
        acc = None if new_acc is None else torch.where(
            keep, new_acc, new_acc.new_zeros(()))
        return value, mask, acc

    _eop._row_budget_chunks = patched_chunks
    if suppress:
        model._coe_write_back = _types.MethodType(patched_write, model)
    pipeline._finar_patched = scenario
    logger.info("feedback scenario %r installed (%s)", scenario,
                "suppressing" if suppress else "audit only")
    try:
        yield
    finally:
        _eop._row_budget_chunks = orig_chunks
        # delattr, not re-assignment: the originals may have been INHERITED,
        # and assigning them back would pin a copy onto the instance.
        for obj, name in ((model, "_coe_write_back"),
                          (pipeline, "_finar_patched")):
            try:
                object.__delattr__(obj, name)
            except AttributeError:
                pass
        if hasattr(model, _ATTR):
            delattr(model, _ATTR)
        logger.info("feedback scenario %r removed", scenario)
