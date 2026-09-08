#!/usr/bin/env python3
"""Restrict WHICH variates are fed back into EO v4's second recursion pass.

FiNAR feeds the whole intermediate forecast back into the next pass. This asks
which half of it earns the improvement — the model's own target trajectory
(self-regression) or its forecast of the covariates (covariate regression) —
by running the same checkpoint under three feedback regimes:

    full        every variate is fed back (stock EO v4 evaluation)
    self_only   targets are fed back; PAST-ONLY covariates are not
    cov_only    covariates are fed back; targets are not

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
`EOPipeline.predict_quantiles_tasks` lays every task out as contiguous rows:
targets first (`item_blocks` records `(start, n_targets)`), then one row per
past covariate, and `future_content_mask[i]` is 1 exactly for the covariates
that expose future values. So:

    target                 row index < start + n_targets
    known-future covariate not a target and future_content_mask.any()
    past-only covariate    not a target and not future_content_mask.any()

`self_only` suppresses PAST-ONLY covariates and leaves known-future ones alone:
a known-future covariate's forecast region is observed, not predicted, so it is
not feedback at all and blanking it would remove real information the scenario
is not about. In a mixed task that means only some of the covariates are
suppressed, which is intended.
"""

from __future__ import annotations

import contextlib
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


def task_row_roles(inputs):
    """``(is_target, is_known_future)`` per row, for one batch of fev tasks.

    Walks `inputs` in the SAME order `predict_quantiles_tasks` does — every
    target row, then one row per past covariate, per task — so the labels line
    up with the rows the model actually sees. A change to that layout breaks
    this loudly (a length mismatch against the value channel) rather than
    silently mislabelling rows.

    A target is never counted as known-future even if it carried future
    content: the scenarios are defined on the target/covariate split first.
    """
    is_target, is_known = [], []
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
        is_target += [True] * n_t + [False] * len(past_cov)
        is_known += [False] * n_t + [n in future_cov for n in past_cov]
    return (torch.tensor(is_target, dtype=torch.bool),
            torch.tensor(is_known, dtype=torch.bool))


def feedback_rows(scenario: str, is_target, is_known_future):
    """Boolean per row: may this variate's forecast be fed into the next pass?"""
    if scenario == "full":
        return torch.ones_like(is_target)
    if scenario == "self_only":
        # Targets yes; known-future covariates yes (their future is observed,
        # so it is not feedback); past-only covariates no.
        return is_target | is_known_future
    if scenario == "cov_only":
        return ~is_target
    raise ValueError(f"unknown scenario {scenario!r}; known: {SCENARIOS}")


@contextlib.contextmanager
def feedback_scenario(pipeline, scenario: str):
    """Run a block with `scenario`'s feedback restriction in force.

    Patches two objects and restores both, so a failure inside the block cannot
    leave a process scoring under a regime it did not ask for.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; known: {SCENARIOS}")
    if scenario == "full":
        yield
        return

    model = pipeline.model
    eo = getattr(model, "eo_config", None)
    if not getattr(eo, "coe_bottleneck", False):
        raise SystemExit(
            "coe_bottleneck is not True on this checkpoint, so passes exchange "
            "hidden state instead of a written-back forecast and there is no "
            "per-variate feedback to restrict. This experiment is undefined "
            "for it.")

    # INSTANCE attributes, not class ones. Assigning to
    # `type(pipeline)._tasks_forward` / `EOModelV4._coe_write_back` would put
    # the scenario on every instance in the process, so a second pipeline doing
    # a stock evaluation would silently be scored under this regime; it would
    # also miss a subclass that overrides _coe_write_back (the base assignment
    # loses to the subclass in the MRO) while still affecting unrelated
    # instances; and restoring an INHERITED method by assignment would leave it
    # permanently shadowed on the subclass. Binding to the two objects we were
    # handed has none of those properties, and `self._coe_write_back(...)`
    # inside the model finds an instance attribute first either way.
    import types as _types

    if hasattr(pipeline, "_finar_patched"):
        raise RuntimeError(
            "feedback_scenario is already active on this pipeline; nesting two "
            "regimes would make the inner one record the outer's patch as its "
            "original and leave a wrapper installed on exit.")
    orig_forward = pipeline._tasks_forward     # bound; may be inherited
    orig_write = model._coe_write_back

    def patched_forward(self, chunk, *a, **kw):
        """Stash this CHUNK's row roles, then run the stock forward.

        Hooked at _tasks_forward and not at predict_quantiles_tasks because the
        latter splits `inputs` with `_row_budget_chunks` under a row budget and
        calls this once per chunk. Roles computed over the whole input list
        would be the wrong length for every chunk after the first, and would
        silently suppress the wrong variates rather than fail.
        """
        is_target, is_known = task_row_roles(chunk)
        allow = feedback_rows(scenario, is_target, is_known)
        setattr(self.model, _ATTR, allow.to(self.device))
        try:
            return orig_forward(chunk, *a, **kw)
        finally:
            if hasattr(self.model, _ATTR):
                delattr(self.model, _ATTR)

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
        mask = torch.where(keep, new_mask, validity)
        acc = None if new_acc is None else torch.where(
            keep, new_acc, torch.zeros_like(new_acc))
        return value, mask, acc

    pipeline._tasks_forward = _types.MethodType(patched_forward, pipeline)
    model._coe_write_back = _types.MethodType(patched_write, model)
    pipeline._finar_patched = scenario
    logger.info("feedback scenario %r installed", scenario)
    try:
        yield
    finally:
        # delattr, not re-assignment: the originals may have been INHERITED,
        # and assigning them back would pin a copy onto the instance.
        for obj, name in ((pipeline, "_tasks_forward"),
                          (model, "_coe_write_back"),
                          (pipeline, "_finar_patched")):
            try:
                object.__delattr__(obj, name)
            except AttributeError:
                pass
        if hasattr(model, _ATTR):
            delattr(model, _ATTR)
        logger.info("feedback scenario %r removed", scenario)
