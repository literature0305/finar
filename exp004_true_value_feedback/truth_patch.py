#!/usr/bin/env python3
"""Feed the TRUE horizon into EO v4's second pass instead of its own forecast.

exp004 asks why forecast-space iterative refinement helps, by replacing what
pass 2 is handed:

    intermediate_pred_new = (1 - alpha) * intermediate_pred + alpha * target

alpha = 0 is the stock run (the blend is the identity, which is a free
consistency check); alpha = 1 hands pass 2 the ground truth and measures what
the second pass does with a perfect input.

THIS IS LABEL LEAKAGE BY CONSTRUCTION at any alpha > 0. The numbers are a
diagnostic upper bound and are not comparable to a benchmark score. Nothing
here should ever be quoted as model performance.

=============================================================================
THE TRUTH IS NOT IN THE MODEL'S INPUT, AND THAT IS THE WHOLE PROBLEM
=============================================================================
tsm-trainer already implements this blend — `EOModelV4._interpolate_initial_value`
does `v <- v*(1-c) + target*c` — but it is TRAINING-ONLY (`_init_options_active`
is `self.training and warmup_active()`), and the "target" it reaches for is the
`context` tensor, which at training time carries the future and at evaluation
time does not. The update region is masked at eval; the horizon truth lives in
the BENCHMARK ADAPTER, as the labels it scores against.

So the truth has to be carried from the adapter to the model. `TruthStore` does
that, and the two capture helpers below install one wrapper per benchmark at
the point where the adapter holds the labels and the model inputs together:

    fev-bench   `_window_to_task_inputs(past_data, future_data, task, ...)`
                takes the future window and RETURNS the very dicts the model is
                given, so `id(dict) -> truth` needs no frame inspection.
    GIFT-Eval   `Dataset.test_data` yields `(input_entry, label_entry)` pairs;
                the model is handed tensors rebuilt from the inputs, so
                identity is gone and the truth is keyed by a digest of the
                context — see `fingerprint` for why not an ordinal queue.

=============================================================================
WHERE THE BLEND IS APPLIED, AND WHY THERE
=============================================================================
On `med` — pass 1's median, the INPUT to `_coe_write_back` — and never on its
outputs. The write-back returns three things that must agree:

    value_channel  the median written over the update region
    mask_channel   the "filled at depth i" stamp
    acc            the residual accumulator, seeded FROM the value channel

Blending the value channel afterwards and leaving `acc` seeded from the
unblended median is exactly the failure the work order warns about: pass 2 adds
a residual computed against one input to an accumulator holding another.
Blending the INPUT instead makes all three derive from the blended median
through tsm-trainer's own code, so they cannot disagree.

`coe_residual` must be True for the accumulator to exist at all; `run_eval.py`
refuses a checkpoint without it.

SPACES AND SCALING. `med` is in prediction space and normalised with the
loc/scale the encoder used. The raw truth is therefore normalised with THAT
loc/scale (`self.instance_norm(x, loc_scale)`), never a fresh one — a target
normalised on its own statistics is a different scale wearing the same units —
then moved to prediction space the way `_seed_acc` moves the accumulator —
`to_pred_space`, or `_across` when `_crosses_scales` holds (a next-patch
objective together with a per-position scaler, where slot t+P is stored in its
own coordinates). Using a plain shift there would let the blend and the
accumulator disagree about what "moving" means, which is the one thing
`_coe_write_back` defends against. NaN truth becomes 0 BEFORE normalising, so
it lands on the series mean, which is what upstream does.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging

import numpy as np
import torch

logger = logging.getLogger("finar_exp004")

#: Where the per-row truth is parked between the chunk hook that can build it
#: and the write-back that needs it. On the MODEL, not a global: two pipelines
#: in one process would otherwise share it.
_ATTR = "_finar_truth_rows"


def fingerprint(target) -> bytes:
    """A content key for one task's context, exact under float32 round-trip.

    Used where identity is lost: the pipeline rebuilds input dicts from bare
    tensors (`convert_context_to_inputs`), so `id()` is gone by the time the
    chunk hook sees them. An ordinal queue would be worse — the multivariate
    GIFT route reorders and realigns entries, and a positional mismatch keeps
    the row COUNTS right, so the chunk hook's length check would pass while
    every row carried the wrong truth.

    OVER THE WHOLE ARRAY, not a tail. An earlier version digested the last 256
    steps "because the pipeline truncates the context from the left" — it does,
    but inside the model forward, well downstream of `_row_budget_chunks`,
    which sees the array the adapter built. A tail digest therefore bought
    nothing and manufactured a collision surface: measured, it made 2,907 of
    17,765 GIFT entries ambiguous and so untreated.

    The variate count is mixed in, so two tasks that share values but differ in
    width cannot collide.
    """
    a = np.ascontiguousarray(np.asarray(target, dtype=np.float32))
    h = hashlib.blake2b(a.tobytes(), digest_size=16)
    h.update(str(np.atleast_2d(a).shape[0]).encode())
    return h.digest()


class TruthStore:
    """Horizon truth for the rows the model is about to see.

    Two lookup modes because the two adapters hand the model different things.
    fev passes the input dicts straight through, so `by_id` is exact and free.
    GIFT-Eval rebuilds tensors from its entries, so identity is gone and the
    key has to come from the content — see `fingerprint`.

    A fingerprint that maps to two DIFFERENT truths is dropped rather than
    guessed: two identical contexts with different futures cannot be told
    apart, and silently picking one would put the wrong answer in front of the
    model. Those rows fall through to `n_misses`, which `run_eval.py` reports.
    """

    def __init__(self):
        self.by_id: dict[int, np.ndarray] = {}
        self.by_fp: dict[bytes, np.ndarray] = {}
        self._ambiguous: set[bytes] = set()
        self.n_hits = 0
        self.n_misses = 0

    def add_fp(self, target, truth) -> None:
        fp = fingerprint(target)
        if fp in self._ambiguous:
            return
        prev = self.by_fp.get(fp)
        if prev is not None and not np.array_equal(prev, truth):
            self._ambiguous.add(fp)
            self.by_fp.pop(fp, None)
            return
        self.by_fp[fp] = truth

    def take(self, inp) -> np.ndarray | None:
        """The ``(n_targets, H)`` truth for one task input, or None."""
        t = self.by_id.get(id(inp))
        if t is None and self.by_fp:
            t = self.by_fp.get(fingerprint(inp["target"]))
        if t is None:
            self.n_misses += 1
        else:
            self.n_hits += 1
        return t

    @property
    def n_ambiguous(self) -> int:
        return len(self._ambiguous)


class _RecordingTestData:
    """A gluonts ``TestData`` that records ``(context -> future)`` as it is read.

    Every access path — pair iteration, ``.input`` and ``.label`` — is served
    from the same underlying pair walk, so the truth is recorded exactly once
    per entry however the adapter chooses to consume the split. Anything else
    on the object is forwarded untouched.
    """

    def __init__(self, inner, store: "TruthStore"):
        self._inner = inner
        self._store = store

    def _pairs(self):
        for inp, lbl in self._inner:
            self._store.add_fp(inp["target"], np.atleast_2d(
                np.asarray(lbl["target"], dtype=np.float32)))
            yield inp, lbl

    def __iter__(self):
        return self._pairs()

    def __len__(self):
        return len(self._inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def input(self):
        return _HalfView(self, 0)

    @property
    def label(self):
        return _HalfView(self, 1)


class _HalfView:
    """One side of a recording pair walk, with the pair's length."""

    def __init__(self, pairs, which: int):
        self._pairs, self._which = pairs, which

    def __iter__(self):
        for pair in self._pairs:
            yield pair[self._which]

    def __len__(self):
        return len(self._pairs)


@contextlib.contextmanager
def capture_truth(benchmark: str, store: TruthStore):
    """Record the horizon truth the adapter is about to score against.

    One wrapper per benchmark, installed on the adapter module's globals and
    removed on exit. tsm-trainer is imported, never written to.
    """
    if benchmark.startswith("fev"):
        import fev
        import benchmarks.fev_bench as fb

        # TWO patches, in the order the adapter calls them. fev STRIPS the
        # target from `future_data` — it is the answer — so the horizon truth
        # is not an argument to `_window_to_task_inputs` at all; it lives on
        # the window, behind `get_ground_truth()`, whose own docstring says
        # "This data should never be provided to the model!". Providing it is
        # precisely this experiment, which is why every result here is a
        # diagnostic and not a score.
        #
        # `get_input_data` is called once per window immediately before the
        # task inputs are built, so stashing the window's truth there and
        # consuming it in the next call is safe for the single-threaded loop
        # the adapter runs. The consumer checks the length it got.
        orig_input = fev.EvaluationWindow.get_input_data
        orig_build = fb._window_to_task_inputs
        pending: dict = {"truth": None}

        def patched_input(self, *a, **kw):
            out = orig_input(self, *a, **kw)
            try:
                gt = self.get_ground_truth()
                pending["truth"] = gt
            except Exception as e:                     # noqa: BLE001
                logger.warning("no ground truth for this window (%s: %s)",
                               type(e).__name__, e)
                pending["truth"] = None
            return out

        def patched_build(past_data, future_data, task, *a, **kw):
            inputs = orig_build(past_data, future_data, task, *a, **kw)
            gt = pending["truth"]
            if gt is None:
                return inputs
            cols = list(task.target_columns)
            # COLUMNAR. `gt[i]` decodes a whole Arrow row into Python objects —
            # every value converted twice, per series, per column. Reading each
            # column once for the window is the same values in two lines.
            by_col = {c: np.asarray(gt[c], dtype=np.float32) for c in cols}
            n = min(len(inputs), len(gt))
            for i in range(n):
                store.by_id[id(inputs[i])] = np.stack(
                    [by_col[c][i] for c in cols], axis=0)
            # Released at the window boundary: a `datasets.Dataset` held here
            # would outlive its window and sit in RAM for the whole run.
            pending["truth"] = None
            return inputs

        # THE THIRD PATCH, for the path a checkpoint WITHOUT group attention
        # takes. fev-bench picks its input-construction mode once per run
        # (`fev_bench.py`, `mode = ...`), and only `covariate-aware` calls
        # `_window_to_task_inputs`. A forecaster whose `supports_covariates` is
        # False — for EO, `config.group_attention` reduced to false — runs in
        # `independent (group-id ignored)` mode, where the model is handed the
        # bare context arrays `_window_to_contexts` builds and the hook above
        # never fires.
        #
        # That is a limitation of WHERE the truth was captured, not of the
        # experiment: exp004 replaces a variate's own pass-1 forecast with its
        # own true future, which is self-feedback along time. It needs no
        # cross-variate path and is perfectly well defined for a
        # channel-independent model.
        #
        # BY FINGERPRINT, not id: the pipeline rebuilds input dicts from these
        # bare arrays (`convert_context_to_inputs`) before `_row_budget_chunks`
        # sees them, so identity is gone by then — the same reason the
        # GIFT-Eval branch below is fingerprinted.
        orig_ctx = fb._window_to_contexts

        def patched_contexts(past_data, target_col, *a, **kw):
            out = orig_ctx(past_data, target_col, *a, **kw)
            gt = pending["truth"]
            if gt is None:
                return out
            cols = ([target_col] if isinstance(target_col, str)
                    else list(target_col))
            by_col = {c: np.asarray(gt[c], dtype=np.float32) for c in cols}
            # The builder's own order: series-major, target-column-minor. One
            # context row per (series, column), so each row's truth is that one
            # column's horizon — a (1, H) block, since the rebuilt dict's
            # target is 1-D and `_row_truth` reads n_targets = 1 for it.
            k = 0
            for i in range(min(len(past_data), len(gt))):
                for c in cols:
                    if k >= len(out):
                        break
                    store.add_fp(out[k], by_col[c][i][None, :])
                    k += 1
            # `pending` is deliberately NOT cleared here. On the covariate-aware
            # path BOTH builders run for the same window, contexts first, and
            # clearing would leave `patched_build` with nothing — turning the
            # mode that already worked into one that captures nothing.
            return out

        fev.EvaluationWindow.get_input_data = patched_input
        fb._window_to_task_inputs = patched_build
        fb._window_to_contexts = patched_contexts
        try:
            yield
        finally:
            fev.EvaluationWindow.get_input_data = orig_input
            fb._window_to_task_inputs = orig_build
            fb._window_to_contexts = orig_ctx
        return

    if benchmark.startswith("gift"):
        import gift_eval.data as gd

        # The SOURCE module, not the adapter's namespace: both gift adapters do
        # `from gift_eval.data import Dataset` INSIDE the function, so the name
        # is resolved per call and an attribute on the adapter module would
        # never be read.
        orig_ds = gd.Dataset

        class _Dataset(orig_ds):
            """Records `(context -> future)` as the adapter's split is consumed.

            A subclass, not a proxy: the adapter reads `freq`,
            `prediction_length`, `target_dim` and more off this object, and a
            wrapper would have to forward every one of them correctly forever.

            THE SAME ARGUMENT ONE LEVEL DOWN, which the first version of this
            file abandoned and paid for. `test_data` is a gluonts `TestData`,
            and the multivariate adapter never iterates it as pairs before the
            forward — it iterates `test_data.input` (`gift_eval_hf_mul.py:273`)
            and only walks pairs afterwards, over the to_univariate view. A
            proxy that overrode `__iter__` and forwarded everything else
            therefore recorded NOTHING on the multivariate route, while the
            univariate tasks still produced hits through the parent adapter —
            so the global "did any truth arrive" gate stayed green and the
            multivariate half ran silently at alpha = 0. Measured: 20 entries
            iterated, 0 recorded.

            `input` and `label` are therefore served from the pair iteration
            too, so every access path funnels through one recording point.
            """

            @property
            def test_data(self):
                return _RecordingTestData(super().test_data, store)

        gd.Dataset = _Dataset
        try:
            yield
        finally:
            gd.Dataset = orig_ds
        return

    raise ValueError(f"no truth capture for benchmark {benchmark!r}")


def _row_truth(inputs, store: TruthStore):
    """``(rows, H)`` truth for one chunk, NaN on rows with none.

    Laid out by the pipeline's own rule — `_task_n_rows`: every variate of the
    task in array order, then one row per past covariate — so it lines up with
    the rows the write-back will see. Covariate rows are NaN: their future is
    either observed already (known-future) or not scored (past-only), and
    blending a covariate towards a target's truth would be nonsense.
    """
    widths, found = [], []
    for inp in inputs:
        tgt = inp["target"]
        n_t = 1 if getattr(tgt, "ndim", 1) == 1 else int(np.shape(tgt)[0])
        n_c = len(inp.get("past_covariates") or {})
        t = store.take(inp)
        found.append((n_t, n_c, None if t is None else
                      np.asarray(t, dtype=np.float32)))
        if t is not None:
            widths.append(found[-1][2].shape[1])
    if not widths:
        # No truth anywhere in this chunk: return None rather than allocate,
        # fill and transfer a block the write-back would mask out entirely.
        return None
    # SIZED FROM THE DATA. A fixed cap allocated `rows x 4096` per chunk when
    # these horizons are 8-720, and a horizon past the cap was silently clipped
    # — the unfilled tail read as "no truth" and degraded towards alpha = 0 on
    # the far end with no counter moving.
    H = max(widths)
    buf = np.full((sum(a + b for a, b, _ in found), H), np.nan, dtype=np.float32)
    r = 0
    for n_t, n_c, t in found:
        if t is not None:
            k = min(t.shape[0], n_t)
            buf[r:r + k, :t.shape[1]] = t[:k]
        r += n_t + n_c
    return torch.from_numpy(buf)


#: The tsm-trainer commit that introduced the prediction-space helpers this
#: module blends through. Named in the error so the fix is one `git log` away.
_REQUIRED_COMMIT = "a53691ba (2026-09-07, 'feat(eo-v4): a TimesFM-3.0 arm')"

#: What `truth_feedback` reaches for, and where. Checked BEFORE the model is
#: loaded, because the alternative is what this check exists to stop: an
#: ImportError raised from inside the patch, one benchmark into an alpha sweep,
#: after minutes of checkpoint loading — and raised identically for the
#: alpha=0.0 arm, which does no blending at all and looks like it should be safe.
_REQUIRED = (
    ("aed.model_base", ("to_pred_space",)),
    ("aed.eo_pipeline", ("_row_budget_chunks",)),
)
#: `instance_norm` is deliberately NOT here: it is an INSTANCE attribute, so
#: `hasattr(EOModelV4, ...)` is False even on a good checkout and the check
#: would refuse every repo. It predates all of these anyway.
_REQUIRED_MODEL_ATTRS = ("_coe_write_back", "_pred_shift", "_across",
                         "_crosses_scales")


def require_repo_support() -> None:
    """Refuse a repo whose EO v4 predates the helpers this experiment blends with.

    `to_pred_space`, `_pred_shift`, `_across` and `_crosses_scales` all arrived
    in ONE commit; a checkout older than it has none of them, and there is no
    fallback to write — the older write-back handles prediction-space
    coordinates differently, so blending the truth there would need different
    arithmetic, not a different import.
    """
    import importlib

    missing = []
    for mod_name, names in _REQUIRED:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            raise SystemExit(f"cannot import {mod_name} from --repo: {e}")
        missing += [f"{mod_name}.{n}" for n in names if not hasattr(mod, n)]
    try:
        from aed.eo_model_v4 import EOModelV4
    except ImportError as e:
        raise SystemExit(f"cannot import aed.eo_model_v4 from --repo: {e}")
    missing += [f"EOModelV4.{n}" for n in _REQUIRED_MODEL_ATTRS
                if not hasattr(EOModelV4, n)]
    if missing:
        raise SystemExit(
            f"this --repo checkout is missing {', '.join(missing)}.\n"
            f"They arrived together in {_REQUIRED_COMMIT}, and exp004 blends "
            f"the truth in the prediction space they define — an older "
            f"checkout needs different arithmetic, not a different import.\n"
            f"Update the tsm-trainer checkout --repo points at, or point "
            f"--repo at one at or after that commit.")


@contextlib.contextmanager
def truth_feedback(pipeline, alpha: float, store: TruthStore):
    """Blend the truth into what pass 2 is handed, for the duration of a block.

    Two patches, both removed on exit:

    * `eo_pipeline._row_budget_chunks` — the one seam every route into the
      model shares (`predict`, `embed`, `predict_repeats` and
      `predict_quantiles_tasks` all chunk through it), so the per-row truth is
      stashed for whichever route the benchmark takes. Computed per CHUNK: the
      budget splits the input list, and truth computed once over everything
      would be the wrong length for every chunk after the first.
    * `EOModelV4._coe_write_back` — where the blend is applied, to the INPUT.

    alpha = 0 installs nothing: the blend is the identity, so the arm is
    bit-identical to a stock run rather than "the patch with a zero weight",
    which makes it a real control for the rest of the sweep.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if alpha == 0.0:
        yield
        return

    model = pipeline.model
    if not getattr(getattr(model, "eo_config", None), "coe_residual", False):
        raise SystemExit(
            "coe_residual is not True on this checkpoint. Pass 2 then predicts "
            "an absolute forecast rather than a residual against the "
            "accumulator, so there is no accumulator to keep consistent with "
            "the blended input and this experiment is not defined for it.")
    # THE SAME attribute the sibling experiment's patch uses. Two different
    # names would each detect only themselves, so composing this with
    # exp002's feedback restriction in one process would install both over
    # `_row_budget_chunks` and `_coe_write_back`, fire neither guard, and tear
    # down in the wrong order — silently, which is the one thing the guard
    # exists to prevent.
    if hasattr(pipeline, "_finar_patched"):
        raise RuntimeError(
            f"a finar pipeline patch is already active "
            f"({getattr(pipeline, '_finar_patched')!r}); nesting two would "
            f"leave a wrapper installed on exit")

    import types as _types

    import aed.eo_pipeline as _eop
    from aed.model_base import to_pred_space

    orig_chunks = _eop._row_budget_chunks
    orig_write = model._coe_write_back

    import inspect as _inspect
    _write_sig = _inspect.signature(orig_write)   # bound: no `self`

    def patched_chunks(items, n_rows, max_rows):
        for chunk in orig_chunks(items, n_rows, max_rows):
            t = _row_truth(chunk, store)
            setattr(model, _ATTR, None if t is None else t.to(pipeline.device))
            yield chunk

    def patched_write(self, *args, **kwargs):
        """Blend `med` towards the truth, then run the stock write-back.

        The stock function then derives value_channel, mask_channel and acc
        from the blended median through its own code, so the three cannot
        disagree — which is what makes the residual correct by construction
        rather than by a second edit.
        """
        # The miss path is a single getattr: whenever a chunk had no truth,
        # binding the signature first would be pure-Python reflection thrown
        # away on every call.
        truth = getattr(self, _ATTR, None)
        if truth is None:
            return orig_write(*args, **kwargs)
        bound = _write_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        med = bound.arguments["med"]
        update_mask = bound.arguments["update_mask"]
        loc_scale = bound.arguments["loc_scale"]
        if truth.shape[0] != med.shape[0]:
            # Loud: the rows this saw are not the rows the truth was built for,
            # and blending them would corrupt every one of them quietly.
            raise RuntimeError(
                f"truth has {truth.shape[0]} rows but the write-back has "
                f"{med.shape[0]} — the hook is not aligned with the chunk")

        # Place the horizon truth on the update positions, in order, in VALUE
        # space so `instance_norm` can be the encoder's own.
        #
        # cumsum + gather, NOT a per-row loop: the loop this replaces called
        # `counts[r].item()` per row, and every one of those drains the CUDA
        # launch queue at the exact point the encoder is trying to keep it
        # full. The cost was linear in rows per chunk, so it grew with
        # --batch-size. `rank` is each update slot's position within its own
        # row's run, which is precisely the index into that row's truth.
        W = truth.shape[1]
        rank = update_mask.cumsum(1).sub_(1)
        # `in_range` BEFORE the clamp, and the clamp NOT in place: clamping
        # `rank` first makes `rank < W` vacuously true, so update positions past
        # the truth's width blend against its last value instead of being left
        # alone. That is silent — it moved one task's alpha=1 MASE by 0.09 while
        # every guard stayed green, because the horizon (28) and the update
        # region (a patch multiple, 32) differ by less than a patch.
        in_range = rank < W
        src = truth.to(med.dtype).gather(1, rank.clamp(0, W - 1))
        have = update_mask & in_range & torch.isfinite(src)
        flat = torch.where(have, src, torch.zeros((), dtype=med.dtype,
                                                  device=med.device))

        # THE ENCODER'S loc/scale, never a fresh one, and nan_to_num BEFORE
        # so a missing truth normalises to the series mean rather than
        # propagating. InstanceNorm with a supplied loc_scale is clamp then
        # (x - loc) / scale, which introduces no NaN from finite input, so one
        # pass is enough.
        normed, _ = self.instance_norm(flat, loc_scale)
        normed = normed.to(med.dtype)
        # MIRRORS `_seed_acc`, which is the function whose output this has to
        # stay consistent with. A plain `to_pred_space` is wrong exactly when
        # `_crosses_scales` holds — a next-patch objective AND a per-position
        # scaler — because value slot t+P is stored in ITS OWN coordinates
        # while output t is read in t's. `_across` is upstream's own name for
        # that move, and using anything else here would let the blend and the
        # accumulator disagree about what "moving" means.
        p = self._pred_shift()
        normed = (self._across(normed, p, loc_scale, to_pred_space)
                  if self._crosses_scales(p, loc_scale)
                  else to_pred_space(normed, p))
        # `torch.where`, NOT a lerp with a zero weight. lerp is a + w*(b - a),
        # so a non-finite `normed` off `have` propagates through 0 * NaN and
        # poisons rows the blend was supposed to leave alone — measured, as one
        # task's alpha=1 MASE moving 1.154 -> 1.248 when this was a lerp.
        # `where` discards the unselected branch instead of arithmetic on it.
        blended = torch.where(have, (1.0 - alpha) * med + alpha * normed, med)
        bound.arguments["med"] = blended
        return orig_write(*bound.args, **bound.kwargs)

    _eop._row_budget_chunks = patched_chunks
    model._coe_write_back = _types.MethodType(patched_write, model)
    pipeline._finar_patched = f"truth_feedback(alpha={alpha:g})"
    logger.info("true-value feedback installed, alpha=%.3f", alpha)
    try:
        yield
    finally:
        _eop._row_budget_chunks = orig_chunks
        # delattr, not re-assignment: the originals may have been INHERITED,
        # and assigning them back would pin a copy onto the instance.
        for obj, name in ((model, "_coe_write_back"),
                          (pipeline, "_finar_patched"), (model, _ATTR)):
            try:
                object.__delattr__(obj, name)
            except AttributeError:
                pass
        logger.info("true-value feedback removed (truth hits=%d misses=%d)",
                    store.n_hits, store.n_misses)
