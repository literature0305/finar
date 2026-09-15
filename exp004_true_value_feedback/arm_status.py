#!/usr/bin/env python3
"""May a scored alpha arm be published? One definition, two readers.

`run_eval.py` decides it from a live `TruthStore` and records the answer;
`build_table.py` reads the recorded answer back. Those were two copies of the
same rule, which is how a producer and a consumer come to disagree about what
counts as publishable — so the rule itself lives here and both call it.

FAILING CLOSED IS THE POINT. An arm with `alpha > 0` whose verdict cannot be
read is refused, not accepted. The pre-fix code wrote `results.csv` BEFORE
`truth.json`, so a run interrupted between the two leaves a results directory
with numbers and no verdict — and "no verdict" for a leakage arm has to mean
"do not publish", or the one state the guard exists for slips through.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def refusal_from_counts(alpha: float, hits: int, misses: int, ambiguous: int,
                        where: str) -> str | None:
    """The rule. Returns why this arm may not be published, or None.

    alpha = 0 installs no capture at all, so it has nothing to cover and is
    exempt by construction.
    """
    if alpha <= 0:
        return None
    if hits == 0:
        return (f"{where}: the truth never reached the model ({misses} "
                f"misses). The capture seam did not fire — this arm is a "
                f"stock run wearing an alpha label.")
    if misses or ambiguous:
        # PARTIAL coverage is the failure the hits==0 check cannot see, and it
        # is worse than total failure because it still produces a number. Rows
        # that miss get ordinary model feedback while the rest get the truth,
        # so the arm is a blend of two treatments reported as one — and it
        # looks entirely normal.
        #
        # Ambiguity is the likely source at scale: two identical contexts with
        # different futures cannot be told apart, so `TruthStore.add_fp` drops
        # the fingerprint rather than guess, and constant or all-missing series
        # make that collision common.
        return (f"{where}: the truth reached only {hits}/{hits + misses} rows "
                f"({misses} misses, {ambiguous} ambiguous fingerprints). A "
                f"partially treated arm mixes true-value feedback with "
                f"ordinary feedback and reports the mixture as one number, so "
                f"it is refused rather than published.")
    return None


def recorded_verdict(dest: Path, alpha: float) -> str | None:
    """The verdict already on disk for one arm, or None if it may be published.

    Reads `truth.json`. Files written before the `refused` field existed carry
    only the counts, so the verdict is derived from them through the same rule
    — old results directories are covered without re-scoring them.
    """
    where = f"{dest.parent.name}/{dest.name}"
    truth = dest / "truth.json"
    if not truth.is_file():
        if alpha <= 0:
            return None
        return (f"{where}: no truth.json, so whether the truth reached the "
                f"model is unknown. An alpha > 0 arm with no verdict is "
                f"refused rather than assumed clean.")
    try:
        recorded = json.loads(truth.read_text())
    except (OSError, ValueError) as exc:
        if alpha <= 0:
            return None
        return f"{where}: truth.json is unreadable ({exc}); arm refused."
    if isinstance(recorded, dict) and "refused" in recorded:
        return recorded["refused"]
    try:
        hits = int(recorded["truth_hits"])
        misses = int(recorded["truth_misses"])
        ambiguous = int(recorded["ambiguous_fingerprints"])
    except (KeyError, TypeError, ValueError):
        if alpha <= 0:
            return None
        return (f"{where}: truth.json carries no usable coverage counts, so "
                f"the arm cannot be shown to be fully treated; refused.")
    return refusal_from_counts(alpha, hits, misses, ambiguous, where)


def write_atomic(path: Path, write) -> None:
    """`write(tmp)` then rename into place.

    A crash partway through `to_csv` used to leave a syntactically valid csv
    holding the first N tasks. Existence alone made the next run skip it, and
    the table then averaged over a truncated population — a wrong number, not
    a missing one.
    """
    tmp = path.with_name(path.name + ".part")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
