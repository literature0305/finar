#!/usr/bin/env python3
"""Is what is on disk the benchmark the paper reports on?

Called by `prepare_data.sh` after every download, and safe to run on its own.
Shape is the hard check — row count and variate count — because a truncated or
re-exported file trains happily and produces an MSE that simply is not
comparable to Table 10. The sha256 is advisory: the same series legitimately
round-trips to different bytes through a different mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# CPU cap BEFORE `data`, which imports numpy/pandas/torch: OpenMP and BLAS
# size their pools at load time. See finar_cpu.py.
sys.path.append(str(next(d for d in Path(__file__).resolve().parents
                         if (d / "finar_cpu.py").is_file())))
from finar_cpu import limit_cpu  # noqa: E402

limit_cpu(quiet=True)

from data import DATASETS, dataset_path  # noqa: E402


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _shape(path: Path, has_header: bool) -> tuple[int, int]:
    """`(data rows, value columns)` without loading the file into memory."""
    with open(path, "r", encoding="utf-8") as fh:
        first = fh.readline()
        if not first:
            return 0, 0
        columns = len(first.rstrip("\n").split(","))
        rows = sum(1 for _ in fh) + (0 if has_header else 1)
    return rows, columns


def verify(data_root: str, names) -> int:
    worst = 0
    for name in names:
        spec = DATASETS[name]
        path = Path(dataset_path(data_root, name))
        if not path.is_file():
            print(f"  [MISSING] {name:<9} {path}")
            worst = max(worst, 2)
            continue
        header = path.suffix == ".csv"
        rows, columns = _shape(path, header)
        # The date column is not a variate; Solar's headerless file has none.
        variates = columns - 1 if header else columns
        problems = []
        if rows != spec["rows"]:
            problems.append(f"{rows} rows, expected {spec['rows']}")
        if variates != spec["enc_in"]:
            problems.append(f"{variates} variates, expected {spec['enc_in']}")
        digest = _sha256(path)[:16]
        same = digest == spec["sha256"]
        if problems:
            print(f"  [BAD]     {name:<9} {'; '.join(problems)}  ({path})")
            worst = max(worst, 2)
        else:
            note = "sha256 matches the reference copy" if same else (
                f"sha256 {digest} differs from the reference "
                f"{spec['sha256']} — shape is right, so this is a different "
                f"mirror, not a different dataset")
            print(f"  [OK]      {name:<9} {rows} x {variates}  {note}")
            if not same:
                worst = max(worst, 1)
    return worst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--datasets", nargs="*", default=sorted(DATASETS))
    p.add_argument("--strict", action="store_true",
                   help="also fail when only the sha256 differs")
    args = p.parse_args()
    unknown = [n for n in args.datasets if n not in DATASETS]
    if unknown:
        sys.exit(f"unknown dataset(s): {', '.join(unknown)}")
    print(f"verifying {args.data_root}")
    worst = verify(args.data_root, args.datasets)
    if worst >= 2 or (worst == 1 and args.strict):
        sys.exit(1)


if __name__ == "__main__":
    main()
