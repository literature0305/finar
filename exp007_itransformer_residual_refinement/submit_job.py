#!/usr/bin/env python3
"""Submit an exp007 training job to the Space server via ssub.

Modelled on tsm-trainer's `scripts/forecasting/training/utils/
space_job_submission.py` — same binary, same argument shape, same
`_sanitize_job_name` rule (ssub rejects a pod name with uppercase or
underscores, and auto-generated names carry both) — but it submits THIS
experiment's `train.sh`, imports nothing from that repo, and needs no config
yaml: exp007 is configured by flags.

ONE job runs every (dataset x horizon) cell in sequence on the same allocation
and evaluates each as it finishes, so a four-horizon sweep is one submission,
not four.

Usage:
    # normally reached through train.sh --mode job
    bash train.sh --mode job --dataset ETTh1 --pred-len "96 192 336 720" \
        --refinement on --train-depth 3 --depth 3

    # or directly: everything after `--` is train.sh's own argument list
    python submit_job.py --ngpu 1 -- \
        --dataset ETTh1 --pred-len "96 192 336 720" --refinement on

    # print the ssub command without submitting
    python submit_job.py --dry-run -- --dataset ETTh1 --pred-len 96
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# ── ssub defaults, as tsm-trainer's submitters use them ──────────────────
SSUB_BIN = "/group-volume/share/space-cli/ssub"
PRIORITY = "3"
EXP_ID = "370"
IMAGE = "sr-asr/tsm-trainer-torch2.6.0-cu121-fev0.8-20260608"
# Both buckets in ONE ssub argument, SPACE-separated — that is the list format
# ssub reads, not a comma. The space lives inside a single argv element, which
# holds only because this hands subprocess.run a LIST; rendering the command
# into a shell string would split it into `bucket=sdp-time-series` and a stray
# `sdp-time-series2`.
BUCKET = "sdp-time-series sdp-time-series2"

HERE = Path(__file__).resolve().parent

def build_combined_cmd(args) -> str:
    """The semicolon-separated command string ssub runs in the container.

    `args.train_args` is train.sh's OWN argv, minus the submission flags —
    train.sh strips those and hands the rest through untouched. Re-declaring
    each of them here meant a new flag had to be spelled in five places, and a
    missed one failed only inside a submitted container, hours later.

    Every word goes through `shlex.quote`, including the paths. Python's `repr`
    is NOT shell quoting — it renders an apostrophe with a backslash, which
    does not escape inside shell single quotes, so a path such as
    `/scratch/O'Brien/runs` would be reparsed into something else entirely.
    """
    q = shlex.quote
    train_sh = HERE / "train.sh"
    words = ["--mode", "local"] + list(args.train_args)
    return "; ".join([
        f"cd {q(str(HERE))}",
        f"bash {q(str(train_sh))} " + " ".join(q(w) for w in words),
    ])


#: Arguments whose value is a path. The container cd's to this directory, so a
#: path relative to the SUBMITTER's cwd would not resolve there.
_PATH_ARGS = ("--data-root", "--out", "--reference")


def absolutise_paths(words: list[str]) -> list[str]:
    out = list(words)
    for i, word in enumerate(out[:-1]):
        if word in _PATH_ARGS and not os.path.isabs(out[i + 1]):
            out[i + 1] = str((Path.cwd() / out[i + 1]).resolve())
    return out


def _sanitize_job_name(name: str) -> str:
    """Lowercase alphanumerics and inner hyphens only, starting with a letter.

    Auto-generated names violate this easily — a dataset is `ETTh1` and a
    variant is `coe3-res-bn` — and ssub rejects the whole submission with
    'Invalid Pod Name' rather than fixing it.
    """
    s = re.sub(r"[^a-z0-9-]+", "-", name.lower())
    s = re.sub(r"-{2,}", "-", s)
    s = re.sub(r"^[^a-z]+", "", s).rstrip("-")
    return s if len(s) >= 2 else "exp007-job"


def submit(job_name: str, combined_cmd: str, args) -> None:
    ssub_cmd = [
        SSUB_BIN,
        f"job_name={job_name}",
        f"cmd={combined_cmd}",
        f"priority={args.priority}",
        f"exp_id={args.exp_id}",
        f"image={args.image}",
        f"ngpu={args.ngpu}",
        f"gpu_type={args.gpu_type}",
        f"bucket={BUCKET}",
    ]
    print("=" * 70)
    print(f"  Job: {job_name}")
    print("=" * 70)
    for arg in ssub_cmd:
        print(f"    {arg}")
    print("\n  combined_cmd:")
    for part in combined_cmd.split("; "):
        print(f"    {part}")
    print(flush=True)
    if args.dry_run:
        print("  [DRY RUN] Skipped submission.\n")
        return
    try:
        result = subprocess.run(ssub_cmd, text=True)
    except FileNotFoundError:
        sys.exit(f"  ERROR: ssub not found at {SSUB_BIN}. Log in first: "
                 f"/group-volume/share/space-cli/space login --regin n6")
    if result.returncode != 0:
        # Exit non-zero. Printing a warning and returning 0 is how a caller
        # ends up reporting success for a job that was never submitted, and
        # finding out hours later that no results appeared.
        sys.exit(f"  ERROR: ssub exited with code {result.returncode} — the "
                 f"job was NOT submitted.")
    print("  Submitted successfully.\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit an exp007 training job to Space.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--gpu-type", default="A100",
                   choices=["A100", "H100", "2080ti"])
    p.add_argument("--ngpu", default="1",
                   help="exp007 trains one cell at a time on one device; more "
                        "GPUs do not make a cell faster (default: 1)")
    p.add_argument("--priority", default=PRIORITY)
    p.add_argument("--exp-id", default=EXP_ID)
    p.add_argument("--image", default=IMAGE)
    p.add_argument("--job-name", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("train_args", nargs="*",
                   help="after `--`: the train.sh arguments to run in the "
                        "container, forwarded verbatim")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.train_args = absolutise_paths(args.train_args)

    dataset = _flag_value(args.train_args, "--dataset") or "exp007"
    refined = (_flag_value(args.train_args, "--refinement") or "off").lower()
    name = args.job_name or (
        f"exp007-{dataset.split()[0]}-"
        f"{'refine' if refined in ('true', 'on') else 'base'}")
    sanitized = _sanitize_job_name(name)
    if sanitized != name:
        print(f"[submit_job] job name {name!r} is not a valid pod name — "
              f"using {sanitized!r}")
    print("=" * 70)
    print(" FiNAR exp007 — Job Submission")
    print("=" * 70)
    print(f"  workdir   : {HERE}")
    print(f"  train.sh  : {' '.join(args.train_args)}")
    print(f"  gpu-type  : {args.gpu_type}   ngpu: {args.ngpu}")
    print("=" * 70)
    submit(sanitized, build_combined_cmd(args), args)
    print("Done." + (" [DRY RUN] No jobs submitted." if args.dry_run
                     else " 1 job submitted."))


def _flag_value(words, flag: str) -> str | None:
    for i, word in enumerate(words[:-1]):
        if word == flag:
            return words[i + 1]
    return None


if __name__ == "__main__":
    main()
