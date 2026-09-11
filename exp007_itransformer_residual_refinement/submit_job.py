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

    # or directly
    python submit_job.py --dataset ETTh1 --pred-len "96 192 336 720" \
        --refinement true --train-depth 3 --depth 3 --ngpu 1

    # print the ssub command without submitting
    python submit_job.py --dataset ETTh1 --pred-len 96 --dry-run
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

#: train.sh flags this forwards verbatim, as (our attribute, train.sh flag).
#: Everything here is a value flag; the two switches are handled separately.
FORWARD = [
    ("seq_len", "--seq-len"), ("data_root", "--data-root"), ("out", "--out"),
    ("refinement", "--refinement"), ("train_depth", "--train-depth"),
    ("depth", "--depth"), ("eval_depths", "--eval-depths"),
    ("coe_residual", "--coe-residual"),
    ("coe_bottleneck", "--coe-bottleneck"),
    ("coe_stochastic_repeat", "--coe-stochastic-repeat"),
    ("coe_backprop", "--coe-backprop"),
    ("coe_internal_loss", "--coe-internal-loss"),
    ("coe_future_marks", "--coe-future-marks"),
    ("epochs", "--epochs"), ("batch_size", "--batch-size"), ("lr", "--lr"),
    ("seed", "--seed"), ("num_workers", "--num-workers"),
    ("loader_workers", "--loader-workers"), ("reference", "--reference"),
]


def build_combined_cmd(args) -> str:
    """The semicolon-separated command string ssub runs in the container.

    Every word goes through `shlex.quote`, including the paths. Python's
    `repr` is NOT shell quoting — it renders an apostrophe with a backslash,
    which does not escape inside shell single quotes, so a path such as
    `/scratch/O'Brien/runs` would be reparsed into something else entirely.
    """
    train_sh = HERE / "train.sh"
    q = shlex.quote
    parts = ["--mode local",
             f"--dataset {q(args.dataset)}",
             f"--pred-len {q(str(args.pred_len))}"]
    for attr, flag in FORWARD:
        value = getattr(args, attr, None)
        if value is not None and value != "":
            parts.append(f"{flag} {q(str(value))}")
    if args.skip_precheck:
        parts.append("--skip-precheck")
    if args.no_reference:
        parts.append("--no-reference")
    if args.overwrite:
        parts.append("--overwrite")
    return "; ".join([
        f"cd {q(str(HERE))}",
        f"bash {q(str(train_sh))} " + " ".join(parts),
    ])


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
    p.add_argument("--dataset", required=True,
                   help="one or more dataset names, space separated")
    p.add_argument("--pred-len", default="96",
                   help="one or more horizons, space separated")
    p.add_argument("--seq-len", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--refinement", default=None, help="true or false")
    p.add_argument("--train-depth", default=None)
    p.add_argument("--depth", default=None)
    p.add_argument("--eval-depths", default=None)
    p.add_argument("--coe-residual", default=None)
    p.add_argument("--coe-bottleneck", default=None)
    p.add_argument("--coe-stochastic-repeat", default=None)
    p.add_argument("--coe-backprop", default=None)
    p.add_argument("--coe-internal-loss", default=None)
    p.add_argument("--coe-future-marks", default=None)
    p.add_argument("--epochs", default=None)
    p.add_argument("--batch-size", default=None)
    p.add_argument("--lr", default=None)
    p.add_argument("--seed", default=None)
    p.add_argument("--num-workers", default=None,
                   help="CPU cap inside the container")
    p.add_argument("--loader-workers", default=None)
    p.add_argument("--reference", default=None,
                   help="official checkout for the in-container precheck")
    p.add_argument("--skip-precheck", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="replace run directories holding a different config")
    p.add_argument("--no-reference", action="store_true",
                   help="let the in-container precheck run without the "
                        "official checkout (it refuses by default)")
    # ── compute ──
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Absolute, because the container cd's to this directory and a path
    # relative to the SUBMITTER's cwd would not resolve there.
    for attr in ("data_root", "out", "reference"):
        value = getattr(args, attr)
        if value and not os.path.isabs(value):
            setattr(args, attr, str((Path.cwd() / value).resolve()))

    name = args.job_name or (
        f"exp007-{args.dataset.split()[0]}-"
        f"{'refine' if str(args.refinement).lower() in ('true', 'on') else 'base'}")
    sanitized = _sanitize_job_name(name)
    if sanitized != name:
        print(f"[submit_job] job name {name!r} is not a valid pod name — "
              f"using {sanitized!r}")
    print("=" * 70)
    print(" FiNAR exp007 — Job Submission")
    print("=" * 70)
    print(f"  workdir   : {HERE}")
    print(f"  datasets  : {args.dataset}")
    print(f"  horizons  : {args.pred_len}")
    print(f"  refinement: {args.refinement}")
    print(f"  gpu-type  : {args.gpu_type}   ngpu: {args.ngpu}")
    print("=" * 70)
    submit(sanitized, build_combined_cmd(args), args)
    print("Done." + (" [DRY RUN] No jobs submitted." if args.dry_run
                     else " 1 job submitted."))


if __name__ == "__main__":
    main()
