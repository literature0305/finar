#!/usr/bin/env python3
"""Check that every launcher emits arguments the script it calls accepts.

WHY THIS EXISTS
---------------
This bug class has bitten twice, and `--help` catches neither case:

  * `--alpha` was documented in the header and forwarded to python, but the
    shell's `case` block had no arm for it, so every documented invocation of
    the new feature exited 2.
  * `--num-workers` was passed unconditionally as `--num-workers "${THREADS}"`
    after THREADS's default became empty, so the launcher emitted the option
    with an empty-string argument and argparse rejected it.

Both are a mismatch between what the launcher BUILDS and what the parser
ACCEPTS, and both survive `--help` — the parser is fine in isolation; it is the
generated command line that is wrong.

WHAT IT CHECKS, for every (launcher x stage x optional flag):
  1. every `--flag` the launcher emits is declared by the target script
  2. no flag is emitted with an empty-string value
  3. every flag in the header's OPTIONS block has a `case` arm, and vice versa

HOW. The launchers resolve their interpreter as `PY="${PYTHON:-...}"`, so this
points PYTHON at a stub that prints its argv one token per line and exits. The
real command line is therefore observed exactly — including EMPTY arguments,
which `--dry-run` cannot show, because it echoes the command and an empty string
leaves no trace in echoed text. That is why the first version of this checker
could not have caught the `--num-workers ""` bug it was written for.

Nothing real is executed: the stub replaces the interpreter, so stages that
would write data are safe to cover.

    python check_launchers.py          # exits non-zero on any finding
"""

from __future__ import annotations

import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
STAGES = ("eval", "table", "build", "all")

#: Values that satisfy a flag's `choices=`, so a legitimate rejection is not
#: reported as a launcher bug.
SAMPLE = {"--alpha": "0 1", "--scenarios": "full", "--benchmarks": "fev",
          "--horizons": "16", "--task-subset": "0 1",
          "--fev-subset": "multivariate",
          # exp007
          "--dataset": "ETTh1", "--datasets": "ETTh1", "--pred-len": "96",
          "--refinement": "on", "--coe-backprop": "last",
          "--coe-residual": "true", "--coe-bottleneck": "true",
          "--coe-stochastic-repeat": "true", "--coe-internal-loss": "false",
          "--coe-future-marks": "true", "--eval-depths": "1 2",
          "--depths": "1 2", "--device": "cpu", "--gpu-type": "A100",
          # job mode is the only way submit_job.py is reached, so the sweep
          # covers it here rather than leaving that arm unchecked.
          "--mode": "job"}

#: What a launcher needs before it reaches a python call, and which `--stage`
#: values it understands (`(None,)` for one that has no `--stage`). Keyed by
#: filename, with None as the fallback for every `run_exp*.sh`.
#:
#: `--out` is pinned away from the repo because these invocations really do run
#: the launcher: the stub replaces the INTERPRETER, not the shell around it, so
#: a `mkdir -p "${OUT}"` happens for real.
CHECK_DIR = "/tmp/finar-check-launchers"
INVOCATION = {
    "train.sh": (f"--dataset ETTh1 --out {CHECK_DIR}", (None,)),
    "eval.sh": (f"--out {CHECK_DIR}", ("eval", "table", "all")),
    # --verify-only, or the check would download 420 MB of datasets.
    "prepare_data.sh": (f"--verify-only --data-root {CHECK_DIR}", (None,)),
    None: ("--ckpt /tmp/fake", STAGES),
}
_NUMERIC = re.compile(r"depth|size|iters|^--n$|length|series|shards|context|"
                      r"max-|workers")


def _sample(flag: str) -> str:
    return SAMPLE.get(flag, "2" if _NUMERIC.search(flag) else "/tmp/x")


def _declared(py: pathlib.Path) -> set[str]:
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', py.read_text()))


def check_flag_docs(sh: pathlib.Path) -> list[str]:
    """Header OPTIONS block vs the `case` arms, both directions."""
    s = sh.read_text()
    head = s.split("\nset -", 1)[0]
    # a flag anywhere in a help line, so `--a PATH / --b PATH` counts both
    documented = set(re.findall(r'(--[a-z][a-z0-9-]*)', head))
    parsed = set(re.findall(r'^\s+(--[a-z][a-z0-9-]*)\)', s, re.M))
    parsed |= set(re.findall(r'^\s+-h\|(--[a-z]+)\)', s, re.M))
    out = []
    for f in sorted(parsed - documented - {"--help"}):
        out.append(f"{sh.name}: {f} is parsed but not in the header")
    for f in sorted(documented - parsed - {"--help"}):
        # only flags that look like an OPTIONS entry, not prose mentions
        if re.search(rf'^#\s{{2,}}{re.escape(f)}(?=[\s|])', head, re.M):
            out.append(f"{sh.name}: {f} is documented but has no `case` arm")
    return out


STUB = r"""#!/usr/bin/env python3
import sys
for a in sys.argv[1:]:
    sys.stdout.write("ARG\t" + a + "\n")
"""


def check_emitted(sh: pathlib.Path, stub: pathlib.Path) -> list[str]:
    s = sh.read_text()
    valued = re.findall(r'^\s+(--[a-z-]+)\)\s+[A-Z_]+="\$2"', s, re.M)
    boolean = re.findall(r'^\s+(--[a-z-]+)\)\s+[A-Z_]+="[^$][^"]*";\s*shift\s*;;',
                         s, re.M)
    skip = {"--ckpt", "--repo", "--out", "--stage", "--dry-run"}
    combos = [""] + [f"{f} {_sample(f)}" for f in valued if f not in skip] \
                  + [f for f in boolean if f != "--dry-run"]
    required, stages = INVOCATION.get(sh.name, INVOCATION[None])
    out = []
    for stage in stages:
        for extra in combos:
            stage_arg = "" if stage is None else f"--stage {stage}"
            r = subprocess.run(
                f'bash "{sh}" {required} {stage_arg} {extra}',
                shell=True, capture_output=True, text=True,
                env={**os.environ, "PYTHON": str(stub)})
            # One invocation's argv per run of the stub; a launcher may call it
            # more than once (eval then table), so split on the script token.
            toks = [l[4:] for l in r.stdout.splitlines() if l.startswith("ARG\t")]
            runs, cur = [], []
            for t in toks:
                if t.endswith(".py") and cur:
                    runs.append(cur); cur = [t]
                else:
                    cur.append(t)
            if cur:
                runs.append(cur)
            for toks in runs:
                py = next((pathlib.Path(t) for t in toks if t.endswith(".py")), None)
                if py is None or not py.is_file():
                    continue
                known = _declared(py)
                where = f"{sh.name} stage={stage} [{extra or 'no flags'}]"
                for i, t in enumerate(toks):
                    if not t.startswith("--"):
                        continue
                    if t not in known:
                        out.append(f"{where}: {py.name} does not accept {t}")
                    if i + 1 < len(toks) and toks[i + 1] == "":
                        out.append(f"{where}: {t} emitted with an empty value")
    return out


def main() -> int:
    findings = []
    launchers = sorted(p for pattern in ("exp*/run_exp*.sh", "exp*/train.sh",
                                        "exp*/eval.sh", "exp*/prepare_data.sh")
                       for p in ROOT.glob(pattern))
    if not launchers:
        print(f"no launcher scripts under {ROOT}")
        return 1
    with tempfile.TemporaryDirectory() as td:
        stub = pathlib.Path(td) / "python-stub"
        stub.write_text(STUB)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        for sh in launchers:
            findings += check_flag_docs(sh)
            findings += check_emitted(sh, stub)
    for f in sorted(set(findings)):
        print(f"  {f}")
    print(f"  {len(launchers)} launcher(s): {len(set(findings))} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
