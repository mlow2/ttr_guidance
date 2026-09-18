"""Standard dated run-directory naming: data/_dated/<YYYYMMDD.HHmmSS>_<name>_<git>."""
import datetime
import os
import re
import subprocess

DATED_ROOT = os.path.join("data", "_dated")

_STAMP_RE = re.compile(r"^\d{8}\.\d{6}_")


def git_short_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def dated_stamp():
    return datetime.datetime.now().strftime("%Y%m%d.%H%M%S")


def dated_run_dir(name, base=DATED_ROOT, git_hash=None, create=False):
    """Return ``<base>/<YYYYMMDD.HHmmSS>_<name>_<git>`` (optionally mkdir it)."""
    gh = git_hash if git_hash is not None else git_short_hash()
    path = os.path.join(base, f"{dated_stamp()}_{name}_{gh}")
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def resolve_run_dir(output_dir, name, *, base=DATED_ROOT, create=False):
    """Resolve an output dir: None -> fresh dated dir; an un-stamped explicit path is dated in place."""
    if output_dir is None:
        path = dated_run_dir(name, base=base)
    else:
        head, tail = os.path.split(os.path.normpath(output_dir))
        if _STAMP_RE.match(tail):
            path = output_dir
        else:
            path = dated_run_dir(tail or name, base=head or base)
    if create:
        os.makedirs(path, exist_ok=True)
    return path
