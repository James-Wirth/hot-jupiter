from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def runtime_metadata() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    packages = {}
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    for package in (
        "numpy",
        "scipy",
        "numba",
        "rebound",
        "pandas",
        "pyarrow",
        "joblib",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256": source_hash.hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
