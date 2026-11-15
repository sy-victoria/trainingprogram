from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return Path(os.environ.get("TP_DATA", "/root/autodl-tmp/trainingprogram-data")).resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
        raise ValueError("ID must contain only letters, digits, dot, dash or underscore")
    return value


def confined(root, relative):
    root = Path(root).resolve()
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or ":" in str(relative):
        raise ValueError("Absolute paths, traversal and drive paths are forbidden")
    path = root / candidate
    # Reject links even if they currently point inside the workspace.
    if any(p.is_symlink() for p in [path, *path.parents] if p != root and root in p.parents):
        raise ValueError("Symlinks are forbidden")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Path leaves workspace")
    return path


@contextlib.contextmanager
def exclusive(path):
    """OS advisory lock, automatically released after a crash (no stale lock flag)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        f.seek(0)
        if not f.read(1):
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError("Another process owns this run") from e
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def command_output(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"code": p.returncode, "stdout": p.stdout[-12000:], "stderr": p.stderr[-1000:]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}


def gpu_device():
    """Physical selector shared by CUDA and nvidia-smi (single GPU experiments)."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
    if not value or "," in value or value == "-1":
        raise ValueError("Set CUDA_VISIBLE_DEVICES to one GPU index or UUID")
    return value


def environment():
    import importlib.metadata
    import psutil
    return {
        "python": sys.version, "platform": platform.platform(),
        "cpu_count_visible": os.cpu_count(), "memory_visible_bytes": psutil.virtual_memory().total,
        "git": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "gpu": command_output(["nvidia-smi", "--id=" + gpu_device(), "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"]),
        "packages": {p.metadata["Name"]: p.version for p in importlib.metadata.distributions() if p.metadata["Name"]},
        "cgroup": {str(p): p.read_text().strip() for p in
                   [Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/cpu.max")] if p.exists()},
    }
