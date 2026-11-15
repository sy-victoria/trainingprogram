"""Explain readiness using container quotas rather than host totals."""
from pathlib import Path
import subprocess

from .common import command_output


def quotas(root=Path("/sys/fs/cgroup")):
    root = Path(root)
    result = {"memory_bytes": None, "cpu_cores": None}
    memory = root / "memory.max"
    if memory.exists():
        text = memory.read_text().strip()
        if text != "max":
            result["memory_bytes"] = int(text)
    else:
        memory = root / "memory/memory.limit_in_bytes"
        if memory.exists():
            value = int(memory.read_text())
            if value < 2**60:
                result["memory_bytes"] = value
    cpu = root / "cpu.max"
    if cpu.exists():
        quota, period = cpu.read_text().split()
        if quota != "max":
            result["cpu_cores"] = int(quota) / int(period)
    else:
        quota = root / "cpu/cpu.cfs_quota_us"
        period = root / "cpu/cpu.cfs_period_us"
        if quota.exists() and period.exists() and int(quota.read_text()) > 0:
            result["cpu_cores"] = int(quota.read_text()) / int(period.read_text())
    return result


def gpu_info():
    result = command_output(["nvidia-smi", "--id=0", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
    if result.get("code") != 0:
        return {"available": False, "detail": result}
    try:
        name, driver, memory = result["stdout"].strip().splitlines()[0].split(",")
        return {"available": True, "name": name.strip(), "driver": driver.strip(), "memory_mib": float(memory)}
    except (ValueError, IndexError):
        return {"available": False, "detail": result}


def check(stage="agent", size="0.8B"):
    from .models import llama_dir, tools_python, verify_llama
    errors = []
    resources = quotas()
    if stage == "agent":
        import importlib.util
        for module in ("langgraph", "langgraph.checkpoint.sqlite", "httpx"):
            if importlib.util.find_spec(module) is None:
                errors.append(f"Missing Python module: {module}")
        return {"ready": not errors, "stage": stage, "quotas": resources, "errors": errors,
                "note": "Agent package readiness only; no model or sandbox result implied"}
    gpu = gpu_info()
    if not gpu["available"]:
        errors.append("No GPU visible. Use GPU startup mode before model preparation or experiments.")
    elif gpu["memory_mib"] < 20 * 1024:
        errors.append("The default experiment configuration targets a GPU with at least 20GiB VRAM.")
    minimum = {"0.8B": 6, "2B": 12, "4B": 24}[size] * 1024**3
    if resources["memory_bytes"] is not None and resources["memory_bytes"] < minimum:
        errors.append(f"Container memory quota is below {minimum // 1024**3}GiB for {size} preparation.")
    if resources["cpu_cores"] is not None and resources["cpu_cores"] < 2:
        errors.append("Container CPU quota is below 2 cores; do not build/run in AutoDL no-GPU mode.")
    for path in [Path(tools_python()), llama_dir() / "build/bin/llama-server", llama_dir() / "build/bin/llama-quantize"]:
        try:
            if not path.is_file():
                errors.append(f"Missing tool: {path}")
        except OSError as exc:
            errors.append(f"Cannot access tool {path}: {exc}")
    if (llama_dir() / ".git").exists():
        try:
            verify_llama()
        except (ValueError, subprocess.CalledProcessError, OSError) as e:
            errors.append(str(e))
    return {"ready": not errors, "stage": stage, "size": size, "gpu": gpu, "quotas": resources, "errors": errors}


def require_gpu(size="0.8B"):
    result = check("gpu", size)
    if not result["ready"]:
        raise RuntimeError("Preflight failed BEFORE model download: " + " | ".join(result["errors"]))
    return result
