"""Sequential, resumable matrix runner; model calls are never silently repeated."""
from __future__ import annotations

from contextlib import contextmanager
import socket
import subprocess
import sys
import time

import httpx

from .common import data_root, read_json, write_json, exclusive, safe_name, ROOT, digest, sha256
from .evaluation import ensure_config, task_filename, generate, speed, judge
from .models import prepare, load_artifact, cleanup, versions
from .readiness import require_gpu
from .reporting import summarize
from .sandbox import DockerSandbox


def jobs(experiment_id, sizes, mode):
    safe_name(experiment_id)
    if not sizes or len(set(sizes)) != len(sizes) or any(s not in versions()["models"] for s in sizes):
        raise ValueError("Use distinct supported sizes: 0.8B, 2B, 4B")
    if mode not in {"smoke", "subset", "full"}:
        raise ValueError("Invalid experiment mode")
    return [{"run": safe_name(f"{experiment_id}-{size}-{variant}-{mode}"), "size": size,
             "variant": variant, "mode": mode} for size in sizes for variant in ("BF16", "Q4_K_M")]


def generated(run):
    folder = data_root() / "runs" / run
    if not (folder / "config.json").exists():
        return False
    config = read_json(folder / "config.json")
    for task_id in config["task_ids"]:
        path = folder / "samples" / task_filename(task_id)
        if not path.exists():
            return False
        record = read_json(path)
        if record.get("config_hash") != digest(config):
            raise ValueError("Saved sample config mismatch")
        if record["status"] == "started":
            return False
    return True


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@contextmanager
def managed_service(size, variant, port=8080, timeout=300):
    # Do not mistake an unrelated service for one launched by this runner.
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is already in use. Stop the other model service first.")
    _, artifact = load_artifact(size, variant)
    alias = "tp-" + artifact["sha256"][:16]
    from .client import ModelClient
    args = [sys.executable, "-m", "trainingprogram.cli", "model", "serve", "--size", size,
            "--variant", variant, "--port", str(port)]
    log_path = data_root() / "diagnostics" / f"controller-{size}-{variant}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + timeout
            with httpx.Client(timeout=2, trust_env=False) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Model service exited. Inspect {log_path} and services/{port}.log")
                    try:
                        if client.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                            ModelClient(f"http://127.0.0.1:{port}/v1", alias).verify(alias)
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Model service startup timed out")
                    time.sleep(1)
            yield alias
        finally:
            stop_process(process)


def run_matrix(experiment_id, sizes, mode="subset", judging="defer", cleanup_completed=False, port=8080):
    if judging not in {"required", "defer"}:
        raise ValueError("judging must be required or defer")
    matrix = jobs(experiment_id, sizes, mode)
    folder = data_root() / "experiments" / experiment_id
    source_hash = digest({p.name: sha256(p) for p in sorted((ROOT / "src/trainingprogram").glob("*.py"))})
    config = {"jobs": matrix, "judging": judging, "cleanup_completed": cleanup_completed,
              "port": port, "harness_hash": source_hash, "versions": versions()}
    with exclusive(data_root() / ".experiment.lock"):
        ensure_config(folder, config)
        state = read_json(folder / "status.json") if (folder / "status.json").exists() else {"jobs": {}}
        state["status"] = "running"
        write_json(folder / "status.json", state)
        try:
            if judging == "required":
                DockerSandbox().probe()  # fail before GPU work if judging is required
            for size in sizes:
                current = [j for j in matrix if j["size"] == size]
                needs_service = [j for j in current if not generated(j["run"]) or
                                 not (data_root() / "runs" / j["run"] / "speed.json").exists()]
                if needs_service:
                    require_gpu(size)
                    prepare(size)
                for job in current:
                    run = job["run"]
                    state["current"] = run
                    write_json(folder / "status.json", state)
                    if job in needs_service:
                        with managed_service(size, job["variant"], port):
                            generate(run, size, job["variant"], mode, port)
                            if not (data_root() / "runs" / run / "speed.json").exists():
                                speed(run, port)
                    state["jobs"][run] = {"generation": "complete", "judgment": "deferred"}
                    if judging == "required":
                        judge(run)
                        state["jobs"][run]["judgment"] = "attempted"
                    write_json(folder / "status.json", state)
                # Explicit opt-in; keep the last size available for the Agent demo.
                if cleanup_completed and size != sizes[-1]:
                    for target in ("source", "BF16", "Q4_K_M"):
                        cleanup(size, target)
            summarize()
            state["status"] = "generation_complete" if judging == "defer" else "judging_finished"
            state.pop("error", None)
            write_json(folder / "status.json", state)
        except BaseException as e:
            state["status"] = "interrupted" if isinstance(e, KeyboardInterrupt) else "failed"
            state["error"] = f"{type(e).__name__}: {e}"[:2000]
            write_json(folder / "status.json", state)
            raise
    return {"experiment": experiment_id, **state, "note": "See summary for completeness and actual scores"}


def judge_matrix(experiment_id):
    folder = data_root() / "experiments" / safe_name(experiment_id)
    config = read_json(folder / "config.json")
    DockerSandbox().probe()
    with exclusive(data_root() / ".experiment.lock"):
        for job in config["jobs"]:
            judge(job["run"])
        summarize()
    return {"experiment": experiment_id, "status": "judging_finished", "note": "Check summary for infrastructure errors or missing samples"}
