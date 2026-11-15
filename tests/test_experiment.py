from contextlib import contextmanager
from pathlib import Path

import pytest

from trainingprogram import experiment
from trainingprogram.common import read_json, write_json, digest


def fake_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_DATA", str(tmp_path))
    calls = []
    monkeypatch.setattr(experiment, "require_gpu", lambda size: calls.append(("preflight", size)))
    monkeypatch.setattr(experiment, "prepare", lambda size: calls.append(("prepare", size)))
    @contextmanager
    def service(size, variant, port):
        calls.append(("start", size, variant))
        try:
            yield "fake"
        finally:
            calls.append(("stop", size, variant))
    monkeypatch.setattr(experiment, "managed_service", service)
    def generate(run, size, variant, mode, port):
        calls.append(("generate", run))
        config = {"task_ids": ["HumanEval/0"]}
        write_json(tmp_path / "runs" / run / "config.json", config)
        write_json(tmp_path / "runs" / run / "samples/HumanEval_0.json",
                   {"status": "generated", "config_hash": digest(config)})
    monkeypatch.setattr(experiment, "generate", generate)
    def speed(run, port):
        calls.append(("speed", run))
        write_json(tmp_path / "runs" / run / "speed.json", {"records": []})
    monkeypatch.setattr(experiment, "speed", speed)
    monkeypatch.setattr(experiment, "cleanup", lambda s, t: calls.append(("cleanup", s, t)))
    monkeypatch.setattr(experiment, "summarize", lambda: calls.append(("summary",)))
    return calls


def test_six_jobs_resume_without_restarting_models(tmp_path, monkeypatch):
    calls = fake_runtime(tmp_path, monkeypatch)
    result = experiment.run_matrix("demo", ["0.8B", "2B", "4B"], cleanup_completed=True)
    assert result["status"] == "generation_complete"
    assert len([c for c in calls if c[0] == "generate"]) == 6
    assert len([c for c in calls if c[0] == "cleanup"]) == 6
    assert not any(c[0] == "cleanup" and c[1] == "4B" for c in calls)
    calls.clear()
    experiment.run_matrix("demo", ["0.8B", "2B", "4B"], cleanup_completed=True)
    assert not any(c[0] in {"preflight", "prepare", "start", "generate", "speed"} for c in calls)


def test_failure_stops_service_and_resumes_missing_jobs(tmp_path, monkeypatch):
    calls = fake_runtime(tmp_path, monkeypatch)
    original = experiment.generate
    def failing(run, size, variant, mode, port):
        if variant == "Q4_K_M":
            raise RuntimeError("test interruption")
        original(run, size, variant, mode, port)
    monkeypatch.setattr(experiment, "generate", failing)
    with pytest.raises(RuntimeError):
        experiment.run_matrix("demo", ["0.8B"])
    assert calls[-1] == ("stop", "0.8B", "Q4_K_M")
    assert read_json(tmp_path / "experiments/demo/status.json")["status"] == "failed"
    calls.clear()
    monkeypatch.setattr(experiment, "generate", original)
    experiment.run_matrix("demo", ["0.8B"])
    assert len([c for c in calls if c[0] == "generate"]) == 1


def test_matrix_config_cannot_change(tmp_path, monkeypatch):
    fake_runtime(tmp_path, monkeypatch)
    experiment.run_matrix("demo", ["0.8B"])
    with pytest.raises(ValueError, match="configuration changed"):
        experiment.run_matrix("demo", ["0.8B", "2B"])


def test_required_judge_fails_before_gpu_work(tmp_path, monkeypatch):
    from trainingprogram.sandbox import SandboxUnavailable
    calls = fake_runtime(tmp_path, monkeypatch)
    def unavailable(self):
        raise SandboxUnavailable("No Docker")
    monkeypatch.setattr(experiment.DockerSandbox, "probe", unavailable)
    with pytest.raises(SandboxUnavailable):
        experiment.run_matrix("demo", ["0.8B"], judging="required")
    assert not calls


def test_managed_service_stopped_on_startup_timeout(tmp_path, monkeypatch):
    fake_runtime(tmp_path, monkeypatch)
    # Restore original context manager replaced by fake_runtime.
    import importlib
    importlib.reload(experiment)
    class Process:
        stopped = False
        def poll(self):
            return 0 if self.stopped else None
        def terminate(self):
            self.stopped = True
        def wait(self, **kwargs):
            return 0
    process = Process()
    monkeypatch.setattr(experiment.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(experiment, "load_artifact", lambda *a: (None, {"sha256": "abc"}))
    class Client:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, *args):
            class Response:
                status_code = 503
            return Response()
    monkeypatch.setattr(experiment.httpx, "Client", Client)
    # Bind a temporary free port only for discovery; no service is started.
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(RuntimeError, match="startup timed out"):
        with experiment.managed_service("0.8B", "BF16", port, timeout=0):
            raise AssertionError("unreachable")
    assert process.stopped
