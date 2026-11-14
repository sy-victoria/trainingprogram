import subprocess
import pytest
from trainingprogram.sandbox import DockerSandbox, SandboxUnavailable


def test_missing_docker_fails_closed(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(SandboxUnavailable):
        DockerSandbox().run(None, ["python", "-c", "raise SystemExit('must not run on host')"])


def test_image_missing_fails_closed(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "missing"))
    with pytest.raises(SandboxUnavailable):
        DockerSandbox().probe()
