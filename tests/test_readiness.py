from pathlib import Path
from trainingprogram.readiness import quotas, check
import pytest


def test_cgroup_v2_reports_actual_no_gpu_mode_quota(tmp_path):
    (tmp_path / "memory.max").write_text("2147483648\n")
    (tmp_path / "cpu.max").write_text("50000 100000\n")
    assert quotas(tmp_path) == {"memory_bytes": 2 * 1024**3, "cpu_cores": 0.5}


def test_cgroup_v1(tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "cpu").mkdir()
    (tmp_path / "memory/memory.limit_in_bytes").write_text(str(90 * 1024**3))
    (tmp_path / "cpu/cpu.cfs_quota_us").write_text("2500000")
    (tmp_path / "cpu/cpu.cfs_period_us").write_text("100000")
    assert quotas(tmp_path) == {"memory_bytes": 90 * 1024**3, "cpu_cores": 25}


def test_unlimited_quotas(tmp_path):
    (tmp_path / "memory.max").write_text("max")
    (tmp_path / "cpu.max").write_text("max 100000")
    assert quotas(tmp_path) == {"memory_bytes": None, "cpu_cores": None}


def test_no_gpu_fails_before_downloading(tmp_path, monkeypatch):
    from trainingprogram import readiness
    monkeypatch.setenv("TP_DATA", str(tmp_path))
    monkeypatch.setattr(readiness, "gpu_info", lambda: {"available": False})
    monkeypatch.setattr(readiness, "quotas", lambda: {"memory_bytes": 2 * 1024**3, "cpu_cores": 0.5})
    result = check("gpu")
    assert not result["ready"]
    assert any("No GPU" in x for x in result["errors"])
    assert any("memory quota" in x for x in result["errors"])
    assert not (tmp_path / "models").exists()


def test_agent_check_never_requires_gpu(monkeypatch):
    from trainingprogram import readiness
    def forbidden():
        raise AssertionError("agent-only check should not query GPU")
    monkeypatch.setattr(readiness, "gpu_info", forbidden)
    assert check("agent")["ready"]


def test_inaccessible_tool_is_reported(tmp_path, monkeypatch):
    from trainingprogram import readiness
    monkeypatch.setenv("TP_DATA", str(tmp_path))
    monkeypatch.setattr(readiness, "gpu_info", lambda: {"available": False})
    def denied(self):
        raise PermissionError("test denied")
    monkeypatch.setattr(Path, "is_file", denied)
    result = check("gpu")
    assert not result["ready"]
    assert any("Cannot access tool" in error for error in result["errors"])


def test_selected_physical_gpu_is_used_for_readiness_and_monitor(monkeypatch):
    from trainingprogram import readiness, evaluation
    from types import SimpleNamespace
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-device")
    queries = []
    def query(args):
        queries.append(args)
        return {"code": 0, "stdout": "RTX 4090, 580.126, 24564"}
    monkeypatch.setattr(readiness, "command_output", query)
    assert readiness.gpu_info()["available"]
    def sample(args, **kwargs):
        queries.append(args)
        return SimpleNamespace(returncode=0, stdout="123")
    monkeypatch.setattr(evaluation.subprocess, "run", sample)
    monitor = evaluation.ResourceMonitor()
    monitor.sample()
    assert monitor.gpu == 123
    assert all("--id=GPU-test-device" in args for args in queries)
    assert "GPU-test-device" in monitor.result()["gpu_scope"]


def test_multiple_devices_are_rejected(monkeypatch):
    from trainingprogram.common import gpu_device
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match="one GPU"):
        gpu_device()
