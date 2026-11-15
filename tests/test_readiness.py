from pathlib import Path
from trainingprogram.readiness import quotas, check


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
