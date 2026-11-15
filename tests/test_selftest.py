import zipfile
from trainingprogram.selftest import run_selftest
from trainingprogram.reporting import export_bundle, summarize
from trainingprogram.common import read_json


def test_offline_demo_isolated_from_model_scores(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_DATA", str(tmp_path / "data"))
    monkeypatch.setattr("trainingprogram.evaluation.ResourceMonitor.sample", lambda self: None)
    result = run_selftest()
    assert result["simulated"] and all(result["checks"].values())
    assert not result["real_model_called"]
    assert not result["generated_code_executed"]
    directory = summarize()
    assert read_json(directory + "/results.json") == {"runs": [], "pairs": [], "agents": []}
    archive = export_bundle(tmp_path / "result.zip")
    with zipfile.ZipFile(archive) as z:
        assert not any(name.startswith("offline/") for name in z.namelist())
