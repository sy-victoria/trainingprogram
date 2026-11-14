import json

import pytest

from trainingprogram.common import write_json, read_json, digest
from trainingprogram.evaluation import judge


def setup_run(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_DATA", str(tmp_path))
    run = tmp_path / "runs/demo"
    config = {"task_ids": ["HumanEval/0", "HumanEval/1"], "evalplus": "pinned"}
    write_json(run / "config.json", config)
    write_json(run / "problems.json", {i: {"task_id": i} for i in config["task_ids"]})
    for index in range(2):
        write_json(run / "samples" / f"HumanEval_{index}.json",
                   {"task_id": f"HumanEval/{index}", "config_hash": digest(config), "status": "generated",
                    "code": "def f(): pass", "extraction_error": None})
    return run


def test_judge_resume_skips_finished_and_retries_infra(tmp_path, monkeypatch):
    run = setup_run(tmp_path, monkeypatch)
    class FakeSandbox:
        calls = 0
        def probe(self):
            return "sha256:pinned"
        def run(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                return {"returncode": 1, "timeout": False, "output": "infrastructure failed"}
            return {"returncode": 0, "timeout": False, "output": json.dumps({"status": "pass", "passed": True})}
    sandbox = FakeSandbox()
    judge("demo", sandbox)
    assert read_json(run / "judgments/HumanEval_1.json")["status"] == "infrastructure_error"
    judge("demo", sandbox)
    assert sandbox.calls == 3
    assert read_json(run / "judgments/HumanEval_1.json")["passed"] is True


def test_judge_configuration_change_rejected(tmp_path, monkeypatch):
    run = setup_run(tmp_path, monkeypatch)
    write_json(run / "judge/config.json", {"image_id": "different"})
    class FakeSandbox:
        def probe(self):
            return "new-image"
    with pytest.raises(ValueError, match="configuration changed"):
        judge("demo", FakeSandbox())
