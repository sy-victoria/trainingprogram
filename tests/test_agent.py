import json
import shutil
from pathlib import Path

import pytest

from trainingprogram.aci import ACI
from trainingprogram.agent import create_run, invoke_run
from trainingprogram.common import write_json, read_json, ROOT


class FakeClient:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return {"text": json.dumps(next(self.actions)), "finish_reason": "stop"}


PATCH = {"tool": "apply_patch", "args": {"path": "solution.py",
         "before": "return sum(values) / len(values)", "after": "return sum(values) / len(values) if values else None"}}
FINISH = {"tool": "finish", "args": {"summary": "done"}}


def make_run(tmp_path, budget=10):
    path = tmp_path / "run"
    create_run(path, "boundary", "http://localhost:9999/v1", "fake", budget)
    return path


def test_approval_survives_new_graph_and_applies_once(tmp_path):
    run = make_run(tmp_path)
    result = invoke_run(run, client=FakeClient([PATCH]))
    assert result["status"] == "awaiting_approval"
    assert "if values" not in (run / "workspace/solution.py").read_text()
    # New graph + fresh SQLite connection emulates a restarted process.
    result = invoke_run(run, decision="approve", client=FakeClient([FINISH]))
    assert result["status"] == "finished"
    before = (run / "workspace/solution.py").read_text()
    invoke_run(run, client=FakeClient([]))
    assert (run / "workspace/solution.py").read_text() == before


@pytest.mark.parametrize("decision", ["reject", "feedback"])
def test_rejection_never_writes(tmp_path, decision):
    run = make_run(tmp_path)
    invoke_run(run, client=FakeClient([PATCH]))
    original = (run / "workspace/solution.py").read_bytes()
    invoke_run(run, decision=decision, feedback="inspect first", client=FakeClient([FINISH]))
    assert (run / "workspace/solution.py").read_bytes() == original


def test_budget_is_durable(tmp_path):
    run = make_run(tmp_path, budget=1)
    client = FakeClient([{"tool": "list_files", "args": {}}])
    result = invoke_run(run, client=client)
    assert result["status"] == "budget_exhausted"
    assert client.calls == 1
    assert read_json(run / "budget.json")["calls"] == 1


def test_crash_during_request_respects_reserved_budget(tmp_path):
    class Crash:
        def chat(self, *a, **k):
            raise KeyboardInterrupt
    run = make_run(tmp_path, budget=1)
    with pytest.raises(KeyboardInterrupt):
        invoke_run(run, client=Crash())
    result = invoke_run(run, client=FakeClient([]))
    assert result["status"] == "budget_exhausted"


def test_approval_detects_external_file_change(tmp_path):
    run = make_run(tmp_path)
    invoke_run(run, client=FakeClient([PATCH]))
    (run / "workspace/solution.py").write_text("# externally modified\n")
    invoke_run(run, decision="approve", client=FakeClient([FINISH]))
    assert (run / "workspace/solution.py").read_text() == "# externally modified\n"
    assert not read_json(run / "tools/step-001.json")["ok"]


@pytest.mark.parametrize("path", ["../escape.py", "/tmp/escape.py", "C:/escape.py"])
def test_path_escape_rejected(tmp_path, path):
    aci = ACI(tmp_path, tmp_path / "journal")
    with pytest.raises(ValueError):
        aci.read(path)


def test_replay_recovers_pending_patch_journal(tmp_path):
    run = make_run(tmp_path)
    aci = ACI(run / "workspace", run / "journal")
    prepared = aci.prepare_patch(PATCH["args"])
    assert aci.apply(prepared, "one")["status"] == "applied"
    journal = read_json(run / "journal/one.json")
    write_json(run / "journal/one.json", {**journal, "status": "pending"})
    assert aci.apply(prepared, "one")["status"] == "already_applied"
    assert read_json(run / "journal/one.json")["status"] == "done"


def test_tool_failure_returns_to_model(tmp_path):
    class TimedOutSandbox:
        def run(self, *a, **k):
            return {"returncode": -1, "timeout": True, "output": "timed out"}
    run = make_run(tmp_path)
    invoke_run(run, client=FakeClient([{"tool": "run_tests", "args": {}}, FINISH]), sandbox=TimedOutSandbox())
    assert read_json(run / "tools/step-001.json")["result"]["timeout"] is True


def test_test_files_cannot_be_patched(tmp_path):
    run = make_run(tmp_path)
    aci = ACI(run / "workspace", run / "journal")
    with pytest.raises(ValueError):
        aci.prepare_patch({"path": "tests/test_solution.py", "before": "assertEqual", "after": "assertTrue"})


def test_suite_pause_and_resume_requires_human_decisions(tmp_path, monkeypatch):
    from trainingprogram import agent
    monkeypatch.setenv("TP_DATA", str(tmp_path))
    class VerifiedClient:
        def __init__(self, *args):
            pass
        def verify(self, *args):
            pass
    monkeypatch.setattr("trainingprogram.client.ModelClient", VerifiedClient)
    decisions = []
    def invoke(directory, decision=None, feedback=""):
        if decision:
            decisions.append(decision)
            return {"status": "finished", "pending": []}
        return {"status": "awaiting_approval", "pending": [{"patch": "fake"}]}
    monkeypatch.setattr(agent, "invoke_run", invoke)
    monkeypatch.setattr("builtins.input", lambda prompt: "q")
    assert agent.run_suite("suite", "http://localhost/v1", "fake")["status"] == "paused"
    assert not decisions
    monkeypatch.setattr("builtins.input", lambda prompt: "a")
    result = agent.run_suite("suite", "http://localhost/v1", "fake")
    assert result["status"] == "finished"
    assert decisions == ["approve"] * 3
