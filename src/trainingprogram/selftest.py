"""No GPU, model download, network or generated-code execution needed."""
import json
import uuid

from .agent import create_run, invoke_run
from .common import data_root, write_json
from .evaluation import ensure_config, generate_samples, GENERATION


class ScriptedClient:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        answer = next(self.answers)
        return {"text": json.dumps(answer) if isinstance(answer, dict) else answer,
                "finish_reason": "stop", "usage": {}, "simulated": True}


class SimulatedSandbox:
    def run(self, *args, **kwargs):
        return {"simulated": True, "returncode": None, "timeout": False,
                "output": "OFFLINE SELFTEST: no tests or generated code were executed"}


def run_selftest():
    directory = data_root() / "offline" / ("selftest-" + uuid.uuid4().hex[:12])
    agent_run = directory / "agent"
    create_run(agent_run, "boundary", "offline://scripted", "SIMULATED-NOT-A-MODEL", budget=6)
    patch = {"tool": "apply_patch", "args": {"path": "solution.py", "before": "return sum(values) / len(values)",
              "after": "return sum(values) / len(values) if values else None"}}
    initial = (agent_run / "workspace/solution.py").read_bytes()
    result = invoke_run(agent_run, client=ScriptedClient([{"tool": "read_file", "args": {"path": "solution.py"}}, patch]))
    if result["status"] != "awaiting_approval" or (agent_run / "workspace/solution.py").read_bytes() != initial:
        raise AssertionError("Approval pause failed")
    # New graph / SQLite connection, simulating an application restart.
    result = invoke_run(agent_run, decision="approve", client=ScriptedClient([
        {"tool": "run_tests", "args": {}}, {"tool": "finish", "args": {"summary": "Offline mechanism demonstration only"}}
    ]), sandbox=SimulatedSandbox())
    if result["status"] != "finished" or b"if values else None" not in (agent_run / "workspace/solution.py").read_bytes():
        raise AssertionError("Checkpoint resume failed")
    evaluation = directory / "evaluation"
    config = {"task_ids": ["HumanEval/0"], "generation": GENERATION, "simulated": True}
    ensure_config(evaluation, config)
    client = ScriptedClient(["```python\ndef identity(x):\n    return x\n```"])
    generate_samples(evaluation, config, {"HumanEval/0": {"prompt": "Implement identity", "entry_point": "identity"}}, client)
    generate_samples(evaluation, config, {}, client)
    if client.calls != 1:
        raise AssertionError("Completed sample was regenerated")
    report = {"kind": "offline_mechanism_test", "simulated": True, "directory": str(directory),
              "checks": {"approval_pause": True, "checkpoint_resume": True, "patch_write": True,
                         "tool_feedback": True, "sample_resume": True},
              "real_model_called": False, "generated_code_executed": False,
              "note": "Excluded from model reports and normal result exports"}
    write_json(directory / "SELFTEST.json", report)
    return report
