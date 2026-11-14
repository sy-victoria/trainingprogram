"""Opt-in real container tests, executed by the Linux CI job."""
import json
import os
from pathlib import Path

import pytest

from trainingprogram.sandbox import DockerSandbox

pytestmark = pytest.mark.skipif(os.environ.get("TP_DOCKER_TESTS") != "1", reason="Requires built Docker image")


def test_real_official_judge():
    problem = {"task_id": "HumanEval/0", "prompt": "def f(x):\n", "canonical_solution": "    return x + 1\n",
               "entry_point": "f", "base_input": [[1], [2]], "plus_input": [[-2], [0]], "atol": 0}
    sandbox = DockerSandbox()
    for code, passed in [("def f(x):\n    return x + 1\n", True), ("def f(x):\n    return x\n", False)]:
        result = sandbox.run(None, ["python", "-I", "/opt/judge_worker.py"], timeout=30,
                             stdin=json.dumps({"problem": problem, "code": code}))
        assert result["returncode"] == 0, result["output"]
        assert json.loads(result["output"])["passed"] is passed


def test_real_wall_timeout():
    result = DockerSandbox().run(None, ["python", "-I", "-c", "while True: pass"], timeout=2)
    assert result["timeout"]


def test_real_sandbox_readonly_and_network(tmp_path):
    tmp_path.chmod(0o755)
    (tmp_path / "value.txt").write_text("unchanged")
    (tmp_path / "value.txt").chmod(0o644)
    script = """
import socket
from pathlib import Path
assert Path('/work/value.txt').read_text() == 'unchanged'
try:
    Path('/work/value.txt').write_text('changed')
except OSError:
    pass
else:
    raise AssertionError('workspace writable')
s = socket.socket()
s.settimeout(1)
assert s.connect_ex(('1.1.1.1', 443)) != 0
print('ok')
"""
    result = DockerSandbox().run(tmp_path, ["python", "-I", "-c", script], timeout=10)
    assert result["returncode"] == 0, result["output"]
    assert (tmp_path / "value.txt").read_text() == "unchanged"
