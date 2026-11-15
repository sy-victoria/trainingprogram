import os
import shutil
import subprocess
from pathlib import Path

import pytest
from trainingprogram.common import ROOT


@pytest.mark.skipif(os.name != "posix", reason="Linux bootstrap integration with fake conda")
def test_agent_only_bootstrap_never_queries_gpu_or_installs_cuda(tmp_path):
    conda = tmp_path / "fake-conda"
    conda.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TP_TEST_COMMANDS"\nexit 0\n')
    conda.chmod(0o755)
    nvidia = tmp_path / "nvidia-smi"
    nvidia.write_text('#!/bin/sh\necho called > "$TP_TEST_GPU"\nexit 1\n')
    nvidia.chmod(0o755)
    env = {**os.environ, "CONDA_EXE": str(conda), "TP_DATA": str(tmp_path / "data"),
           "TP_TEST_COMMANDS": str(tmp_path / "commands.txt"), "TP_TEST_GPU": str(tmp_path / "gpu.txt"),
           "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    result = subprocess.run(["bash", str(ROOT / "scripts/bootstrap.sh"), "--agent-only"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Agent environment ready" in result.stdout
    assert not (tmp_path / "gpu.txt").exists()
    commands = (tmp_path / "commands.txt").read_text()
    assert "tp-tools" not in commands
    assert "cuda-toolkit" not in commands
    assert "--index-url https://pypi.org/simple" in commands
