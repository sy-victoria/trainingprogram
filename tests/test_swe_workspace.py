import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trainingprogram.swe_tasks import load_verified, public_issue
from trainingprogram.swe_workspace import DockerWorkspace, capture


def test_output_is_drained_but_bounded():
    result = capture([sys.executable, '-c', 'print("x"*200000)'], limit=100)
    assert result['returncode'] == 0
    assert result['truncated'] and result['output_bytes'] >= 200000
    assert len(result['output']) == 100


@pytest.mark.skipif(os.environ.get('TP_SWE_WORKSPACE_TESTS') != '1', reason='Requires pinned SWE image and trusted local dataset')
def test_real_repository_patch_replay_restart_timeout_and_export(tmp_path):
    root = Path(os.environ['TP_SWE_DATA'])
    instances = load_verified(root/'verified-v5.json', root/'verified-v5.manifest.json')
    image = 'swebench/sweb.eval.x86_64.django_1776_django-10914@sha256:f26f0db931c6c1e785ab8287e1ea4f5f948846267db159deaf90b3fe7adea4e9'
    issue = public_issue(instances['django__django-10914'], image)
    workspace = DockerWorkspace(tmp_path/'task', issue, memory_gib=2)
    try:
        workspace.start()
        setup = workspace.call('shell', {'command': "printf 'before\\n' > agent_fixture.txt"}, 'setup')
        assert setup['returncode'] == 0, setup
        read = workspace.call('read_file', {'path': 'agent_fixture.txt'}, 'read')
        assert json.loads(read['output']) == 'before\n'
        args = dict(path='agent_fixture.txt', before='before', after='after',
                    expected_sha256=hashlib.sha256(b'before\n').hexdigest())
        first = workspace.call('apply_patch', args, 'patch')
        assert first['returncode'] == 0, first
        assert workspace.call('apply_patch', args, 'patch') == first
        with pytest.raises(ValueError, match='reused'):
            workspace.call('apply_patch', {**args, 'after': 'different'}, 'patch')
        escaped = workspace.call('read_file', {'path': '../../etc/passwd'}, 'escape')
        assert escaped['returncode'] != 0
        workspace.stop()
        restored = DockerWorkspace(tmp_path/'task', issue, memory_gib=2)
        restored.start()
        read = restored.call('read_file', {'path': 'agent_fixture.txt'}, 'read-after-resume')
        assert json.loads(read['output']) == 'after\n'
        assert '+after' in restored.patch()
        timed = restored.call('shell', {'command': 'sleep 10'}, 'timeout', timeout=1)
        assert timed['timeout'] and timed['workspace_stopped']
        assert not restored._inspect()['State']['Running']
        restored.start()
        assert '+after' in restored.patch()
        # Simulate an out-of-band edit; a new controller must refuse recovery.
        subprocess.run(['docker','exec',workspace.name,'bash','-lc',"printf 'tampered\\n' > agent_fixture.txt"], check=True)
        restored.stop()
        with pytest.raises(RuntimeError, match='outside'):
            DockerWorkspace(tmp_path/'task', issue, memory_gib=2).start()
    finally:
        if workspace._inspect() is not None:
            workspace.stop()
            subprocess.run(['docker','rm',workspace.name], check=True, capture_output=True)
