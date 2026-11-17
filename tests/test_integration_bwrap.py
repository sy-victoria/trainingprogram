"""Opt-in real Linux user-namespace and cgroup validation; never host execution."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

import pytest
from trainingprogram.bubblewrap import BubblewrapSandbox

pytestmark = pytest.mark.skipif(os.environ.get('TP_BWRAP_TESTS') != '1', reason='Requires Linux bwrap and user systemd')


def test_filesystem_network_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('TP_TEST_SECRET', 'must-not-cross-boundary')
    (tmp_path / 'input.txt').write_text('trusted fixture')
    script = '''import os,socket
from pathlib import Path
assert 'TP_TEST_SECRET' not in os.environ
assert not Path('/home/sunyue').exists()
assert not Path('/backup/trainingprogram-data').exists()
assert not Path('/proc/1').exists()
assert len(os.sched_getaffinity(0)) <= 2
try: os.sched_setaffinity(0, set(range(1024)))
except OSError: pass
else: raise AssertionError('CPU affinity can be widened')
assert Path('/work/input.txt').read_text() == 'trusted fixture'
for name in ('/work/input.txt','/usr/should-not-exist'):
    try: Path(name).write_text('changed')
    except OSError: pass
    else: raise AssertionError('read-only mount writable')
s=socket.socket(); s.settimeout(1)
assert s.connect_ex(('1.1.1.1',443)) != 0
print('isolated')
'''
    result = BubblewrapSandbox().run(tmp_path, ['python','-I','-c',script])
    assert result['returncode'] == 0, result
    assert result['output'].strip() == 'isolated'
    assert (tmp_path / 'input.txt').read_text() == 'trusted fixture'


def test_aggregate_resource_limits(monkeypatch):
    sandbox = BubblewrapSandbox()
    sandbox.probe()
    original = sandbox.command
    units = []
    def command(unit, *args):
        units.append(unit)
        return original(unit, *args)
    monkeypatch.setattr(sandbox, 'command', command)
    with concurrent.futures.ThreadPoolExecutor(1) as executor:
        future = executor.submit(sandbox.run, None, ['python','-I','-c','import time; time.sleep(5)'], 15)
        deadline = time.monotonic()+5
        group = ''
        while time.monotonic()<deadline:
            if units:
                result = subprocess.run(['systemctl','--user','show',units[0],'-p','ControlGroup','--value'], capture_output=True,text=True)
                group = result.stdout.strip()
                if group:
                    break
            time.sleep(.1)
        assert group.startswith('/user.slice/')
        root=Path('/sys/fs/cgroup')/group.lstrip('/')
        assert (root/'memory.max').read_text().strip() == str(2*1024**3)
        assert (root/'memory.swap.max').read_text().strip() == '0'
        assert (root/'pids.max').read_text().strip() == '128'
        assert future.result()['returncode'] == 0


def test_wall_timeout_with_unread_stdin():
    result=BubblewrapSandbox().run(None,['python','-I','-c','import time; time.sleep(60)'],timeout=2,stdin='x'*2000000)
    assert result['timeout']


def test_official_judge_correct_and_wrong():
    problem=dict(task_id='HumanEval/0',prompt='def f(x):\n',canonical_solution='    return x+1\n',
                 entry_point='f',base_input=[[1],[2]],plus_input=[[-2],[0]],atol=0)
    sandbox=BubblewrapSandbox()
    for code,expected in [('def f(x):\n    return x+1\n',True),('def f(x):\n    return x\n',False)]:
        result=sandbox.run(None,['python','-I','/opt/judge_worker.py'],timeout=30,
                           stdin=json.dumps(dict(problem=problem,code=code)))
        assert result['returncode'] == 0, result
        assert json.loads(result['output'])['passed'] is expected


def test_mbpp_official_input_deserialization():
    # MBPP/2 serializes tuple arguments as JSON lists; failing to deserialize
    # would silently judge against different input semantics.
    problem=dict(task_id='Mbpp/2', prompt='def tuple_input(x):\n',
                 canonical_solution='    return isinstance(x, tuple)\n',
                 entry_point='tuple_input', base_input=[[[1,2]]], plus_input=[[[3]]], atol=0)
    sandbox=BubblewrapSandbox()
    for code,expected in [('def tuple_input(x):\n    return isinstance(x, tuple)\n', True),
                          ('def tuple_input(x):\n    return False\n', False)]:
        result=sandbox.run(None,['python','-I','/opt/judge_worker.py'],timeout=30,
                           stdin=json.dumps(dict(dataset='mbpp',problem=problem,code=code)))
        assert result['returncode']==0, result
        assert json.loads(result['output'])['passed'] is expected
