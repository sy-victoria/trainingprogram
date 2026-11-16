"""Opt-in Linux isolation: minimal mounts + user systemd cgroup limits."""
import json
import os
from pathlib import Path
import subprocess
import threading
import uuid

from .common import ROOT, digest, sha256
from .sandbox import SandboxUnavailable


class BubblewrapSandbox:
    def __init__(self):
        self.python = Path(os.environ.get('TP_BWRAP_PYTHON', '/nonexistent')).absolute()
        self.image_id = None
        self.prefixes = []

    def probe(self):
        if not self.python.is_file():
            raise SandboxUnavailable('Set TP_BWRAP_PYTHON to the isolated judge environment Python')
        try:
            result = subprocess.run([str(self.python), '-I', '-c',
                'import sys,json,importlib.metadata as m; print(json.dumps(dict(prefixes=[sys.prefix,sys.base_prefix], packages={p.metadata["Name"]:p.version for p in m.distributions()})))'],
                capture_output=True, text=True, timeout=30, check=True)
            runtime = json.loads(result.stdout)
            self.prefixes = sorted(set(runtime['prefixes']))
            for prefix in self.prefixes:
                if Path(prefix) in (Path('/'), Path('/home'), Path('/backup')):
                    raise SandboxUnavailable('Runtime prefix is too broad')
            self.image_id = 'bwrap-systemd-v1:' + digest({
                'runtime': runtime, 'python_hash': sha256(self.python),
                'worker_hash': sha256(ROOT / 'docker/judge_worker.py'),
                'sandbox_hash': sha256(Path(__file__)),
            })
            check = self.run(None, ['python', '-I', '-c', "print('sandbox-ok')"], timeout=20, probe=False)
            if check['returncode'] != 0 or check['output'].strip() != 'sandbox-ok':
                raise SandboxUnavailable(check['output'])
            return self.image_id
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise SandboxUnavailable(str(exc)) from exc

    def command(self, unit, workspace, command, timeout):
        if not command or command[0] != 'python':
            raise ValueError('Only Python workloads are supported')
        args = ['systemd-run', '--user', '--quiet', '--wait', '--pipe', '--collect', '--unit', unit,
                '-p', 'MemoryMax=2G', '-p', 'MemorySwapMax=0', '-p', 'TasksMax=128',
                '-p', 'CPUAffinity=' + ' '.join(map(str, sorted(os.sched_getaffinity(0))[:2])),
                '-p', 'SystemCallFilter=~sched_setaffinity', '-p', 'NoNewPrivileges=yes',
                '-p', 'SystemCallErrorNumber=EPERM',
                '-p', 'KillMode=control-group',
                '-p', f'RuntimeMaxSec={timeout + 2}', '-p', 'TimeoutStopSec=2',
                '-p', 'LimitFSIZE=10485760', '-p', 'LimitNOFILE=256',
                'bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
                '--clearenv', '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib']
        if Path('/lib64').exists():
            args += ['--ro-bind', '/lib64', '/lib64']
        for prefix in self.prefixes:
            args += ['--ro-bind', prefix, prefix]
        args += ['--dir', '/proc', '--ro-bind', '/proc/meminfo', '/proc/meminfo',
                 '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/work',
                 '--ro-bind', str(ROOT / 'docker/judge_worker.py'), '/opt/judge_worker.py',
                 '--setenv', 'PATH', str(self.python.parent) + ':/usr/bin',
                 '--setenv', 'HOME', '/tmp', '--setenv', 'TMPDIR', '/tmp',
                 '--setenv', 'PYTHONDONTWRITEBYTECODE', '1',
                 '--setenv', 'OMP_NUM_THREADS', '1', '--setenv', 'OPENBLAS_NUM_THREADS', '1']
        if workspace is not None:
            workspace = Path(workspace).resolve()
            if any(p.is_symlink() for p in workspace.rglob('*')):
                raise ValueError('Sandbox workspace must not contain symlinks')
            args += ['--ro-bind', str(workspace), '/work']
        return args + ['--chdir', '/work', '--', str(self.python), *command[1:]]

    def run(self, workspace, command, timeout=60, stdin=None, probe=True):
        if probe and self.image_id is None:
            self.probe()
        unit = 'tp-sandbox-' + uuid.uuid4().hex
        p = subprocess.Popen(self.command(unit, workspace, command, timeout),
                             stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = bytearray()
        def drain():
            while chunk := p.stdout.read(4096):
                output.extend(chunk)
                if len(output) > 65536:
                    del output[:-65536]
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        def feed():
            try:
                p.stdin.write(stdin.encode())
                p.stdin.close()
            except (OSError, BrokenPipeError):
                pass
        if stdin is not None:
            threading.Thread(target=feed, daemon=True).start()
        timed_out = False
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            try:
                subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, timeout=15)
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait(timeout=10)
        reader.join(timeout=5)
        return dict(returncode=p.returncode, timeout=timed_out,
                    output=output.decode(errors='replace'), image_id=self.image_id)
