"""Capture the exact small source files used by an experiment, never model weights."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
destination = Path(os.environ['TP_DATA']) / 'diagnostics'
files = {}
for pattern in ('src/**/*.py', 'tasks/**/*.py', 'tasks/**/*.json', 'configs/*.json',
                'requirements/*', 'docker/*', 'scripts/*.sh', 'scripts/*.py', 'pyproject.toml'):
    for path in sorted(root.glob(pattern)):
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024**2:
            continue
        raw = path.read_bytes()
        files[path.relative_to(root).as_posix()] = {
            'sha256': hashlib.sha256(raw).hexdigest(), 'content': raw.decode('utf-8')}
identity = hashlib.sha256(json.dumps({k:v['sha256'] for k,v in files.items()},sort_keys=True).encode()).hexdigest()
git = subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True)
record = {'snapshot_sha256':identity,'git_head':git.stdout.strip(),
          'note':'File hashes and content are authoritative even if working tree differs from git HEAD.',
          'files':files}
destination.mkdir(parents=True,exist_ok=True)
target=destination / ('source-snapshot-'+identity[:16]+'.json')
if not target.exists():
    target.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(target)
