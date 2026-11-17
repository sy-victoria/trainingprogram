"""Persistent per-issue containers; no host mounts or evaluation data in tasks."""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from pathlib import Path

from .common import digest, read_json, write_json, safe_name, exclusive
from .evaluation import ensure_config


def capture(argv, timeout=60, limit=65536):
    """Drain output continuously while retaining only a bounded prefix."""
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    retained = bytearray()
    total = 0

    def drain():
        nonlocal total
        while chunk := process.stdout.read(8192):
            total += len(chunk)
            retained.extend(chunk[:max(0, limit - len(retained))])

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        process.wait(timeout=10)
    reader.join(timeout=10)
    if reader.is_alive():
        raise RuntimeError("Output stream did not close")
    return {"returncode": process.returncode, "timeout": timed_out,
            "output": retained.decode("utf-8", errors="replace"),
            "output_bytes": total, "truncated": total > limit}


FILE_TOOL = r'''
import hashlib,json,os,pathlib,sys
root=pathlib.Path('/testbed').resolve()
tool,args=json.loads(sys.argv[1])
def path(name):
    p=root/name
    if p.is_symlink() or not p.resolve().is_relative_to(root) or '.git' in p.relative_to(root).parts:
        raise ValueError('Path leaves repository or accesses git internals')
    return p.resolve()
def read(p):
    if p.stat().st_size>200000: raise ValueError('File too large; use shell to inspect a bounded range')
    return p.read_text()
if tool=='read_file':
    result=read(path(args['path']))
elif tool=='apply_patch':
    p=path(args['path']); before=read(p)
    if hashlib.sha256(before.encode()).hexdigest()!=args['expected_sha256']:
        raise ValueError('File changed since patch preparation')
    if not args['before'] or before.count(args['before'])!=1 or args['before']==args['after']:
        raise ValueError('Patch must have one unique match and change the file')
    after=before.replace(args['before'],args['after'],1)
    if len(after.encode())>200000: raise ValueError('Patched file too large')
    temp=p.with_name(p.name+'.tp-edit')
    if temp.exists() or temp.is_symlink(): raise ValueError('Temporary path already exists')
    with temp.open('x') as f:
        f.write(after); f.flush(); os.fsync(f.fileno())
    os.replace(temp,p)
    result={'path':args['path'],'sha256':hashlib.sha256(after.encode()).hexdigest()}
else: raise ValueError('Unknown file tool')
print(json.dumps(result))
'''

VERIFY_TREE = r'''
import hashlib,json,subprocess,sys
def entries(commit):
    raw=subprocess.check_output(['git','-C','/testbed','ls-tree','-rz',commit])
    result=[]
    for entry in raw.split(b'\0'):
        if not entry: continue
        meta,path=entry.split(b'\t',1)
        mode,kind,blob=meta.split()
        # Official images may add a setup commit changing regular executable bits.
        if mode in (b'100644',b'100755'): mode=b'100644'
        result.append((mode,kind,blob,path))
    return result
base,head=sys.argv[1:]
a,b=entries(base),entries(head)
if a!=b: raise ValueError('Image source content differs from dataset base commit')
print(hashlib.sha256(repr(a).encode()).hexdigest())
'''


class DockerWorkspace:
    def __init__(self, directory, issue, memory_gib=8, cpus=2):
        self.directory = Path(directory).resolve()
        self.issue = issue
        self.identity = digest({"directory": str(self.directory), "issue": issue})
        self.name = "tp-swe-" + self.identity[:20]
        self.config = {"issue": issue, "memory_gib": memory_gib, "cpus": cpus,
                       "network": "none", "pids": 512, "schema": 1}
        if "@sha256:" not in issue["image"]:
            raise ValueError("Pinned image required")
        if not 1 <= memory_gib <= 12 or not 0 < cpus <= 4:
            raise ValueError("Invalid local worker resource limits")

    def _docker(self, *args, timeout=60, limit=65536):
        return capture(["docker", *args], timeout, limit)

    def _checked(self, *args, timeout=60):
        result = self._docker(*args, timeout=timeout)
        if result["returncode"] or result["timeout"]:
            raise RuntimeError(result["output"])
        return result["output"].strip()

    def _inspect(self):
        result = self._docker("container", "inspect", self.name)
        if result["returncode"]:
            # A daemon failure is not evidence of a missing container.
            if "No such" in result["output"]:
                return None
            raise RuntimeError(result["output"])
        item = json.loads(result["output"])[0]
        if item["Config"].get("Labels", {}).get("trainingprogram.owner") != self.identity:
            raise RuntimeError("Container ownership mismatch")
        if item["Config"]["Image"] != self.issue["image"] or item.get("Mounts"):
            raise RuntimeError("Unexpected image or host mounts")
        host = item["HostConfig"]
        if host["NetworkMode"] != "none" or host.get("Privileged"):
            raise RuntimeError("Container isolation changed")
        return item

    def start(self):
        with exclusive(self.directory / ".workspace.lock"):
            ensure_config(self.directory, self.config)
            item = self._inspect()
            if item is None:
                # Do not recreate a disappeared workspace and pretend checkpoint recovery.
                if (self.directory / "created.json").exists():
                    raise RuntimeError("Task container missing; cannot restore workspace from conversation alone")
                self._checked("create", "--name", self.name, "--label", "trainingprogram.owner=" + self.identity,
                    "--network", "none", "--memory", f"{self.config['memory_gib']}g", "--memory-swap", f"{self.config['memory_gib']}g",
                    "--cpus", str(self.config["cpus"]), "--pids-limit", "512", "--cap-drop", "ALL",
                    "--security-opt", "no-new-privileges", "--workdir", "/testbed", "--entrypoint", "/bin/bash",
                    self.issue["image"], "-lc", "sleep infinity")
                write_json(self.directory / "created.json", {"container": self.name})
            self._checked("start", self.name)
            head = self._checked("exec", self.name, "git", "-C", "/testbed", "rev-parse", "HEAD")
            baseline_path = self.directory / "baseline.json"
            if baseline_path.exists():
                matches = head == read_json(baseline_path)["image_head"]
            else:
                try:
                    content_hash = self._checked("exec", self.name, "/opt/miniconda3/bin/python", "-c",
                                                 VERIFY_TREE, self.issue["base_commit"], head)
                    write_json(baseline_path, {"image_head": head, "dataset_base": self.issue["base_commit"],
                                              "source_content_hash": content_hash})
                    matches = True
                except RuntimeError:
                    matches = False
            if not matches:
                self._checked("stop", "--time", "1", self.name)
                raise RuntimeError("Repository source or checkpoint HEAD differs from verified baseline")
        return self.name

    def stop(self):
        if self._inspect() is not None:
            self._checked("stop", "--time", "1", self.name)

    def call(self, tool, args, operation_id, timeout=60):
        safe_name(operation_id)
        if tool not in {"read_file", "apply_patch", "shell"}:
            raise ValueError("Unknown workspace tool")
        if not 0 < timeout <= 300:
            raise ValueError("Tool timeout out of range")
        key = digest({"tool": tool, "args": args, "timeout": timeout})
        journal = self.directory / "operations" / (operation_id + ".json")
        with exclusive(self.directory / ".workspace.lock"):
            if journal.exists():
                previous = read_json(journal)
                if previous["key"] != key:
                    raise ValueError("Operation ID reused with different arguments")
                if previous["status"] != "finished":
                    raise RuntimeError("Interrupted operation has uncertain effects; do not silently replay")
                return previous["result"]
            item = self._inspect()
            if item is None or not item["State"]["Running"]:
                raise RuntimeError("Workspace not running; explicit start required")
            if tool == "shell":
                if set(args) != {"command"} or not isinstance(args["command"], str) or len(args["command"]) > 16000:
                    raise ValueError("Invalid shell request")
                argv = ["bash", "-lc", args["command"]]
            else:
                argv = ["/opt/miniconda3/bin/python", "-c", FILE_TOOL, json.dumps([tool, args])]
            write_json(journal, {"key": key, "status": "started", "tool": tool, "args": args})
            result = self._docker("exec", "--workdir", "/testbed", self.name, *argv, timeout=timeout)
            if result["timeout"]:
                # Killing docker exec's client alone does not kill its process in the container.
                self._checked("stop", "--time", "1", self.name)
                result["workspace_stopped"] = True
            write_json(journal, {"key": key, "status": "finished", "tool": tool, "args": args, "result": result})
            return result

    def patch(self):
        self._inspect()
        # -N includes new files without staging their content; required for SWE fixes.
        self._checked("exec", self.name, "git", "-C", "/testbed", "add", "-N", ".")
        baseline = read_json(self.directory / "baseline.json")["image_head"]
        result = self._docker("exec", self.name, "git", "-C", "/testbed", "diff", "--binary", baseline, limit=2_000_000)
        if result["returncode"] or result["timeout"] or result["truncated"]:
            raise RuntimeError("Patch export failed or exceeds 2MB")
        return result["output"]
