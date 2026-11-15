"""Docker-only execution. No unsafe subprocess fallback for generated programs."""
from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

IMAGE = "trainingprogram-eval:0.1"


class SandboxUnavailable(RuntimeError):
    pass


class DockerSandbox:
    def __init__(self, image=IMAGE):
        self.image = image
        self.image_id = None

    def probe(self):
        try:
            p = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", self.image],
                               capture_output=True, text=True, timeout=20)
            if p.returncode:
                raise SandboxUnavailable("Docker/image unavailable. See docs/SANDBOX.md; generation can continue separately.")
            self.image_id = p.stdout.strip()
            result = self.run(None, ["python", "-I", "-c", "print('sandbox-ok')"], timeout=20, probe=False)
            if result["returncode"] != 0:
                raise SandboxUnavailable(result["output"])
            return self.image_id
        except (OSError, subprocess.TimeoutExpired) as e:
            raise SandboxUnavailable("Docker is unavailable; no host execution fallback") from e

    def run(self, workspace, command, timeout=60, stdin=None, probe=True):
        if probe and self.image_id is None:
            self.probe()
        name = "tp-" + uuid.uuid4().hex
        args = ["docker", "run", "--rm", "--name", name, "--network=none", "--read-only",
                "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=128",
                "--memory=2g", "--memory-swap=2g", "--cpus=2", "--user=65534:65534",
                "--ulimit", "nofile=256:256", "--ulimit", "fsize=10485760:10485760",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
                "-e", "HOME=/tmp", "-e", "XDG_CACHE_HOME=/tmp/cache",
                "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "OMP_NUM_THREADS=1"]
        if workspace is not None:
            workspace = Path(workspace).resolve()
            if any(p.is_symlink() for p in workspace.rglob("*")):
                raise ValueError("Sandbox workspace must not contain symlinks")
            args += ["--mount", f"type=bind,src={workspace},dst=/work,readonly", "-w", "/work"]
        if stdin is not None:
            args += ["-i"]
        args += [self.image_id or self.image, *command]
        # stdout goes to a capped stream reader instead of unbounded PIPE buffering.
        import threading
        p = subprocess.Popen(args, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        chunks = bytearray()

        def drain():
            while True:
                chunk = p.stdout.read(4096)
                if not chunk:
                    break
                chunks.extend(chunk)
                if len(chunks) > 65536:
                    del chunks[:-65536]

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        writer = None
        if stdin is not None:
            def feed():
                try:
                    p.stdin.write(stdin.encode())
                    p.stdin.close()
                except (BrokenPipeError, OSError):
                    pass  # A failed container may exit without reading input.
            writer = threading.Thread(target=feed, daemon=True)
            writer.start()
        try:
            p.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also remove a container whose docker CLI exited early; only our UUID is targeted.
            try:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                pass
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait(timeout=10)
        thread.join(timeout=5)
        if writer:
            writer.join(timeout=5)
        return {"returncode": p.returncode, "timeout": timed_out,
                "output": chunks.decode("utf-8", errors="replace"), "image_id": self.image_id}
