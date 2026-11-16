from __future__ import annotations

import hashlib
from pathlib import Path
from .common import confined, read_json, write_json, sha256, safe_name, digest
from .sandbox import get_sandbox


class ACI:
    def __init__(self, workspace, journal, sandbox=None):
        self.workspace = Path(workspace).resolve()
        self.journal = Path(journal)
        self.sandbox = sandbox or get_sandbox()

    def files(self):
        paths = []
        for p in sorted(self.workspace.rglob("*")):
            if p.is_symlink():
                continue
            if p.is_file() and not any(x.startswith(".") or x == "__pycache__" for x in p.relative_to(self.workspace).parts):
                confined(self.workspace, p.relative_to(self.workspace))
                paths.append(p)
        return paths[:500]

    def read(self, path):
        target = confined(self.workspace, path)
        if target.stat().st_size > 100_000:
            raise ValueError("File exceeds 100KB tool limit")
        return target.read_text(encoding="utf-8")

    def prepare_patch(self, args):
        if set(args) != {"path", "before", "after"}:
            raise ValueError("apply_patch requires exactly path, before and after")
        if not all(isinstance(x, str) for x in args.values()):
            raise ValueError("Patch arguments must be strings")
        path = confined(self.workspace, args["path"])
        if path.name.startswith("test_") or "tests" in path.relative_to(self.workspace).parts:
            raise ValueError("Editing acceptance tests is forbidden")
        text = self.read(args["path"])
        if not args["before"] or args["before"] == args["after"] or text.count(args["before"]) != 1:
            raise ValueError("before must match exactly once and after must differ")
        result = text.replace(args["before"], args["after"], 1)
        if len(result.encode()) > 100_000:
            raise ValueError("Patched file exceeds 100KB")
        return {**args, "expected_sha256": sha256(path), "result_sha256": hashlib.sha256(result.encode()).hexdigest(),
                "result_text": result}

    def apply(self, prepared, operation_id):
        safe_name(operation_id)
        journal = self.journal / (operation_id + ".json")
        target = confined(self.workspace, prepared["path"])
        key = digest(prepared)
        current = sha256(target)
        if journal.exists():
            record = read_json(journal)
            if record["key"] != key:
                raise ValueError("Operation ID reused for a different patch")
            if current == prepared["result_sha256"]:
                write_json(journal, {**record, "status": "done"})
                return {"status": "already_applied", "path": prepared["path"]}
        if current != prepared["expected_sha256"]:
            raise ValueError("File changed since approval was requested; refusing stale patch")
        write_json(journal, {"key": key, "status": "pending", "path": prepared["path"]})
        # Durable intent before atomic replacement makes replay safe after a crash.
        import os
        tmp = target.with_name(target.name + ".tp-tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            f.write(prepared["result_text"])
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        write_json(journal, {"key": key, "status": "done", "path": prepared["path"]})
        return {"status": "applied", "path": prepared["path"]}

    def call(self, action, operation_id):
        name, args = action["tool"], action.get("args", {})
        if name == "list_files":
            return [str(p.relative_to(self.workspace)) for p in self.files()]
        if name == "read_file":
            return self.read(args["path"])
        if name == "search_code":
            query = args["query"]
            if not isinstance(query, str) or not query:
                raise ValueError("query must be a nonempty literal string")
            results = []
            for p in self.files():
                try:
                    lines = self.read(p.relative_to(self.workspace)).splitlines()
                except (ValueError, UnicodeError):
                    continue
                for i, line in enumerate(lines, 1):
                    if query in line:
                        results.append({"path": str(p.relative_to(self.workspace)), "line": i, "text": line[:500]})
            return results[:100]
        if name == "apply_patch":
            return self.apply(action["prepared"], operation_id)
        if name == "run_tests":
            if args:
                raise ValueError("run_tests accepts no arguments; command is fixed")
            return self.sandbox.run(self.workspace, ["python", "-m", "unittest", "discover", "-s", "tests", "-v"], timeout=60)
        raise ValueError(f"Unknown tool: {name}")
