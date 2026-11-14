from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .common import data_root, write_json, environment, safe_name


def main():
    parser = argparse.ArgumentParser(description="Coding agent and Qwen quantization coursework")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    model = sub.add_parser("model").add_subparsers(dest="operation", required=True)
    for operation in ("prepare", "serve", "cleanup"):
        p = model.add_parser(operation)
        p.add_argument("--size", choices=["0.8B", "2B", "4B"], required=True)
        if operation == "serve":
            p.add_argument("--variant", choices=["BF16", "Q4_K_M"], required=True)
            p.add_argument("--port", type=int, default=8080)
        if operation == "cleanup":
            p.add_argument("--target", choices=["source", "BF16", "Q4_K_M"], required=True)
    evaluation = sub.add_parser("eval").add_subparsers(dest="operation", required=True)
    for operation in ("generate", "judge", "speed"):
        p = evaluation.add_parser(operation)
        p.add_argument("--run", required=True)
        if operation in {"generate", "speed"}:
            p.add_argument("--port", type=int, default=8080)
        if operation == "generate":
            p.add_argument("--size", choices=["0.8B", "2B", "4B"], required=True)
            p.add_argument("--variant", choices=["BF16", "Q4_K_M"], required=True)
            p.add_argument("--mode", choices=["smoke", "subset", "full"], default="smoke")
    agent = sub.add_parser("agent").add_subparsers(dest="operation", required=True)
    for operation in ("run", "resume", "status", "verify"):
        p = agent.add_parser(operation)
        p.add_argument("--run", required=True)
        if operation == "run":
            p.add_argument("--task", choices=["logic", "boundary", "feature"], required=True)
            p.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
            p.add_argument("--model", required=True, help="Alias printed by tp model serve")
            p.add_argument("--budget", type=int, default=10)
        if operation == "resume":
            p.add_argument("--decision", choices=["approve", "reject", "feedback"])
            p.add_argument("--feedback", default="")
    sub.add_parser("summary")
    export = sub.add_parser("export")
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = environment()
            root = data_root()
            existing = root
            while not existing.exists():
                existing = existing.parent
            result["disk"] = {"target": str(root), "free_bytes": shutil.disk_usage(existing).free}
            from .sandbox import DockerSandbox
            try:
                result["sandbox"] = DockerSandbox().probe()
            except Exception as e:
                result["sandbox"] = {"available": False, "reason": str(e)}
            write_json(root / "diagnostics" / "environment.json", result)
        elif args.command == "model":
            from . import models
            if args.operation == "prepare":
                result = models.prepare(args.size)
            elif args.operation == "serve":
                result = models.serve(args.size, args.variant, args.port)
            else:
                result = models.cleanup(args.size, args.target)
        elif args.command == "eval":
            from . import evaluation
            if args.operation == "generate":
                result = evaluation.generate(args.run, args.size, args.variant, args.mode, args.port)
            elif args.operation == "judge":
                result = evaluation.judge(args.run)
            else:
                result = evaluation.speed(args.run, args.port)
        elif args.command == "agent":
            from . import agent
            directory = data_root() / "agents" / safe_name(args.run)
            if args.operation == "run":
                if not 1 <= args.budget <= 100:
                    raise ValueError("budget must be between 1 and 100")
                agent.create_run(directory, args.task, args.base_url, args.model, args.budget)
            if args.operation == "verify":
                from .tasks import verify_agent
                result = verify_agent(directory)
            else:
                result = agent.invoke_run(directory, getattr(args, "decision", None), getattr(args, "feedback", ""),
                                          status_only=args.operation == "status")
        elif args.command == "summary":
            from .reporting import summarize
            result = summarize()
        else:
            from .reporting import export_bundle
            result = export_bundle(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print("Interrupted. Persisted samples/checkpoints are retained.")
        raise SystemExit(130)
    except Exception as e:
        parser.exit(1, f"{type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
