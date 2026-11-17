from __future__ import annotations

import ast
import gzip
import hashlib
import json
import random
import re
import subprocess
import threading
import time
from pathlib import Path

import httpx
import psutil

from .client import ModelClient
from .common import ROOT, data_root, write_json, read_json, digest, exclusive, environment, safe_name, sha256
from .models import load_artifact, versions
from .common import gpu_device
from .sandbox import DockerSandbox, SandboxUnavailable, get_sandbox

DATA_URL = "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz"
GENERATION = {"temperature": 0, "seed": 2026, "max_tokens": 2048, "thinking": False,
              "prompt_version": "full-function-v1", "samples_per_task": 1}
PROMPT = "Implement the Python function below. Return the complete function and required imports in one Python code block. Do not include tests or explanations.\n\n"


def dataset(benchmark="humaneval"):
    specs = {"humaneval": ("HumanEvalPlus", "v0.1.10", 164),
             "mbpp": ("MbppPlus", "v0.2.0", 378)}
    if benchmark not in specs:
        raise ValueError("Unsupported benchmark")
    name, version, count = specs[benchmark]
    url = f"https://github.com/evalplus/{name.lower()}_release/releases/download/{version}/{name}.jsonl.gz"
    path = data_root() / "datasets" / f"{name}-{version}.json"
    with exclusive(path.with_suffix(".lock")):
        if not path.exists():
            with httpx.Client(timeout=180, follow_redirects=True) as client:
                r = client.get(url)
                r.raise_for_status()
            raw = gzip.decompress(r.content)
            problems = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            if len(problems) != count or len({p["task_id"] for p in problems}) != count:
                raise ValueError(f"Unexpected {name} dataset shape")
            write_json(path, {"url": url, "version": version, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                              "problems": {p["task_id"]: p for p in problems}})
        data = read_json(path)
        if len(data["problems"]) != count or data["version"] != version or data["url"] != url:
            raise ValueError("Cached dataset metadata mismatch")
    return data


def select_tasks(problems, mode):
    ids = sorted(problems)
    random.Random(2026).shuffle(ids)
    return ids[:{"smoke": 2, "subset": 30, "full": len(ids)}[mode]]


def extract_code(text, entry_point):
    # Do not execute, repair, or cherry-pick multiple candidates.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    code = blocks[0].strip() if blocks else text.strip()
    try:
        tree = ast.parse(code)
        if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry_point for n in tree.body):
            return code, "missing_entry_point"
    except (SyntaxError, ValueError):
        return code, "syntax_error"
    return code + "\n", None


class ResourceMonitor:
    """Sampled process RSS and whole-device GPU memory, not allocator peak counters."""
    def __init__(self, pid=None):
        self.pid = pid
        self.done = threading.Event()
        self.rss = None
        self.gpu = None
        self.thread = None

    def sample(self):
        if self.pid:
            try:
                rss = psutil.Process(self.pid).memory_info().rss
                self.rss = max(self.rss or 0, rss)
            except psutil.Error:
                pass
        try:
            p = subprocess.run(["nvidia-smi", "--id=" + gpu_device(), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=3)
            if p.returncode == 0:
                value = float(p.stdout.strip())
                self.gpu = max(self.gpu or 0, value)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass

    def __enter__(self):
        self.sample()
        def loop():
            while not self.done.wait(1):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join(timeout=5)

    def result(self):
        return {"peak_server_rss_bytes_sampled": self.rss, "peak_gpu_device_mib_sampled": self.gpu,
                "sampling_interval_s": 1, "gpu_scope": f"entire GPU {gpu_device()}; run without other GPU workloads"}


def ensure_config(run, config):
    path = Path(run) / "config.json"
    if path.exists():
        if read_json(path) != config:
            raise ValueError("Run configuration changed. Choose a NEW --run ID; results cannot be mixed.")
    else:
        write_json(path, config)


def task_filename(task_id):
    if not re.fullmatch(r"(?:HumanEval|Mbpp)/\d+", task_id):
        raise ValueError("Unexpected task ID")
    return task_id.replace("/", "_") + ".json"


def generate(run_id, size, variant, mode="smoke", port=8080, client=None,
             benchmark="humaneval", generation=None):
    data = dataset(benchmark)
    settings = dict(GENERATION)
    if generation is not None:
        if set(generation) - {"seed", "temperature", "top_p", "max_tokens"}:
            raise ValueError("Unsupported generation override")
        settings.update(generation)
    if not 0 <= settings["temperature"] <= 2 or not 0 < settings.get("top_p", 1) <= 1 or settings["max_tokens"] <= 0:
        raise ValueError("Invalid generation settings")
    ids = select_tasks(data["problems"], mode)
    _, artifact = load_artifact(size, variant)
    service = read_json(data_root() / "services" / f"{port}.json")
    alias = "tp-" + artifact["sha256"][:16]
    client = client or ModelClient(f"http://127.0.0.1:{port}/v1", alias)
    client.verify(alias)
    if service["alias"] != alias:
        raise ValueError("Service manifest points to another artifact")
    run = data_root() / "runs" / safe_name(run_id)
    config = {"schema": 1, "size": size, "variant": variant, "artifact_sha256": artifact["sha256"],
              "artifact_bytes": artifact["bytes"], "model_revision": artifact["revision"],
              "llama_cpp": artifact["llama_cpp"], "evalplus": versions()["evalplus"],
              "dataset_sha256": data["raw_sha256"], "task_ids": ids, "mode": mode,
              "generation": settings, "server": {k: service[k] for k in ("context", "threads", "parallel")},
              "harness_hash": digest({p.name: sha256(p) for p in sorted((ROOT / "src/trainingprogram").glob("*.py"))})}
    config["benchmark"] = benchmark
    current_environment = environment()
    config["hardware"] = {k: current_environment.get(k) for k in ("platform", "gpu", "cgroup")}
    with exclusive(run / ".lock"):
        ensure_config(run, config)
        if not (run / "environment.json").exists():
            write_json(run / "environment.json", current_environment)
            write_json(run / "artifact.json", artifact)
            write_json(run / "problems.json", {i: data["problems"][i] for i in ids})
        generate_samples(run, config, data["problems"], client, service.get("pid"))
    return str(run)


def generate_samples(run, config, problems, client, pid=None):
    run = Path(run)
    for task_id in config["task_ids"]:
        out = run / "samples" / task_filename(task_id)
        if out.exists():
            old = read_json(out)
            if old["task_id"] != task_id or old["config_hash"] != digest(config):
                raise ValueError("Sample/config mismatch")
            if old["status"] == "started":
                write_json(out, {**old, "status": "interrupted", "error": "Request interrupted; not silently resampled"})
            continue
        record = {"task_id": task_id, "config_hash": digest(config), "status": "started"}
        write_json(out, record)
        with ResourceMonitor(pid) as monitor:
            try:
                settings = config["generation"]
                sampling = {key: settings[key] for key in ("temperature", "top_p") if key in settings}
                result = client.chat([{"role": "user", "content": PROMPT + problems[task_id]["prompt"]}],
                                     max_tokens=settings["max_tokens"], seed=settings["seed"], **sampling)
                code, error = extract_code(result["text"], problems[task_id]["entry_point"])
                record.update(status="generated", response=result, code=code, extraction_error=error,
                              truncated=result["finish_reason"] == "length")
            except Exception as e:
                record.update(status="generation_error", error=type(e).__name__)
        record["resources"] = monitor.result()
        write_json(out, record)
        print(f"{task_id}: {record['status']}", flush=True)


def judge(run_id, sandbox=None):
    run = data_root() / "runs" / safe_name(run_id)
    sandbox = sandbox or get_sandbox()
    # Probe before any result mutation. Missing isolation never becomes a model failure.
    image = sandbox.probe()
    config = read_json(run / "config.json")
    problems = read_json(run / "problems.json")
    with exclusive(run / ".lock"):
        judge_config = {"image_id": image, "evalplus": config["evalplus"], "wall_timeout_s": 180,
                        "min_time_limit_s": 0.2, "gt_time_limit_factor": 4.0,
                        "generation_config_hash": digest(config)}
        ensure_config(run / "judge", judge_config)
        for task_id in config["task_ids"]:
            target = run / "judgments" / task_filename(task_id)
            sample_path = run / "samples" / task_filename(task_id)
            if not sample_path.exists():
                continue
            sample = read_json(sample_path)
            if sample.get("config_hash") != digest(config):
                raise ValueError("Sample was generated with a different configuration")
            sample_hash = digest(sample)
            if target.exists():
                existing = read_json(target)
                if existing["sample_hash"] != sample_hash:
                    raise ValueError("Sample changed after judging")
                if existing["status"] != "infrastructure_error":
                    continue
            result = {"task_id": task_id, "sample_hash": sample_hash, "passed": False}
            if sample["status"] != "generated" or sample.get("extraction_error"):
                result.update(status="invalid_generation", reason=sample.get("extraction_error") or sample["status"])
            else:
                response = sandbox.run(None, ["python", "-I", "/opt/judge_worker.py"], timeout=180,
                                       stdin=json.dumps({"problem": problems[task_id], "code": sample["code"],
                                                         "dataset": config.get("benchmark", "humaneval")}))
                if response["timeout"]:
                    # Includes reference execution; do not mislabel infrastructure/reference timeout as model failure.
                    result.update(status="infrastructure_error", reason="container_wall_timeout")
                elif response["returncode"] != 0:
                    result.update(status="infrastructure_error", reason=response["output"][-3000:])
                else:
                    try:
                        result.update(json.loads(response["output"]))
                    except (ValueError, TypeError):
                        result.update(status="infrastructure_error", reason="Invalid judge output")
            write_json(target, result)
            print(f"{task_id}: {result['status']}", flush=True)
    return str(run / "judgments")


SPEED_PROMPTS = [
    "Write a Python function that sums a list of integers. Return only code.",
    "Write a Python function that reverses a string. Return only code.",
    "Write a Python function that checks whether a number is prime. Return only code.",
    "Write a Python function that merges two sorted lists. Return only code.",
    "Write a Python function that counts words in a string. Return only code.",
]


def speed(run_id, port=8080):
    run = data_root() / "runs" / safe_name(run_id)
    config = read_json(run / "config.json")
    alias = "tp-" + config["artifact_sha256"][:16]
    client = ModelClient(f"http://127.0.0.1:{port}/v1", alias)
    client.verify(alias)
    service = read_json(data_root() / "services" / f"{port}.json")
    if service["alias"] != alias or any(service[k] != config["server"][k] for k in ("context", "threads", "parallel")):
        raise ValueError("Speed service configuration differs from the generation run")
    if "hardware" in config:
        current = environment()
        if {k: current.get(k) for k in ("platform", "gpu", "cgroup")} != config["hardware"]:
            raise ValueError("Speed measurement hardware differs from the generation run")
    with exclusive(run / ".lock"):
        speed_config = {"generation_config_hash": digest(config), "prompts": SPEED_PROMPTS,
                        "repeats": 3, "max_tokens": 256, "warmup_per_session": 1}
        ensure_config(run / "speed", speed_config)
        if (run / "speed.json").exists():
            return str(run / "speed.json")
        client.chat([{"role": "user", "content": SPEED_PROMPTS[0]}], max_tokens=256)
        records = []
        for repeat in range(3):
            for i, prompt in enumerate(SPEED_PROMPTS):
                path = run / "speed" / f"repeat-{repeat}-prompt-{i}.json"
                if path.exists():
                    record = read_json(path)
                    if record["config_hash"] != digest(speed_config):
                        raise ValueError("Speed sample configuration mismatch")
                    if record["status"] == "started":
                        record["status"] = "interrupted"
                        write_json(path, record)
                else:
                    record = {"repeat": repeat, "prompt_id": i, "status": "started", "response": {},
                              "config_hash": digest(speed_config)}
                    write_json(path, record)
                    with ResourceMonitor(service["pid"]) as monitor:
                        try:
                            record["response"] = client.chat([{"role": "user", "content": prompt}], max_tokens=256)
                            record["status"] = "completed"
                        except Exception as e:
                            record.update(status="request_error", error=type(e).__name__)
                    record["resources"] = monitor.result()
                    write_json(path, record)
                records.append(record)
        write_json(run / "speed.json", {**speed_config, "records": records,
                                        "note": "tokens/s includes prefill; llama-server prefix cache may be warm"})
    return str(run / "speed.json")
