from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from collections import Counter
from pathlib import Path

from .common import ROOT, data_root, read_json, write_json, sha256, exclusive, environment


def versions():
    return read_json(ROOT / "configs" / "versions.json")


def model_dir(size):
    if size not in versions()["models"]:
        raise ValueError("Supported sizes: 0.8B, 2B, 4B")
    return data_root() / "models" / size


def tensor_dtypes(folder):
    counts = Counter()
    for p in sorted(Path(folder).glob("*.safetensors")):
        with p.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            if length > 100_000_000:
                raise ValueError("Unexpected safetensors header length")
            header = json.loads(f.read(length))
        for name, meta in header.items():
            if name != "__metadata__":
                counts[meta["dtype"]] += 1
    if not counts:
        raise ValueError("No safetensors weights found")
    return dict(counts)


def tools_python():
    return os.environ.get("TP_TOOLS_PYTHON", "/root/miniconda3/envs/tp-tools/bin/python")


def llama_dir():
    return data_root() / "tools" / "llama.cpp"


def verify_llama():
    p = subprocess.run(["git", "-C", str(llama_dir()), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    if p.stdout.strip() != versions()["llama_cpp"]:
        raise ValueError("llama.cpp checkout differs from pinned revision")


def prepare(size):
    from huggingface_hub import snapshot_download
    from .readiness import require_gpu
    require_gpu(size)
    directory = model_dir(size)
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive(data_root() / "models" / ".prepare.lock"):
        # Existing complete artifacts do not need a source redownload after cleanup.
        if all((directory / f"{v}.{ext}").exists() for v in ("BF16", "Q4_K_M") for ext in ("gguf", "json")):
            for variant in ("BF16", "Q4_K_M"):
                _, manifest = load_artifact(size, variant)
                if manifest["revision"] != versions()["models"][size]["revision"] or manifest["llama_cpp"] != versions()["llama_cpp"]:
                    raise ValueError("Existing model revision differs from the project lock")
            return str(directory)
        # Conservative allowance for source + BF16 + quantized artifact + temporary files.
        minimum = {"0.8B": 6, "2B": 14, "4B": 26}[size] * 1024**3
        if not (directory / "source").exists() and shutil.disk_usage(directory).free < minimum:
            raise RuntimeError(f"Need at least {minimum // 1024**3} GiB free before starting {size}")
        verify_llama()
        spec = versions()["models"][size]
        source = directory / "source"
        snapshot_download(spec["repo"], revision=spec["revision"], local_dir=source,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"])
        dtypes = tensor_dtypes(source)
        if "BF16" not in dtypes or any(d not in {"BF16", "F32", "I64", "I32", "BOOL"} for d in dtypes):
            raise RuntimeError(f"Cannot preserve original BF16 baseline for dtype mix {dtypes}; no silent fallback")
        source_hashes = {p.name: sha256(p) for p in sorted(source.iterdir()) if p.is_file()}
        base_manifest = {**spec, "size": size, "source_dtypes": dtypes, "source_hashes": source_hashes,
                         "llama_cpp": versions()["llama_cpp"], "baseline": "BF16",
                         "note": "GGUF text inference; conversion may retain norms in F32; vision tower is not evaluated"}
        write_json(directory / "source-manifest.json", base_manifest)
        baseline = directory / "BF16.gguf"
        quant = directory / "Q4_K_M.gguf"
        convert = [tools_python(), str(llama_dir() / "convert_hf_to_gguf.py"), str(source),
                   "--outfile", str(baseline) + ".partial", "--outtype", "bf16"]
        for target, command in [
            (baseline, convert),
            (quant, [str(llama_dir() / "build/bin/llama-quantize"), str(baseline), str(quant) + ".partial", "Q4_K_M", "8"])
        ]:
            manifest_path = directory / (target.stem + ".json")
            if target.exists() and manifest_path.exists():
                old = read_json(manifest_path)
                if old["source_hashes"] != source_hashes or old["llama_cpp"] != versions()["llama_cpp"] or old["sha256"] != sha256(target):
                    raise ValueError("Existing artifact mismatch; use explicit cleanup before rebuilding")
                continue
            subprocess.run(command, check=True)
            os.replace(str(target) + ".partial", target)
            write_json(manifest_path, {**base_manifest, "variant": target.stem, "sha256": sha256(target),
                                       "bytes": target.stat().st_size, "command": command})
        write_json(directory / "environment.json", environment())
    return str(directory)


def load_artifact(size, variant, verify=True):
    if variant not in {"BF16", "Q4_K_M"}:
        raise ValueError("Unsupported variant")
    directory = model_dir(size)
    manifest = read_json(directory / (variant + ".json"))
    path = directory / (variant + ".gguf")
    if verify and sha256(path) != manifest["sha256"]:
        raise ValueError("Model artifact hash mismatch")
    return path, manifest


def serve(size, variant, port=8080):
    from .readiness import require_gpu
    require_gpu(size)
    import signal
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    verify_llama()
    path, manifest = load_artifact(size, variant)
    alias = "tp-" + manifest["sha256"][:16]
    args = [str(llama_dir() / "build/bin/llama-server"), "-m", str(path), "--alias", alias,
            "--host", "127.0.0.1", "--port", str(port), "-ngl", "99", "-c", "8192", "-np", "1",
            "--threads", "8", "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}']
    root = data_root() / "services"
    root.mkdir(parents=True, exist_ok=True)
    key = str(port)
    with exclusive(root / (key + ".lock")):
        with (root / (key + ".log")).open("a", encoding="utf-8") as log:
            env = os.environ.copy()
            tool_prefix = Path(tools_python()).parent.parent
            env["LD_LIBRARY_PATH"] = ":".join([str(tool_prefix / "lib"), str(tool_prefix / "targets/x86_64-linux/lib"), env.get("LD_LIBRARY_PATH", "")])
            p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
            write_json(root / (key + ".json"), {"pid": p.pid, "alias": alias, "artifact": manifest,
                                               "command": args, "context": 8192, "threads": 8,
                                               "parallel": 1, "port": port})
            print(f"Model alias: {alias}; log: {root / (key + '.log')}", flush=True)
            try:
                code = p.wait()
                if code:
                    raise RuntimeError(f"llama-server exited with {code}; inspect its log")
            finally:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        p.kill()
                (root / (key + ".json")).unlink(missing_ok=True)


def cleanup(size, target):
    directory = model_dir(size).resolve()
    if target not in {"source", "BF16", "Q4_K_M"}:
        raise ValueError("Choose source, BF16 or Q4_K_M")
    for status in (data_root() / "services").glob("*.json"):
        if read_json(status).get("artifact", {}).get("size") == size:
            raise RuntimeError("Stop the model service before cleanup")
    path = directory / (target if target == "source" else target + ".gguf")
    if path.is_symlink() or not path.resolve().is_relative_to(directory):
        raise ValueError("Unsafe cleanup target")
    with exclusive(data_root() / "models" / ".prepare.lock"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    return f"Removed only {path}; manifests retained"
