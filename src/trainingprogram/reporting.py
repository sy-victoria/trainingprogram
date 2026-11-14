from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import statistics
import zipfile
from pathlib import Path

from .common import data_root, read_json, write_json, digest
from .evaluation import task_filename


def score(config, judgments):
    expected = config["task_ids"]
    valid = [judgments[t] for t in expected if t in judgments and judgments[t]["status"] != "infrastructure_error"]
    correct = sum(x.get("passed") is True for x in valid)
    complete = len(valid) == len(expected)
    return {"n": len(expected), "judged": len(valid), "passed": correct, "complete": complete,
            "pass_at_1": correct / len(expected) if complete and expected else None,
            "infrastructure_errors": sum(judgments.get(t, {}).get("status") == "infrastructure_error" for t in expected)}


def summarize():
    rows, paired, records = [], [], []
    for path in sorted((data_root() / "runs").glob("*/config.json")):
        config = read_json(path)
        run = path.parent
        judgments = {x["task_id"]: x for p in (run / "judgments").glob("*.json") for x in [read_json(p)]}
        result = score(config, judgments)
        samples = [read_json(p) for p in (run / "samples").glob("*.json")]
        row = {"run": run.name, "size": config["size"], "variant": config["variant"], "mode": config["mode"],
               **result, "artifact_gib": config["artifact_bytes"] / 1024**3,
               "truncated": sum(x.get("truncated", False) for x in samples),
               "generation_errors": sum(x["status"] != "generated" for x in samples),
               "format_errors": sum(bool(x.get("extraction_error")) for x in samples)}
        for key in ("peak_server_rss_bytes_sampled", "peak_gpu_device_mib_sampled"):
            values = [x.get("resources", {}).get(key) for x in samples]
            row[key] = max((v for v in values if v is not None), default=None)
        speed_path = run / "speed.json"
        speed = read_json(speed_path)["records"] if speed_path.exists() else []
        for key in ("ttft_s", "elapsed_s", "output_tokens_per_s"):
            values = [x["response"].get(key) for x in speed]
            row["speed_median_" + key] = statistics.median(v for v in values if v is not None) if any(v is not None for v in values) else None
        rows.append(row)
        pair_config = {k: config[k] for k in ("size", "model_revision", "llama_cpp", "evalplus", "dataset_sha256", "task_ids", "generation", "server", "harness_hash")}
        if (run / "judge/config.json").exists():
            pair_config["judge"] = {k: v for k, v in read_json(run / "judge/config.json").items() if k != "generation_config_hash"}
        env = read_json(run / "environment.json")
        pair_config["hardware"] = {k: env.get(k) for k in ("platform", "gpu", "cgroup")}
        records.append((digest(pair_config), row, judgments))
    for group, baseline, base_judgments in records:
        if baseline["variant"] != "BF16" or not baseline["complete"]:
            continue
        for other_group, quant, quant_judgments in records:
            if other_group != group or quant["variant"] != "Q4_K_M" or not quant["complete"]:
                continue
            paired.append({"size": baseline["size"], "baseline_run": baseline["run"], "quant_run": quant["run"],
                           "delta_pp": 100 * (quant["pass_at_1"] - baseline["pass_at_1"]),
                           "disk_reduction_pct": 100 * (1 - quant["artifact_gib"] / baseline["artifact_gib"]),
                           "regressions": [t for t in base_judgments if base_judgments[t].get("passed") and not quant_judgments[t].get("passed")],
                           "improvements": [t for t in base_judgments if not base_judgments[t].get("passed") and quant_judgments[t].get("passed")]})
    directory = data_root() / "summary"
    write_json(directory / "results.json", {"runs": rows, "pairs": paired})
    if rows:
        with (directory / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# 实验结果（自动汇总）", "", "未判题或存在基础设施错误时不输出 pass@1。子集成绩不能当作完整榜单成绩。", "",
             "| Run | 模型 | 精度 | 已判题/总数 | pass@1 |", "|---|---|---|---|---|"]
    for r in rows:
        value = f"{r['pass_at_1']:.2%}" if r["pass_at_1"] is not None else "待完成"
        lines.append(f"| {r['run']} | {r['size']} | {r['variant']} | {r['judged']}/{r['n']} | {value} |")
    lines += ["", "## 同配置配对比较", ""]
    for p in paired:
        lines.append(f"- {p['baseline_run']} → {p['quant_run']}：{p['delta_pp']:+.2f} 个百分点；文件体积减少 {p['disk_reduction_pct']:.1f}%。")
    lines += ["", "## 报告待补充", "", "- 真实 Agent 任务通过率与审批/恢复演示。", "- 量化回归与改善题目的原因分析。",
              "- 速度指标包含 prefill；显存为整卡采样峰值，不能当作精确分配器峰值。", "- Qwen3.8 与大规模模型尚未覆盖。", ""]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return str(directory)


SECRET_KEY = re.compile(r"^(?:api[_-]?key|access[_-]?token|authorization|password|secret|credential)$", re.I)
SECRET_TEXT = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")


def redact(value):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if SECRET_KEY.match(k) else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return SECRET_TEXT.sub("[REDACTED]", value)
    return value


def export_bundle(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Export destination already exists; choose a new filename")
    manifest = []
    root = data_root()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Only application-owned result trees, never environment variables, caches or models.
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in ("runs", "agents", "summary", "diagnostics"):
            for path in sorted((root / folder).rglob("*")):
                if not path.is_file() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
                    continue
                if path.suffix not in {".json", ".jsonl", ".csv", ".md", ".py", ".txt"} or path.name.startswith("."):
                    continue
                if path.stat().st_size > 100 * 1024**2:
                    raise ValueError(f"Unexpectedly large result file: {path.name}")
                raw = path.read_text(encoding="utf-8-sig")
                if path.suffix == ".json":
                    raw = json.dumps(redact(json.loads(raw)), ensure_ascii=False, indent=2)
                else:
                    raw = SECRET_TEXT.sub("[REDACTED]", raw)
                content = raw.encode()
                name = path.relative_to(root).as_posix()
                archive.writestr(name, content)
                manifest.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        archive.writestr("EXPORT_MANIFEST.json", json.dumps({"files": manifest, "note": "No model weights or checkpoints; known secrets redacted"}, indent=2))
    return str(destination)
