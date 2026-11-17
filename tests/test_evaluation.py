import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from trainingprogram.client import ModelClient
from trainingprogram.common import write_json, read_json, digest
from trainingprogram.evaluation import ensure_config, extract_code, generate_samples, select_tasks, GENERATION
from trainingprogram.reporting import score, export_bundle, redact


def test_config_change_rejected(tmp_path):
    ensure_config(tmp_path, {"seed": 1})
    ensure_config(tmp_path, {"seed": 1})
    with pytest.raises(ValueError, match="NEW"):
        ensure_config(tmp_path, {"seed": 2})


@pytest.mark.parametrize("text,error", [
    ("```python\ndef f(x):\n    return x\n```", None),
    ("<think>reasoning</think>\ndef f(x):\n    return x", None),
    ("def broken(", "syntax_error"),
    ("def other():\n    return 1", "missing_entry_point"),
])
def test_extraction(text, error):
    code, found = extract_code(text, "f")
    assert found == error


def test_subset_prefixes_match():
    problems = {f"HumanEval/{i}": {} for i in range(164)}
    assert select_tasks(problems, "smoke") == select_tasks(problems, "subset")[:2]
    assert select_tasks(problems, "subset") == select_tasks(problems, "full")[:30]


def test_infra_not_scored_and_invalid_counted():
    config = {"task_ids": ["a", "b"]}
    judgments = {"a": {"status": "pass", "passed": True}, "b": {"status": "infrastructure_error", "passed": False}}
    assert score(config, judgments)["pass_at_1"] is None
    judgments["b"] = {"status": "invalid_generation", "passed": False}
    assert score(config, judgments)["pass_at_1"] == 0.5


def test_interrupted_request_not_resampled(tmp_path, monkeypatch):
    config = {"task_ids": ["HumanEval/0"], "generation": GENERATION}
    write_json(tmp_path / "samples/HumanEval_0.json", {"task_id": "HumanEval/0", "status": "started", "config_hash": digest(config)})
    class NoCalls:
        def chat(self, *a, **k):
            raise AssertionError("must not resample")
    generate_samples(tmp_path, config, {}, NoCalls())
    assert read_json(tmp_path / "samples/HumanEval_0.json")["status"] == "interrupted"


@pytest.mark.parametrize("sampling", [{}, {"temperature": 0.2, "top_p": 0.95, "seed": 2028}])
def test_http_sse_and_resume_end_to_end(tmp_path, monkeypatch, sampling):
    monkeypatch.setattr("trainingprogram.evaluation.ResourceMonitor.sample", lambda self: None)
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [{"choices": [{"delta": {"content": "```python\ndef f(x):\n    return x\n```"}, "finish_reason": None}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 15}}]
            for c in chunks:
                self.wfile.write(("data: " + json.dumps(c) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ModelClient(f"http://127.0.0.1:{server.server_port}/v1")
        config = {"task_ids": ["HumanEval/0"], "generation": {**GENERATION, **sampling}}
        problems = {"HumanEval/0": {"entry_point": "f", "prompt": "def f(x):\n    pass"}}
        generate_samples(tmp_path, config, problems, client)
        generate_samples(tmp_path, config, problems, client)
        assert len(received) == 1
        assert received[0]["temperature"] == sampling.get("temperature", 0)
        assert received[0]["seed"] == sampling.get("seed", 2026)
        if "top_p" in sampling:
            assert received[0]["top_p"] == sampling["top_p"]
        assert received[0]["chat_template_kwargs"]["enable_thinking"] is False
        record = read_json(tmp_path / "samples/HumanEval_0.json")
        assert record["extraction_error"] is None
        assert record["response"]["usage"]["completion_tokens"] == 15
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_export_allowlist_and_redaction(tmp_path, monkeypatch):
    import zipfile
    monkeypatch.setenv("TP_DATA", str(tmp_path / "data"))
    write_json(tmp_path / "data/runs/demo/config.json", {"api_key": "secret", "max_tokens": 2048})
    write_json(tmp_path / "data/models/weights.json", {"should_not": "export"})
    path = export_bundle(tmp_path / "bundle.zip")
    with zipfile.ZipFile(path) as z:
        assert "models/weights.json" not in z.namelist()
        assert json.loads(z.read("runs/demo/config.json"))["api_key"] == "[REDACTED]"
        assert json.loads(z.read("runs/demo/config.json"))["max_tokens"] == 2048


def test_summary_pairs_only_matching_configs(tmp_path, monkeypatch):
    from trainingprogram.reporting import summarize
    monkeypatch.setenv("TP_DATA", str(tmp_path))
    for variant, passed in [("BF16", True), ("Q4_K_M", False)]:
        run = tmp_path / "runs" / variant
        config = {"size": "2B", "variant": variant, "mode": "smoke", "model_revision": "same",
                  "llama_cpp": "same", "evalplus": "same", "dataset_sha256": "same",
                  "task_ids": ["HumanEval/0"], "generation": GENERATION, "server": {},
                  "harness_hash": "same", "artifact_bytes": 100 if variant == "BF16" else 30}
        write_json(run / "config.json", config)
        write_json(run / "environment.json", {"platform": "same"})
        write_json(run / "judgments/HumanEval_0.json", {"task_id": "HumanEval/0", "status": "pass" if passed else "fail", "passed": passed})
    summarize()
    result = read_json(tmp_path / "summary/results.json")
    assert result["pairs"][0]["delta_pp"] == -100
    assert result["pairs"][0]["disk_reduction_pct"] == 70
    assert result["pairs"][0]["regressions"] == ["HumanEval/0"]
    config["generation"] = {**GENERATION, "seed": 4}
    write_json(tmp_path / "runs/Q4_K_M/config.json", config)
    summarize()
    assert not read_json(tmp_path / "summary/results.json")["pairs"]
