import pytest
from trainingprogram import evaluation
from trainingprogram.common import write_json, read_json


def test_speed_resume_preserves_completed_and_interrupted_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_DATA", str(tmp_path))
    monkeypatch.setattr(evaluation.ResourceMonitor, "sample", lambda self: None)
    run = tmp_path / "runs/demo"
    config = {"artifact_sha256": "abc", "server": {"context": 8192, "threads": 8, "parallel": 1}}
    write_json(run / "config.json", config)
    write_json(tmp_path / "services/8080.json", {"pid": 1, "alias": "tp-abc", **config["server"]})
    class Client:
        calls = 0
        crash = True
        def verify(self, alias):
            assert alias == "tp-abc"
        def chat(self, *args, **kwargs):
            self.calls += 1
            if self.crash and self.calls == 3:
                raise KeyboardInterrupt
            return {"text": "pass", "elapsed_s": 1, "output_tokens_per_s": 1}
    client = Client()
    monkeypatch.setattr(evaluation, "ModelClient", lambda *args, **kwargs: client)
    with pytest.raises(KeyboardInterrupt):
        evaluation.speed("demo")
    first = read_json(run / "speed/repeat-0-prompt-0.json")
    assert first["status"] == "completed"
    assert read_json(run / "speed/repeat-0-prompt-1.json")["status"] == "started"
    client.calls, client.crash = 0, False
    evaluation.speed("demo")
    assert client.calls == 14  # new warmup + 13 not-yet-started samples
    assert read_json(run / "speed/repeat-0-prompt-0.json") == first
    records = read_json(run / "speed.json")["records"]
    assert len(records) == 15
    assert records[1]["status"] == "interrupted"
    evaluation.speed("demo")
    assert client.calls == 14
