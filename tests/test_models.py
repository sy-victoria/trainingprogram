import json
import struct
import pytest
from trainingprogram.models import tensor_dtypes, cleanup
from trainingprogram.common import write_json


def test_safetensors_dtype_read_without_loading_weights(tmp_path):
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
                         "__metadata__": {"format": "pt"}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)
    assert tensor_dtypes(tmp_path) == {"BF16": 1}


def test_cleanup_only_requested_artifact_and_preserves_results(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_DATA", str(tmp_path))
    folder = tmp_path / "models/0.8B"
    folder.mkdir(parents=True)
    (folder / "BF16.gguf").write_text("artifact")
    write_json(folder / "BF16.json", {"manifest": True})
    write_json(tmp_path / "runs/example/config.json", {"results": True})
    cleanup("0.8B", "BF16")
    assert not (folder / "BF16.gguf").exists()
    assert (folder / "BF16.json").exists()
    assert (tmp_path / "runs/example/config.json").exists()


def test_cleanup_rejects_active_model(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_DATA", str(tmp_path))
    write_json(tmp_path / "services/8080.json", {"artifact": {"size": "0.8B"}})
    with pytest.raises(RuntimeError, match="Stop"):
        cleanup("0.8B", "BF16")


def test_prepare_rejects_no_gpu_before_download(tmp_path, monkeypatch):
    from trainingprogram.models import prepare
    monkeypatch.setenv("TP_DATA", str(tmp_path))
    def reject(size):
        raise RuntimeError("No GPU")
    def no_download(*args, **kwargs):
        raise AssertionError("download should not be attempted")
    monkeypatch.setattr("trainingprogram.readiness.require_gpu", reject)
    monkeypatch.setattr("huggingface_hub.snapshot_download", no_download)
    with pytest.raises(RuntimeError, match="No GPU"):
        prepare("0.8B")
    assert not (tmp_path / "models").exists()
