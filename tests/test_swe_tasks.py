import hashlib
import json

import pytest

from trainingprogram.swe_tasks import load_verified, public_issue


def test_public_issue_excludes_gold_tests_hints_and_future_fields():
    row = dict(instance_id="org__repo-1", repo="org/repo", base_commit="a"*40,
               problem_statement="Repair the issue", image="swebench/instance:latest",
               patch="secret gold", test_patch="secret tests", eval_script="hidden",
               FAIL_TO_PASS=["hidden name"], PASS_TO_PASS=["another hidden name"],
               hints_text="solution hint", future_field="unknown secret")
    result = public_issue(row, "swebench/instance@sha256:" + "b"*64)
    assert set(result) == {"instance_id", "repo", "base_commit", "problem_statement", "image"}
    assert "secret" not in json.dumps(result)
    with pytest.raises(ValueError, match="immutable"):
        public_issue(row, row["image"])
    with pytest.raises(ValueError, match="repository"):
        public_issue(row, "other/instance@sha256:" + "b"*64)


def test_verified_manifest_rejects_missing_duplicate_and_changed_instances(tmp_path):
    path, meta = tmp_path/"dataset.json", tmp_path/"manifest.json"
    rows = [{"instance_id": f"org__repo-{i}"} for i in range(500)]
    def save(values):
        raw = json.dumps(values).encode()
        path.write_bytes(raw)
        meta.write_text(json.dumps(dict(repo="SWE-bench/SWE-bench_Verified", split="test",
            revision="a"*40, sha256=hashlib.sha256(raw).hexdigest(),
            instance_ids=[x["instance_id"] for x in values])))
    save(rows)
    assert len(load_verified(path, meta)) == 500
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        load_verified(path, meta)
    save(rows[:-1])
    with pytest.raises(ValueError, match="500"):
        load_verified(path, meta)
    save(rows[:-1]+[rows[0]])
    with pytest.raises(ValueError, match="500"):
        load_verified(path, meta)
