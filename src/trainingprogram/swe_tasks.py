"""Separate public issue inputs from the trusted SWE-bench evaluation payload."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def load_verified(path, manifest_path):
    raw = Path(path).read_bytes()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    if manifest.get("repo") != "SWE-bench/SWE-bench_Verified" or manifest.get("split") != "test":
        raise ValueError("Not the locked official Verified test set")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
        raise ValueError("Dataset revision must be a commit")
    if hashlib.sha256(raw).hexdigest() != manifest.get("sha256"):
        raise ValueError("Dataset content hash differs from manifest")
    rows = json.loads(raw)
    ids = [row["instance_id"] for row in rows]
    if len(rows) != 500 or len(set(ids)) != 500 or ids != manifest.get("instance_ids"):
        raise ValueError("Verified requires the exact 500-instance manifest")
    return {row["instance_id"]: row for row in rows}


def public_issue(instance, pinned_image):
    """Strict allowlist: never expose gold patch, eval script or hidden test lists.

    The returned dictionary may be mounted into an agent controller; the original
    dataset must remain accessible only to the trusted evaluator.
    """
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", pinned_image):
        raise ValueError("Agent environment must use an immutable image digest")
    expected_repo = instance["image"].split("@")[0].rsplit(":", 1)[0]
    if pinned_image.split("@")[0] != expected_repo:
        raise ValueError("Pinned image repository differs from official instance image")
    commit = instance["base_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Invalid base commit")
    return {
        "instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "base_commit": commit,
        "problem_statement": instance["problem_statement"],
        "image": pinned_image,
    }
