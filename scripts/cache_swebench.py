"""Cache a pinned official dataset for the trusted harness, never an agent mount."""
import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--revision')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Dataset destination exists; verify its manifest instead of overwriting')
    repo = 'SWE-bench/SWE-bench_Verified'
    revision = args.revision or HfApi().dataset_info(repo).sha
    data = load_dataset(repo, split='test', revision=revision)
    rows = list(data)
    if len(rows) != 500 or len({r['instance_id'] for r in rows}) != 500:
        raise ValueError('Verified must contain exactly 500 unique instances')
    required = {'image', 'eval_script', 'log_parser', 'eval_type', 'FAIL_TO_PASS', 'PASS_TO_PASS'}
    if any(required - row.keys() for row in rows):
        raise ValueError('Dataset schema does not match pinned SWE-bench 5 harness')
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        f.write(payload)
    manifest = {'repo': repo, 'revision': revision, 'split': 'test', 'instances': 500,
                'sha256': hashlib.sha256(payload).hexdigest(),
                'contains_gold': True, 'access': 'trusted_harness_only',
                'instance_ids': [r['instance_id'] for r in rows]}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'instance_ids'}), flush=True)


if __name__ == '__main__':
    main()
