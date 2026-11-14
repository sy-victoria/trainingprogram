"""Executed only inside the resource-limited, offline Docker container."""
import json
import sys

from evalplus.eval import untrusted_check
from evalplus.gen.util import trusted_exec


def main():
    payload = json.load(sys.stdin)
    problem = payload["problem"]
    code = payload["code"]
    results = {}
    # Same oracle execution and untrusted_check as EvalPlus, restricted to one task.
    for kind in ("base", "plus"):
        inputs = problem[kind + "_input"]
        expected, timing = trusted_exec(problem["prompt"] + problem["canonical_solution"], inputs,
                                        problem["entry_point"], record_time=True)
        status, details = untrusted_check("humaneval", code, inputs, problem["entry_point"],
                                          expected=expected, atol=problem["atol"], ref_time=timing,
                                          fast_check=True, min_time_limit=0.2, gt_time_limit_factor=4.0)
        results[kind] = status
    passed = results["base"] == results["plus"] == "pass"
    print(json.dumps({"status": "pass" if passed else "fail", "passed": passed, **results}))


if __name__ == "__main__":
    main()
