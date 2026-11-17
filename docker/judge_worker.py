"""Executed only inside the resource-limited, offline sandbox."""
import json
import sys

from evalplus.eval import untrusted_check
from evalplus.gen.util import trusted_exec


def main():
    payload = json.load(sys.stdin)
    problem = payload["problem"]
    code = payload["code"]
    dataset = payload.get("dataset", "humaneval")
    if dataset not in {"humaneval", "mbpp"}:
        raise ValueError("Unsupported dataset")
    output_not_none = False
    if dataset == "mbpp":
        from evalplus.data.mbpp import mbpp_deserialize_inputs
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
        problem = dict(problem)
        for kind in ("base", "plus"):
            problem[kind + "_input"] = mbpp_deserialize_inputs(problem["task_id"], problem[kind + "_input"])
        output_not_none = problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS
    results = {}
    # Same oracle execution and untrusted_check as EvalPlus, restricted to one task.
    for kind in ("base", "plus"):
        inputs = problem[kind + "_input"]
        expected, timing = trusted_exec(problem["prompt"] + problem["canonical_solution"], inputs,
                                        problem["entry_point"], record_time=True, output_not_none=output_not_none)
        status, details = untrusted_check(dataset, code, inputs, problem["entry_point"],
                                          expected=expected, atol=problem["atol"], ref_time=timing,
                                          fast_check=True, min_time_limit=0.2, gt_time_limit_factor=4.0)
        results[kind] = status
    passed = results["base"] == results["plus"] == "pass"
    print(json.dumps({"status": "pass" if passed else "fail", "passed": passed, **results}))


if __name__ == "__main__":
    main()
