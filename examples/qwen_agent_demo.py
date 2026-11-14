"""Run in its own environment: requirements/qwen-example.txt."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("QWEN_AGENT_MAX_LLM_CALL_PER_RUN", "5")
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tasks/logic/workspace"


@register_tool("read_file")
class ReadFile(BaseTool):
    description = "Read a UTF-8 file in the task repository."
    parameters = [{"name": "path", "type": "string", "description": "Relative file path", "required": True}]

    def call(self, params, **kwargs):
        path = (FIXTURE / json.loads(params)["path"]).resolve()
        if not path.is_relative_to(FIXTURE.resolve()) or path.is_symlink():
            raise ValueError("Path leaves fixture")
        if path.stat().st_size > 100_000:
            raise ValueError("File too large")
        return path.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    args = parser.parse_args()
    agent = Assistant(llm={"model": args.model, "model_server": args.base_url, "api_key": "EMPTY",
                           "generate_cfg": {"temperature": 0, "max_tokens": 2048}},
                      function_list=["read_file"], system_message="Read the file using the tool, then suggest a minimal fix. Do not edit files.")
    last = []
    for response in agent.run(messages=[{"role": "user", "content": "Read solution.py and explain the bug in is_prime. Suggest a minimal fix."}]):
        last = response
    print(json.dumps(last, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
