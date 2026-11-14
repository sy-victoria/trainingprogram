"""Minimal Deep Agents comparison; no shell or write tools are exposed."""
import argparse
from pathlib import Path
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tasks/logic/workspace"


def read_repository_file(path: str) -> str:
    """Read a UTF-8 file in the task repository by relative path."""
    target = (FIXTURE / path).resolve()
    if not target.is_relative_to(FIXTURE.resolve()) or target.is_symlink():
        raise ValueError("Path leaves fixture")
    if target.stat().st_size > 100_000:
        raise ValueError("File too large")
    return target.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    args = parser.parse_args()
    model = ChatOpenAI(model=args.model, base_url=args.base_url, api_key="EMPTY", temperature=0,
                       max_tokens=2048, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    agent = create_deep_agent(model=model, tools=[read_repository_file],
                              system_prompt="Use read_repository_file to inspect solution.py and suggest a minimal fix. Do not edit files or delegate.")
    result = agent.invoke({"messages": [{"role": "user", "content": "Read solution.py and explain the bug in is_prime. Suggest a minimal fix."}]},
                          {"recursion_limit": 12})
    for message in result["messages"]:
        print(message.pretty_repr())


if __name__ == "__main__":
    main()
