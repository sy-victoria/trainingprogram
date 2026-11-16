from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver

from .aci import ACI
from .client import ModelClient
from .common import ROOT, write_json, read_json, exclusive, environment, safe_name, data_root

SYSTEM = """You are a coding agent. Use tools to inspect and repair the provided Python repository.
Return exactly one JSON object per turn, without markdown or reasoning text:
{"tool":"list_files","args":{}}
{"tool":"read_file","args":{"path":"solution.py"}}
{"tool":"search_code","args":{"query":"literal text"}}
{"tool":"apply_patch","args":{"path":"solution.py","before":"exact unique text","after":"replacement text"}}
{"tool":"run_tests","args":{}}
{"tool":"finish","args":{"summary":"what changed and what was verified"}}
Never edit tests. Run tests before finishing. Patches need human approval.
Only claim tests passed if a tool result confirms it. All paths are relative to the workspace.
"""


class AgentState(TypedDict, total=False):
    messages: list
    action: dict
    step: int
    approval: str
    status: str
    summary: str


def parse_action(text):
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        if match:
            text = match.group(1)
    action = json.loads(text)
    if not isinstance(action, dict) or action.get("tool") not in {
        "list_files", "read_file", "search_code", "apply_patch", "run_tests", "finish"
    } or not isinstance(action.get("args"), dict):
        raise ValueError("Invalid tool action schema")
    return {"tool": action["tool"], "args": action["args"]}


def build_graph(run, client, saver, budget=10, sandbox=None):
    run = Path(run)
    aci = ACI(run / "workspace", run / "journal", sandbox=sandbox)

    def decide(state):
        ledger = read_json(run / "budget.json") if (run / "budget.json").exists() else {"calls": 0}
        if ledger["calls"] >= budget:
            return {"status": "budget_exhausted", "summary": "Model call budget exhausted", "action": {"tool": "finish", "args": {}}}
        step = ledger["calls"] + 1
        write_json(run / "budget.json", {"calls": step})  # reserve before the HTTP call
        messages = list(state["messages"])
        try:
            result = client.chat(messages)
            write_json(run / "model" / f"{step:03}.json", result)
            action = parse_action(result["text"])
            messages.append({"role": "assistant", "content": result["text"]})
            if action["tool"] == "apply_patch":
                action["prepared"] = aci.prepare_patch(action["args"])
            if action["tool"] == "finish":
                return {"messages": messages, "action": action, "step": step,
                        "status": "finished", "summary": str(action["args"].get("summary", ""))}
            return {"messages": messages, "action": action, "step": step, "status": "running"}
        except Exception as e:
            # Error type only for HTTP errors: do not store headers/tokens.
            error = str(e)[:1000] if isinstance(e, (ValueError, KeyError)) else type(e).__name__
            write_json(run / "model" / f"{step:03}.error.json", {"error": error})
            return {"messages": messages + [{"role": "user", "content": "Action failed: " + error}],
                    "action": {"tool": "invalid", "args": {}}, "step": step, "status": "running"}

    def approve(state):
        decision = interrupt({"kind": "patch_approval", "step": state["step"], "patch": state["action"]["args"]})
        if not isinstance(decision, dict) or decision.get("decision") not in {"approve", "reject", "feedback"}:
            raise ValueError("Decision must be approve, reject or feedback")
        write_json(run / "approvals" / f"{state['step']:03}.json", decision)
        if decision["decision"] == "approve":
            return {"approval": "approve"}
        return {"approval": "reject", "messages": state["messages"] + [
            {"role": "user", "content": "Patch NOT approved. " + str(decision.get("feedback", ""))}]}

    def execute(state):
        operation = f"step-{state['step']:03}"
        try:
            value = aci.call(state["action"], operation)
            result = {"ok": True, "result": value}
        except Exception as e:
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"[:2000]}
        write_json(run / "tools" / (operation + ".json"), {"action": state["action"], **result})
        return {"messages": state["messages"] + [{"role": "user", "content": json.dumps(result, ensure_ascii=False)[:16000]}]}

    graph = StateGraph(AgentState)
    graph.add_node("decide", decide)
    graph.add_node("approve", approve)
    graph.add_node("execute", execute)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", lambda s: END if s["action"]["tool"] == "finish" else
                                "approve" if s["action"]["tool"] == "apply_patch" else
                                "decide" if s["action"]["tool"] == "invalid" else "execute")
    graph.add_conditional_edges("approve", lambda s: "execute" if s["approval"] == "approve" else "decide")
    graph.add_edge("execute", "decide")
    return graph.compile(checkpointer=saver)


def create_run(directory, task, base_url, model, budget=10):
    directory = Path(directory)
    if directory.exists():
        raise ValueError("Run already exists; use resume or a new ID")
    source = ROOT / "tasks" / safe_name(task)
    spec = read_json(source / "task.json")
    directory.mkdir(parents=True)
    shutil.copytree(source / "workspace", directory / "workspace")
    write_json(directory / "config.json", {"task": task, "prompt": spec["prompt"], "base_url": base_url,
                                           "model": model, "budget": budget})
    write_json(directory / "environment.json", environment())


def invoke_run(directory, decision=None, feedback="", status_only=False, client=None, sandbox=None):
    directory = Path(directory)
    config = read_json(directory / "config.json")
    with exclusive(directory / ".lock"), SqliteSaver.from_conn_string(str(directory / "checkpoint.sqlite")) as saver:
        graph = build_graph(directory, client or ModelClient(config["base_url"], config["model"]),
                            saver, config["budget"], sandbox)
        options = {"configurable": {"thread_id": "task"}, "recursion_limit": max(100, config["budget"] * 5)}
        snapshot = graph.get_state(options)
        if not status_only:
            paused = any(t.interrupts for t in snapshot.tasks)
            if paused and decision is None:
                status_only = True
            else:
                if decision and not paused:
                    raise ValueError("No pending approval")
                if decision:
                    value = Command(resume={"decision": decision, "feedback": feedback})
                elif not snapshot.values:
                    value = {"messages": [{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": config["prompt"]}], "status": "running"}
                elif snapshot.next:
                    value = None  # resume an interrupted process, not a new conversation
                else:
                    value = None
                    status_only = True
                if not status_only:
                    graph.invoke(value, options)
            snapshot = graph.get_state(options)
        result = {"status": snapshot.values.get("status", "new"), "next": list(snapshot.next),
                  "summary": snapshot.values.get("summary", ""),
                  "pending": [i.value for t in snapshot.tasks for i in t.interrupts]}
        if result["pending"]:
            result["status"] = "awaiting_approval"
        write_json(directory / "status.json", result)
        return result


def run_suite(suite_id, base_url, model, budget=10, judging="defer"):
    """Run three tasks with explicit interactive approval; quitting preserves state."""
    from .evaluation import ensure_config
    from .tasks import verify_agent
    from .sandbox import get_sandbox
    from .client import ModelClient
    safe_name(suite_id)
    if judging == "required":
        get_sandbox().probe()
    ModelClient(base_url, model).verify(model)
    folder = data_root() / "agent_suites" / suite_id
    config = {"base_url": base_url, "model": model, "budget": budget, "judging": judging,
              "tasks": ["logic", "boundary", "feature"]}
    with exclusive(folder / ".lock"):
        ensure_config(folder, config)
        results = {}
        for task in config["tasks"]:
            directory = data_root() / "agents" / safe_name(f"{suite_id}-{task}")
            if not directory.exists():
                create_run(directory, task, base_url, model, budget)
            else:
                saved = read_json(directory / "config.json")
                if any(saved[k] != config[k] for k in ("base_url", "model", "budget")) or saved["task"] != task:
                    raise ValueError("Existing task belongs to another suite configuration")
            state = invoke_run(directory)
            while state["pending"]:
                print(json.dumps({"task": task, "approval": state["pending"]}, ensure_ascii=False, indent=2))
                choice = input("[a] approve, [r] reject, [f] feedback, [q] pause suite: ").strip().lower()
                if choice == "q":
                    result = {"status": "paused", "current_task": task, "tasks": results}
                    write_json(folder / "status.json", result)
                    return result
                if choice not in {"a", "r", "f"}:
                    continue
                feedback = input("Feedback: ") if choice in {"r", "f"} else ""
                state = invoke_run(directory, {"a": "approve", "r": "reject", "f": "feedback"}[choice], feedback)
            results[task] = state
            if judging == "required":
                results[task]["acceptance"] = verify_agent(directory)
            write_json(folder / "status.json", {"status": "running", "tasks": results})
        result = {"status": "finished", "tasks": results, "judging": judging}
        write_json(folder / "status.json", result)
        return result
