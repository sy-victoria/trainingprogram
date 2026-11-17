"""Durable accounting shared by framework adapters, not a shared agent scheduler."""
from __future__ import annotations

import time
from pathlib import Path

from .common import digest, exclusive, read_json, write_json


class BudgetExceeded(RuntimeError):
    pass


class UncertainRequest(RuntimeError):
    pass


class AgentLedger:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.path = self.directory / "ledger.json"
        self.lock = self.directory / ".ledger.lock"
        self.config = dict(config)
        required = {"max_calls", "max_total_tokens", "max_active_seconds"}
        if set(config) != required or any(type(config[k]) is not int or config[k] <= 0 for k in required):
            raise ValueError("Positive integer call, token and active-time limits required")
        with exclusive(self.lock):
            if self.path.exists():
                if self._read()["config"] != config:
                    raise ValueError("Budget configuration changed; use a new run")
            else:
                write_json(self.path, {"config": config, "calls": 0, "tokens": 0,
                    "active_seconds": 0.0, "inflight": None, "events": [], "terminal_error": None})

    def _read(self):
        return read_json(self.path)

    def reserve_model(self, request, prompt_tokens, max_output_tokens):
        """Prompt count must come from the actual backend tokenizer, not a guess.

        Reserve one request before sending it. Unknown outcomes never refund a call
        or permit silently sampling another answer after a crash.
        """
        if type(prompt_tokens) is not int or prompt_tokens < 0 or type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("Exact nonnegative prompt count and positive output limit required")
        with exclusive(self.lock):
            state = self._read()
            if state["terminal_error"]:
                raise BudgetExceeded(state["terminal_error"])
            if state["inflight"]:
                raise UncertainRequest("An earlier request has no persisted outcome; do not resample")
            if state["calls"] >= self.config["max_calls"] or state["active_seconds"] >= self.config["max_active_seconds"]:
                raise BudgetExceeded("Call or active-time limit reached")
            remaining = self.config["max_total_tokens"] - state["tokens"] - prompt_tokens
            if remaining < 1:
                raise BudgetExceeded("No token allowance remains for the response")
            allowed = min(max_output_tokens, remaining)
            state["calls"] += 1
            ticket = {"id": state["calls"], "request_hash": digest(request), "prompt_tokens": prompt_tokens,
                      "max_output_tokens": allowed, "started_at": time.time()}
            state["inflight"] = ticket
            write_json(self.path, state)
            return dict(ticket)

    def complete_model(self, ticket, response, usage, active_seconds):
        if active_seconds < 0:
            raise ValueError("Negative duration")
        if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
            raise ValueError("Actual model usage is required; missing usage is not zero")
        with exclusive(self.lock):
            state = self._read()
            if state["inflight"] != ticket:
                raise ValueError("Request ticket is stale or already completed")
            # Save the raw response before finalizing its accounting. A crash here
            # remains an uncertain request, never grounds for a fresh generation.
            write_json(self.directory / "model" / f"{ticket['id']:05}.json", {"ticket": ticket, "response": response, "usage": usage})
            state["tokens"] += usage["prompt_tokens"] + usage["completion_tokens"]
            state["active_seconds"] += active_seconds
            mismatch = (usage["prompt_tokens"] != ticket["prompt_tokens"] or
                        usage["completion_tokens"] > ticket["max_output_tokens"])
            state["events"].append({"request_id": ticket["id"], "usage": usage, "active_seconds": active_seconds})
            state["inflight"] = None
            if mismatch:
                state["terminal_error"] = "Backend usage differs from tokenizer/reservation; protocol validation failed"
            write_json(self.path, state)
            if mismatch:
                raise BudgetExceeded(state["terminal_error"])

    def record_tool_time(self, seconds):
        if seconds < 0:
            raise ValueError("Negative duration")
        with exclusive(self.lock):
            state = self._read()
            state["active_seconds"] += seconds
            write_json(self.path, state)

    def remaining_seconds(self):
        state = self._read()
        return max(0, self.config["max_active_seconds"] - state["active_seconds"])


class ApprovalStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "approval.json"
        self.lock = self.directory / ".approval.lock"

    def request(self, action, workspace_fingerprint):
        with exclusive(self.lock):
            if self.path.exists():
                existing = read_json(self.path)
                if existing["status"] == "pending":
                    if existing["action"] == action and existing["workspace_fingerprint"] == workspace_fingerprint:
                        return existing
                    raise ValueError("Another approval is pending")
                sequence = existing["sequence"] + 1
            else:
                sequence = 1
            record = {"sequence": sequence, "action": action, "workspace_fingerprint": workspace_fingerprint,
                      "status": "pending", "requested_at": time.time()}
            write_json(self.path, record)
            return record

    def decide(self, sequence, decision, reviewer, feedback, workspace_fingerprint):
        if decision not in {"approve", "reject", "feedback"} or reviewer not in {"human", "assistant", "fixed_policy"}:
            raise ValueError("Invalid decision or reviewer attribution")
        if reviewer == "fixed_policy" and feedback:
            raise ValueError("Autonomous comparison must not receive solution feedback")
        with exclusive(self.lock):
            record = read_json(self.path)
            if record["sequence"] != sequence or record["status"] != "pending":
                raise ValueError("Approval is stale or already decided")
            if record["workspace_fingerprint"] != workspace_fingerprint:
                raise ValueError("Workspace changed since approval request")
            record.update(status="decided", decision=decision, reviewer=reviewer, feedback=feedback, decided_at=time.time())
            write_json(self.directory / "approvals" / f"{sequence:05}.json", record)
            write_json(self.path, record)
            return record
