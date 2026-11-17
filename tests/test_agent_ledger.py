import pytest

from trainingprogram.agent_ledger import AgentLedger, ApprovalStore, BudgetExceeded, UncertainRequest
from trainingprogram.common import read_json

CONFIG = dict(max_calls=2, max_total_tokens=100, max_active_seconds=10)


def test_unknown_request_survives_restart_without_refunding_budget(tmp_path):
    ledger = AgentLedger(tmp_path, CONFIG)
    ledger.reserve_model({'messages': []}, 10, 50)
    resumed = AgentLedger(tmp_path, CONFIG)
    with pytest.raises(UncertainRequest):
        resumed.reserve_model({'messages': []}, 10, 50)
    assert read_json(tmp_path/'ledger.json')['calls'] == 1
    with pytest.raises(ValueError, match='changed'):
        AgentLedger(tmp_path, {**CONFIG, 'max_calls': 100})


def test_usage_and_time_enforce_limits_including_tool_execution(tmp_path):
    ledger = AgentLedger(tmp_path, CONFIG)
    ticket = ledger.reserve_model({'messages': []}, 70, 80)
    assert ticket['max_output_tokens'] == 30
    ledger.complete_model(ticket, {'content': 'answer'}, dict(prompt_tokens=70, completion_tokens=20), 2)
    with pytest.raises(BudgetExceeded):
        ledger.reserve_model({}, 10, 50)
    ledger.record_tool_time(8)
    assert ledger.remaining_seconds() == 0
    with pytest.raises(BudgetExceeded):
        ledger.reserve_model({}, 0, 1)


def test_missing_usage_and_backend_mismatch_are_not_silently_accepted(tmp_path):
    ledger = AgentLedger(tmp_path, CONFIG)
    ticket = ledger.reserve_model({}, 10, 20)
    with pytest.raises(ValueError, match='usage'):
        ledger.complete_model(ticket, {}, {}, 1)
    with pytest.raises(BudgetExceeded, match='Backend'):
        ledger.complete_model(ticket, {}, dict(prompt_tokens=12, completion_tokens=5), 1)
    assert read_json(tmp_path/'model/00001.json')['usage']['prompt_tokens'] == 12
    with pytest.raises(BudgetExceeded):
        ledger.reserve_model({}, 10, 20)


def test_approval_resume_stale_workspace_and_reviewer_attribution(tmp_path):
    action = {'tool': 'apply_patch', 'path': 'file.py', 'after': 'fix'}
    ApprovalStore(tmp_path).request(action, 'snapshot-a')
    resumed = ApprovalStore(tmp_path)
    assert resumed.request(action, 'snapshot-a')['sequence'] == 1
    with pytest.raises(ValueError, match='changed'):
        resumed.decide(1, 'approve', 'assistant', '', 'snapshot-b')
    with pytest.raises(ValueError, match='feedback'):
        resumed.decide(1, 'approve', 'fixed_policy', 'Use this solution', 'snapshot-a')
    result = resumed.decide(1, 'reject', 'assistant', 'Does not support required inputs', 'snapshot-a')
    assert result['reviewer'] == 'assistant' and result['decision'] == 'reject'
    with pytest.raises(ValueError, match='stale'):
        resumed.decide(1, 'approve', 'assistant', '', 'snapshot-a')
