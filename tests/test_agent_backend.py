import json

import httpx
import pytest

from trainingprogram.agent_backend import CountedChatBackend
from trainingprogram.agent_ledger import AgentLedger, BudgetExceeded, UncertainRequest
from trainingprogram.common import read_json


def setup(tmp_path, handler, tokens=100):
    ledger=AgentLedger(tmp_path, dict(max_calls=3,max_total_tokens=tokens,max_active_seconds=60))
    return ledger,CountedChatBackend('http://model/v1','artifact-alias',ledger,seed=2028,
        temperature=.2,top_p=.95,max_output_tokens=40,context_tokens=100,
        transport=httpx.MockTransport(handler))


def test_exact_backend_count_caps_tokens_and_preserves_tools(tmp_path):
    observed=[]
    def handle(request):
        if request.method=='GET':
            return httpx.Response(200,json={'data':[{'id':'artifact-alias'}]})
        body=json.loads(request.content); observed.append(body)
        if request.url.path.endswith('input_tokens'):
            return httpx.Response(200,json={'input_tokens':80})
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'ok'}}],
            'usage':{'prompt_tokens':80,'completion_tokens':10}})
    ledger,backend=setup(tmp_path,handle)
    request={'messages':[{'role':'user','content':'repair'}],
             'tools':[{'type':'function','function':{'name':'read_file','parameters':{'type':'object'}}}],
             'temperature':1,'seed':1,'max_tokens':999}
    backend.chat(request)
    assert observed[0]['tools']==observed[1]['tools']==request['tools']
    assert observed[1]['max_tokens']==20 and observed[1]['seed']==2028
    assert observed[1]['temperature']==.2 and observed[1]['top_p']==.95
    assert read_json(tmp_path/'ledger.json')['tokens']==90
    assert request['seed']==1  # caller payload is not mutated


def test_network_failure_leaves_unknown_request_instead_of_retrying(tmp_path):
    requests=[]
    def handle(request):
        requests.append(request.url.path)
        if request.method=='GET': return httpx.Response(200,json={'data':[{'id':'artifact-alias'}]})
        if request.url.path.endswith('input_tokens'): return httpx.Response(200,json={'input_tokens':10})
        raise httpx.ReadError('connection lost')
    ledger,backend=setup(tmp_path,handle)
    with pytest.raises(httpx.ReadError): backend.chat({'messages':[{'role':'user','content':'x'}]})
    with pytest.raises(UncertainRequest): backend.chat({'messages':[{'role':'user','content':'x'}]})
    assert requests.count('/v1/chat/completions')==1


def test_no_context_truncation_or_counter_fallback(tmp_path):
    def handle(request):
        if request.method=='GET': return httpx.Response(200,json={'data':[{'id':'artifact-alias'}]})
        assert request.url.path.endswith('input_tokens')
        return httpx.Response(200,json={'input_tokens':101})
    ledger,backend=setup(tmp_path,handle)
    with pytest.raises(BudgetExceeded,match='Context'):
        backend.chat({'messages':[{'role':'user','content':'x'}]})
    assert read_json(tmp_path/'ledger.json')['calls']==0
