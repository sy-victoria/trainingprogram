import json

import httpx

from trainingprogram.agent_gateway import model_gateway
from trainingprogram.agent_ledger import BudgetExceeded


def test_authenticated_gateway_streams_tools_and_usage_without_resampling():
    class Backend:
        model = 'artifact'
        calls = 0
        def chat(self, request):
            self.calls += 1
            if self.calls > 1:
                raise BudgetExceeded('limit')
            return {'id': 'reply-1', 'created': 1, 'model': self.model,
                    'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message':
                        {'role': 'assistant', 'content': None, 'tool_calls': [
                            {'id': 'tool-1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"x.py"}'}}]}}],
                    'usage': {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30}}
    backend = Backend()
    with model_gateway(backend) as endpoint, httpx.Client(trust_env=False) as client:
        url = endpoint['base_url']+'/chat/completions'
        assert client.post(url, json={}).status_code == 401
        assert backend.calls == 0
        headers = {'Authorization': 'Bearer '+endpoint['api_key']}
        response = client.post(url, headers=headers, json={'stream': True, 'messages': [{'role':'user','content':'read'}]})
        assert response.status_code == 200
        chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        assert chunks[0]['choices'][0]['delta']['tool_calls'][0]['index'] == 0
        assert chunks[-1]['usage']['total_tokens'] == 30
        assert backend.calls == 1
        assert client.post(url, headers=headers, json={}).status_code == 402
