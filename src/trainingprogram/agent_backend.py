"""A counted model transport; framework adapters retain their native scheduling."""
from __future__ import annotations

import copy
import time

import httpx

from .agent_ledger import AgentLedger, BudgetExceeded
from .common import write_json


class CountedChatBackend:
    def __init__(self, base_url, model, ledger: AgentLedger, *, seed, temperature,
                 top_p, max_output_tokens, context_tokens, transport=None):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.ledger = ledger
        self.settings = dict(seed=seed, temperature=temperature, top_p=top_p,
                             max_tokens=max_output_tokens,
                             chat_template_kwargs={'enable_thinking': False})
        self.context_tokens = context_tokens
        self.transport = transport
        if max_output_tokens < 1 or context_tokens < 1:
            raise ValueError('Positive context and output limits required')

    def chat(self, request):
        payload = copy.deepcopy(request)
        if payload.get('model', self.model) != self.model or payload.get('n', 1) != 1:
            raise ValueError('Model changes and multiple candidates are forbidden')
        if not isinstance(payload.get('messages'), list) or not payload['messages']:
            raise ValueError('Nonempty messages required')
        # Disallow per-framework generation overrides and hidden extra samplers.
        allowed = {'model','messages','tools','tool_choice','parallel_tool_calls','functions','function_call',
                   'stream','stream_options','temperature','top_p','seed','max_tokens','max_completion_tokens',
                   'chat_template_kwargs','n'}
        if set(payload) - allowed:
            raise ValueError('Unsupported request fields: ' + ', '.join(sorted(set(payload)-allowed)))
        payload.pop('stream_options', None)
        payload.pop('max_completion_tokens', None)
        payload.update(self.settings, model=self.model, stream=False, n=1)
        remaining = self.ledger.remaining_seconds()
        if remaining <= 0:
            raise BudgetExceeded('Active time limit reached')
        started = time.monotonic()
        with httpx.Client(transport=self.transport, timeout=min(remaining, 300), trust_env=False) as client:
            models = client.get(self.base_url + '/models')
            models.raise_for_status()
            if [x['id'] for x in models.json()['data']] != [self.model]:
                raise ValueError('Actual service does not match configured model artifact')
            count = client.post(self.base_url + '/chat/completions/input_tokens', json=payload)
            count.raise_for_status()
            prompt_tokens = count.json()['input_tokens']
            if type(prompt_tokens) is not int or prompt_tokens < 0:
                raise ValueError('Invalid backend token count')
            available_context = self.context_tokens - prompt_tokens
            if available_context < 1:
                raise BudgetExceeded('Context exhausted; no silent history truncation')
            self.ledger.record_tool_time(time.monotonic() - started)
            ticket = self.ledger.reserve_model(payload, prompt_tokens,
                min(self.settings['max_tokens'], available_context))
            payload['max_tokens'] = ticket['max_output_tokens']
            write_json(self.ledger.directory / 'requests' / f"{ticket['id']:05}.json",
                       {'ticket': ticket, 'request': payload})
            started = time.monotonic()
            response = client.post(self.base_url + '/chat/completions', json=payload,
                                   timeout=min(300, self.ledger.remaining_seconds()))
            response.raise_for_status()
            result = response.json()
            self.ledger.complete_model(ticket, result, result.get('usage', {}), time.monotonic()-started)
        choices = result.get('choices', [])
        if len(choices) != 1 or not choices[0].get('finish_reason'):
            raise ValueError('Incomplete or multiple-choice response; recorded but not silently retried')
        return result
