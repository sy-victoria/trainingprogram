"""Loopback OpenAI-compatible adapter for native framework clients.

Responses are fully persisted before delivery. SSE is buffered compatibility
output, not a measurement of model TTFT. Benchmark speed uses its separate path.
"""
import contextlib
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from .agent_ledger import BudgetExceeded, UncertainRequest


@contextlib.contextmanager
def model_gateway(backend):
    key = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, code, payload):
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def authorized(self):
            if self.headers.get('Authorization') != 'Bearer ' + key:
                self.send_json(401, {'error': {'message': 'Unauthorized', 'type': 'authentication_error'}})
                return False
            return True

        def do_GET(self):
            if not self.authorized(): return
            if self.path != '/v1/models':
                self.send_json(404, {'error': {'message': 'Unknown route'}})
                return
            self.send_json(200, {'object': 'list', 'data': [{'id': backend.model, 'object': 'model'}]})

        def do_POST(self):
            if not self.authorized(): return
            if self.path != '/v1/chat/completions':
                self.send_json(404, {'error': {'message': 'Unknown route'}})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 2_000_000:
                    raise ValueError('Invalid request size')
                request = json.loads(self.rfile.read(size))
                result = backend.chat(request)
            except (BudgetExceeded, UncertainRequest) as exc:
                self.send_json(402, {'error': {'message': str(exc), 'type': 'budget_or_recovery_required'}})
                return
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(400, {'error': {'message': str(exc), 'type': 'invalid_request_error'}})
                return
            except Exception as exc:
                self.send_json(502, {'error': {'message': type(exc).__name__, 'type': 'upstream_error'}})
                return
            if not request.get('stream'):
                self.send_json(200, result)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            choice = result['choices'][0]
            delta = dict(choice['message'])
            if delta.get('tool_calls'):
                delta['tool_calls'] = [dict(call, index=i) for i, call in enumerate(delta['tool_calls'])]
            meta = {k: result[k] for k in ('id', 'created', 'model') if k in result}
            meta['object'] = 'chat.completion.chunk'
            chunks = [dict(meta, choices=[{'index': 0, 'delta': delta, 'finish_reason': None}]),
                      dict(meta, choices=[{'index': 0, 'delta': {}, 'finish_reason': choice['finish_reason']}]),
                      dict(meta, choices=[], usage=result.get('usage', {}))]
            for chunk in chunks:
                self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {'base_url': f'http://127.0.0.1:{server.server_port}/v1', 'api_key': key}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
