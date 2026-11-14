import argparse
import time
import httpx
import psutil

parser = argparse.ArgumentParser()
parser.add_argument('--pid', type=int, required=True)
args = parser.parse_args()
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    if not psutil.pid_exists(args.pid):
        raise SystemExit('Model service exited; inspect services/8080.log')
    try:
        r = httpx.get('http://127.0.0.1:8080/health', timeout=2, trust_env=False)
        if r.status_code == 200:
            break
    except httpx.HTTPError:
        pass
    time.sleep(2)
else:
    raise SystemExit('Model service did not become ready within 300 seconds')
