"""Run the two minimal framework examples against the same verified model artifact."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from trainingprogram.common import ROOT, data_root, exclusive, write_json
from trainingprogram.evaluation import ensure_config
from trainingprogram.experiment import managed_service
from trainingprogram.models import load_artifact

parser=argparse.ArgumentParser()
parser.add_argument('--id',default='frameworks-4b-v1')
parser.add_argument('--size',choices=['0.8B','2B','4B'],default='4B')
parser.add_argument('--variant',choices=['BF16','Q4_K_M'],default='BF16')
parser.add_argument('--port',type=int,default=28082)
args=parser.parse_args()
from trainingprogram.common import safe_name
folder=data_root()/'diagnostics'/'framework-comparison'/safe_name(args.id)
environments=Path(os.environ.get('TP_FRAMEWORK_ENVS',str(data_root()/'tools/framework-envs')))
_, artifact=load_artifact(args.size,args.variant)
alias='tp-'+artifact['sha256'][:16]
config=dict(size=args.size,variant=args.variant,model=alias,artifact_sha256=artifact['sha256'],
            prompt='Read solution.py and explain the bug in is_prime. Suggest a minimal fix.',
            caveat='Read-only minimal examples, not three equivalent full SWE agents; budgets differ by framework.')
with exclusive(folder/'.lock'):
    ensure_config(folder,config)
    frameworks={'qwen':'qwen_agent_demo.py','deepagents':'deepagents_demo.py'}
    pending=[name for name in frameworks if not (folder/(name+'.json')).exists()]
    if pending:
        with managed_service(args.size,args.variant,args.port):
            for framework in pending:
                python=environments/framework/'bin/python'
                command=[str(python),str(ROOT/'examples'/frameworks[framework]),'--model',alias,
                         '--base-url',f'http://127.0.0.1:{args.port}/v1']
                started=time.monotonic()
                try:
                    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=300)
                    record=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,timeout=False)
                except subprocess.TimeoutExpired as exc:
                    def decoded(value):
                        return value.decode(errors='replace') if isinstance(value,bytes) else value or ''
                    record=dict(returncode=None,stdout=decoded(exc.stdout),stderr=decoded(exc.stderr),timeout=True)
                except OSError as exc:
                    record=dict(returncode=None,stdout='',stderr=str(exc),timeout=False)
                record.update(framework=framework,elapsed_s=time.monotonic()-started,command=command,
                              model=alias,real_model_service=True)
                write_json(folder/(framework+'.json'),record)
                print(framework,record['returncode'],record['timeout'],flush=True)
