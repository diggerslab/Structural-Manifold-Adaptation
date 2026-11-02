"""Concurrent simulation against one Pi0.5 service; no GPU allocation or SSH."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from evaluation_contract import read_plan,bind_output,summarize


def main():
    p=argparse.ArgumentParser()
    for key in ['assets','plan','output','checkpoint-id','url']:p.add_argument('--'+key,required=True)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--sim-python',default=sys.executable)
    p.add_argument('--summary-only',action='store_true')
    a=p.parse_args()
    if a.workers<1:raise ValueError('Positive worker count required.')
    entries=read_plan(a.plan);out=Path(a.output).resolve()
    bind_output(out,a.plan,a.checkpoint_id,a.assets)
    processes=[];handles=[]
    try:
        if not a.summary_only:
            if (out/'STOP').exists():raise RuntimeError('Inspect the prior error; remove only this run STOP marker before retry.')
            (out/'logs').mkdir(exist_ok=True)
            env=dict(os.environ,MUJOCO_GL='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            for i in range(a.workers):
                log=(out/'logs'/f'worker-{i}.log').open('a');handles.append(log)
                command=[a.sim_python,str(Path(__file__).parent/'scripts/eval_worker.py')]
                for key in ['assets','plan','url','checkpoint_id']:
                    command.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
                command.extend(['--output',str(out),'--workers',str(a.workers),'--worker',str(i)])
                processes.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT))
            while any(child.poll() is None for child in processes):
                if any(child.poll() not in [None,0] for child in processes):
                    (out/'STOP').touch()
                    raise RuntimeError('Simulator error; preserved completed cases. Inspect worker logs.')
                time.sleep(2)
            if any(child.returncode!=0 for child in processes):raise RuntimeError('Simulator worker failed.')
    finally:
        for child in processes:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        for handle in handles:handle.close()
        summary=summarize(entries,out,a.checkpoint_id)
        (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
