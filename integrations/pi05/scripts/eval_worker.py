"""One process per simulator window; use the same paired plan for both methods."""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE),str(HERE/'adapters')]
from evaluation_contract import read_plan


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


async def work(args):
    from r5_benchmark import R5Benchmark
    from vla_eval.connection import Connection
    from vla_eval.runners.sync_runner import SyncEpisodeRunner
    entries=read_plan(args.plan);out=Path(args.output)
    benchmark=R5Benchmark(args.assets);runner=SyncEpisodeRunner()
    try:
        for index,entry in enumerate(entries):
            if index%args.workers!=args.worker: continue
            if (out/'STOP').exists(): break
            eid=entry['episode_id'];result_path=out/'episodes'/(eid+'.json')
            lock=out/'locks'/(eid+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
            with lock.open('a+') as handle:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if result_path.exists():
                    old=json.loads(result_path.read_text())
                    if old['checkpoint_id']!=args.checkpoint_id: raise ValueError('Mixed checkpoint.')
                    if old.get('valid'): continue
                attempts=out/'attempts'/(eid+'.jsonl');attempts.parent.mkdir(parents=True,exist_ok=True)
                prior=[json.loads(s) for s in attempts.read_text().splitlines()] if attempts.exists() else []
                number=sum(x.get('event')=='START' for x in prior)+1
                if number>3: raise RuntimeError('Three attempts exhausted for '+eid)
                base=dict(entry,checkpoint_id=args.checkpoint_id,attempt=number,start_unix=time.time())
                with attempts.open('a') as f:
                    f.write(json.dumps(dict(base,event='START'))+'\n');f.flush();os.fsync(f.fileno())
                conn=Connection(args.url,timeout=300,max_retries=1);begin=time.monotonic()
                benchmark.save_frames=entry['slot']<2
                try:
                    await conn.connect(benchmark='sma_paired_3x10')
                    result=await asyncio.wait_for(runner.run_episode(benchmark,dict(entry,name=entry['instruction']),conn,max_steps=520),timeout=1800)
                    metrics=result['metrics']
                    row=dict(base,event='END',valid=True,success=bool(metrics['success']),steps=result['steps'],
                             termination=metrics['termination'],metrics=metrics,wall_seconds=time.monotonic()-begin)
                    if benchmark.save_frames and benchmark.frames:
                        import imageio.v2 as imageio
                        video=out/'videos'/f'{eid}__{number}.mp4';video.parent.mkdir(parents=True,exist_ok=True)
                        try:
                            imageio.mimsave(video,benchmark.frames,fps=20,macro_block_size=1,quality=6);row['video']=str(video)
                        except Exception as exc: row['video_error']=repr(exc)
                    atomic(result_path,row)
                    with attempts.open('a') as f: f.write(json.dumps(row)+'\n')
                    print(json.dumps({k:row[k] for k in ['episode_id','success','steps','wall_seconds']}),flush=True)
                except Exception as exc:
                    row=dict(base,event='END',valid=False,success=None,steps=benchmark.steps,error=repr(exc),traceback=traceback.format_exc())
                    atomic(result_path,row)
                    with attempts.open('a') as f: f.write(json.dumps(row)+'\n')
                    (out/'STOP').touch()
                    raise
                finally:
                    await conn.close();benchmark.frames=[]
    finally: benchmark.cleanup()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','plan','url','output','checkpoint-id']:p.add_argument('--'+key,required=True)
    p.add_argument('--workers',type=int,required=True);p.add_argument('--worker',type=int,required=True)
    asyncio.run(work(p.parse_args()))
