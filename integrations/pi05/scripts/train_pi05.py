#!/usr/bin/env python3
"""R8 native full-parameter JAX/FSDP4, scalar time-LR, explicit 20-minute fit clock."""
import dataclasses,functools,hashlib,importlib.util,json,os,shutil,signal,sys,time
from pathlib import Path
ROOT=Path(os.environ['SMA_RUN_DIR']).resolve()
ADAPTERS=Path(__file__).resolve().parents[1]/'adapters'
sys.path[:0]=[str(ROOT/'code/openpi/src'),str(ROOT/'code/openpi/packages/openpi-client/src'),str(ADAPTERS)]
import flax.nnx as nnx
import jax,jax.numpy as jnp,numpy as np,optax,orbax.checkpoint as ocp
from torch.utils.data import DataLoader
from openpi.models import model as M
from openpi.shared import array_typing as AT
from openpi.training import checkpoints as C,sharding as S,weight_loaders as W
from pi05_data import model_config,data_config,collate_numpy
from hdf5_data import RankBatchSampler
from balanced_windows import TimedPiWindows,Consumption
from fit_clock import FitClock
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def link_or_copy(src,dst):
 try:os.link(src,dst)
 except OSError as error:
  if error.errno!=18:raise
  shutil.copy2(src,dst)
 return dst
def hashes(p):
 result=[]
 for f in sorted(Path(p).rglob('*')):
  if f.is_file():
   h=hashlib.sha256()
   with f.open('rb') as s:
    for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
   result.append(dict(path=str(f.relative_to(p)),bytes=f.stat().st_size,sha256=h.hexdigest()))
 return result
@dataclasses.dataclass(frozen=True)
class UnitSchedule:
 def create(self):return lambda step:jnp.array(1.,jnp.float32)
def step_fn(cfg,rng,state,batch,lr):
 model=nnx.merge(state.model_def,state.params);model.train();obs,actions,mask=batch
 def loss_fn(model):
  native,score=model.compute_loss(jax.random.fold_in(rng,state.step),obs,actions,train=True,return_r8_components=True)
  loss=jnp.sum(native*mask)/jnp.maximum(mask.sum(),1)
  per_sample=jnp.sum(score*mask,axis=1)/jnp.maximum(mask.sum(axis=1),1)
  # Main loss remains the native all-padded-dim flow objective; scoring uses real7.
  reconstructed=jnp.sum(jnp.sum(native*mask,axis=1))/jnp.maximum(mask.sum(),1)
  return loss,(per_sample,mask.sum(axis=1),jnp.abs(loss-reconstructed))
 (loss,aux),grads=nnx.value_and_grad(loss_fn,argnums=nnx.DiffState(0,cfg.trainable_filter),has_aux=True)(model)
 params=state.params.filter(cfg.trainable_filter);updates,op=state.tx.update(grads,state.opt_state,params)
 updates=jax.tree.map(lambda x:x*lr,updates);new=optax.apply_updates(params,updates);nnx.update(model,new)
 norms={}
 for path,v in grads.flat_state().items():
  name='/'.join(map(str,path));group='vision' if '/img/' in '/'+name+'/' else ('action_expert' if '_1' in name else 'language') if 'llm' in name else 'projection_state'
  norms[group]=norms.get(group,0.)+jnp.sum(v.value.astype(jnp.float32)**2)
 change=optax.global_norm(jax.tree.map(lambda a,b:a-b,new,params))
 info=dict(loss=loss,gradient_norm=optax.global_norm(grads),parameter_delta_norm=change,per_sample_score=aux[0],valid_denominator=aux[1],aggregate_error=aux[2],lr=lr,**{'gradient_'+k:jnp.sqrt(v) for k,v in norms.items()})
 return dataclasses.replace(state,step=state.step+1,params=nnx.state(model),opt_state=op),info
def main():
 c=json.loads((ROOT/'configs/CHAIN.json').read_text());out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
 assert c['method'] in ['U_A','U_B','U'] and c['G']==64 and c['initialization']=='ORIGINAL_OFFICIAL_C0'
 assert not (out/'TRAINING_COMPLETE.json').exists(),'Already finished; no duplicate training'
 clock=FitClock(out,c['save_reserve_seconds']);stop=Path(os.environ['R5_STOP_FILE']);job=os.environ.get('SLURM_JOB_ID','local');start_wall=time.time()
 for sig in [signal.SIGUSR1,signal.SIGTERM]:signal.signal(sig,lambda *_:stop.touch())
 cfg=dataclasses.replace(model_config(ROOT),name='sma_pi05',freeze_filter=nnx.Nothing,lr_schedule=UnitSchedule(),weight_loader=W.CheckpointWeightLoader(c['c0']+'/params'))
 assert jax.device_count()==4
 jax.config.update('jax_compilation_cache_dir',str(out/'jax_cache'))
 mesh=S.make_mesh(4);ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(S.DATA_AXIS));rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
 train_rng,init_rng=jax.random.split(jax.random.key(c['seed']))
 sp=importlib.util.spec_from_file_location('r8_native_train',ROOT/'code/openpi/scripts/train.py');native=importlib.util.module_from_spec(sp);sp.loader.exec_module(native)
 resume_root=Path(c.get('resume_output',str(Path(os.environ['R8_RESUME_BASE'])/'pi05'/out.name) if os.environ.get('R8_RESUME_BASE') else str(out/'resume')));resume_root.mkdir(parents=True,exist_ok=True)
 atomic(out/'FULL_RESUME_LOCATION.json',dict(path=str(resume_root),primary_model_output=str(out),cross_filesystem_copy_included_in_T_fit=True))
 manager=ocp.CheckpointManager(resume_root,item_handlers={'assets':C.CallbackHandler(),'train_state':ocp.PyTreeCheckpointHandler(),'params':ocp.PyTreeCheckpointHandler()},options=ocp.CheckpointManagerOptions(max_to_keep=1,create=True,async_options=ocp.AsyncOptions(timeout_secs=600)))
 resuming=manager.latest_step() is not None
 if resuming:raise RuntimeError('Explicit recovery audit required before resuming interrupted short chain; no clock reset')
 state,ss=native.init_train_state(cfg,init_rng,mesh,resume=False);jax.block_until_ready(state)
 class DataRef:
  def data_config(self):return data_config(ROOT)
 dataset=TimedPiWindows(ROOT);consumption=Consumption(dataset,10);ref=DataRef()
 loader=DataLoader(dataset,batch_sampler=RankBatchSampler(0,10000,0,1,64),num_workers=4,multiprocessing_context='spawn',persistent_workers=True,collate_fn=collate_numpy,timeout=300)
 def batch(x):
  mask=x.pop('valid_action_mask');actions=x.pop('actions');return jax.device_put((M.Observation.from_dict(x),actions,mask),ds)
 def event(kind,**kw):
  row=dict(event=kind,unix=time.time(),T_fit=clock.elapsed(),model='pi05',method=c['method'],job_id=job,**kw)
  with (out/'TRAIN_TIMING.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
  print(json.dumps(row),flush=True)
 selected={p for p,_ in state.params.filter(cfg.trainable_filter).flat_state().items()}
 atomic(out/'trainable_scope.json',dict(parameters=[dict(path='/'.join(map(str,p)),shape=list(v.value.shape),dtype=str(v.value.dtype),optimizer_member=p in selected) for p,v in state.params.flat_state().items()],trainable_scalars=sum(int(np.prod(v.value.shape)) for p,v in state.params.flat_state().items()),native_full_parameter_filter=True,no_lora=True))
 # AOT lower/compile only: not a single calibration update is executed or retained.
 compile_batch=batch(collate_numpy([dataset[i] for i in range(64)]));step=jax.jit(functools.partial(step_fn,cfg),in_shardings=(rep,ss,ds,rep),out_shardings=(ss,rep),donate_argnums=(1,))
 tick=time.monotonic()
 with S.set_mesh(mesh),AT.disable_typechecking():compiled=step.lower(train_rng,state,compile_batch,jnp.array(c['peak_lr'],jnp.float32)).compile()
 compile_wall=time.monotonic()-tick
 def compare_native(st,b):
  model=nnx.merge(st.model_def,st.params);model.train();obs,actions,mask=b
  key=jax.random.fold_in(train_rng,0)
  old=model.compute_loss(key,obs,actions,train=True)
  new,real=model.compute_loss(key,obs,actions,train=True,return_r8_components=True)
  return jnp.max(jnp.abs(old-new)),jnp.sum(old*mask)/mask.sum(),jnp.sum(real*mask)/mask.sum()
 with S.set_mesh(mesh):
  parity=jax.jit(compare_native,in_shardings=(ss,ds),out_shardings=rep)(state,compile_batch)
 parity=np.asarray(jax.device_get(parity));assert np.isfinite(parity).all() and parity[0]<=1e-7
 atomic(out/'NATIVE_LOSS_PARITY.json',dict(status='PASS',max_element_difference=float(parity[0]),native_scalar=float(parity[1]),auxiliary_real7_scalar=float(parity[2]),retained_updates=0))
 del compile_batch
 # Reset the finite uniform data cursor; compiling did not modify model/optimizer/RNG.
 assert int(state.step)==0
 atomic(out/'PREPARE_RECEIPT.json',dict(source=c['c0'],model_state_step=int(state.step),optimizer_updates_executed=0,AOT_compile_only=True,compile_seconds=compile_wall,full_C0_load=True,source_config_sha256=hashlib.sha256((ROOT/'configs/CHAIN.json').read_bytes()).hexdigest(),uniform_schedule_sha256=hashlib.sha256(Path(dataset.schedule.values.filename).read_bytes()).hexdigest(),no_target_data=True))
 # Compile/parity opened parent HDF5 handles and tokenizer objects. Spawn workers from a clean reader only.
 for file in dataset.files.values():file.close()
 dataset.files.clear();dataset.transform=None
 clock.start();event('FIT_START',T_prepare=clock.prepare_seconds,compile_wall=compile_wall);it=iter(loader)
 n=0;last_saved=-1;loop_wall=0.;first_nonzero_checked=False;consumed=out/'SAMPLE_CONSUMPTION.jsonl'
 def save(seconds=None):
  nonlocal last_saved
  assert shutil.disk_usage(out).free>16*1024**3 and shutil.disk_usage(resume_root).free>40*1024**3,'Atomic model/resume storage guard on each R8 filesystem'
  tick=time.monotonic();jax.block_until_ready(state);C.save_state(manager,state,ref,n);manager.wait_until_finished();directory=resume_root/str(n)
  assert directory.is_dir();atomic(directory/'R8_SAMPLER_STATE.json',dict(seed=c['seed'],cursor=n*64,successful_updates=n,probability_version=0,sampler='frozen source schedule',train_rng='jax seed split folded by successful step',consumption=consumption.serializable(),clock=clock.state()))
  if seconds:
   dest=out/('model_%dmin'%(seconds//60));tmp=out/(dest.name+'.tmp');tmp.mkdir(exist_ok=True)
   for key in ['params','assets']:shutil.copytree(directory/key,tmp/key,copy_function=link_or_copy,dirs_exist_ok=True)
   shutil.copy2(ROOT/'configs/CHAIN.json',tmp/'CONFIG.json');files=hashes(tmp);tmp.replace(dest)
   row=clock.commit(seconds,n,time.monotonic()-tick);row.update(files=files,G=64,seed=c['seed'],method=c['method'],method_version=1,consumption=consumption.summary(),full_inference_weights=True,T_ready_to_checkpoint=time.time()-start_wall)
   atomic(dest/'COMPLETE.json',row);row['T_fit']=clock.elapsed();row['budget_status']='ON_TIME' if row['T_fit']<=seconds else 'TIME_BUDGET_MISS';atomic(dest/'COMPLETE.json',row);clock.complete[seconds]=row
  atomic(out/'FIT_RESUME.json',clock.state());atomic(out/'CONSUMPTION_RESUME.json',consumption.serializable());atomic(out/'DATA_CONSUMPTION.json',consumption.summary());last_saved=n
  event('CHECKPOINT',node_seconds=seconds,successful_updates=n,checkpoint_write_seconds=time.monotonic()-tick)
 while not clock.done():
  if stop.exists() or time.time()>=float(os.environ['R5_STOP_AT_UNIX']):
   if n and n!=last_saved:save()
   break
  due=clock.due()
  if due:
   if n:save(due)
   else:
    clock.complete[due]=dict(status='MISSING');atomic(out/('MISSING_%d.json'%due),dict(status='MISSING',T_fit=clock.elapsed()))
   continue
  tick=time.monotonic();lr=clock.lr(c['peak_lr']);b=batch(next(it))
  with S.set_mesh(mesh):state,info=compiled(train_rng,state,b,jnp.array(lr,jnp.float32))
  vals=jax.device_get(info);jax.block_until_ready(state);assert all(np.isfinite(v).all() for v in vals.values()),'nonfinite loss/gradient/update'
  assert float(vals['aggregate_error'])<1e-5,'per-sample aggregate mismatch'
  if lr>0 and not first_nonzero_checked:
   for name in ['gradient_vision','gradient_language','gradient_action_expert','gradient_projection_state','parameter_delta_norm']:assert float(vals[name])>0,name
   atomic(out/'FIRST_NONZERO_UPDATE_CHECK.json',{k:float(v) for k,v in vals.items() if np.ndim(v)==0});first_nonzero_checked=True
  rows=dataset.schedule.batch(n)
  with consumed.open('a') as f:
   f.write(json.dumps(dict(step=n+1,probability_version=0,rows=rows.tolist(),per_sample_score=np.asarray(vals['per_sample_score']).tolist(),valid_denominator=np.asarray(vals['valid_denominator']).tolist(),method=c['method'],sampling='task trajectory uniform fixed global64 schedule',T_fit=clock.elapsed()))+'\n')
  consumption.add(n);n=int(state.step);dt=time.monotonic()-tick;clock.last_step=dt;loop_wall+=dt
  if n==1 or n%25==0:
   row=dict(status='TRAINING',successful_updates=n,T_fit=clock.elapsed(),T_prepare=clock.prepare_seconds,seconds_per_update=dt,optimizer_window_wall=loop_wall,lr=lr,loss=float(vals['loss']),gradient_norm=float(vals['gradient_norm']),G=64,windows=n*64)
   atomic(out/'PROGRESS.json',row);event('PROGRESS',**{k:v for k,v in row.items() if k!='T_fit'})
 complete=all(k in clock.complete for k in clock.nodes);manager.close();jax.block_until_ready(state)
 atomic(out/('TRAINING_COMPLETE.json' if complete else 'SEGMENT_EXIT.json'),dict(status='COMPLETE' if complete else 'PARTIAL',successful_updates=n,T_fit=clock.elapsed(),T_prepare=clock.prepare_seconds,T_ready_to_checkpoint=time.time()-start_wall,fit_GPUh=4*clock.elapsed()/3600,nodes=clock.complete,G=64,seed=c['seed'],extra_calibration_updates=0))
 event('FIT_END',successful_updates=n,completed=complete)
if __name__=='__main__':main()
