#!/usr/bin/env python3
"""Frozen original C0 summaries and bounded probes, zero learning updates."""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(os.environ['SMA_RUN_DIR']).resolve();P=json.loads((ROOT/'configs/PATHS.json').read_text());OUT=Path(P['remote_data'])/'cache/structure'
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'adapters'),str(ROOT/'code/openpi/src'),str(ROOT/'code/openpi/packages/openpi-client/src')]
import flax.nnx as nnx,jax,jax.numpy as jnp,numpy as np
from torch.utils.data import DataLoader
from openpi.models import model as M
from openpi.training import sharding as S
from adaptive_windows import AdaptivePiWindows
from pi05_data import model_config,collate_numpy
from sma.structure import build_graph as build_core_graph
def build_graph(arrays,visual,out,model,feature_receipt):
 g,report=build_core_graph(arrays,visual)
 np.savez_compressed(Path(out)/('structure_'+model+'.npz'),**g)
 (Path(out)/('STRUCTURE_'+model+'.json')).write_text(json.dumps(report,indent=2))
def atomic(p,x):t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
class TicketBatches:
 def __init__(self,windows,ids):self.windows=windows;self.ids=np.asarray(ids)
 def __iter__(self):
  for start in range(0,len(self.ids),64):
   ids=self.ids[start:start+64].tolist();ids+=ids[-1:]*(64-len(ids))
   yield [(int(self.windows[i,0]),int(self.windows[i,1]),int(i)) for i in ids]
 def __len__(self):return (len(self.ids)+63)//64
def main():
 assert jax.device_count()==4
 if (OUT/'PI_STRUCTURE_COMPLETE.json').exists():print('Immutable C0 structure already exists');return
 started=time.monotonic();arrays=dict(np.load(OUT/'numeric_windows.npz'));windows=arrays['windows'];cfg=model_config(ROOT)
 c=json.loads((ROOT/'configs/PI05.json').read_text());model=cfg.model.load(M.restore_params(Path(c['c0'])/'params',dtype=jnp.bfloat16));model.eval();definition,params=nnx.split(model);del model
 mesh=S.make_mesh(4);rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec());ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(S.DATA_AXIS));ps=S.fsdp_sharding(params,mesh);params=jax.device_put(params,ps)
 dataset=AdaptivePiWindows(ROOT)
 def loader(ids):return DataLoader(dataset,batch_sampler=TicketBatches(windows,ids),num_workers=4,multiprocessing_context='spawn',persistent_workers=False,collate_fn=collate_numpy,timeout=180)
 def obs_batch(x):
  mask=x.pop('valid_action_mask');actions=x.pop('actions');return jax.device_put((M.Observation.from_dict(x),actions,mask),ds)
 def features(parameters,batch):
  model=nnx.merge(definition,parameters);model.eval();obs=M.preprocess_observation(None,batch[0],train=False);tokens,mask,_=model.embed_prefix(obs);nt=obs.tokenized_prompt.shape[1]
  image=(tokens[:,:-nt,:].astype(jnp.float32)*mask[:,:-nt,None]).sum(1)/jnp.maximum(mask[:,:-nt].sum(1,keepdims=True),1)
  language=(tokens[:,-nt:,:].astype(jnp.float32)*mask[:,-nt:,None]).sum(1)/jnp.maximum(mask[:,-nt:].sum(1,keepdims=True),1)
  # Fixed unlearned projection. No new encoder, PCA training, future action or target image.
  projection=jax.random.normal(jax.random.key(7042),(image.shape[-1],32),dtype=jnp.float32)/jnp.sqrt(image.shape[-1])
  return jnp.concatenate([image@projection,language@projection],axis=-1)
 compiled=jax.jit(features,in_shardings=(ps,ds),out_shardings=rep)
 fpath=OUT/'features_pi05.npy';feature_wall=0.
 if not (OUT/'FEATURES_pi05_COMPLETE.json').exists():
  z=np.lib.format.open_memmap(OUT/'features_pi05.tmp.npy',mode='w+',dtype=np.float32,shape=(len(windows),64));cursor=0;t=time.monotonic()
  for x in loader(np.arange(len(windows))):
   if time.time()>float(os.environ['R5_STOP_AT_UNIX'])-120:raise RuntimeError('Allocation cutoff before frozen feature completion; no gradients executed')
   with S.set_mesh(mesh):result=compiled(params,obs_batch(x))
   y=np.asarray(jax.device_get(result));n=min(64,len(windows)-cursor);z[cursor:cursor+n]=y[:n];cursor+=n
   if cursor%4096==0:print(json.dumps(dict(stage='FROZEN_C0_FEATURES',windows=cursor,total=len(windows),wall=time.monotonic()-t)),flush=True)
  z.flush();del z;(OUT/'features_pi05.tmp.npy').replace(fpath);feature_wall=time.monotonic()-t
  atomic(OUT/'FEATURES_pi05_COMPLETE.json',dict(status='COMPLETE',source_C0_fingerprint=str(c['c0']),source_norm_sha256=sha(ROOT/'data_manifests/TRAIN_STATS.json'),windows=len(windows),feature_dim=64,projection='fixed Gaussian key7042, image and masked language mean each32dim',parameters_frozen=True,optimizer_updates=0,future_actions_used_in_features=False,target_appearance_read=False,feature_sha256=sha(fpath),bytes=fpath.stat().st_size,work_seconds=feature_wall,allocated_GPUs=4,preprocess_GPUh=4*feature_wall/3600))
 receipt=json.loads((OUT/'FEATURES_pi05_COMPLETE.json').read_text());z=np.load(fpath,mmap_mode='r')
 if not (OUT/'structure_pi05.npz').exists():build_graph(arrays,z,OUT,'pi05',receipt)
 graph=dict(np.load(OUT/'structure_pi05.npz'));graph['probe_window_ids']=graph['representative_window_ids']
 def score(parameters,batch,key):
  model=nnx.merge(definition,parameters);model.eval();obs,actions,mask=batch
  native,real=model.compute_loss(key,obs,actions,train=False,return_r8_components=True)
  return (real*mask).sum(1)/jnp.maximum(mask.sum(1),1),mask.sum(1)
 scorer=jax.jit(score,in_shardings=(ps,ds,rep),out_shardings=rep)
 for method,g in [('S',graph)]:
  probe=g['probe_window_ids'];assert all(np.sum(arrays['tasks'][probe]==t)<=32 for t in range(1,11))
  values=[];den=[];t=time.monotonic()
  for i,x in enumerate(loader(probe)):
   with S.set_mesh(mesh):value,n=scorer(params,obs_batch(x),jax.random.fold_in(jax.random.key(9081),i))
   values.extend(np.asarray(jax.device_get(value)).tolist());den.extend(np.asarray(jax.device_get(n)).tolist())
  values=np.asarray(values[:len(probe)]);den=np.asarray(den[:len(probe)]);assert np.isfinite(values).all()
  region_probe=g['regions'][probe];initial=np.asarray([values[region_probe==c].mean() for c in range(len(g['region_tasks']))]);assert np.isfinite(initial).all()
  np.savez_compressed(OUT/('sampler_pi05_'+method+'.npz'),**g,initial_loss=initial,probe_scores=values,probe_denominator=den)
  atomic(OUT/('PROBE_pi05_'+method+'.json'),dict(status='COMPLETE',method=method,initial_residual='real7-dimensional native flow velocity MSE, valid-time mask; native main loss unchanged',probe_windows=probe.tolist(),probe_region_ids=region_probe.tolist(),fixed_probe_noise_seed=9081,fixed_batch_positions=True,noise_samples_per_window=1,maximum_windows_per_task=32,score_seconds=time.monotonic()-t,epsilon=max(float(np.median(initial))*.1,1e-12),zero_optimizer_updates=True))
 atomic(OUT/'PI_STRUCTURE_COMPLETE.json',dict(status='COMPLETE',unix=time.time(),total_prepare_work_seconds=time.monotonic()-started,allocated_GPUs=4,preprocessing_GPUh=4*(time.monotonic()-started)/3600,learning_updates=0,source='original official C0',all_train_windows=len(windows),features=receipt,all_artifacts=[dict(path=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in OUT.glob('*pi05*') if f.is_file()]))
 print('Frozen Pi C0 structure/probes complete; zero learning updates',flush=True)
if __name__=='__main__':main()
