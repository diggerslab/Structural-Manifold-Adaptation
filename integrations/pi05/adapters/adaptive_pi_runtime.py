"""Native Pi data/score adapter for the shared external allocation mechanism."""
import json,time
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from adaptive_windows import AdaptivePiWindows,GlobalTicketSampler
from sma import SMASampler as AllocationSampler
from pi05_data import collate_numpy
from consumption import ActualConsumption
class AdaptiveRuntime:
 def __init__(self,root,cfg):
  self.root=Path(root);self.cfg=cfg;self.out=Path(cfg['output']);p=json.loads((Path(cfg['r8_root'])/'configs/PATHS.json').read_text());cache=Path(p['remote_data'])/'cache/structure'
  source='S' if cfg['method'] in ['C','S2'] else cfg['method'];g=dict(np.load(cache/('sampler_pi05_'+source+'.npz')))
  self.graph=g;self.dataset=AdaptivePiWindows(root);self.sampler=AllocationSampler(self.dataset.episodes,g['windows'],g['regions'],g['region_tasks'],g['K'],g['initial_loss'],cfg['method'],cfg['seed'],priority_eligible=g.get('priority_eligible'));self.tickets=GlobalTicketSampler(self.sampler)
  self.loader=DataLoader(self.dataset,batch_sampler=self.tickets,num_workers=4,multiprocessing_context='spawn',persistent_workers=True,collate_fn=collate_numpy,timeout=300,prefetch_factor=2)
  self.consumption=ActualConsumption(self.dataset);self.next_calibration=60.;self.probe_cursor=0;self.sampling_work=0.;self.scoring_work=0.
  self.write_distribution(self.sampler.update_sampler(0.))
 def write_distribution(self,r):
  with (self.out/'SAMPLER_DISTRIBUTIONS.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 def compile_batch(self):
  ids=self.graph['probe_window_ids'][:64].tolist();ids+=ids[-1:]*(64-len(ids));return collate_numpy([self.dataset[(int(self.graph['windows'][i,0]),int(self.graph['windows'][i,1]),int(i))] for i in ids])
 def observed(self,step,vals,fit_seconds):
  t=time.monotonic();d=self.tickets.finish(step,np.asarray(vals['per_sample_score']),np.asarray(vals['valid_denominator']),fit_seconds);self.consumption.add_draw(d);self.sampling_work+=time.monotonic()-t;return d
 def maybe_calibrate(self,clock,score_callback):
  if clock.elapsed()<self.next_calibration:return
  next_node=next((n for n in clock.nodes if n not in clock.complete),1200)
  if clock.elapsed()+clock.reserve+8>=next_node:return
  begin=time.monotonic()
  if self.cfg['method']!='C':
   probe=self.graph['probe_window_ids'];chunks=(len(probe)+63)//64;batch_id=self.probe_cursor%chunks;ids=probe[batch_id*64:(batch_id+1)*64].tolist();valid=len(ids);ids+=ids[-1:]*(64-valid)
   raw=collate_numpy([self.dataset[(int(self.graph['windows'][i,0]),int(self.graph['windows'][i,1]),int(i))] for i in ids])
   scores,den=score_callback(raw,batch_id)
   d=dict(window_ids=ids[:valid],region_ids=self.graph['regions'][ids[:valid]].tolist());self.sampler.observe(d,np.asarray(scores)[:valid],np.asarray(den)[:valid],clock.elapsed(),probe=True);self.probe_cursor+=1
   with (self.out/'PROBE_EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(T_fit=clock.elapsed(),batch_id=batch_id,window_ids=ids[:valid],probability_version_before=self.sampler.version,extra_forward_windows=64,valid_probe_windows=valid,scores=np.asarray(scores)[:valid].tolist()))+'\n')
  self.scoring_work+=time.monotonic()-begin;t=time.monotonic();self.write_distribution(self.sampler.update_sampler(clock.elapsed()));self.sampling_work+=time.monotonic()-t
  self.next_calibration=(int(clock.elapsed()//60)+1)*60
 def state_dict(self):return dict(tickets=self.tickets.state_dict(),next_calibration=self.next_calibration,probe_cursor=self.probe_cursor,sampling_seconds=self.sampling_work+self.tickets.draw_seconds,scoring_seconds=self.scoring_work,consumption=self.consumption.serializable())
