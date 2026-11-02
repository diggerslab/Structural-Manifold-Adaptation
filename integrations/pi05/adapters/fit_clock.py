"""Two explicit clocks; synchronous completed checkpoint defines each node."""
import json,math,time
from pathlib import Path
class FitClock:
 def __init__(self,out,reserve=65):
  self.out=Path(out);self.prepare_start=time.monotonic();self.fit_start=None;self.prior=0.;self.nodes=[300,600,900,1200];self.complete={};self.reserve=float(reserve);self.last_step=2.
  p=self.out/'FIT_RESUME.json'
  if p.exists():
   s=json.loads(p.read_text());self.prior=s['T_fit'];self.complete={int(k):v for k,v in s['nodes'].items()};self.reserve=max(self.reserve,s['reserve'])
 def start(self):self.prepare_seconds=time.monotonic()-self.prepare_start;self.fit_start=time.monotonic()
 def elapsed(self):return self.prior+(time.monotonic()-self.fit_start if self.fit_start else 0.)
 def lr(self,peak):
  t=self.elapsed()
  if t<30:return peak*t/30
  x=min(1,max(0,(t-30)/1170));return peak*(.2+.8*.5*(1+math.cos(math.pi*x)))
 def due(self):
  for n in self.nodes:
   if n not in self.complete:return n if self.elapsed()+self.reserve+self.last_step>=n else None
  return None
 def done(self):return all(n in self.complete for n in self.nodes) or self.elapsed()>=1200
 def commit(self,budget,step,save_seconds):
  t=self.elapsed();row=dict(budget_seconds=budget,T_fit=t,successful_updates=step,step=step,budget_status='ON_TIME' if t<=budget else 'TIME_BUDGET_MISS',T_prepare=self.prepare_seconds,checkpoint_write_seconds=save_seconds)
  self.complete[budget]=row;self.reserve=max(self.reserve,save_seconds*1.3+5);return row
 def state(self):return dict(T_fit=self.elapsed(),nodes=self.complete,reserve=self.reserve)
