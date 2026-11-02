"""Framework-independent counts of actual consumed TRAIN windows."""
class ActualConsumption:
 def __init__(self,dataset,horizon=10):
  self.dataset=dataset;self.horizon=horizon;self.updates=0;self.windows=0;self.valid_steps=0;self.by_task={};self.by_episode={};self.unique=set();self.parents=set()
 def add_draw(self,d):
  for eid,start in d['episode_start']:
   e=self.dataset.episodes[eid];key='kitchen:task%02d'%e['task_id'];self.by_task[key]=self.by_task.get(key,0)+1;self.by_episode[str(eid)]=self.by_episode.get(str(eid),0)+1;self.parents.add(e['episode_group_id']);self.unique.add((eid,start));self.valid_steps+=min(self.horizon,e['length']-start)
  self.updates+=1;self.windows+=len(d['episode_start'])
 def summary(self):return dict(successful_updates=self.updates,windows=self.windows,valid_action_steps=self.valid_steps,by_cell=self.by_task,by_episode=self.by_episode,unique_episode_start_windows=len(self.unique),unique_source_groups=len(self.parents))
 def serializable(self):return dict(self.summary(),unique_windows=sorted(self.unique),source_groups=sorted(self.parents))
