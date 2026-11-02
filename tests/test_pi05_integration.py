import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from sma import SMASampler,build_graph
from sma.demo import synthetic_dataset

ROOT=Path(__file__).resolve().parents[1]
PI=ROOT/'integrations/pi05'


def module(name):
    spec=importlib.util.spec_from_file_location('pi_test_'+name,PI/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def selected_class(file,name,namespace):
    tree=ast.parse((PI/file).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(file),'exec'),namespace)
    return namespace[name]


class PiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)

    def sampler(self):
        episodes,arrays,features=synthetic_dataset()
        graph,_=build_graph(arrays,features)
        sampler=SMASampler(episodes,graph['windows'],graph['regions'],graph['region_tasks'],graph['K'],np.ones(len(graph['region_tasks'])),priority_eligible=graph['priority_eligible'])
        return sampler,graph

    def test_native_update_equal_between_arms(self):
        functions=[]
        for file in ['train_pi05.py','train_pi05_adaptive.py']:
            tree=ast.parse((PI/'scripts'/file).read_text())
            functions.append(ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='step_fn')))
        self.assertEqual(*functions)
        cfg=json.loads((PI/'configs/common.json').read_text())
        self.assertEqual(cfg['peak_lr'],2.5e-5)
        self.assertEqual(cfg['G'],64)
        self.assertEqual(cfg['maximum_T_fit'],1200)

    def test_pending_prefetch_is_replayed(self):
        sampler,graph=self.sampler()
        cls=selected_class('adapters/adaptive_windows.py','GlobalTicketSampler',{'time':time,'np':np})
        tickets=cls(sampler)
        first=tickets.draw(0);second=tickets.draw(1);third=tickets.draw(2)
        tickets.finish(0,np.ones(64),np.full(64,10),1.)
        saved=json.loads(json.dumps(tickets.state_dict()))
        fresh,_=self.sampler();restored=cls(fresh,state=saved)
        self.assertEqual(restored.draw(1),second);self.assertEqual(restored.draw(2),third)
        for obj in [tickets,restored]:
            obj.finish(1,np.ones(64),np.full(64,10),2.)
            obj.finish(2,np.ones(64),np.full(64,10),3.)
        self.assertEqual(restored.draw(3),tickets.draw(3))

    def test_minute_probe_is_compatible_with_public_sampler(self):
        sampler,graph=self.sampler();graph['probe_window_ids']=graph['representative_window_ids']
        cls=selected_class('adapters/adaptive_pi_runtime.py','AdaptiveRuntime',
                           {'time':time,'np':np,'json':json,'collate_numpy':lambda x:x})
        runtime=cls.__new__(cls);runtime.sampler=sampler;runtime.graph=graph
        runtime.out=self.out;runtime.cfg={'method':'S2'};runtime.dataset={}
        runtime.dataset=type('Dataset',(),{'__getitem__':lambda self,index:index})()
        runtime.probe_cursor=0;runtime.scoring_work=runtime.sampling_work=0.;runtime.next_calibration=60.
        clock=type('Clock',(),{'elapsed':lambda self:61.,'nodes':[300,600,900,1200],'complete':{},'reserve':65.})()
        runtime.maybe_calibrate(clock,lambda rows,batch_id:(np.ones(len(rows)),np.full(len(rows),10)))
        self.assertEqual(runtime.probe_cursor,1)
        self.assertGreater(sampler.observations.sum(),0)
        self.assertEqual(sampler.counts.sum(),0) # probe is not a training-consumption update
        self.assertEqual(runtime.next_calibration,120.)

    def test_fit_clock_warmup_save_and_stop(self):
        cls=selected_class('adapters/fit_clock.py','FitClock',{'Path':Path,'json':json,'time':time,'math':__import__('math')})
        with patch('time.monotonic',return_value=100.):clock=cls(self.out)
        with patch('time.monotonic',return_value=150.):clock.start()
        with patch('time.monotonic',return_value=165.):self.assertAlmostEqual(clock.lr(2.5e-5),1.25e-5)
        with patch('time.monotonic',return_value=180.):self.assertAlmostEqual(clock.lr(2.5e-5),2.5e-5)
        with patch('time.monotonic',return_value=390.):self.assertEqual(clock.due(),300)
        with patch('time.monotonic',return_value=1350.):
            self.assertTrue(clock.done());self.assertAlmostEqual(clock.lr(2.5e-5),.2*2.5e-5)

    def test_balanced_schedule_and_train_only(self):
        prep=module('prepare_data')
        episodes=[dict(task_id=t,cell_id=t-1,room='kitchen',split='train',hdf5='x.h5',episode='demo_0',length=10,episode_group_id=str(t),instruction='task') for t in range(1,11)]
        prep.validate_episodes(episodes)
        a=prep.uniform_schedule(episodes,42,steps=5);b=prep.uniform_schedule(episodes,42,steps=5)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(320,3))
        counts=np.bincount(a[:,1],minlength=10)
        self.assertLessEqual(counts.max()-counts.min(),1)
        episodes[0]['room']='office'
        with self.assertRaises(ValueError):prep.validate_episodes(episodes)

    def test_real_hdf5_adapter_and_temporal_padding(self):
        import h5py
        prep=module('prepare_data');episodes=[]
        with h5py.File(self.out/'data.h5','w') as f:
            for t in range(1,11):
                d=f.create_group(f'data/demo_{t}')
                d.create_dataset('actions',data=np.full((3,7),t,np.float32))
                d.create_dataset('obs/ee_states',data=np.zeros((3,6),np.float32))
                d.create_dataset('obs/gripper_states',data=np.zeros((3,2),np.float32))
                for name in ['agentview_rgb','eye_in_hand_rgb']:d.create_dataset('obs/'+name,data=np.zeros((3,8,8,3),np.uint8))
                episodes.append(dict(task_id=t,cell_id=t-1,room='kitchen',split='train',hdf5='data.h5',episode=f'demo_{t}',length=3,episode_group_id=str(t),instruction='task'))
        p=self.out/'episodes.json';p.write_text(json.dumps(episodes))
        schedule=np.tile(np.array([[t-1,t-1,2] for t in range(1,11)],np.int32),(64000,1))
        with patch.object(prep,'uniform_schedule',return_value=schedule):prep.prepare(p,self.out/'prepared')
        raw=selected_class('adapters/hdf5_data.py','RawWindows',{'Path':Path,'json':json,'np':np,'h5py':h5py,'OrderedDict':__import__('collections').OrderedDict})
        run=self.out/'run';run.mkdir();(run/'data_manifests').mkdir()
        import shutil
        for source in (self.out/'prepared').iterdir():shutil.copy2(source,run/'data_manifests'/source.name)
        ds=raw(run,10)
        try:
            result=ds.get(0)
            self.assertEqual(result['state'].shape,(8,));self.assertEqual(result['actions'].shape,(10,7))
            self.assertEqual(result['action_mask'].sum(),1)
            self.assertTrue(np.all(result['actions']==1))
        finally:
            for handle in ds.files.values():handle.close()
            del ds.schedule

    def test_evaluation_denominators_and_identity(self):
        ev=module('evaluation_contract')
        entries=[dict(episode_id='case'+str(i),room='kitchen',task_id=i+1,instruction='task',environment_seed=42,policy_seed=42,state_key=str(i),state_hash='abc',slot=i) for i in range(2)]
        plan=self.out/'plan.jsonl';plan.write_text('\n'.join(json.dumps(e) for e in entries))
        ev.read_plan(plan);assets=self.out/'assets';assets.mkdir()
        (assets/'TASKS_30.json').write_text('[]');(assets/'EVAL_DEV_STATES.npz').write_bytes(b'fixture')
        results=self.out/'results';ev.bind_output(results,plan,'model-a',assets)
        with self.assertRaises(ValueError):ev.bind_output(results,plan,'model-b',assets)
        (results/'episodes').mkdir()
        (results/'episodes/case0.json').write_text(json.dumps(dict(episode_id='case0',checkpoint_id='model-a',valid=True,success=True,steps=10)))
        summary=ev.summarize(entries,results,'model-a')
        self.assertEqual(summary['status'],'PARTIAL');self.assertIsNone(summary['final_success_rate'])
        (results/'episodes/case1.json').write_text(json.dumps(dict(episode_id='case1',checkpoint_id='model-a',valid=True,success=False,steps=520)))
        summary=ev.summarize(entries,results,'model-a')
        self.assertEqual(summary['final_success_rate'],.5);self.assertEqual(len(summary['by_cell']),2)
        plan.write_text('\n'.join(json.dumps(entries[0]) for _ in range(2)))
        with self.assertRaises(ValueError):ev.read_plan(plan)

    def test_all_public_integration_sources_parse(self):
        for p in PI.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))


if __name__=='__main__':unittest.main()
