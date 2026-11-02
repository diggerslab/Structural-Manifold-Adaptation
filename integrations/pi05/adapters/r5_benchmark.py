"""Frozen paired-appearance benchmark using the pinned Harness synchronous API."""
import hashlib,json,time
from pathlib import Path
import h5py,numpy as np
from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark
from vla_eval.benchmarks.base import StepResult

class R5Benchmark(LIBEROBenchmark):
    def __init__(self,root,save_frames=False):
        super().__init__(suite='paired_scene',seed=42,num_steps_wait=0,send_wrist_image=True,send_state=True,quat_no_antipodal=True)
        self.root=Path(root)
        self.tasks={(t['task_id'],t['room']):t for t in json.loads((self.root/'TASKS_30.json').read_text())}
        for meta in self.tasks.values():
            for key in ['hdf5','physics_bddl_file_name']:
                meta[key]=str((self.root/meta[key]).resolve())
        self.states=np.load(self.root/'EVAL_DEV_STATES.npz',allow_pickle=False)
        self.save_frames=save_frames;self.frames=[];self.steps=0;self.events=[]
    def get_tasks(self):return list(self.tasks.values())
    def get_action_spec(self):
        from r5_specs import action_spec
        return action_spec()
    def reset(self,task):
        from libero.libero.envs import OffScreenRenderEnv
        meta=self.tasks[(task['task_id'],task['room'])]
        key=(task['task_id'],task['room'])
        if self._env is not None and getattr(self,'_key',None)!=key:self.cleanup()
        if self._env is None:
            self._env=OffScreenRenderEnv(bddl_file_name=meta['physics_bddl_file_name'],camera_heights=256,camera_widths=256)
            self._key=key
        with h5py.File(meta['hdf5'],'r') as f:xml=f['data/demo_0'].attrs['model_file']
        if isinstance(xml,bytes):xml=xml.decode()
        self._env.seed(task['environment_seed']);self._env.reset();self._env.reset_from_xml_string(xml)
        initial=self.states[task['state_key']]
        assert hashlib.sha256(initial.astype('<f8').tobytes()).hexdigest()==task['state_hash']
        obs=self._env.regenerate_obs_from_state(initial)
        after=self._env.sim.get_state().flatten()
        assert np.allclose(after,initial,rtol=0,atol=1e-10),'reset altered frozen physical state'
        self.tid=task['task_id'];self.rule=meta['success_rule'];self.held=0;self.steps=0;self.events=[];self.frames=[]
        self.action_digest=hashlib.sha256();self.raw_actions=[];self.env_step_wall=0.
        self.initial_state_hash=task['state_hash']
        if self.tid==1:self.initial_z=self._position()[2]
        elif self._env.check_success():raise RuntimeError('Frozen state already successful')
        self._capture(obs)
        return obs
    def _position(self):
        return self._env.sim.data.body_xpos[self._env.env.obj_body_id['akita_black_bowl_1']].copy()
    def _capture(self,obs):
        if self.save_frames:self.frames.append(self._extract_frame(obs))
    def step(self,action):
        raw=np.asarray(action.get('actions',action.get('action')),dtype=np.float64).reshape(7)
        if not np.isfinite(raw).all():raise ValueError('Nonfinite policy output')
        self.raw_actions.append(raw.copy());raw[-1]=1 if raw[-1]>=0 else -1
        self.action_digest.update(raw.astype('<f8').tobytes())
        t=time.monotonic();self._env.step(raw.tolist());self.steps+=1
        obs=self._env.regenerate_obs_from_state(self._env.sim.get_state().flatten())
        self.env_step_wall+=time.monotonic()-t
        if self.tid==1:
            assert self.rule=='lift_delta_z_0.05m_eef_distance_0.12m_bilateral_grasp_hold_5_steps'
            grasp=bool(self._env.env._check_grasp(self._env.robots[0].gripper,self._env.env.objects_dict['akita_black_bowl_1']))
            lift=float(self._position()[2]-self.initial_z);distance=float(np.linalg.norm(self._position()-obs['robot0_eef_pos']))
            ok=lift>=.05 and distance<=.12 and grasp
            self.held=self.held+1 if ok else 0;done=self.held>=5
            if self.held==1 or done:self.events.append(dict(step=self.steps,lift=lift,eef_distance=distance,bilateral_grasp=grasp,held_steps=self.held))
        else:
            assert self.rule=='native_bddl_success';done=bool(self._env.check_success())
        self._capture(obs)
        return StepResult(obs=obs,reward=float(done),done=done,info={})
    def get_step_result(self,result):
        return dict(success=bool(result.done),termination='first_success' if result.done else 'horizon_timeout',
            action_sha256=self.action_digest.hexdigest(),initial_state_sha256=self.initial_state_hash,
            env_step_wall=self.env_step_wall,env_steps=self.steps,events=self.events)
    def get_metadata(self):return dict(max_steps=520,control_hz=20,reset_settle_steps=0,formal_unseen_test=False)
