"""Pi0.5 native B1 inference; concurrent simulator sessions have isolated RNG."""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(os.environ['SMA_RUN_DIR']).resolve()
ADAPTERS = Path(__file__).resolve().parents[1]/'adapters'
sys.path[:0] = [str(ADAPTERS),str(ROOT/'code/openpi/src'),str(ROOT/'code/openpi/packages/openpi-client/src')]
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve

class Pi05Server(PredictModelServer):

    def __init__(self, model, checkpoint):
        self.model_name = model
        self.rng = {}
        self.calls = 0
        self.infer_wall = 0.0
        self.checkpoint = checkpoint
        self.stats = json.loads((ROOT / 'data_manifests/TRAIN_STATS.json').read_text())
        super().__init__(chunk_size=10, max_batch_size=1, continuous_inference=False, hz=20.0)
        t = time.monotonic()
        self.load_pi(checkpoint)
        print(json.dumps(dict(event='MODEL_LOADED', model=model, checkpoint=checkpoint, load_wall=time.monotonic() - t, chunk=self.chunk_size)), flush=True)

    def load_pi(self, path):
        import jax.numpy as jnp
        from pi05_data import model_config, data_config, train_norm
        from openpi.models import model as M
        from openpi.policies.policy import Policy
        from openpi import transforms as T
        cfg = model_config(ROOT)
        dc = data_config(ROOT)
        model = cfg.model.load(M.restore_params(Path(path) / 'params', dtype=jnp.bfloat16))
        self.policy = Policy(model, transforms=[*dc.data_transforms.inputs, T.Normalize(train_norm(ROOT), use_quantiles=True), *dc.model_transforms.inputs], output_transforms=[*dc.model_transforms.outputs, T.Unnormalize(train_norm(ROOT), use_quantiles=True), *dc.data_transforms.outputs], sample_kwargs={'num_steps': 10})

    def get_observation_params(self):
        return dict(send_wrist_image=True, send_state=True, quat_no_antipodal=True)

    def get_observation_spec(self):
        from vla_eval.specs import IMAGE_RGB, LANGUAGE, STATE_EEF_POS_AA_GRIP
        return dict(agentview=IMAGE_RGB, wrist=IMAGE_RGB, language=LANGUAGE, state=STATE_EEF_POS_AA_GRIP)

    def get_action_spec(self):
        from r5_specs import action_spec
        return action_spec()

    async def on_episode_start(self, config, ctx):
        await super().on_episode_start(config, ctx)
        seed = int(config['task']['policy_seed'])
        import jax
        self.rng[ctx.session_id] = jax.random.key(seed)

    async def on_episode_end(self, result, ctx):
        await super().on_episode_end(result, ctx)
        self.rng.pop(ctx.session_id, None)

    def predict(self, obs, ctx):
        t = time.monotonic()
        sid = ctx.session_id
        self.policy._rng = self.rng[sid]
        a = self.policy.infer({'observation/image': obs['images']['agentview'], 'observation/wrist_image': obs['images']['wrist'], 'observation/state': np.asarray(obs['states'], np.float32), 'prompt': obs['task_description']})['actions']
        self.rng[sid] = self.policy._rng
        a = np.asarray(a, dtype=np.float32)
        assert a.ndim == 2 and a.shape[1] == 7 and np.isfinite(a).all(), (self.model_name, a.shape)
        dt = time.monotonic() - t
        self.calls += 1
        self.infer_wall += dt
        if self.calls <= 2 or self.calls % 200 == 0:
            print(json.dumps(dict(event='INFERENCE', model=self.model_name, calls=self.calls, wall=dt, cumulative_wall=self.infer_wall, batch_size=1)), flush=True)
        return {'actions': a}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--port',type=int,default=8000)
    p.add_argument('--host',default='127.0.0.1')
    a=p.parse_args()
    serve(Pi05Server('pi05',a.checkpoint),a.host,a.port)
if __name__=='__main__': main()
