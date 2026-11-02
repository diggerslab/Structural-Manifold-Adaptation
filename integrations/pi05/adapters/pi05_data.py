"""Native OpenPI transforms on the common unfiltered HDF5 schedule."""
import dataclasses,json,os
from pathlib import Path
import numpy as np
from hdf5_data import RawWindows

def train_norm(root):
    from openpi.shared.normalize import NormStats
    stats=json.loads((Path(root)/'data_manifests/TRAIN_STATS.json').read_text())
    return {k:NormStats(**{n:np.asarray(stats[k][n],np.float32) for n in ['mean','std','q01','q99']}) for k in ['state','actions']}

def model_config(root):
    from openpi.training import config as C
    from openpi.training import optimizer as O
    from openpi.training import weight_loaders as W
    import flax.nnx as nnx
    root=Path(root);cfg=json.loads((root/'configs/PI05.json').read_text())
    base=C.get_config('pi05_libero')
    return dataclasses.replace(base,name='sma_pi05_pi05',exp_name='seed42',batch_size=64,num_train_steps=10000,
        fsdp_devices=4,freeze_filter=nnx.Nothing,ema_decay=None,
        lr_schedule=O.CosineDecaySchedule(warmup_steps=500,peak_lr=5e-5,decay_steps=10000,decay_lr=1e-5),
        optimizer=O.AdamW(clip_gradient_norm=1.0),weight_loader=W.CheckpointWeightLoader(cfg['c0']+'/params'),
        wandb_enabled=False)

def data_config(root):
    from openpi.training import config as C
    from openpi import transforms as T
    from openpi.policies.libero_policy import LiberoInputs,LiberoOutputs
    cfg=model_config(root)
    return C.DataConfig(repo_id='sma_pi05',asset_id='sma_pi05',norm_stats=train_norm(root),use_quantile_norm=True,
        data_transforms=T.Group(inputs=[LiberoInputs(model_type=cfg.model.model_type)],outputs=[LiberoOutputs()]),
        model_transforms=C.ModelTransformFactory()(cfg.model))

class PiWindows(RawWindows):
    def __init__(self,root):
        super().__init__(root,10);self.transform=None
    def __getitem__(self,index):
        from openpi import transforms as T
        if self.transform is None:
            c=data_config(self.root)
            self.transform=T.compose([*c.data_transforms.inputs,T.Normalize(c.norm_stats,use_quantiles=c.use_quantile_norm),*c.model_transforms.inputs])
        x=self.get(index)
        raw={'observation/image':np.ascontiguousarray(x['image'][::-1,::-1]),
             'observation/wrist_image':np.ascontiguousarray(x['wrist_image'][::-1,::-1]),
             'observation/state':x['state'],'actions':x['actions'],'prompt':x['prompt']}
        y=self.transform(raw);y['valid_action_mask']=x['action_mask'].astype(np.float32)
        return y

def collate_numpy(rows):
    import jax
    return jax.tree.map(lambda *x:np.stack(x),*rows)
