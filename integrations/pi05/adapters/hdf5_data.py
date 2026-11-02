"""Shared TRAIN windows. No dataset edits, filtering, or episode crossing."""
import json
from collections import OrderedDict
from pathlib import Path
import h5py
import numpy as np

class RawWindows:

    def __init__(self, root, horizon):
        self.root = Path(root)
        d = self.root / 'data_manifests'
        self.episodes = json.loads((d / 'EPISODES_1500.json').read_text())
        self.schedule = np.load(d / 'TRAIN_WINDOWS_seed42_G64_U10000.npy', mmap_mode='r')
        self.stats = json.loads((d / 'TRAIN_STATS.json').read_text())
        self.horizon = int(horizon)
        self.files = OrderedDict()

    def __len__(self):
        return len(self.schedule)

    def get(self, index):
        cell, eid, t = map(int, self.schedule[index])
        e = self.episodes[eid]
        assert e['split'] == 'train' and e['cell_id'] == cell
        path = e['hdf5']
        if path not in self.files:
            self.files[path] = h5py.File(path, 'r', rdcc_nbytes=8 * 1024 * 1024)
            if len(self.files) > 30:
                self.files.popitem(last=False)[1].close()
        d = self.files[path]['data'][e['episode']]
        n = min(self.horizon, e['length'] - t)
        a = np.zeros((self.horizon, 7), dtype=np.float32)
        a[:n] = d['actions'][t:t + n]
        a[n:] = a[n - 1]
        state = np.concatenate([d['obs/ee_states'][t], d['obs/gripper_states'][t]]).astype(np.float32)
        return {'image': d['obs/agentview_rgb'][t], 'wrist_image': d['obs/eye_in_hand_rgb'][t], 'state': state, 'actions': a, 'action_mask': np.arange(self.horizon) < n, 'prompt': e['instruction'], 'index': int(index), 'cell_id': cell, 'episode_index': eid, 'start': t, 'valid_action_steps': n}

class RankBatchSampler:

    def __init__(self, start_step, end_step, rank, world_size, microbatch, G=64):
        assert G % (world_size * microbatch) == 0
        self.start, self.end, self.rank, self.world, self.micro, self.G = (start_step, end_step, rank, world_size, microbatch, G)
        self.local = G // world_size

    def __iter__(self):
        for step in range(self.start, self.end):
            base = step * self.G + self.rank * self.local
            for offset in range(0, self.local, self.micro):
                yield list(range(base + offset, base + offset + self.micro))

    def __len__(self):
        return (self.end - self.start) * self.local // self.micro
