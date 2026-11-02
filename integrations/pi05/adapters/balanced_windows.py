"""Stateless full-global-batch sampling, with an unbounded timed-training cursor."""
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
from hdf5_data import RawWindows
from pi05_data import PiWindows

class SourceSchedule:
    """Exact prefix of the already-frozen kitchen-only R6 source sampler."""

    def __init__(self, path):
        self.values = np.load(path, mmap_mode='r')
        assert self.values.shape == (640000, 3)

    def __len__(self):
        return len(self.values)

    def batch(self, step):
        assert 0 <= step < 10000, 'Frozen source schedule exhausted'
        return self.values[step * 64:(step + 1) * 64]

    def __getitem__(self, index):
        return self.values[index]

def initialize(raw, root, horizon):
    raw.root = Path(root)
    d = raw.root / 'data_manifests'
    raw.episodes = json.loads((d / 'EPISODES_1500.json').read_text())
    cfg = json.loads((raw.root / 'configs/SOURCE_TIME.json').read_text())
    assert cfg['arm'] == 'S' and all((e['room'] == 'kitchen' and e['split'] == 'train' for e in raw.episodes))
    raw.schedule = SourceSchedule(d / f"TRAIN_WINDOWS_seed{cfg.get('data_seed', cfg.get('seed', 42))}_G64_U10000.npy")
    raw.stats = json.loads((d / 'TRAIN_STATS.json').read_text())
    raw.horizon = horizon
    raw.files = OrderedDict()

class TimedPiWindows(PiWindows):

    def __init__(self, root):
        initialize(self, root, 10)
        self.transform = None

class Consumption:

    def __init__(self, dataset, horizon, state=None):
        self.dataset = dataset
        self.horizon = horizon
        self.state = state or {'successful_updates': 0, 'windows': 0, 'valid_action_steps': 0, 'by_cell': {}, 'unique_windows': []}
        self.unique = set(map(tuple, self.state.pop('unique_windows', [])))

    def add(self, step):
        for cell, eid, start in self.dataset.schedule.batch(step):
            e = self.dataset.episodes[int(eid)]
            key = e['room'] + ':task' + str(e['task_id']).zfill(2)
            self.state['by_cell'][key] = self.state['by_cell'].get(key, 0) + 1
            self.unique.add((int(eid), int(start)))
            self.state['valid_action_steps'] += min(self.horizon, e['length'] - int(start))
        self.state['successful_updates'] += 1
        self.state['windows'] += 64

    def serializable(self):
        return dict(self.state, unique_windows=sorted(self.unique))

    def summary(self):
        return dict(self.state, unique_episode_start_windows=len(self.unique))
