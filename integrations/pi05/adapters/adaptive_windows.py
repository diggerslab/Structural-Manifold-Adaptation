"""Explicit global64 window tickets, traceable prefetch and resumable sampling state."""
import json, time
from collections import OrderedDict
from pathlib import Path
import numpy as np
from hdf5_data import RawWindows
from pi05_data import PiWindows

class TicketRaw:

    def get(self, index):
        eid, start, occurrence = map(int, index)
        e = self.episodes[eid]
        saved = self.schedule
        self.schedule = {occurrence: (e['cell_id'], eid, start)}
        try:
            return RawWindows.get(self, occurrence)
        finally:
            self.schedule = saved

def initialize(ds, root, horizon):
    ds.root = Path(root)
    d = ds.root / 'data_manifests'
    ds.episodes = json.loads((d / 'EPISODES_1500.json').read_text())
    assert all((e['split'] == 'train' and e['room'] == 'kitchen' for e in ds.episodes))
    ds.stats = json.loads((d / 'TRAIN_STATS.json').read_text())
    ds.schedule = {}
    ds.files = OrderedDict()
    ds.horizon = horizon

class AdaptivePiWindows(TicketRaw, PiWindows):

    def __init__(self, root):
        initialize(self, root, 10)
        self.transform = None

class GlobalTicketSampler:
    """Pi usesworld1/micro64; Torch rank0 creates tickets and broadcasts global64."""

    def __init__(self, sampler, start_step=0, state=None):
        self.sampler = sampler
        self.next_step = start_step
        self.completed = start_step
        self.pending = {}
        self.draw_seconds = 0.0
        if state:
            self.next_step = state['next_step']
            self.completed = state['completed']
            self.pending = {int(k): v for k, v in state['pending'].items()}
            self.draw_seconds = state.get('draw_seconds', 0.0)
            sampler.restore(state['sampler'])

    def draw(self, step):
        if step not in self.pending:
            assert step == self.next_step, (step, self.next_step)
            begin = time.monotonic()
            self.pending[step] = self.sampler.draw(64)
            self.next_step += 1
            self.draw_seconds += time.monotonic() - begin
        d = self.pending[step]
        return [(e, t, step * 64 + i) for i, (e, t) in enumerate(d['episode_start'])]

    def finish(self, step, loss, denominator, fit_seconds):
        assert step == self.completed
        draw = self.pending.pop(step)
        self.sampler.observe(draw, loss, denominator, fit_seconds)
        self.completed += 1
        return draw

    def state_dict(self):
        return dict(next_step=self.next_step, completed=self.completed, pending=self.pending, draw_seconds=self.draw_seconds, sampler=self.sampler.state_dict(), prefetch_semantics='Pending immutable tickets retain old probability_version; restore replays them before fresh draws')

    def __iter__(self):
        step = self.completed
        while step < 10000:
            yield self.draw(step)
            step += 1

    def __len__(self):
        return 10000 - self.completed
