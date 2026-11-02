"""Prepare TRAIN-only HDF5 windows, statistics and a uniform control schedule."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from sma.data import make_windows


def uniform_schedule(episodes, seed=42, steps=10000):
    """New deterministic balanced schedule; import the frozen array for exact historical draws."""
    rng = np.random.default_rng(seed)
    pools = {t: [i for i, e in enumerate(episodes) if e['task_id'] == t] for t in range(1, 11)}
    if not all(pools.values()):
        raise ValueError('The Pi0.5 reference expects task IDs 1..10 with TRAIN examples for each.')
    rows = []
    for step in range(steps):
        batch = []
        for i in range(64):
            task = (step * 64 + i) % 10 + 1
            eid = int(rng.choice(pools[task]))
            batch.append([episodes[eid]['cell_id'], eid, int(rng.integers(episodes[eid]['length']))])
        rows.extend(np.asarray(batch)[rng.permutation(64)].tolist())
    return np.asarray(rows, np.int32)


def validate_episodes(episodes):
    if not episodes or any(e.get('split') != 'train' or e.get('room') != 'kitchen' for e in episodes):
        raise ValueError('Use kitchen TRAIN only; do not pass evaluation trajectories.')
    if {e['task_id'] for e in episodes} != set(range(1, 11)):
        raise ValueError('Expected ten kitchen tasks numbered 1..10.')
    for e in episodes:
        for key in ['hdf5', 'episode', 'length', 'instruction', 'cell_id', 'episode_group_id', 'task_id']:
            if key not in e:
                raise ValueError('Missing episode field: ' + key)
        if e['length'] <= 0:
            raise ValueError('Empty trajectory.')


def prepare(episode_path, destination, stats_path=None, schedule_path=None, seed=42):
    import h5py
    episode_path, out = Path(episode_path).resolve(), Path(destination).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'EPISODES_1500.json').exists():
        raise FileExistsError('Use a new data directory; existing prepared data is not overwritten.')
    episodes = json.loads(episode_path.read_text())
    validate_episodes(episodes)
    states, actions = [], []
    for e in episodes:
        e['hdf5'] = str((episode_path.parent / e['hdf5']).resolve())
        e['source_group'] = str(e['episode_group_id'])
        with h5py.File(e['hdf5'], 'r') as f:
            d = f['data'][e['episode']]
            state = np.concatenate([d['obs/ee_states'][:], d['obs/gripper_states'][:]], axis=-1)
            action = d['actions'][:]
            if state.shape != (e['length'], 8) or action.shape != (e['length'], 7):
                raise ValueError('Require state xyz/axis-angle/gripper2 and seven OSC action coordinates.')
            for key in ['agentview_rgb', 'eye_in_hand_rgb']:
                if d['obs'][key].shape[0] != e['length']:
                    raise ValueError('Images and actions are not aligned.')
            states.append(state); actions.append(action)
    arrays = make_windows(episodes, states, actions, horizon=10)
    np.savez_compressed(out / 'numeric_windows.npz', **arrays)
    if stats_path:
        stats = json.loads(Path(stats_path).read_text())
    else:
        stats = {}
        for name, rows in [('state', states), ('actions', actions)]:
            x = np.concatenate(rows).astype(np.float64)
            stats[name] = {k: v.tolist() for k, v in dict(mean=x.mean(0), std=x.std(0),
                q01=np.quantile(x, .01, axis=0), q99=np.quantile(x, .99, axis=0)).items()}
    for name, dim in [('state', 8), ('actions', 7)]:
        for key in ['mean', 'std', 'q01', 'q99']:
            value = np.asarray(stats[name][key])
            if value.shape != (dim,) or not np.isfinite(value).all():
                raise ValueError('Invalid frozen TRAIN statistics.')
    if schedule_path:
        schedule = np.load(schedule_path, allow_pickle=False)
        if schedule.shape != (640000, 3) or not np.issubdtype(schedule.dtype, np.integer):
            raise ValueError('Frozen schedule must be integer [640000,3].')
        for cell, eid, start in schedule:
            if not (0 <= eid < len(episodes) and 0 <= start < episodes[eid]['length'] and cell == episodes[eid]['cell_id']):
                raise ValueError('Frozen schedule does not match episode order.')
    else:
        schedule = uniform_schedule(episodes, seed)
    np.save(out / f'TRAIN_WINDOWS_seed{seed}_G64_U10000.npy', schedule)
    (out / 'TRAIN_STATS.json').write_text(json.dumps(stats, indent=2) + '\n')
    (out / 'EPISODES_1500.json').write_text(json.dumps(episodes, indent=2) + '\n')
    (out / 'DATA_PREPARATION.json').write_text(json.dumps(dict(episodes=len(episodes),
        frames=sum(e['length'] for e in episodes), seed=seed, train_only=True,
        frozen_schedule_imported=bool(schedule_path), frozen_stats_imported=bool(stats_path)), indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--stats')
    p.add_argument('--schedule')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    prepare(a.episodes, a.output, a.stats, a.schedule, a.seed)
