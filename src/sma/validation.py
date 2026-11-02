"""Input contracts for the public, array-only API."""
import hashlib
import json
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_clock(seconds, previous):
    require(np.isfinite(seconds) and seconds >= previous, 'Elapsed fit time must be finite and monotonic.')


def validate_sampler_inputs(episodes, windows, regions, tasks, kernel, loss, eligible, batch):
    require(isinstance(batch, int) and batch > 0 and batch % 2 == 0, 'Global batch size must be positive and even.')
    require(len(episodes) > 0 and all(e.get('split') == 'train' for e in episodes), 'Only training episodes may enter SMA.')
    require(windows.ndim == 2 and windows.shape[1] == 2 and len(windows) > 0, 'windows must be nonempty [N,2] episode/start indices.')
    require(np.all(windows >= 0) and np.all(windows[:, 0] < len(episodes)), 'Invalid window indices.')
    require(len(np.unique(windows, axis=0)) == len(windows), 'Duplicate episode/start window.')
    require(set(windows[:, 0]) == set(range(len(episodes))), 'Every declared episode must have a window.')
    n = len(tasks)
    require(tasks.ndim == 1 and n > 0, 'region_tasks must be nonempty and one-dimensional.')
    require(regions.shape == (len(windows),) and set(regions) == set(range(n)), 'Region IDs must be contiguous and nonempty.')
    episode_tasks = np.asarray([e['task_id'] for e in episodes])
    require(np.all(episode_tasks[windows[:, 0]] == tasks[regions]), 'Windows and regions disagree about task identity.')
    require(kernel.shape == (n, n) and np.isfinite(kernel).all(), 'Invalid diffusion kernel.')
    require(np.allclose(kernel, kernel.T, atol=1e-12) and np.linalg.eigvalsh(kernel).min() > -1e-10, 'Kernel must be symmetric positive semidefinite.')
    a = np.asarray(loss)
    require(a.shape == (n,) and np.isfinite(a).all() and np.all(a >= 0), 'initial_loss must contain finite nonnegative regional scores.')
    if eligible is not None:
        require(np.asarray(eligible).shape == (n,), 'priority_eligible has wrong shape.')


def sampler_fingerprint(episodes, windows, regions, tasks, kernel, loss, eligible, batch):
    h = hashlib.sha256()
    h.update(json.dumps(episodes, sort_keys=True, separators=(',', ':')).encode())
    for value in [windows, regions, tasks, kernel, np.asarray(loss, np.float64),
                  np.ones(len(tasks), bool) if eligible is None else np.asarray(eligible, bool)]:
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    h.update(str(batch).encode())
    return h.hexdigest()


def validate_observation(draw, loss, denominator, seconds, sampler):
    validate_clock(seconds, max(sampler.clock, float(sampler.last_seen.max())))
    ids = np.asarray(draw['window_ids'])
    regions = np.asarray(draw['region_ids'])
    require(ids.ndim == 1 and len(ids) > 0 and np.issubdtype(ids.dtype, np.integer), 'Window IDs must be a nonempty integer vector.')
    require(np.all((ids >= 0) & (ids < len(sampler.windows))), 'Unknown observed window.')
    require(np.array_equal(regions, sampler.regions[ids]), 'Feedback does not match ticket region IDs.')
    loss, denominator = np.asarray(loss), np.asarray(denominator)
    require(loss.shape == ids.shape and np.isfinite(loss).all() and np.all(loss >= 0), 'Provide one finite nonnegative normalized score per window.')
    require(denominator.shape == ids.shape and np.isfinite(denominator).all() and np.all(denominator > 0), 'Every window needs a positive valid denominator.')


def validate_structure(arrays, visual):
    n = len(arrays['windows'])
    require(n > 0, 'Empty structural dataset.')
    for key in ['windows', 'tasks', 'groups', 'state', 'quat', 'actions', 'mask', 'phase']:
        require(len(arrays[key]) == n, f'{key} is not aligned with windows.')
    require(arrays['state'].shape == (n, 8), 'Reference state schema is xyz + axis-angle + two gripper positions (8D).')
    require(arrays['quat'].shape == (n, 4), 'quaternions must be [N,4], xyzw.')
    require(np.allclose(np.linalg.norm(arrays['quat'], axis=1), 1, atol=1e-5), 'Unit quaternions required.')
    action, mask = arrays['actions'], arrays['mask']
    require(action.ndim == 3 and action.shape[0] == n and action.shape[2] == 7, 'Actions must be [N,H,7].')
    require(mask.shape == action.shape[:2] and mask.dtype == bool, 'mask must be boolean [N,H].')
    require(np.all(mask[:, 0]) and np.all(mask.sum(axis=1) > 0), 'Every window needs a valid first action.')
    require(np.all(mask[:, 1:] <= mask[:, :-1]), 'Valid actions must form a contiguous prefix.')
    for key in ['state', 'quat', 'actions']:
        require(np.isfinite(arrays[key]).all(), f'Nonfinite {key}.')
    z = np.asarray(visual)
    require(z.ndim == 2 and z.shape[0] == n and z.shape[1] > 0 and np.isfinite(z).all(), 'Frozen features must be finite [N,D].')
