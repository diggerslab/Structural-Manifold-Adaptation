"""Build the reference 8D-state / 7D-action windows from in-memory episodes."""
import numpy as np
from .validation import require


def make_windows(episodes, states, actions, horizon=10):
    """Return metadata arrays; observations/features stay in the caller's loader.

    Each episode has task_id, source_group and split='train'. Each state sequence
    is [T,8] = xyz, axis-angle radians, two gripper positions; actions are [T,7].
    Gripper action convention must be consistent across the training dataset.
    """
    require(isinstance(horizon, int) and horizon > 0, 'Positive horizon required.')
    require(len(episodes) == len(states) == len(actions) > 0, 'Episode arrays differ in length.')
    rows = {k: [] for k in ['windows', 'tasks', 'groups', 'state', 'actions', 'mask', 'phase']}
    for eid, (episode, state, action) in enumerate(zip(episodes, states, actions)):
        require(episode.get('split') == 'train', 'Statistics and graph must use TRAIN only.')
        state, action = np.asarray(state, np.float32), np.asarray(action, np.float32)
        require(state.shape == (len(action), 8) and action.shape == (len(action), 7) and len(action) > 0, 'Expected aligned [T,8] states and [T,7] actions.')
        require(np.isfinite(state).all() and np.isfinite(action).all(), 'Nonfinite episode.')
        for t in range(len(action)):
            count = min(horizon, len(action) - t)
            chunk = np.zeros((horizon, 7), np.float32)
            chunk[:count] = action[t:t + count]
            signs = chunk[:count, -1] >= 0
            values = [[eid, t], episode['task_id'], str(episode['source_group']), state[t],
                      chunk, np.arange(horizon) < count, 2 * int(signs[0]) + int(np.any(signs != signs[0]))]
            for key, value in zip(rows, values):
                rows[key].append(value)
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    rot = arrays['state'][:, 3:6].astype(np.float64)
    angle = np.linalg.norm(rot, axis=1)
    scale = 0.5 * np.sinc(angle / (2 * np.pi))
    arrays['quat'] = np.c_[rot * scale[:, None], np.cos(angle / 2)].astype(np.float32)
    return arrays
