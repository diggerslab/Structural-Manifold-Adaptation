"""Structure-aware sampling; the native model objective stays unchanged."""
import copy
import numpy as np
from .math import bounded_scale, coverage_coefficients
from .validation import validate_sampler_inputs, sampler_fingerprint, validate_observation, validate_clock

class SMASampler:
    """Draw once globally, then shard. S2 mixes equal uniform and priority halves."""

    def __init__(self, episodes, windows, regions, region_tasks, K, initial_loss, method='S2', seed=42, state=None, priority_eligible=None, global_batch_size=64):
        if not method in ['U', 'S2']:
            raise ValueError('Invalid SMA inputs or state.')
        self.method = method
        self.episodes = episodes
        for values in [windows, regions, region_tasks]:
            if not np.issubdtype(np.asarray(values).dtype, np.integer):
                raise ValueError('Window, region and task indices must be integers.')
        self.windows = np.asarray(windows, np.int64)
        self.regions = np.asarray(regions, np.int64)
        self.region_tasks = np.asarray(region_tasks, np.int64)
        self.K = np.asarray(K, np.float64)
        self.n = len(region_tasks)
        self.rng = np.random.default_rng(seed)
        self.version = 0
        self.drawn = 0
        validate_sampler_inputs(episodes, self.windows, self.regions, self.region_tasks, self.K, initial_loss, priority_eligible, global_batch_size)
        self.tasks = tuple(sorted(set((int(t) for t in self.region_tasks))))
        self.task_weight = 1.0 / len(self.tasks)
        self.task_cap = max(0.3, self.task_weight)
        self.global_batch_size = global_batch_size
        self.fingerprint = sampler_fingerprint(episodes, self.windows, self.regions, self.region_tasks, self.K, initial_loss, priority_eligible, global_batch_size)
        if not len(self.windows) == len(self.regions):
            raise ValueError('Invalid SMA inputs or state.')
        if not all((e['split'] == 'train' for e in episodes)):
            raise ValueError('Invalid SMA inputs or state.')
        self.region_windows = [np.flatnonzero(self.regions == c) for c in range(self.n)]
        if not all((len(x) for x in self.region_windows)):
            raise ValueError('Invalid SMA inputs or state.')
        self.region_episode_windows = []
        for ids in self.region_windows:
            by = {}
            for w in ids:
                by.setdefault(int(self.windows[w, 0]), []).append(int(w))
            self.region_episode_windows.append({e: np.asarray(x, np.int64) for e, x in by.items()})
        self.task_episodes = {t: [i for i, e in enumerate(episodes) if e['task_id'] == t] for t in self.tasks}
        self.episode_windows = {e: np.flatnonzero(self.windows[:, 0] == e) for e in range(len(episodes))}
        self.q = np.zeros(self.n)
        self.priority_eligible = np.ones(self.n, bool) if priority_eligible is None else np.asarray(priority_eligible, bool)
        for t in self.tasks:
            eligible = (self.region_tasks == t) & self.priority_eligible
            if not eligible.any():
                raise ValueError('Invalid SMA inputs or state.')
            self.q[eligible] = self.task_weight / np.sum(eligible)
        initial_loss = np.asarray(initial_loss, np.float64)
        if not (initial_loss.shape == (self.n,) and np.isfinite(initial_loss).all()):
            raise ValueError('Invalid SMA inputs or state.')
        self.initial = initial_loss
        self.epsilon = max(float(np.median(initial_loss)) * 0.1, 1e-12)
        self.ema = initial_loss.copy()
        self.last_seen = np.zeros(self.n)
        self.observations = np.zeros(self.n, np.int64)
        self.p_hat = self.q.copy()
        self.counts = np.zeros(self.n, np.int64)
        self.prob_region = self.q.copy()
        self.clock = 0.0
        self.uniform_window = np.zeros(len(windows))
        for e, ids in self.episode_windows.items():
            if len(ids):
                self.uniform_window[ids] = self.task_weight / len(self.task_episodes[episodes[e]['task_id']]) / len(ids)
        if not abs(self.uniform_window.sum() - 1) < 1e-12:
            raise ValueError('Invalid SMA inputs or state.')
        self.uniform_region = np.bincount(self.regions, weights=self.uniform_window, minlength=self.n)
        self.coverage_target = 0.5 * self.uniform_region + 0.5 * self.q if method in ['S2'] else self.q.copy()
        if method in ['S2']:
            self.p_hat = self.coverage_target.copy()
        if state:
            self.restore(state)
        else:
            self.update_sampler(0.0)

    def update_sampler(self, fit_seconds):
        validate_clock(fit_seconds, self.clock)
        self.clock = float(fit_seconds)
        r = np.clip(self.ema / np.maximum(self.initial, self.epsilon), 0.5, 2)
        r[(fit_seconds - self.last_seen > 180) | (self.observations == 0)] = 1.0
        coverage, marginal = coverage_coefficients(self.p_hat, self.coverage_target, self.K, mean_weights=self.q) if self.method in ['S2'] else (np.ones(self.n), np.zeros(self.n))
        if self.method == 'U':
            r = np.ones(self.n)
        v = self.q * r * coverage
        task_raw = np.asarray([v[self.region_tasks == t].sum() for t in self.tasks])
        task_p = bounded_scale(task_raw, total=1, lo=1e-12, hi=self.task_cap)
        for task_index, t in enumerate(self.tasks):
            ids = self.region_tasks == t
            v[ids] *= task_p[task_index] / v[ids].sum()
        self.prob_region = v
        self.version += 1
        final_task = 0.5 * self.task_weight + 0.5 * task_p
        if not (abs(v.sum() - 1) < 1e-10 and np.min(final_task) >= 0.5 * self.task_weight - 1e-10 and (np.max(final_task) <= 0.5 * (self.task_weight + self.task_cap) + 1e-10)):
            raise ValueError('Invalid SMA inputs or state.')
        return dict(version=self.version, fit_seconds=self.clock, task_probability=final_task.tolist(), region_priority_probability=v.tolist(), r=r.tolist(), c=coverage.tolist(), p_hat=self.p_hat.tolist(), q=self.q.tolist(), coverage_target=self.coverage_target.tolist(), uniform_region=self.uniform_region.tolist(), coverage_energy=float((self.p_hat - self.coverage_target) @ self.K @ (self.p_hat - self.coverage_target)), marginal=marginal.tolist(), age=(fit_seconds - self.last_seen).tolist())

    def _uniform(self, n):
        return self.rng.choice(len(self.windows), n, p=self.uniform_window)

    def draw(self, G=None):
        G = self.global_batch_size if G is None else G
        if G != self.global_batch_size:
            raise ValueError('Batch size must match sampler configuration.')
        half = G // 2
        if self.method == 'U':
            ids = self._uniform(G)
            branch = np.zeros(G, np.int8)
        else:
            prioritized = []
            for c in self.rng.choice(self.n, half, p=self.prob_region):
                pool = self.region_episode_windows[c]
                e = int(self.rng.choice(list(pool)))
                prioritized.append(int(self.rng.choice(pool[e])))
            ids = np.concatenate([self._uniform(half), np.asarray(prioritized, np.int64)])
            branch = np.r_[np.zeros(half, np.int8), np.ones(half, np.int8)]
        perm = self.rng.permutation(G)
        ids = ids[perm]
        branch = branch[perm]
        prob = []
        for w in ids:
            c = int(self.regions[w])
            e = int(self.windows[w, 0])
            pool = self.region_episode_windows[c]
            pv = self.prob_region[c] / len(pool) / len(pool[e])
            prob.append(self.uniform_window[w] if self.method == 'U' else 0.5 * self.uniform_window[w] + 0.5 * pv)
        self.drawn += G
        result = dict(window_ids=ids.tolist(), episode_start=self.windows[ids].tolist(), region_ids=self.regions[ids].tolist(), probability=prob, branch=branch.tolist(), probability_version=self.version, global_G=G)
        return result

    def observe(self, draw, loss, valid_denominator, fit_seconds, probe=False):
        validate_observation(draw, loss, valid_denominator, fit_seconds, self)
        regions = np.asarray(draw['region_ids'])
        loss = np.asarray(loss, np.float64)
        den = np.asarray(valid_denominator, np.float64)
        if not (len(loss) == len(regions) and np.isfinite(loss).all() and (den > 0).all()):
            raise ValueError('Invalid SMA inputs or state.')
        for c in np.unique(regions):
            mask = regions == c
            n = int(mask.sum())
            value = float(np.mean(loss[mask]))
            alpha = 1 - np.exp(-n / 16)
            self.ema[c] = (1 - alpha) * self.ema[c] + alpha * value
            self.last_seen[c] = fit_seconds
            self.observations[c] += n
        if not probe:
            counts = np.bincount(regions, minlength=self.n)
            alpha = 1 - np.exp(-len(regions) / 2048)
            self.p_hat = (1 - alpha) * self.p_hat + alpha * counts / len(regions)
            self.counts += counts

    def state_dict(self):
        return dict(schema_version=1, fingerprint=self.fingerprint, method=self.method, rng=copy.deepcopy(self.rng.bit_generator.state), version=self.version, drawn=self.drawn, epsilon=self.epsilon, initial=self.initial.tolist(), ema=self.ema.tolist(), last_seen=self.last_seen.tolist(), observations=self.observations.tolist(), p_hat=self.p_hat.tolist(), counts=self.counts.tolist(), prob_region=self.prob_region.tolist(), clock=self.clock)

    def restore(self, s):
        if s.get('fingerprint') != self.fingerprint or s.get('schema_version') != 1:
            raise ValueError('Sampler checkpoint belongs to a different dataset, graph or configuration.')
        if not (s['method'] == self.method and np.allclose(s['initial'], self.initial) and (s['epsilon'] == self.epsilon)):
            raise ValueError('Invalid SMA inputs or state.')
        self.rng.bit_generator.state = s['rng']
        self.version = s['version']
        self.drawn = s['drawn']
        self.clock = s['clock']
        for k in ['ema', 'last_seen', 'observations', 'p_hat', 'counts', 'prob_region']:
            value = np.asarray(s[k])
            if value.shape != (self.n,) or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError('Malformed sampler checkpoint field: ' + k)
            setattr(self, k, value.copy())
        for value in [self.p_hat, self.prob_region]:
            if not np.isclose(value.sum(), 1., atol=1e-10):
                raise ValueError('Checkpoint distribution is not normalized.')
