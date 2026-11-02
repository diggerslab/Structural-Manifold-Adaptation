"""Four-block structural distance and within-task diffusion graph."""
import hashlib
import numpy as np
from .math import diffusion_kernel
from .validation import validate_structure

def robust(x):
    med = np.median(x, axis=0)
    lo, hi = np.quantile(x, [0.25, 0.75], axis=0)
    scale = hi - lo
    floor = max(float(np.median(scale[scale > 1e-08])) * 0.01, 1e-06) if np.any(scale > 1e-08) else 1.0
    scale = np.maximum(scale, floor)
    return (np.clip((x - med) / scale, -10, 10), dict(median=med.tolist(), IQR_floor=floor, scale=scale.tolist()))

class StructuralMetric:

    def __init__(self, arrays, visual):
        validate_structure(arrays, visual)
        self.a = arrays
        self.scales = {}
        self.visual, self.scales['visual_language'] = robust(np.asarray(visual, np.float32))
        self.position, self.scales['position'] = robust(arrays['state'][:, :3])
        self.gripper, self.scales['gripper'] = robust(np.c_[arrays['state'][:, 6:8], arrays['actions'][:, 0, 6], (arrays['actions'][:, :, 6] * arrays['mask']).sum(1) / arrays['mask'].sum(1)])
        valid = arrays['actions'][arrays['mask']]
        _, actscale = robust(valid)
        self.scales['expert_actions'] = actscale
        self.actions = np.clip((arrays['actions'] - np.asarray(actscale['median'])) / np.asarray(actscale['scale']), -10, 10)

    def distance(self, left, right, barrier=True):
        left, right = (np.asarray(left), np.asarray(right))

        def mse(x):
            return ((x[left, None, :] - x[None, right, :]) ** 2).mean(-1)
        visual = mse(self.visual)
        pos = mse(self.position)
        dot = np.clip(np.abs(self.a['quat'][left] @ self.a['quat'][right].T), 0, 1)
        angle = 2 * np.arccos(dot) / np.pi
        geometry = 0.5 * (pos + angle ** 2)
        gripper = mse(self.gripper)
        am = self.a['mask'][left, None, :] & self.a['mask'][None, right, :]
        ae = ((self.actions[left, None, :, :] - self.actions[None, right, :, :]) ** 2).mean(-1)
        act = (ae * am).sum(-1) / np.maximum(am.sum(-1), 1)
        result = np.sqrt(np.maximum(0, (visual + geometry + gripper + act) / 4))
        if barrier:
            result = np.where(self.a['phase'][left, None] == self.a['phase'][None, right], result, np.inf)
        return result

def build_graph(arrays, visual):
    metric = StructuralMetric(arrays, visual)
    assign = np.full(len(arrays['windows']), -1, np.int32)
    centers = []
    region_tasks = []
    Ws = []
    reports = []
    for task in np.unique(arrays['tasks']):
        task = int(task)
        ids = np.flatnonzero(arrays['tasks'] == task)
        candidate = []
        for eid in sorted(set(arrays['windows'][ids, 0])):
            trajectory = ids[arrays['windows'][ids, 0] == eid]
            candidate.extend(trajectory[np.unique(np.round(np.linspace(0, len(trajectory) - 1, min(4, len(trajectory)))).astype(int))].tolist())
        for phase in sorted(set(arrays['phase'][ids])):
            if not any((arrays['phase'][i] == phase for i in candidate)):
                candidate.append(int(ids[np.flatnonzero(arrays['phase'][ids] == phase)[0]]))
        candidate = np.asarray(candidate, np.int64)
        candidate = candidate[np.argsort([hashlib.sha256(('R8_REGION42:' + str(arrays['windows'][i].tolist())).encode()).hexdigest() for i in candidate])]
        d = metric.distance(candidate, candidate)
        chosen = []
        for phase in sorted(set(arrays['phase'][ids])):
            candidates = np.flatnonzero(arrays['phase'][candidate] == phase)
            if len(candidates):
                chosen.append(int(candidates[0]))
        while len(chosen) < min(32, len(candidate)):
            closest = np.min(d[:, chosen], axis=1)
            closest[chosen] = -1
            i = int(np.argmax(closest))
            chosen.append(i)
        c = candidate[chosen]

        def map_to_centers(c):
            blocks = []
            for j in range(0, len(ids), 256):
                distances = metric.distance(ids[j:j + 256], c)
                if not np.isfinite(distances).any(axis=1).all():
                    raise ValueError('Missing compatible phase center')
                blocks.append(np.argmin(distances, axis=1))
            return np.concatenate(blocks)
        local = map_to_centers(c)
        alive = np.unique(local)
        c = c[alive]
        local = map_to_centers(c)
        while True:
            support = [len(set(arrays['groups'][ids[local == i]])) for i in range(len(c))]
            merged = False
            for i, groups in enumerate(support):
                if groups >= 2:
                    continue
                row = metric.distance([c[i]], c)[0]
                row[i] = np.inf
                if not np.isfinite(row).any():
                    continue
                c = np.delete(c, i)
                local = map_to_centers(c)
                merged = True
                break
            if not merged:
                break
        nc = len(c)
        distances = metric.distance(c, c)
        W = np.zeros((nc, nc), np.float64)
        neighbor_values = []
        pairs = []
        for i in range(nc):
            row = distances[i].copy()
            row[i] = np.inf
            near = np.argsort(row)[:4]
            for j in near:
                if np.isfinite(row[j]):
                    pairs.append((i, int(j), float(row[j])))
                    neighbor_values.append(float(row[j]))
        scale = max(float(np.median(neighbor_values)), 1e-06) if neighbor_values else 1.0
        for i, j, dij in pairs:
            if dij <= 2 * scale:
                W[i, j] = np.exp(-(dij / scale) ** 2)
        W = np.maximum(W, W.T)
        np.fill_diagonal(W, 0)
        offset = len(centers)
        assign[ids] = local + offset
        centers.extend(c.tolist())
        region_tasks.extend([task] * nc)
        Ws.append(W)
        support = [len(set(arrays['groups'][ids[local == i]])) for i in range(nc)]
        reports.append(dict(task_id=task, regions=nc, candidate_windows=len(candidate), source_groups_per_region=support, low_support_flagged=[offset + i for i, n in enumerate(support) if n < 2], phase_compatibility_barrier=True, neighbor_k_max=4, kernel_scale=scale, edge_cutoff_multiple=2.0))
    n = len(centers)
    W = np.zeros((n, n))
    off = 0
    for block in Ws:
        nc = len(block)
        W[off:off + nc, off:off + nc] = block
        off += nc
    L, K = diffusion_kernel(W)
    if not np.min(assign) >= 0:
        raise ValueError('Invalid SMA inputs or state.')
    priority_eligible = np.ones(n, bool)
    for report in reports:
        priority_eligible[report['low_support_flagged']] = False
    if not all((priority_eligible[np.asarray(region_tasks) == task].any() for task in np.unique(arrays['tasks']))):
        raise ValueError('Invalid SMA inputs or state.')
    graph = dict(windows=arrays['windows'], regions=assign, region_tasks=np.asarray(region_tasks), representative_window_ids=np.asarray(centers), priority_eligible=priority_eligible, W=W, L=L, K=K)
    report = dict(scales=metric.scales, task_graphs=reports, actual_regions=n, all_windows=len(assign), all_windows_mapped=True)
    return (graph, report)
