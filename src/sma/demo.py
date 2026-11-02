"""CPU-only synthetic integration example. This is not a robot evaluation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .data import make_windows
from .structure import build_graph
from .sampler import SMASampler


def synthetic_dataset(task_ids=tuple(range(1, 11)), seed=7):
    rng = np.random.default_rng(seed)
    episodes, states, actions, features = [], [], [], []
    for task in task_ids:
        for episode in range(3):
            length = 12 + episode
            state = rng.normal(0, 0.15, (length, 8)).astype(np.float32)
            state[:, 0] += task / 10
            state[:, 6:8] = 0.02
            action = rng.normal(0, 0.1, (length, 7)).astype(np.float32)
            action[:, -1] = -1
            action[length // 2:, -1] = 1
            episodes.append(dict(task_id=task, source_group=f't{task}-source{episode}', split='train'))
            states.append(state)
            actions.append(action)
            features.append(rng.normal(size=(length, 64)).astype(np.float32))
    return episodes, make_windows(episodes, states, actions), np.concatenate(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('demo-output'))
    args = parser.parse_args()
    start = time.perf_counter()
    episodes, arrays, frozen_features = synthetic_dataset()
    graph, _ = build_graph(arrays, frozen_features)
    # Real integrations replace this with fixed-noise TRAIN scores of frozen C0.
    initial = np.linspace(0.4, 1.6, len(graph['region_tasks']))
    sampler = SMASampler(episodes, graph['windows'], graph['regions'],
                         graph['region_tasks'], graph['K'], initial,
                         priority_eligible=graph['priority_eligible'])
    trace = []
    for step in range(8):
        ticket = sampler.draw()
        # Stand-in for normalized per-window scores returned by a native trainer.
        region = np.asarray(ticket['region_ids'])
        score = initial[region] * (0.7 + 0.05 * step + 0.1 * (region % 3))
        elapsed = (step + 1) * 5.0
        sampler.observe(ticket, score, np.ones(len(region)), elapsed)
        trace.append(sampler.update_sampler(elapsed))
    args.output.mkdir(parents=True, exist_ok=True)
    state = json.loads(json.dumps(sampler.state_dict()))
    restored = SMASampler(episodes, graph['windows'], graph['regions'], graph['region_tasks'],
                          graph['K'], initial, priority_eligible=graph['priority_eligible'], state=state)
    equal = sampler.draw() == restored.draw()
    result = dict(example='synthetic_only', windows=len(graph['windows']),
                  regions=len(initial), global_batch=64, uniform_samples=32,
                  priority_samples=32, resume_next_draw_exact=equal,
                  cpu_wall_seconds=time.perf_counter() - start, trace=trace)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (args.output / 'sampler-state.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'trace'}, indent=2))
    if not equal:
        raise RuntimeError('Resume mismatch')


if __name__ == '__main__':
    main()
