"""Adapter contract for an existing VLA trainer (framework-neutral).

Call this function with your real TRAIN-only data, frozen C0 features and model.
The package's runnable synthetic demo is `python -m sma.demo`.
"""
import time
from sma import SMASampler, build_graph


def adapt(episodes, arrays, frozen_features, fixed_c0_region_scores,
          load_windows, native_train_step, steps=100):
    """native_train_step returns (already_normalized_scores, valid_counts).

    It performs the optimizer update using the ORIGINAL policy objective.
    Scores are detached monitoring values in ticket order, not a new loss.
    fixed_c0_region_scores(graph) evaluates one fixed representative per region.
    """
    graph, report = build_graph(arrays, frozen_features)
    initial = fixed_c0_region_scores(graph)
    sampler = SMASampler(episodes, graph['windows'], graph['regions'],
                         graph['region_tasks'], graph['K'], initial,
                         priority_eligible=graph['priority_eligible'])
    start = time.monotonic()
    for _ in range(steps):
        ticket = sampler.draw()
        batch = load_windows(ticket['episode_start'])
        scores, valid_counts = native_train_step(batch)
        seconds = time.monotonic() - start
        sampler.observe(ticket, scores, valid_counts, seconds)
        sampler.update_sampler(seconds)
    return sampler.state_dict(), report
