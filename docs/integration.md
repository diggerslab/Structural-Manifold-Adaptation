# Integration with a VLA trainer

## Input contract

`make_windows(episodes, states, actions, horizon=10)` accepts:

- `episodes`: JSON-serializable metadata, one dictionary per trajectory with
  integer `task_id`, string `source_group`, and `split="train"`.
- `states[e]`: `[T,8]`: xyz, axis-angle rotation in radians, two gripper positions.
- `actions[e]`: `[T,7]`: six continuous control coordinates and one gripper action.

Source groups identify independent source demonstrations. Paired appearances of
the same original trajectory must share a source group. Keep coordinate frames,
units and gripper sign conventions consistent. The reference uses a ten-step
chunk of OSC controls; the library does not silently convert another action type.

The returned arrays contain episode/start indices, task IDs, source groups,
states, xyzw unit quaternions, padded actions, masks and phase codes. Windows are
ordered by episode and then start time. Supply frozen `[N,D]` features in exactly
this order. The graph accepts other D; the reference π0.5 setup uses64.

## Frozen feature recipe for π0.5

The core supplies the array API; the [Pi0.5 integration](../integrations/pi05/README.md)
supplies a pinned upstream model setup and concrete feature/loss adapters. The
research recipe extracts frozen initial-policy prefix embeddings with evaluation
preprocessing. Separately take masked means of image and language tokens, project
each to32 dimensions using a fixed Gaussian projection, and concatenate them.
The π0.5 recipe uses JAX key7042 and the same projection matrix for both streams
when their input dimension matches. Torch backbones use their native extraction
and generator, which are not numerically interchangeable with JAX.

Use the same TRAIN-derived normalization as your corresponding training arm.
Record feature revision, checkpoint identity, preprocessing and projection RNG.
Do not regenerate frozen features after every policy update.

## Residual hook

For a flow-matching policy, the reference score is MSE of the **real seven action
coordinates**, averaged over valid timesteps. Mask padded timesteps. Exclude
dummy dimensions from the monitoring score, while retaining the complete native
loss for the optimizer. Initial regional scores use the frozen starting model,
one representative window per region, fixed noise/timestep settings and one
sample per window. Keep each window's probe RNG/batch position fixed.

Other policy families need their own native residual definition. The API accepting
their scores does not imply that their semantics match π0.5 or that transfer
performance is established. The trainer returns detached CPU scores and valid
counts in the global ticket order.

## Distributed execution

1. Rank0 creates one sampler and draws one global ticket.
2. Broadcast/scatter that ticket to workers, respecting gradient accumulation.
3. Run the unchanged native optimizer update.
4. Gather normalized scores and valid counts in the exact ticket order.
5. Call `observe` and `update_sampler` once on rank0 after consumption.

The reference has global batch64. Other positive even sizes are supported, but
changing them changes training exposure and is not the original experiment.
Control rank-side dropout/augmentation/model RNG separately; the sampler does not
own those streams. Never use validation or evaluation outcomes to set priorities.

The minimal callback example updates allocation every optimizer step. A trainer
with a different refresh cadence must document that cadence and preserve it in
matched comparisons. The package is the S2 allocation mechanism, not a promise
of identical asynchronous loader behavior in every host trainer.

## Checkpoint and resume

Save `sampler.state_dict()` alongside the model, optimizer, scheduler, RNG streams,
data cursor and the immutable graph/features. Restore with `state=...` using the
same ordered input metadata and arrays. State is JSON serializable and bound to
a dataset/graph/configuration fingerprint; mismatched input is rejected.

The core test demonstrates exact continuation of **sampler draws**. A real
prefetching loader must additionally serialize outstanding tickets and their
order, or drain them before checkpointing. Restoring the sampler while discarding
already-drawn pending tickets changes the consumed stream. Do not call sampler
restore alone an end-to-end trainer recovery.

## Experimental comparisons

Use the same starting checkpoint, TRAIN split, native model objective, optimizer,
wall-time definition and evaluation protocol for uniform and SMA arms. The
`method="U"` control performs task/episode/time uniform sampling. Separate feature
preparation, model loading/compilation, actual training, final saving and evaluation
costs. Report negative results and model-specific differences as well as gains.

No simulator or model is executed by the default tests. The Pi0.5 integration
includes ordinary/SMA training and evaluation code. Running it requires the
separately installed model dependencies, weights, data and benchmark assets.
