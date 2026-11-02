# Paired evaluation asset contract

These assets are **inputs**, not generated sample results. Use the same immutable
files for the initial, baseline and SMA checkpoints.

`TASKS_30.json` is a list with one entry for each room/task combination:

```json
{
  "room": "kitchen",
  "task_id": 1,
  "hdf5": "hdf5/kitchen/task_01_demo.hdf5",
  "physics_bddl_file_name": "bddl/task_01.bddl",
  "success_rule": "lift_delta_z_0.05m_eef_distance_0.12m_bilateral_grasp_hold_5_steps"
}
```

Paths may be absolute or relative to this assets directory. Each HDF5 file must
contain the paired scene XML in `data/demo_0.attrs['model_file']`. XML asset paths
must resolve in the simulator environment. Rooms are `kitchen`, `office`,
`living_room`. Task 1 uses the published custom lift/grasp rule; tasks 2..10 use
`native_bddl_success`. Task 1 expects the object `akita_black_bowl_1`.

`EVAL_DEV_STATES.npz` maps each plan `state_key` to a flattened MuJoCo simulator
state. State hashes are SHA256 of `state.astype('<f8').tobytes()`. The wrapper
checks that resetting preserved the exact initial state, before executing actions.

`EVAL_PLAN.jsonl` contains one object per episode, for example:

```json
{"episode_id":"EVAL_kitchen_task01_00","room":"kitchen","task_id":1,"instruction":"pick up the black bowl and hold it above the table","environment_seed":42,"policy_seed":42,"state_key":"kitchen_task01_00","state_hash":"REPLACE_WITH_ACTUAL_SHA256","slot":0}
```

For the reference full panel use 3 rooms × 10 tasks × 50 cases = 1500 entries.
Each task/slot must retain its matched physical state and policy random stream
across rooms and checkpoints. Historical task sets were previously exposed to
development; this panel should not be relabeled as an untouched holdout.

`slot` 0 and 1 are recorded as video regardless of success. All 1500 result
metadata records are counted, including unsuccessful episodes. The generic
aggregator also supports a smaller plan, but its denominator is always the
actual plan; a small panel is never labeled a full 1500-case result.
