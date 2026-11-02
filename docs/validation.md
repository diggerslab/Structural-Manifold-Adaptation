# Release validation

## Pi0.5 integration, v0.2.0

All 21 local CPU tests passed on 2026-11-02 (13 core tests plus 8 integration
tests). Integration checks cover baseline/SMA native update-function equality,
prefetched-ticket restoration, minute-based residual probes against the public
sampler, fit-time schedule/checkpoint boundaries, TRAIN-only data validation,
actual synthetic HDF5 reads and temporal padding, result identity and complete
versus partial denominators, and Python syntax across the published integration.
The optional score hook was diffed against the pinned upstream OpenPI model;
unrelated historical training configs and logging edits were excluded.

No new GPU training or simulator rollout was run for this release. This validates
the CPU contracts and source extraction, not a clean-environment end-to-end
reproduction or a new success-rate measurement.

## Core, v0.1.0

Version0.1.0 was checked on2026-11-02 with Python3.12 and NumPy2.3.5 on CPU.

- Original S2 reference graph versus packaged graph: all eight exported arrays
  matched on390 synthetic windows and84 regions.
- Eight consecutive batch64 draws and updates: indices, probabilities, residuals,
  coverage coefficients and distribution state matched the reference within
  floating-point comparison tolerances (1e-12 in the local reference check).
- Thirteen unit tests passed, including exact JSON checkpoint continuation,
  noncontiguous task IDs, energy marginals, phase barriers and padded-action masks.
- Built and installed the wheel into a separate directory, then ran the CPU demo
  from that installed package successfully.

The repository's CI reruns tests on supported Python versions. These checks cover
the released numerical mechanism; they do not execute a GPU policy trainer or
robot evaluation. Frozen encoder extraction and trainer-side recovery remain
integration responsibilities documented in `integration.md`.
