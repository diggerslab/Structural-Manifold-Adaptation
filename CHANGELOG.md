# Changelog

## 0.2.0 — 2026-11-02

- Publish the Pi0.5 model setup, native JAX/FSDP ordinary and SMA trainers,
  frozen features/probes, HDF5 data preparation, policy server and paired eval.
- Preserve the selected U_B recipe, native objective, minute-based sampler
  refresh and time-budget checkpoint behavior from the research implementation.
- Add CPU integration contracts; document the separate dataset/custom simulator
  dependencies and the absence of a fresh portable GPU reproduction.

## 0.1.0 — 2026-11-02

- Extract the S2 structural graph and mixed sampler as a NumPy-only package.
- Preserve reference ten-task, batch64 numerical behavior on synthetic fixtures.
- Support other integer task IDs and even batch sizes with explicit input checks.
- Add sampler checkpoint identity, CPU demo, integration guide and bilingual README.
- Release under MIT. No pretrained weights, robot data or full trainer included.
