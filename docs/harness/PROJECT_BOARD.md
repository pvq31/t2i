# seethrough3d2 Harness Project Board

## Current Main Entry

Recommended automated loop:

```bash
python3 run_scene_harness.py \
  --scene-text "..." \
  --output inference/saved_scenes/example_harness.pkl \
  --objects "table:1,bowl:1" \
  --max-rounds 4
```

Legacy manual chain remains available:

```text
agent_text2pkl_v5.py -> agent_opinion.py -> agent_reverse.py -> infer2.py
```

## System of Record

- `修改方案5.12.md`: design plan.
- `harness_param_constraints.py`: deterministic pkl-level constraints and validation.
- `run_scene_harness.py`: plan -> execute -> verify -> repair loop.
- `docs/harness/FAILURE_KB.jsonl`: durable failure memory.

## Current Harness Capabilities

- Object manifest generation and strict pkl subject reconciliation.
- Predicate extraction for center, left/right, front/behind, above/below, support.
- Deterministic projection to `subjects_data[*].dims/x/y/z/azimuth` and `camera_data`.
- Dimension range checks, volume ordering, same-type volume normalization.
- JSON validation reports and executable repair plans.
- Repair-plan execution through `agent_reverse.py --repair-plan`.

## Known Limitations

- Predicate extraction is rule-based and covers common English/Chinese relations; ambiguous language should use `--objects` or `--object-manifest`.
- Final image semantic checking is not yet implemented; current checks are pkl/cube-level.
- `infer2.py` is only called after pkl validation passes in the new loop.
