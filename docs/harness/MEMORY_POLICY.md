# Harness Memory Policy

Keep in active context:

- `docs/harness/PROJECT_BOARD.md`
- Current prompt and object manifest.
- Current `validation.json`, `repair_plan.json`, and failing predicates.
- Files being edited.

Compress:

- Old run logs into `runs/<id>/summary.json`.
- Repeated failures into `docs/harness/FAILURE_KB.jsonl`.
- Long stdout into command logs referenced by path.

Discard from active context:

- Old image render artifacts unless the current task is image-level diagnosis.
- Repeated successful intermediate pkl/json dumps.
- Full historical command output once summarized.
