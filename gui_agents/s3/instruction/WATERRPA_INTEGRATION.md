# waterRPA Integration

`Agent-S` now has a shared workflow schema in `workflow_schema.py`. YAML is one serialization format for that schema. `waterRPA` JSON is another input/output format handled by `water_rpa_adapter.py`.

## What changed

- Existing YAML parsing still works through `yaml/yaml_instruction_parser.py`.
- Existing imports from `yaml/yaml_instruction.py` still work through backward-compatible re-exports.
- The executor in `yaml/yaml_instruction_auto_executor.py` now accepts the shared `Workflow` model, supports `screenshot_input`, and includes `run_safe_workflow(...)` with post-step verification plus minimal `on_success` / `on_failure` branching.

## Conversion paths

- `waterRPA JSON -> Workflow`: `load_water_rpa_workflow(path)`
- `Workflow -> waterRPA JSON`: `workflow_to_water_rpa_tasks(workflow, strict=True)`
- `Workflow -> plain dict`: `workflow_to_dict(workflow)`

There is a runnable example in `water_rpa_integration_example.py`.

## Mapping

- `type 1/2/3` -> `mouse_input` with `images.step_image`
- `type 4` -> `text_input`
- `type 5` -> `wait`
- `type 6` -> `scroll_input`
- `type 7` -> `key_input`
- `type 8` -> `hover_input` with `images.step_image`
- `type 9` -> `screenshot_input`

## Current limits

- `waterRPA` only supports a narrow single-action task model. Export from `Workflow` to `waterRPA` fails in `strict=True` mode if a step contains multiple populated action sections.
- `result_image`, `expected_result`, and `element_text` are preserved in exported JSON under `task["agent_s"]`, but `waterRPA` itself does not execute those fields.
- `expected_result` is currently verified by OCR substring matching. For deterministic safe-runner behavior, author it as literal UI text whenever possible; otherwise rely on `result_image`.
