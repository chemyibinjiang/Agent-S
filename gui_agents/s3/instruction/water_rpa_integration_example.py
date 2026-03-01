from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from instruction.water_rpa_adapter import (
        load_water_rpa_workflow,
        workflow_to_water_rpa_tasks,
    )
    from instruction.workflow_schema import workflow_to_dict
except ImportError:
    from gui_agents.s3.instruction.water_rpa_adapter import (
        load_water_rpa_workflow,
        workflow_to_water_rpa_tasks,
    )
    from gui_agents.s3.instruction.workflow_schema import workflow_to_dict


def export_water_rpa_json_as_yaml(
    water_rpa_json_path: str | Path,
    yaml_output_path: str | Path,
) -> Path:
    import yaml

    workflow = load_water_rpa_workflow(water_rpa_json_path)
    output_path = Path(yaml_output_path)
    output_path.write_text(
        yaml.safe_dump(
            workflow_to_dict(workflow),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return output_path


def export_yaml_as_water_rpa_json(
    workflow_yaml_path: str | Path,
    json_output_path: str | Path,
    job_id: str | None = None,
    strict: bool = True,
) -> Path:
    try:
        from instruction.yaml.yaml_instruction_parser import load_instruction
    except ImportError:
        from gui_agents.s3.instruction.yaml.yaml_instruction_parser import load_instruction

    workflow = load_instruction(workflow_yaml_path)
    tasks = workflow_to_water_rpa_tasks(workflow, job_id=job_id, strict=strict)
    output_path = Path(json_output_path)
    output_path.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    instruction_dir = Path(__file__).resolve().parent

    sample_water_rpa_path = REPO_ROOT / ".." / "waterRPA" / "workflow.json"
    sample_yaml_path = instruction_dir / "yaml" / "example_instruction.yaml"

    print("waterRPA -> Agent-S workflow example")
    print(f"  Input JSON: {sample_water_rpa_path}")
    print(
        f"  Output YAML: {instruction_dir / 'yaml' / 'generated_from_water_rpa.yaml'}"
    )

    print("\nAgent-S workflow -> waterRPA example")
    print(f"  Input YAML: {sample_yaml_path}")
    print(f"  Output JSON: {instruction_dir / 'generated_from_yaml.json'}")
    print("\nEdit the sample paths in this script before running it.")


if __name__ == "__main__":
    main()
