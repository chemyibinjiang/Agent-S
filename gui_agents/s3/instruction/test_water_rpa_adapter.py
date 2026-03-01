from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui_agents.s3.instruction.water_rpa_adapter import (
    parse_water_rpa_tasks,
    workflow_to_water_rpa_tasks,
)


def test_water_rpa_round_trip() -> None:
    tasks = [
        {"type": 1.0, "value": "button.png", "retry": 2},
        {"type": 4.0, "value": "hello world", "retry": 1},
        {"type": 6.0, "value": -240, "retry": 1},
        {"type": 9.0, "value": "captures", "retry": 1},
    ]

    workflow = parse_water_rpa_tasks(tasks, workflow_name="demo")
    exported = workflow_to_water_rpa_tasks(workflow, strict=True)

    assert workflow.name == "demo"
    assert workflow.jobs["main"].steps[0].images.step_image == "button.png"
    assert workflow.jobs["main"].steps[1].actions.text_input[0].input_method == "clipboard"
    assert workflow.jobs["main"].steps[2].actions.scroll_input[0].direction == "down"
    assert workflow.jobs["main"].steps[3].actions.screenshot_input[0].path == "captures"
    assert len(exported) == len(tasks)
    assert exported[0]["type"] == 1.0
    assert exported[1]["value"] == "hello world"
    assert exported[2]["value"] == -240
    assert exported[3]["type"] == 9.0


if __name__ == "__main__":
    test_water_rpa_round_trip()
    print("waterRPA adapter tests passed.")
