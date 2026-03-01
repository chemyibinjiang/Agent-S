from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from instruction.workflow_schema import (
        Actions,
        HoverInput,
        Images,
        Job,
        KeyInput,
        Metadata,
        MouseInput,
        ScreenshotInput,
        ScrollInput,
        Software,
        Step,
        TextInput,
        WaitInput,
        Workflow,
        resolve_workflow_paths,
    )
except ImportError:
    from gui_agents.s3.instruction.workflow_schema import (
        Actions,
        HoverInput,
        Images,
        Job,
        KeyInput,
        Metadata,
        MouseInput,
        ScreenshotInput,
        ScrollInput,
        Software,
        Step,
        TextInput,
        WaitInput,
        Workflow,
        resolve_workflow_paths,
    )


WATER_RPA_CMD_TYPES: Dict[float, str] = {
    1.0: "left_click",
    2.0: "double_click",
    3.0: "right_click",
    4.0: "text_input",
    5.0: "wait",
    6.0: "scroll",
    7.0: "hotkey",
    8.0: "hover",
    9.0: "screenshot",
}


def _build_step_metadata(
    task: Dict[str, Any],
    step_index: int,
) -> Dict[str, Any]:
    metadata = task.get("agent_s") if isinstance(task.get("agent_s"), dict) else {}
    default_name = f"Step {step_index + 1}: {WATER_RPA_CMD_TYPES.get(task.get('type'), 'task')}"
    return {
        "id": metadata.get("step_id") or f"step_{step_index + 1}",
        "name": metadata.get("step_name") or default_name,
        "result_image": metadata.get("result_image"),
        "expected_result": metadata.get("expected_result"),
        "element_text": metadata.get("element_text"),
    }


def _parse_hotkey(value: Any) -> List[str]:
    return [part.strip().lower() for part in str(value).split("+") if part.strip()]


def water_rpa_task_to_step(task: Dict[str, Any], step_index: int) -> Step:
    cmd_type = float(task.get("type"))
    cmd_value = task.get("value")
    retry = task.get("retry", 1)
    step_meta = _build_step_metadata(task, step_index)

    images = Images()
    actions = Actions()
    action_text: List[str] = []
    timeout_sec: Optional[float] = None

    if cmd_type == 1.0:
        images.step_image = str(cmd_value)
        actions.mouse_input.append(
            MouseInput(button="left", action="click", clicks=1)
        )
        action_text.append("Single-click the UI element matched by the template image.")
        timeout_sec = 60.0
    elif cmd_type == 2.0:
        images.step_image = str(cmd_value)
        actions.mouse_input.append(
            MouseInput(button="left", action="double_click", clicks=2)
        )
        action_text.append("Double-click the UI element matched by the template image.")
        timeout_sec = 60.0
    elif cmd_type == 3.0:
        images.step_image = str(cmd_value)
        actions.mouse_input.append(
            MouseInput(button="right", action="click", clicks=1)
        )
        action_text.append("Right-click the UI element matched by the template image.")
        timeout_sec = 60.0
    elif cmd_type == 4.0:
        actions.text_input.append(
            TextInput(text=str(cmd_value), input_method="clipboard")
        )
        action_text.append("Paste text into the active input.")
    elif cmd_type == 5.0:
        actions.wait.append(WaitInput(timeout_sec=float(cmd_value)))
        action_text.append(f"Wait for {cmd_value} seconds.")
    elif cmd_type == 6.0:
        scroll_value = int(cmd_value)
        direction = "up" if scroll_value >= 0 else "down"
        actions.scroll_input.append(
            ScrollInput(direction=direction, amount=abs(scroll_value), unit="raw")
        )
        action_text.append(f"Scroll {direction} by {abs(scroll_value)} raw units.")
    elif cmd_type == 7.0:
        keys = _parse_hotkey(cmd_value)
        actions.key_input.append(KeyInput(keys=keys))
        action_text.append(f"Press the hotkey {'+'.join(keys)}.")
    elif cmd_type == 8.0:
        images.step_image = str(cmd_value)
        actions.hover_input.append(HoverInput())
        action_text.append("Move the mouse to the UI element matched by the template image.")
        timeout_sec = 60.0
    elif cmd_type == 9.0:
        actions.screenshot_input.append(ScreenshotInput(path=str(cmd_value)))
        action_text.append("Capture and save a screenshot.")
    else:
        raise ValueError(f"Unsupported waterRPA command type: {cmd_type}")

    images.result_image = step_meta["result_image"]
    return Step(
        id=step_meta["id"],
        name=step_meta["name"],
        description=f"Imported from waterRPA command type {cmd_type}",
        action=action_text,
        actions=actions,
        images=images,
        expected_result=step_meta["expected_result"],
        timeout_sec=timeout_sec,
        retry=int(retry) if retry is not None else 1,
        element_text=step_meta["element_text"],
        extra={"water_rpa": {"type": cmd_type, "raw_value": cmd_value}},
    )


def parse_water_rpa_tasks(
    tasks: List[Dict[str, Any]],
    workflow_name: str = "waterRPA Workflow",
    software_name: str = "waterRPA",
    software_version: str = "legacy",
) -> Workflow:
    steps = [water_rpa_task_to_step(task, idx) for idx, task in enumerate(tasks)]
    return Workflow(
        name=workflow_name,
        metadata=Metadata(
            title=workflow_name,
            software=Software(name=software_name, version=software_version),
            language="zh-CN",
        ),
        on={"workflow_dispatch": {}},
        jobs={
            "main": Job(
                name=workflow_name,
                runs_on="desktop",
                steps=steps,
                extra={"source": "waterRPA"},
            )
        },
        extra={"source": "waterRPA"},
    )


def load_water_rpa_workflow(path: str | Path) -> Workflow:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        tasks = json.load(handle)
    if not isinstance(tasks, list):
        raise ValueError("waterRPA config must be a list of task objects.")
    workflow = parse_water_rpa_tasks(tasks, workflow_name=config_path.stem)
    return resolve_workflow_paths(workflow, config_path.parent)


def _serialize_agent_s_metadata(step: Step) -> Dict[str, Any]:
    return {
        "step_id": step.id,
        "step_name": step.name,
        "result_image": step.images.result_image,
        "expected_result": step.expected_result,
        "element_text": step.element_text,
    }


def step_to_water_rpa_task(step: Step, strict: bool = True) -> Dict[str, Any]:
    actions = step.actions or Actions()
    non_empty_sections = {
        "mouse_input": actions.mouse_input,
        "key_input": actions.key_input,
        "text_input": actions.text_input,
        "scroll_input": actions.scroll_input,
        "hover_input": actions.hover_input,
        "wait": actions.wait,
        "screenshot_input": actions.screenshot_input,
    }
    used_sections = [name for name, value in non_empty_sections.items() if value]

    if len(used_sections) != 1:
        if strict:
            raise ValueError(
                f"Step '{step.id}' cannot be exported to waterRPA because it uses "
                f"{len(used_sections)} action sections."
            )
        return {}

    section = used_sections[0]
    retry = step.retry if step.retry is not None else 1
    task: Dict[str, Any] = {
        "retry": retry,
        "agent_s": _serialize_agent_s_metadata(step),
    }

    if section == "mouse_input":
        if len(actions.mouse_input) != 1:
            raise ValueError(f"Step '{step.id}' has multiple mouse actions.")
        item = actions.mouse_input[0]
        if item.modifiers or item.position:
            raise ValueError(
                f"Step '{step.id}' uses modifiers or explicit coordinates that waterRPA does not support."
            )
        value = step.images.step_image
        if not value:
            raise ValueError(f"Step '{step.id}' is missing step_image for mouse export.")

        if (item.button or "left") == "right":
            task.update({"type": 3.0, "value": value})
        elif (item.action or "click") == "double_click" or (item.clicks or 1) == 2:
            task.update({"type": 2.0, "value": value})
        else:
            task.update({"type": 1.0, "value": value})
    elif section == "text_input":
        if len(actions.text_input) != 1:
            raise ValueError(f"Step '{step.id}' has multiple text actions.")
        task.update({"type": 4.0, "value": actions.text_input[0].text or ""})
    elif section == "wait":
        if len(actions.wait) != 1:
            raise ValueError(f"Step '{step.id}' has multiple wait actions.")
        task.update({"type": 5.0, "value": actions.wait[0].timeout_sec or 0})
    elif section == "scroll_input":
        if len(actions.scroll_input) != 1:
            raise ValueError(f"Step '{step.id}' has multiple scroll actions.")
        item = actions.scroll_input[0]
        value = int(item.amount or 0)
        if (item.direction or "down").lower() in {"down", "left"}:
            value = -value
        task.update({"type": 6.0, "value": value})
    elif section == "key_input":
        if len(actions.key_input) != 1:
            raise ValueError(f"Step '{step.id}' has multiple key actions.")
        task.update({"type": 7.0, "value": "+".join(actions.key_input[0].keys)})
    elif section == "hover_input":
        if len(actions.hover_input) != 1:
            raise ValueError(f"Step '{step.id}' has multiple hover actions.")
        value = step.images.step_image
        if not value:
            raise ValueError(f"Step '{step.id}' is missing step_image for hover export.")
        task.update({"type": 8.0, "value": value})
    elif section == "screenshot_input":
        if len(actions.screenshot_input) != 1:
            raise ValueError(f"Step '{step.id}' has multiple screenshot actions.")
        task.update({"type": 9.0, "value": actions.screenshot_input[0].path or ""})
    else:
        raise ValueError(f"Unsupported export action section: {section}")

    return task


def workflow_to_water_rpa_tasks(
    workflow: Workflow,
    job_id: Optional[str] = None,
    strict: bool = True,
) -> List[Dict[str, Any]]:
    if not workflow.jobs:
        return []

    if job_id is None:
        job = next(iter(workflow.jobs.values()))
    else:
        if job_id not in workflow.jobs:
            raise KeyError(f"Workflow job '{job_id}' does not exist.")
        job = workflow.jobs[job_id]

    tasks: List[Dict[str, Any]] = []
    for step in job.steps:
        task = step_to_water_rpa_task(step, strict=strict)
        if task:
            tasks.append(task)
    return tasks


__all__ = [
    "load_water_rpa_workflow",
    "parse_water_rpa_tasks",
    "step_to_water_rpa_task",
    "water_rpa_task_to_step",
    "workflow_to_water_rpa_tasks",
]
