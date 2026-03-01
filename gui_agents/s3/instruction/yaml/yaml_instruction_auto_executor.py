from __future__ import annotations

from io import BytesIO
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
from cv2.typing import MatLike
import numpy as np
from PIL import Image
import pyautogui
import pytesseract
import pyperclip

from core.observation import Observation
from agents.execution_summary import ExecutionSummary
from utils.tesseract_utils import configure_pytesseract

try:
    from instruction.workflow_schema import Actions, Job, Step, TemplateMatching, Workflow
except ImportError:
    from gui_agents.s3.instruction.workflow_schema import Actions, Job, Step, TemplateMatching, Workflow

configure_pytesseract()


@dataclass
class TemplateMatch:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    scale: float = 1.0


@dataclass
class StepVerificationResult:
    passed: bool
    result_image_required: bool = False
    result_image_matched: bool = True
    result_image_confidence: Optional[float] = None
    expected_result_required: bool = False
    expected_result_matched: bool = True
    observed_text: Optional[str] = None
    message: str = ""


@dataclass
class StepExecutionResult:
    step_id: str
    step_name: str
    before_observation: Observation
    after_observation: Observation
    anchor: Optional[TemplateMatch]
    verification: StepVerificationResult


class SafeWorkflowError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        step_id: Optional[str] = None,
        observation: Optional[Observation] = None,
        verification: Optional["StepVerificationResult"] = None,
    ) -> None:
        super().__init__(message)
        self.step_id = step_id
        self.observation = observation
        self.verification = verification


class StepPreconditionError(SafeWorkflowError):
    pass


class StepVerificationError(SafeWorkflowError):
    pass


def _load_image(path: bytes| str | Path) -> MatLike:
    if isinstance(path, bytes):
        image = Image.open(BytesIO(path)).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    else:
        image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return image


def _pil_image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def capture_desktop_observation() -> Observation:
    screenshot = pyautogui.screenshot()
    screenshot_bytes = _pil_image_to_png_bytes(screenshot)
    return Observation(
        screenshot=screenshot_bytes,
        original_screenshot=screenshot_bytes,
    )


def _capture_observation(
    observation_provider: Optional[Callable[[], Observation]],
) -> Observation:
    if observation_provider is not None:
        obs = observation_provider()
        if obs is None:
            raise ValueError("Observation provider returned None.")
        return obs
    return capture_desktop_observation()


def _resolve_match_threshold(
    matching: TemplateMatching,
    fallback_threshold: float,
) -> float:
    if matching.threshold is None:
        return fallback_threshold
    return max(0.0, min(float(matching.threshold), 1.0))


def _resolve_scale_candidates(matching: TemplateMatching) -> List[float]:
    scales: List[float] = []

    for scale in matching.scales:
        if scale is not None and scale > 0:
            scales.append(float(scale))

    if not scales and (matching.min_scale is not None or matching.max_scale is not None):
        min_scale = float(matching.min_scale if matching.min_scale is not None else 1.0)
        max_scale = float(matching.max_scale if matching.max_scale is not None else min_scale)
        if min_scale > max_scale:
            min_scale, max_scale = max_scale, min_scale

        scale_step = float(matching.scale_step if matching.scale_step is not None else 0.05)
        if scale_step <= 0:
            scale_step = 0.05

        current = min_scale
        generated = 0
        while current <= max_scale + 1e-9 and generated < 200:
            if current > 0:
                scales.append(round(current, 4))
            current += scale_step
            generated += 1

    if not scales:
        scales = [1.0]

    unique_scales = sorted({round(scale, 4) for scale in scales if scale > 0}, key=lambda value: abs(value - 1.0))
    return unique_scales or [1.0]


def _resize_template(template: MatLike, scale: float) -> Optional[MatLike]:
    if scale <= 0:
        return None
    if abs(scale - 1.0) < 1e-6:
        return template

    height, width = template.shape[:2]
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    if scaled_width == width and scaled_height == height:
        return template

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(template, (scaled_width, scaled_height), interpolation=interpolation)


def _match_template(
    screenshot: MatLike,
    template: MatLike,
    *,
    scale: float,
) -> Optional[TemplateMatch]:
    screenshot_height, screenshot_width = screenshot.shape[:2]
    template_height, template_width = template.shape[:2]
    if template_height > screenshot_height or template_width > screenshot_width:
        return None

    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return TemplateMatch(
        x=max_loc[0],
        y=max_loc[1],
        width=template_width,
        height=template_height,
        confidence=float(max_val),
        scale=scale,
    )


def find_template_bbox(
    screenshot_path: bytes,
    template_path: str | Path,
    threshold: float = 0.8,
    matching: Optional[TemplateMatching] = None,
) -> Optional[TemplateMatch]:
    screenshot = _load_image(screenshot_path)
    template = _load_image(Path(template_path))
    config = matching or TemplateMatching()
    best_match: Optional[TemplateMatch] = None

    for scale in _resolve_scale_candidates(config):
        scaled_template = _resize_template(template, scale)
        if scaled_template is None:
            continue
        candidate = _match_template(screenshot, scaled_template, scale=scale)
        if candidate is None:
            continue
        if best_match is None or candidate.confidence > best_match.confidence:
            best_match = candidate

    if best_match is None or best_match.confidence < threshold:
        return None
    return best_match


def _find_text_bbox(image: MatLike, query: str) -> Optional[TemplateMatch]:
    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    for i in range(len(ocr_result["text"])):
        if query.lower() in ocr_result["text"][i].lower():
            return TemplateMatch(
                x=ocr_result["left"][i],
                y=ocr_result["top"][i],
                width=ocr_result["width"][i],
                height=ocr_result["height"][i],
                confidence=1.0,
            )
    return None


def _extract_ocr_text(image: MatLike) -> str:
    return pytesseract.image_to_string(image)


def _normalize_text(text: Optional[str]) -> str:
    return "".join((text or "").lower().split())


def _locate_step_anchor(
    step: Step,
    obs: Observation,
    template_threshold: float = 0.8,
) -> Optional[TemplateMatch]:
    step_image = step.images.step_image if step.images else None
    screenshot = _load_image(obs.original_screenshot)
    match_threshold = _resolve_match_threshold(step.template_matching, template_threshold)

    if step_image:
        bbox = find_template_bbox(
            obs.original_screenshot,
            step_image,
            threshold=match_threshold,
            matching=step.template_matching,
        )
        if bbox is not None:
            return bbox

    if step.element_text:
        return _find_text_bbox(screenshot, step.element_text)

    return None


def _should_require_anchor(step: Step) -> bool:
    return bool((step.images and step.images.step_image) or step.element_text)


def _find_anchor_with_retry(
    step: Step,
    obs: Observation,
    template_threshold: float = 0.8,
    default_timeout_sec: float = 60.0,
    observation_provider: Optional[Callable[[], Observation]] = None,
) -> Tuple[Optional[TemplateMatch], Observation]:
    current_obs = obs
    if not _should_require_anchor(step):
        return None, current_obs

    retry = step.retry if step.retry is not None else 1
    timeout_sec = step.timeout_sec if step.timeout_sec is not None else default_timeout_sec
    attempt = 0
    start_time = time.time()

    while True:
        bbox = _locate_step_anchor(step, current_obs, template_threshold=template_threshold)
        if bbox is not None:
            return bbox, current_obs

        attempt += 1
        timed_out = timeout_sec is not None and (time.time() - start_time) >= timeout_sec
        exhausted_retries = retry != -1 and attempt >= max(retry, 1)
        if timed_out or exhausted_retries:
            return None, current_obs
        time.sleep(0.1)
        current_obs = _capture_observation(observation_provider)


def _center_of_bbox(bbox: TemplateMatch) -> Tuple[int, int]:
    return (bbox.x + bbox.width // 2, bbox.y + bbox.height // 2)


def _resolve_point(position: Optional[Dict[str, Any]], bbox: Optional[TemplateMatch]) -> Optional[Tuple[int, int]]:
    if isinstance(position, dict) and "x" in position and "y" in position:
        return (int(position["x"]), int(position["y"]))
    if bbox is None:
        return None
    return _center_of_bbox(bbox)


def _apply_mouse_input(actions: Actions, bbox: Optional[TemplateMatch]) -> None:
    for item in actions.mouse_input:
        point = _resolve_point(item.position, bbox)
        if point is not None:
            pyautogui.moveTo(point[0], point[1])

        if item.modifiers:
            for key in item.modifiers:
                pyautogui.keyDown(key)

        if item.action in {"press", "long_press"}:
            pyautogui.mouseDown(button=item.button or "left")
            if item.duration_ms:
                time.sleep(item.duration_ms / 1000)
            if item.action != "press":
                pyautogui.mouseUp(button=item.button or "left")
        elif item.action == "release":
            pyautogui.mouseUp(button=item.button or "left")
        elif item.action == "double_click":
            pyautogui.doubleClick(button=item.button or "left")
        else:
            pyautogui.click(button=item.button or "left", clicks=item.clicks or 1)

        if item.modifiers:
            for key in reversed(item.modifiers):
                pyautogui.keyUp(key)


def _apply_key_input(actions: Actions) -> None:
    for item in actions.key_input:
        keys = item.keys or []
        if not keys:
            continue
        if item.hold_ms:
            for key in keys:
                pyautogui.keyDown(key)
            time.sleep(item.hold_ms / 1000)
            for key in reversed(keys):
                pyautogui.keyUp(key)
        elif len(keys) > 1:
            pyautogui.hotkey(*keys)
        else:
            pyautogui.press(keys[0], presses=item.repeat or 1)


def _apply_text_input(actions: Actions) -> None:
    for item in actions.text_input:
        if item.clear_before:
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("backspace")
        if item.text:
            if (item.input_method or "").lower() == "clipboard":
                pyperclip.copy(item.text)
                pyautogui.hotkey("ctrl", "v")
            else:
                pyautogui.write(item.text)


def _apply_scroll_input(actions: Actions, bbox: Optional[TemplateMatch]) -> None:
    for item in actions.scroll_input:
        point = _resolve_point(item.position, bbox)
        if point is not None:
            pyautogui.moveTo(point[0], point[1])

        amount = item.amount or 0
        unit = (item.unit or "notch").lower()
        if unit == "page":
            amount *= 800
        elif unit == "notch":
            amount *= 120
        elif unit == "raw":
            amount = amount

        direction = (item.direction or "down").lower()
        if direction in {"left", "right"}:
            pyautogui.hscroll(amount if direction == "right" else -amount)
        else:
            pyautogui.scroll(amount if direction == "up" else -amount)


def _apply_drag_drop(actions: Actions, bbox: Optional[TemplateMatch]) -> None:
    for item in actions.drag_drop:
        from_point = _resolve_point(item.from_target if isinstance(item.from_target, dict) else None, bbox)
        to_point = _resolve_point(item.to_target if isinstance(item.to_target, dict) else None, bbox)
        if from_point is None and isinstance(item.from_target, str) and bbox is not None:
            from_point = _center_of_bbox(bbox)
        if to_point is None and isinstance(item.to_target, str) and bbox is not None:
            to_point = _center_of_bbox(bbox)
        if from_point is None or to_point is None:
            continue
        pyautogui.moveTo(from_point[0], from_point[1])
        pyautogui.mouseDown(button=item.button or "left")
        pyautogui.moveTo(to_point[0], to_point[1], duration=(item.duration_ms or 200) / 1000)
        pyautogui.mouseUp(button=item.button or "left")


def _apply_hover_input(actions: Actions, bbox: Optional[TemplateMatch]) -> None:
    for item in actions.hover_input:
        point = _resolve_point(None, bbox)
        if point is None:
            continue
        pyautogui.moveTo(point[0], point[1])
        if item.duration_ms:
            time.sleep(item.duration_ms / 1000)


def _apply_window_input(actions: Actions) -> None:
    for item in actions.window_input:
        if item.action == "close":
            pyautogui.hotkey("alt", "f4")
        elif item.action == "minimize":
            pyautogui.hotkey("win", "down")
        elif item.action == "maximize":
            pyautogui.hotkey("win", "up")


def _apply_file_input(actions: Actions) -> None:
    for item in actions.file_input:
        if item.path:
            pyautogui.write(item.path)
        if item.filename:
            pyautogui.write(item.filename)
        pyautogui.press("enter")


def _apply_clipboard_input(actions: Actions) -> None:
    for item in actions.clipboard_input:
        if item.action == "copy":
            pyautogui.hotkey("ctrl", "c")
        elif item.action == "cut":
            pyautogui.hotkey("ctrl", "x")
        elif item.action == "paste":
            pyautogui.hotkey("ctrl", "v")
        elif item.action == "set_text" and item.text:
            pyautogui.write(item.text)


def _apply_wait(actions: Actions) -> None:
    for item in actions.wait:
        if item.timeout_sec:
            time.sleep(item.timeout_sec)


def _apply_screenshot_input(actions: Actions) -> None:
    for item in actions.screenshot_input:
        if not item.path:
            continue

        target = item.path
        if os.path.isdir(target):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(target, f"screenshot_{timestamp}.png")
        else:
            filename = target
            if not filename.endswith((".png", ".jpg", ".bmp")):
                filename += ".png"

        pyautogui.screenshot(filename)


def _has_postconditions(step: Step) -> bool:
    return bool((step.images and step.images.result_image) or step.expected_result)


def verify_step_result(
    step: Step,
    obs: Observation,
    template_threshold: float = 0.8,
) -> StepVerificationResult:
    result_image_required = bool(step.images and step.images.result_image)
    expected_result_required = bool(step.expected_result)
    match_threshold = _resolve_match_threshold(step.template_matching, template_threshold)

    result_image_match = None
    result_image_matched = True
    if result_image_required:
        result_image_match = find_template_bbox(
            obs.original_screenshot,
            step.images.result_image,
            threshold=match_threshold,
            matching=step.template_matching,
        )
        result_image_matched = result_image_match is not None

    observed_text = None
    expected_result_matched = True
    if expected_result_required:
        observed_text = _extract_ocr_text(_load_image(obs.original_screenshot))
        expected_result_matched = (
            _normalize_text(step.expected_result) in _normalize_text(observed_text)
        )

    passed = result_image_matched and expected_result_matched
    problems: List[str] = []
    if result_image_required and not result_image_matched:
        problems.append("result_image not matched")
    if expected_result_required and not expected_result_matched:
        problems.append("expected_result text not found in OCR output")
    message = "Verification passed."
    if problems:
        message = "; ".join(problems)
        if observed_text:
            message += f" OCR='{observed_text.strip()[:200]}'"

    return StepVerificationResult(
        passed=passed,
        result_image_required=result_image_required,
        result_image_matched=result_image_matched,
        result_image_confidence=result_image_match.confidence if result_image_match else None,
        expected_result_required=expected_result_required,
        expected_result_matched=expected_result_matched,
        observed_text=observed_text,
        message=message,
    )


def _verify_step_result_with_retry(
    step: Step,
    initial_obs: Optional[Observation] = None,
    template_threshold: float = 0.8,
    default_timeout_sec: float = 60.0,
    observation_provider: Optional[Callable[[], Observation]] = None,
) -> Tuple[Observation, StepVerificationResult]:
    current_obs = initial_obs or _capture_observation(observation_provider)
    if not _has_postconditions(step):
        return current_obs, StepVerificationResult(
            passed=True,
            message="No postconditions defined.",
        )

    retry = step.retry if step.retry is not None else 1
    timeout_sec = step.timeout_sec if step.timeout_sec is not None else default_timeout_sec
    attempt = 0
    start_time = time.time()

    while True:
        verification = verify_step_result(
            step,
            current_obs,
            template_threshold=template_threshold,
        )
        if verification.passed:
            return current_obs, verification

        attempt += 1
        timed_out = timeout_sec is not None and (time.time() - start_time) >= timeout_sec
        exhausted_retries = retry != -1 and attempt >= max(retry, 1)
        if timed_out or exhausted_retries:
            return current_obs, verification
        time.sleep(0.1)
        current_obs = _capture_observation(observation_provider)


def get_job_steps(workflow: Workflow, job_id: Optional[str] = None) -> List[Step]:
    if not workflow.jobs:
        return []

    if job_id is None:
        first_job = next(iter(workflow.jobs.values()))
        return first_job.steps

    job = workflow.jobs.get(job_id)
    if job is None:
        raise KeyError(f"Workflow job '{job_id}' does not exist.")
    return job.steps


def _get_workflow_job(workflow: Workflow, job_id: Optional[str] = None) -> Job:
    if not workflow.jobs:
        raise SafeWorkflowError("Workflow does not contain any jobs.")

    if job_id is None:
        return next(iter(workflow.jobs.values()))

    job = workflow.jobs.get(job_id)
    if job is None:
        raise KeyError(f"Workflow job '{job_id}' does not exist.")
    return job


def _build_step_lookup(job: Job) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for index, step in enumerate(job.steps):
        if step.id in lookup:
            raise SafeWorkflowError(f"Duplicate step id '{step.id}' in job '{job.name}'.")
        lookup[step.id] = index
    return lookup


def _resolve_branch_step_index(
    branch_target: Optional[str],
    *,
    step: Step,
    branch_name: str,
    step_lookup: Dict[str, int],
    fallback_index: Optional[int],
) -> Optional[int]:
    if branch_target is None:
        return fallback_index
    if branch_target not in step_lookup:
        raise SafeWorkflowError(
            f"Step '{step.id}' references unknown {branch_name} target '{branch_target}'.",
            step_id=step.id,
        )
    return step_lookup[branch_target]


def execute_workflow_step(
    workflow: Workflow,
    obs: Observation,
    step_index: int,
    job_id: Optional[str] = None,
    template_threshold: float = 0.8,
    observation_provider: Optional[Callable[[], Observation]] = None,
) -> Optional[ExecutionSummary]:
    steps = get_job_steps(workflow, job_id=job_id)
    if step_index < 0 or step_index >= len(steps):
        raise IndexError(f"Step index {step_index} is out of range for workflow.")
    return execute_step(
        steps[step_index],
        obs,
        step_index=step_index,
        template_threshold=template_threshold,
        observation_provider=observation_provider,
    )


def execute_job_step(
    job: Job,
    obs: Observation,
    step_index: int,
    template_threshold: float = 0.8,
    observation_provider: Optional[Callable[[], Observation]] = None,
) -> Optional[ExecutionSummary]:
    if step_index < 0 or step_index >= len(job.steps):
        raise IndexError(f"Step index {step_index} is out of range for job.")
    return execute_step(
        job.steps[step_index],
        obs,
        step_index=step_index,
        template_threshold=template_threshold,
        observation_provider=observation_provider,
    )


def execute_step(
    step: Step,
    obs: Observation,
    step_index: int,
    template_threshold: float = 0.8,
    observation_provider: Optional[Callable[[], Observation]] = None,
) -> Optional[ExecutionSummary]:
    _ = step_index
    bbox, resolved_obs = _find_anchor_with_retry(
        step,
        obs,
        template_threshold=template_threshold,
        observation_provider=observation_provider,
    )

    if bbox is None and _should_require_anchor(step):
        return None

    def call_executable():
        actions = step.actions or Actions()
        time.sleep(step.pre_processing_delay_millisec / 1000 if step.pre_processing_delay_millisec else 0.05)
        _apply_mouse_input(actions, bbox)
        _apply_key_input(actions)
        _apply_text_input(actions)
        _apply_scroll_input(actions, bbox)
        _apply_drag_drop(actions, bbox)
        _apply_hover_input(actions, bbox)
        _apply_window_input(actions)
        _apply_file_input(actions)
        _apply_clipboard_input(actions)
        _apply_wait(actions)
        _apply_screenshot_input(actions)
        time.sleep(step.post_processing_delay_millisec / 1000 if step.post_processing_delay_millisec else 0.05)
        post_action_obs = _capture_observation(observation_provider)
        after_obs, verification = _verify_step_result_with_retry(
            step,
            initial_obs=post_action_obs,
            template_threshold=template_threshold,
            observation_provider=observation_provider,
        )
        if not verification.passed:
            raise StepVerificationError(
                f"Step '{step.id}' failed postcondition verification: {verification.message}",
                step_id=step.id,
                observation=after_obs,
                verification=verification,
            )
        return StepExecutionResult(
            step_id=step.id,
            step_name=step.name,
            before_observation=resolved_obs,
            after_observation=after_obs,
            anchor=bbox,
            verification=verification,
        )

    return ExecutionSummary(
        plan=step.description or "",
        plan_code="",  # No code generation in this executor
        executable=call_executable,
        reflection="called by yaml configuration",
    )


def run_safe_workflow(
    workflow: Workflow,
    obs: Optional[Observation] = None,
    job_id: Optional[str] = None,
    template_threshold: float = 0.8,
    observation_provider: Optional[Callable[[], Observation]] = None,
    max_steps: int = 100,
) -> List[StepExecutionResult]:
    job = _get_workflow_job(workflow, job_id=job_id)
    step_lookup = _build_step_lookup(job)
    current_obs = obs or _capture_observation(observation_provider)
    results: List[StepExecutionResult] = []
    step_index = 0 if job.steps else None
    executed_steps = 0

    while step_index is not None:
        if executed_steps >= max_steps:
            raise SafeWorkflowError(
                f"Workflow exceeded max_steps={max_steps}; possible loop detected."
            )
        executed_steps += 1
        step = job.steps[step_index]
        default_next_index = step_index + 1 if step_index + 1 < len(job.steps) else None
        summary = execute_step(
            step,
            current_obs,
            step_index=step_index,
            template_threshold=template_threshold,
            observation_provider=observation_provider,
        )
        if summary is None:
            failure_index = _resolve_branch_step_index(
                step.on_failure,
                step=step,
                branch_name="on_failure",
                step_lookup=step_lookup,
                fallback_index=None,
            )
            if failure_index is None:
                raise StepPreconditionError(
                    f"Step '{step.id}' could not start because its precondition was not met.",
                    step_id=step.id,
                    observation=current_obs,
                )
            step_index = failure_index
            continue

        try:
            result = summary.call_executable()
        except StepVerificationError as exc:
            failure_index = _resolve_branch_step_index(
                step.on_failure,
                step=step,
                branch_name="on_failure",
                step_lookup=step_lookup,
                fallback_index=None,
            )
            if failure_index is None:
                raise
            current_obs = exc.observation or current_obs
            step_index = failure_index
            continue

        if not isinstance(result, StepExecutionResult):
            raise SafeWorkflowError(
                f"Step '{step.id}' did not return a step execution result.",
                step_id=step.id,
            )
        results.append(result)
        current_obs = result.after_observation
        step_index = _resolve_branch_step_index(
            step.on_success,
            step=step,
            branch_name="on_success",
            step_lookup=step_lookup,
            fallback_index=default_next_index,
        )

    return results


__all__ = [
    "SafeWorkflowError",
    "StepExecutionResult",
    "StepPreconditionError",
    "StepVerificationError",
    "StepVerificationResult",
    "TemplateMatch",
    "capture_desktop_observation",
    "verify_step_result",
    "find_template_bbox",
    "execute_job_step",
    "execute_step",
    "execute_workflow_step",
    "get_job_steps",
    "run_safe_workflow",
]
