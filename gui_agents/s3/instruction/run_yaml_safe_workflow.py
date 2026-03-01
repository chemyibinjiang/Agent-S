from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
GUI_AGENTS_ROOT = REPO_ROOT / "gui_agents" / "s3"

for candidate in (REPO_ROOT, GUI_AGENTS_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    from instruction.yaml.yaml_instruction_auto_executor import (
        SafeWorkflowError,
        StepExecutionResult,
        run_safe_workflow,
    )
    from instruction.yaml.yaml_instruction_parser import load_instruction
except ImportError:
    from gui_agents.s3.instruction.yaml.yaml_instruction_auto_executor import (
        SafeWorkflowError,
        StepExecutionResult,
        run_safe_workflow,
    )
    from gui_agents.s3.instruction.yaml.yaml_instruction_parser import load_instruction


def _format_result(result: StepExecutionResult) -> str:
    verification = result.verification
    parts = [f"{result.step_id} ({result.step_name})"]
    if verification.result_image_required:
        if verification.result_image_confidence is not None:
            parts.append(f"result_image={verification.result_image_confidence:.3f}")
        else:
            parts.append("result_image=matched")
    if verification.expected_result_required:
        parts.append(f"expected_result={verification.expected_result_matched}")
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a YAML safe workflow on the current desktop.")
    parser.add_argument("workflow", help="Path to the workflow YAML file.")
    parser.add_argument("--job-id", default=None, help="Optional workflow job id to run.")
    parser.add_argument(
        "--template-threshold",
        type=float,
        default=0.8,
        help="Template matching threshold for step_image/result_image verification.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of executed steps before aborting to prevent infinite loops.",
    )
    args = parser.parse_args()

    workflow_path = Path(args.workflow).resolve()
    workflow = load_instruction(workflow_path)

    print(f"Workflow: {workflow.name}")
    print(f"File: {workflow_path}")
    if "endnote" in workflow.name.lower():
        print("This workflow is the EndNote Find Full Text example.")

    try:
        results = run_safe_workflow(
            workflow,
            job_id=args.job_id,
            template_threshold=args.template_threshold,
            max_steps=args.max_steps,
        )
    except SafeWorkflowError as exc:
        print(f"FAILED: {exc}")
        if exc.step_id:
            print(f"Step: {exc.step_id}")
        return 2

    print("SUCCESS")
    for result in results:
        print(_format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
