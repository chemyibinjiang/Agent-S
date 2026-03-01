"""Backward-compatible schema re-export for YAML-based workflows."""

try:
    from instruction.workflow_schema import *  # type: ignore[F403]
except ImportError:
    from gui_agents.s3.instruction.workflow_schema import *  # type: ignore[F403]

