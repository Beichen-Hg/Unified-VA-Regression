"""Explicit release state for prospective matched training, with no model imports."""
from __future__ import annotations

V2_SCOPE = "two_week_v2_gpu_handoff_and_execution"


def apply_recorded_v2_release(cfg: dict, release: dict) -> bool:
    """Carry one explicit task-wide instruction into derived effective configs.

    Pending records cannot authorize anything. This only changes the caller's
    in-memory effective config, never its historical or candidate source file.
    """
    if release.get("execution_authorized") is not True:
        return False
    if (release.get("scope") != V2_SCOPE
            or release.get("own_holder_handoff_authorized") is not True
            or not release.get("authorization_source")):
        raise PermissionError("the v2 task-wide execution/handoff instruction is incomplete")
    cfg["matched_protocol"].update(
        author_approval="approved", execution_authorized=True,
        author_approval_scope=V2_SCOPE,
    )
    return True


def require_training_release(cfg: dict) -> None:
    protocol = cfg.get("matched_protocol", {}) or {}
    if not protocol.get("enabled"):
        return
    if protocol.get("author_approval") != "approved":
        raise PermissionError("The new recipe/training protocol remains unapproved; a route decision alone does not release training.")
    if protocol.get("execution_authorized") is not True:
        raise PermissionError("Training execution is not released; current authorization covers CPU preparation only.")
