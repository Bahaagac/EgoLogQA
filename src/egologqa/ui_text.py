from __future__ import annotations

from collections.abc import Sequence


def recommended_action_copy(
    action_token: str,
    gate: str,
    fail_reasons: Sequence[str] | None = None,
    warn_reasons: Sequence[str] | None = None,
) -> dict[str, str]:
    fail = list(fail_reasons or [])
    warn = list(warn_reasons or [])

    if action_token == "USE_FULL_SEQUENCE":
        return {
            "what_to_do": "Use the full recording as-is.",
            "why": "No warning or fail reasons were reported for this run.",
        }
    if action_token == "USE_SEGMENTS_ONLY":
        why = (
            "Some parts of the recording did not meet quality checks, so segment-only use is safer."
            if warn or fail
            else "This action limits use to cleaner time ranges."
        )
        return {
            "what_to_do": "Use only the clean segments listed in this report.",
            "why": why,
        }
    if action_token == "FIX_TIME_ALIGNMENT":
        return {
            "what_to_do": "Apply time alignment correction between RGB and depth, then re-run validation.",
            "why": "Sync offset-related timing issues were detected and are often recoverable by alignment.",
        }
    if action_token == "RECAPTURE_OR_SKIP":
        why = (
            "Critical fail reasons were detected and likely cannot be fixed by segment filtering."
            if fail
            else "Critical quality issues were detected."
        )
        return {
            "what_to_do": "Recapture this sequence or skip it for downstream use.",
            "why": why,
        }

    return {
        "what_to_do": "Review the quality sections and choose a conservative handling strategy.",
        "why": (
            "This action token is not recognized by the current UI mapping."
            if gate
            else "No gate context is available."
        ),
    }
