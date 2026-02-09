from __future__ import annotations

from egologqa.ui_text import recommended_action_copy


def test_recommended_action_copy_is_populated_for_known_tokens() -> None:
    for token in (
        "USE_FULL_SEQUENCE",
        "USE_SEGMENTS_ONLY",
        "FIX_TIME_ALIGNMENT",
        "RECAPTURE_OR_SKIP",
    ):
        copy = recommended_action_copy(
            action_token=token,
            gate="WARN",
            fail_reasons=[],
            warn_reasons=["WARN_SYNC_P95_GT_WARN"],
        )
        assert copy["what_to_do"].strip()
        assert copy["why"].strip()


def test_recommended_action_copy_fallback_for_unknown_token() -> None:
    copy = recommended_action_copy(
        action_token="UNKNOWN_ACTION",
        gate="WARN",
        fail_reasons=[],
        warn_reasons=[],
    )
    assert "not recognized" in copy["why"]
