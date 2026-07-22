"""Stable structured explanation helpers for candidate logs."""

from __future__ import annotations

from typing import Any

from .candidate import CandidateDecision
from .features import projection_features, score_components, skill_utility_features


def structured_score(candidate: CandidateDecision, labels: tuple[str, ...]) -> dict[str, Any]:
    return {
        "legacy_sort_key": list(candidate.score[:-1]),
        "components": score_components(candidate, labels),
        "projection": projection_features(candidate),
        "skill_utility": skill_utility_features(candidate),
        "table_pressure": dict(candidate.table_pressure),
        "tactical_utility": dict(candidate.tactical_utility),
        "game_stage": candidate.game_stage,
    }
