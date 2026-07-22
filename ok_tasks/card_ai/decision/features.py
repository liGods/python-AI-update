"""Structured score inputs retained from the legacy candidate record."""

from __future__ import annotations

from typing import Any

from .candidate import CandidateDecision


_STAGE_COMPONENT_LABELS = (
    "stage_structure_protection",
    "stage_control_preservation",
    "stage_initiative",
    "stage_continuous_route",
    "stage_exact_remaining_turns",
    "stage_emergency_block",
)


def score_components(candidate: CandidateDecision, labels: tuple[str, ...]) -> dict[str, Any]:
    """Expose every legacy score component without changing sort semantics."""

    values = candidate.score[:-1]
    components = {
        label: values[index] if index < len(values) else None
        for index, label in enumerate(labels)
    }
    stage_start = len(values) - len(_STAGE_COMPONENT_LABELS)
    for index in range(len(labels), max(len(labels), stage_start)):
        components[f"legacy_component_{index + 1}"] = values[index]
    if stage_start >= len(labels):
        for index, label in enumerate(_STAGE_COMPONENT_LABELS):
            components[label] = values[stage_start + index]
    return components


def projection_features(candidate: CandidateDecision) -> dict[str, Any]:
    projection = candidate.projection
    return {
        "expected_remaining_turns": projection.expected_remaining_turns,
        "worst_remaining_turns": projection.worst_remaining_turns,
        "expected_remaining_cards": projection.expected_remaining_cards,
        "worst_remaining_cards": projection.worst_remaining_cards,
        "enemy_finish_risk": projection.enemy_finish_risk,
        "control_card_cost": projection.control_card_cost,
    }


def skill_utility_features(candidate: CandidateDecision) -> dict[str, Any]:
    """Expose the one projection-based skill utility used by every candidate.

    These values are already part of ``SkillProjection.score_key`` and the live
    legacy sort key.  Keeping them together makes hero benefit comparable for
    ordinary plays, pass projections, and simulator legal actions without
    introducing a second scoring formula.
    """

    projection = candidate.projection
    return {
        "triggered_rules": list(projection.triggered_rules),
        "triggered_skills": list(projection.triggered_skills),
        "resource_value": projection.skill_resource_value,
        "expected_risk": projection.expected_skill_risk,
        "worst_risk": projection.worst_skill_risk,
        "external_skill_cost": projection.external_skill_cost,
        "target_relation_cost": projection.target_relation_cost,
        "resource_changes": dict(projection.resource_changes),
        "random_branch_count": len(projection.random_branches),
        "hand_expansion": dict(candidate.hand_expansion),
    }
