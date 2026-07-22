"""Behavior-preserving soft selection over immutable legal candidates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .candidate import CandidateDecision
from .context import DecisionContext
from .landlord_strategy import select_landlord_candidate
from .farmer_strategy import select_farmer_candidate
from .team_strategy import _NO_TEAM_DECISION, select_team_candidate


def select_soft_candidate(
    context: DecisionContext,
    candidates: Sequence[CandidateDecision],
    *,
    is_bomb: Callable[[str], bool],
    rank_index: Callable[[str], int],
    baseline_turns: Callable[[], int],
    pass_projection: Callable[[], Any],
) -> CandidateDecision | None:
    """Apply the legacy soft-policy order without emitting a new card group.

    ``None`` is the existing pass result.  The rules below intentionally retain
    their historical order while being isolated from legality and hard safety.
    """

    if not candidates:
        return None
    target = context.target
    urgent = context.urgent
    winning = [candidate for candidate in candidates if candidate.terminal]
    teammate_takeover: CandidateDecision | None = None
    teammate_count = context.teammate_count
    nearest_enemy = context.nearest_enemy

    if context.position != "landlord":
        if context.protect_teammate_play or context.teammate_count in {1, 2}:
            team_choice = select_team_candidate(context, candidates, rank_index=rank_index)
            if team_choice is not _NO_TEAM_DECISION:
                return team_choice
        if target and urgent and not context.protect_teammate_play:
            farmer_choice = select_farmer_candidate(
                context, candidates, is_bomb=is_bomb, rank_index=rank_index, baseline_turns=baseline_turns()
            )
            if farmer_choice is not None:
                return farmer_choice

    if not target and not urgent and not winning and context.position != "landlord":
        preliminary = min(candidates, key=lambda candidate: candidate.score)
        if preliminary.action_type == "solo" and preliminary.uses_control:
            ordinary = [
                candidate for candidate in candidates
                if candidate.action_type == "solo"
                and not candidate.uses_control
                and candidate.projection.expected_remaining_turns
                <= preliminary.projection.expected_remaining_turns + 0.5
                and candidate.projection.worst_remaining_turns
                <= preliminary.projection.worst_remaining_turns + 1
            ]
            if ordinary:
                return min(
                    ordinary,
                    key=lambda candidate: max(rank_index(card) for card in candidate.effective_action),
                )

    if target and context.protect_teammate_play and urgent and (teammate_count is None or teammate_count > 2):
        economical = [
            candidate for candidate in candidates
            if not is_bomb(candidate.effective_action) and not candidate.uses_control
        ]
        return min(economical or candidates, key=lambda candidate: candidate.score)

    medium_team_pressure = bool(
        target and context.protect_teammate_play and teammate_count is not None
        and nearest_enemy <= 10 and teammate_count >= 12
        and teammate_count - nearest_enemy >= 4
    )
    if medium_team_pressure:
        economical = [
            candidate for candidate in candidates
            if not is_bomb(candidate.effective_action) and not candidate.uses_control
        ]
        if economical:
            return min(economical, key=lambda candidate: candidate.score)
        pair_twos = [
            candidate for candidate in candidates
            if nearest_enemy <= 8 and candidate.effective_action == "22"
        ]
        if pair_twos:
            return pair_twos[0]

    if target and context.protect_teammate_play and not winning and (teammate_count is None or teammate_count > 5):
        takeover_types = {"straight", "pair_chain", "airplane", "trio_solo", "trio_pair"}
        current_turns = baseline_turns()
        takeover = [
            candidate for candidate in candidates
            if candidate.action_type in takeover_types
            and not is_bomb(candidate.effective_action)
            and not candidate.uses_control
            and candidate.projection.expected_remaining_turns < current_turns
        ]
        if takeover:
            teammate_takeover = min(takeover, key=lambda candidate: candidate.score)
    if target and context.protect_teammate_play and not winning and teammate_takeover is None:
        return None

    landlord_best = select_landlord_candidate(
        context,
        candidates,
        is_bomb=is_bomb,
        rank_index=rank_index,
        baseline_turns=baseline_turns(),
    )
    best = teammate_takeover or landlord_best or min(candidates, key=lambda candidate: candidate.score)
    farmer_route_press: CandidateDecision | None = None
    if target and context.position != "landlord" and not context.protect_teammate_play and not winning:
        current_turns = baseline_turns()
        route = [
            candidate for candidate in candidates
            if not is_bomb(candidate.effective_action)
            and not candidate.uses_control
            and candidate.projection.expected_remaining_turns < current_turns
        ]
        if route:
            farmer_route_press = min(route, key=lambda candidate: candidate.score)
            best = farmer_route_press

    remaining_turns = best.projection.worst_remaining_turns
    if (
        target and context.position != "landlord" and not context.protect_teammate_play and not urgent
        and not winning and nearest_enemy > 10 and remaining_turns > 1 and best.uses_control
    ):
        return None
    reserved_control_only = bool(
        target and context.position != "landlord"
        and all(is_bomb(candidate.effective_action) or candidate.uses_control for candidate in candidates)
    )
    if not urgent and not winning and farmer_route_press is None and reserved_control_only and remaining_turns > 1:
        return None
    if (
        target and context.position != "landlord" and not urgent and not winning
        and farmer_route_press is None and teammate_takeover is None and remaining_turns > 1
        and best.tactical_utility["total"] < 0
    ):
        return None
    if target and not urgent and all(is_bomb(candidate.effective_action) for candidate in candidates) and remaining_turns > 1:
        return None
    if target and not urgent:
        projected_pass = pass_projection()
        if projected_pass.triggered_rules and (
            projected_pass.expected_remaining_turns,
            projected_pass.worst_remaining_turns,
        ) < (
            best.projection.expected_remaining_turns,
            best.projection.worst_remaining_turns,
        ):
            return None
    return best
