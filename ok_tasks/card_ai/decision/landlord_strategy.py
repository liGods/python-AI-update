"""Public-information soft ranking for landlord legal candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .candidate import CandidateDecision
from .context import DecisionContext


@dataclass(frozen=True)
class LandlordScoreWeights:
    route: float
    reclaim: float
    structure: float
    control_preservation: float
    emergency_block: float


LANDLORD_SCORE_WEIGHTS: dict[str, LandlordScoreWeights] = {
    "opening": LandlordScoreWeights(0.80, 0.70, 1.40, 1.50, 0.40),
    "midgame": LandlordScoreWeights(1.30, 1.40, 1.15, 1.10, 0.85),
    "endgame": LandlordScoreWeights(1.65, 1.25, 0.80, 0.65, 1.35),
    "emergency": LandlordScoreWeights(1.45, 1.50, 0.45, 0.15, 2.00),
}

_CONTINUOUS_TYPES = frozenset({"straight", "pair_chain", "airplane", "trio_solo", "trio_pair"})


def landlord_score_key(
    context: DecisionContext,
    candidate: CandidateDecision,
    *,
    is_bomb: Callable[[str], bool],
    rank_index: Callable[[str], int],
    baseline_turns: int,
) -> tuple[object, ...]:
    """Rank a legal landlord candidate; no action is generated here."""

    weights = LANDLORD_SCORE_WEIGHTS.get(candidate.game_stage, LANDLORD_SCORE_WEIGHTS["midgame"])
    emergency = context.urgent or candidate.game_stage == "emergency"
    projection = candidate.projection
    control_cost = float(candidate.uses_control or is_bomb(candidate.effective_action))
    structure_cost = 0.0 if candidate.action_type in _CONTINUOUS_TYPES else 1.0
    route_turns = float(projection.worst_remaining_turns)
    short_route_cost = 0.0 if route_turns <= 3 else route_turns - 3
    reclaim_cost = 0.0 if context.target and candidate.effective_action else 1.0
    emergency_cost = 0.0 if not emergency or projection.enemy_emergency_block or candidate.terminal else 1.0
    high_rank = max((rank_index(card) for card in candidate.effective_action), default=0)
    return (
        0 if candidate.terminal else 1,
        round(weights.emergency_block * emergency_cost, 6),
        -high_rank if emergency else 0,
        round(weights.control_preservation * control_cost, 6) if not emergency else 0.0,
        round(weights.route * short_route_cost, 6),
        round(weights.route * route_turns, 6),
        round(weights.reclaim * reclaim_cost, 6),
        round(weights.structure * structure_cost, 6) if not context.target else 0.0,
        -len(candidate.effective_action) if not context.target else 0,
        high_rank,
        candidate.score,
    )


def select_landlord_candidate(
    context: DecisionContext,
    candidates: Sequence[CandidateDecision],
    *,
    is_bomb: Callable[[str], bool],
    rank_index: Callable[[str], int],
    baseline_turns: int,
) -> CandidateDecision | None:
    """Return the best existing legal candidate for a landlord decision."""

    if context.position != "landlord" or not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: landlord_score_key(
            context,
            candidate,
            is_bomb=is_bomb,
            rank_index=rank_index,
            baseline_turns=baseline_turns,
        ),
    )
