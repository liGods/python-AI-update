"""Public-information soft ranking for farmer candidates against the landlord."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .candidate import CandidateDecision
from .context import DecisionContext


FARMER_ROLE_WEIGHTS = {
    "landlord_down": {"reclaim": 1.35, "structure": 0.85},
    "landlord_up": {"reclaim": 1.00, "structure": 1.25},
}


def select_farmer_candidate(
    context: DecisionContext,
    candidates: Sequence[CandidateDecision],
    *,
    is_bomb: Callable[[str], bool],
    rank_index: Callable[[str], int],
    baseline_turns: int,
) -> CandidateDecision | None:
    """Rank existing legal farmer actions without observing hidden cards."""

    if context.position == "landlord" or not candidates:
        return None
    weights = FARMER_ROLE_WEIGHTS.get(context.position, FARMER_ROLE_WEIGHTS["landlord_down"])
    urgent = context.urgent
    def key(candidate: CandidateDecision) -> tuple[object, ...]:
        projection = candidate.projection
        control_cost = float(candidate.uses_control or is_bomb(candidate.effective_action))
        reclaim_cost = 0.0 if context.target and candidate.effective_action else 1.0
        structure_cost = 0.0 if candidate.action_type in {"straight", "pair_chain", "airplane", "trio_solo", "trio_pair"} else 1.0
        block_cost = 0.0 if not urgent or projection.enemy_emergency_block or candidate.terminal else 1.0
        return (
            0 if candidate.terminal else 1,
            block_cost,
            0.0 if urgent else control_cost,
            round(weights["reclaim"] * reclaim_cost, 6),
            projection.worst_remaining_turns,
            round(weights["structure"] * structure_cost, 6),
            max((rank_index(card) for card in candidate.effective_action), default=99),
            candidate.score,
        )
    return min(candidates, key=key)
