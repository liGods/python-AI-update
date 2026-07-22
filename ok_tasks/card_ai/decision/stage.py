"""Public-information game-stage classification and score weights."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .context import DecisionContext


class GameStage(StrEnum):
    OPENING = "opening"
    MIDGAME = "midgame"
    ENDGAME = "endgame"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class StageContext:
    """Only public state needed to classify the strategic stage."""

    own_card_count: int
    position: str
    enemy_card_counts: tuple[int, ...]
    teammate_card_count: int | None
    table_has_cards: bool
    table_is_teammate: bool
    seen_bombs: int
    seen_jokers: int
    seen_twos: int
    one_turn_finish_risk: bool

    @classmethod
    def from_decision(cls, context: DecisionContext) -> "StageContext":
        history = context.hero_context.history
        seen = [rank for event in history for rank in event.get("ranks", ())]
        seen.extend(context.target)
        counts = {rank: seen.count(rank) for rank in set(seen)}
        seen_bombs = sum(count >= 4 for rank, count in counts.items() if rank not in {"X", "D"})
        nearest_enemy = context.nearest_enemy
        table_size = len(context.target)
        finish_risk = nearest_enemy <= 1 or (table_size > 0 and nearest_enemy == table_size)
        return cls(
            own_card_count=len(context.hand),
            position=context.position,
            enemy_card_counts=context.enemy_counts,
            teammate_card_count=context.teammate_count,
            table_has_cards=bool(context.target),
            table_is_teammate=context.protect_teammate_play,
            seen_bombs=seen_bombs,
            seen_jokers=sum(counts.get(rank, 0) for rank in ("B", "R", "X", "D")),
            seen_twos=counts.get("2", 0),
            one_turn_finish_risk=finish_risk,
        )

    @property
    def nearest_enemy(self) -> int:
        return min(self.enemy_card_counts or (17,))


@dataclass(frozen=True)
class StageWeights:
    structure_protection: float
    control_preservation: float
    initiative: float
    continuous_route: float
    exact_remaining_turns: float
    emergency_block: float


STAGE_WEIGHTS: Mapping[GameStage, StageWeights] = {
    GameStage.OPENING: StageWeights(1.40, 1.35, 0.60, 0.80, 0.55, 0.20),
    GameStage.MIDGAME: StageWeights(0.85, 0.80, 1.35, 1.40, 0.90, 0.65),
    GameStage.ENDGAME: StageWeights(0.60, 0.55, 0.85, 0.95, 1.55, 1.10),
    GameStage.EMERGENCY: StageWeights(0.35, 0.15, 1.50, 0.70, 1.25, 1.70),
}


def classify_game_stage(context: StageContext) -> GameStage:
    if context.one_turn_finish_risk:
        return GameStage.EMERGENCY
    if context.nearest_enemy <= (3 if context.position == "landlord" else 5):
        return GameStage.EMERGENCY
    if context.nearest_enemy <= 6 and (context.seen_bombs >= 2 or context.seen_jokers >= 2 or context.seen_twos >= 4):
        return GameStage.EMERGENCY
    if context.teammate_card_count is not None and context.teammate_card_count <= 2:
        return GameStage.ENDGAME
    if context.own_card_count <= 10 or context.nearest_enemy <= 6:
        return GameStage.ENDGAME
    if context.nearest_enemy <= 10 or context.table_has_cards:
        return GameStage.MIDGAME
    return GameStage.OPENING


def stage_score_components(
    stage: GameStage,
    action_type: str,
    effective_action: str,
    projection: object,
) -> tuple[float, ...]:
    """Score an already-legal action without generating any card group."""

    weights = STAGE_WEIGHTS[stage]
    structure_cost = 0.0 if action_type in {"straight", "pair_chain", "airplane", "trio_solo", "trio_pair"} else 1.0
    control_cost = float(projection.control_card_cost)
    initiative_cost = 0.0 if effective_action else 1.0
    route_cost = float(projection.expected_remaining_turns)
    exact_turn_cost = float(projection.worst_remaining_turns)
    emergency_cost = 0.0 if projection.enemy_emergency_block or projection.terminal else 1.0
    return (
        round(weights.structure_protection * structure_cost, 6),
        round(weights.control_preservation * control_cost, 6),
        round(weights.initiative * initiative_cost, 6),
        round(weights.continuous_route * route_cost, 6),
        round(weights.exact_remaining_turns * exact_turn_cost, 6),
        round(weights.emergency_block * emergency_cost, 6),
    )


def stage_score_key(stage: GameStage, candidate: object) -> tuple[float, ...]:
    """Compatibility wrapper for a CandidateDecision-like object."""

    return stage_score_components(
        stage,
        candidate.action_type,
        candidate.effective_action,
        candidate.projection,
    )


# Backward-compatible name for callers introduced by the earlier refactor.
DecisionStage = GameStage


def classify_stage(context: DecisionContext) -> GameStage:
    return classify_game_stage(StageContext.from_decision(context))
