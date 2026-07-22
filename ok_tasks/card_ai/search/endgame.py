"""Deterministic public-information endgame search with alpha-beta pruning."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Sequence

from .budget import SearchBudget
from .evaluator import evaluate_public_projection
from .transposition import TranspositionEntry, TranspositionTable


@dataclass(frozen=True)
class EndgameSearchResult:
    candidate: object
    nodes: int
    depth: int
    elapsed_ms: float
    reason: str
    triggered: bool


def should_search(context: object, candidates: Sequence[object], rule_best: object) -> bool:
    enemy_counts = tuple(getattr(context, "enemy_counts", (17, 17)))
    total = len(getattr(context, "hand", "")) + sum(enemy_counts)
    close_enemy = min(enemy_counts or (17,)) <= 5
    expensive = bool(getattr(rule_best, "uses_control", False))
    close_score = len(candidates) >= 2 and tuple(candidates[0].score[:-1]) == tuple(candidates[1].score[:-1])
    return close_enemy or total <= 18 or expensive or close_score


def search_public_endgame(context: object, candidates: Sequence[object], rule_best: object, budget_ms: int = 40) -> EndgameSearchResult:
    """Search legal own candidates against an abstract paranoid opponent response.

    Opponent branches contain only public remaining-card counts, never sampled or
    reconstructed hands.  The result therefore remains safe for live decisions.
    """
    budget = SearchBudget(budget_ms)
    if not should_search(context, candidates, rule_best):
        return EndgameSearchResult(rule_best, 0, 0, budget.elapsed_ms, "not_triggered", False)
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.score))
    table = TranspositionTable()
    nodes = 0

    def value(candidate: object, depth: int, alpha: float, beta: float) -> float:
        nonlocal nodes
        nodes += 1
        if budget.expired or depth == 0 or candidate.projection.terminal:
            return evaluate_public_projection(candidate.projection, maximizing=True)
        key = (candidate.effective_action, candidate.projection.worst_remaining_turns, depth)
        cached = table.get_at_depth(key, depth)
        if cached is not None:
            return cached
        # Paranoid opponent: public worst-case continuation is a one-turn delay.
        opponent_value = -evaluate_public_projection(candidate.projection, maximizing=False) - 8.0
        best = min(beta, opponent_value)
        table[key] = TranspositionEntry(depth, best)
        return best

    best = rule_best
    reached = 0
    for depth in range(1, 5):
        if budget.expired:
            break
        reached = depth
        current = best
        current_value = -inf
        for candidate in ordered:
            if budget.expired:
                break
            candidate_value = value(candidate, depth, -inf, inf)
            if candidate_value > current_value:
                current, current_value = candidate, candidate_value
        best = current
    proven_override = best.projection.terminal or (
        best.projection.enemy_emergency_block and not rule_best.projection.enemy_emergency_block
    )
    chosen = best if proven_override else rule_best
    reason = "timeout_rule_fallback" if budget.expired else "proven_public_override" if proven_override else "rule_fallback_no_proof"
    return EndgameSearchResult(chosen, nodes, reached, budget.elapsed_ms, reason, True)
