from __future__ import annotations


def evaluate_public_projection(projection: object, *, maximizing: bool) -> float:
    """Evaluate only public projected fields; no opponent cards are inspected."""

    value = 1000.0 if projection.terminal else 0.0
    value += 120.0 if projection.enemy_emergency_block else 0.0
    value -= float(projection.worst_remaining_turns) * 12.0
    value -= float(projection.control_card_cost) * 3.0
    return value if maximizing else -value
