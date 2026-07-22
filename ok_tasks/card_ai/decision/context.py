"""Immutable context for one public-information card decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DecisionContext:
    hand: str
    target: str
    enemy_counts: tuple[int, ...]
    hero: str | None
    last_action_type: str | None
    policy_id: str
    protect_teammate_play: bool
    hero_state: Mapping[str, Any]
    pressure: Mapping[str, Any]
    hero_context: Any

    @property
    def position(self) -> str:
        return str(self.pressure.get("position", "landlord_down"))

    @property
    def urgent(self) -> bool:
        threshold = 3 if self.position == "landlord" else 5
        return min(self.enemy_counts or (17, 17)) <= threshold

    @property
    def nearest_enemy(self) -> int:
        return min(self.enemy_counts or (17,))

    @property
    def teammate_count(self) -> int | None:
        value = self.pressure.get("teammate_count")
        return int(value) if isinstance(value, int) else None
