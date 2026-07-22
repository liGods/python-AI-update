"""Read-only candidate wrapper; candidates never create card groups themselves."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class CandidateDecision:
    effective_action: str
    physical_action: str
    action_type: str
    projection: Any
    score: tuple[Any, ...]
    hero_skill_evaluation: Mapping[str, Any] | None
    table_pressure: Mapping[str, Any]
    tactical_utility: Mapping[str, Any]
    game_stage: str = "midgame"
    hand_expansion: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze nested score payloads as well as the dataclass fields."""

        if self.hero_skill_evaluation is not None:
            object.__setattr__(self, "hero_skill_evaluation", MappingProxyType(dict(self.hero_skill_evaluation)))
        object.__setattr__(self, "table_pressure", MappingProxyType(dict(self.table_pressure)))
        object.__setattr__(self, "tactical_utility", MappingProxyType(dict(self.tactical_utility)))
        object.__setattr__(self, "hand_expansion", MappingProxyType(dict(self.hand_expansion)))

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CandidateDecision":
        return cls(
            effective_action=str(record["action"]),
            physical_action=str(record["physical_action"]),
            action_type=str(record["action_type"]),
            projection=record["projection"],
            score=tuple(record["score"]),
            hero_skill_evaluation=record.get("hero_skill_evaluation"),
            table_pressure=dict(record.get("table_pressure", {})),
            tactical_utility=dict(record.get("tactical_utility", {})),
            game_stage=str(record.get("game_stage", "midgame")),
            hand_expansion=dict(record.get("hand_expansion", {})),
        )

    @property
    def terminal(self) -> bool:
        return bool(self.projection.terminal)

    @property
    def uses_control(self) -> bool:
        return any(card in {"2", "B", "R"} for card in self.effective_action)
