from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SCHEMA_VERSION = 3
POSITIONS = ("landlord", "landlord_down", "landlord_up")


@dataclass
class CardInstance:
    """A physical or skill-created card whose provenance survives rank changes."""

    card_id: str
    rank: str
    original_rank: str
    owner: str
    source: str = "deck"
    tags: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tags"] = sorted(self.tags)
        return value


@dataclass
class PlayerState:
    position: str
    hero: str | None = None
    hand: list[CardInstance] = field(default_factory=list)
    skill_uses: dict[str, int] = field(default_factory=dict)
    marks: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, reveal_hand: bool = True) -> dict[str, Any]:
        value = {
            "position": self.position,
            "hero": self.hero,
            "hand_count": len(self.hand),
            "skill_uses": dict(self.skill_uses),
            "marks": dict(self.marks),
            "extra": dict(self.extra),
        }
        if reveal_hand:
            value["hand"] = [card.to_dict() for card in self.hand]
        return value


@dataclass(frozen=True)
class LegalAction:
    action_id: str
    kind: Literal["play", "pass", "skill", "interaction"]
    actor: str
    card_ids: tuple[str, ...] = ()
    ranks: tuple[str, ...] = ()
    action_type: str = "none"
    target: str | None = None
    skill: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["card_ids"] = list(self.card_ids)
        value["ranks"] = list(self.ranks)
        return value


@dataclass
class FullGameState:
    game_id: str
    seed: int
    players: dict[str, PlayerState]
    current_player: str = "landlord"
    landlord: str = "landlord"
    target_ranks: list[str] = field(default_factory=list)
    target_card_ids: list[str] = field(default_factory=list)
    target_action_type: str = "none"
    trick_owner: str | None = None
    consecutive_passes: int = 0
    turn_index: int = 0
    next_card_sequence: int = 1000
    terminal: bool = False
    winner: str | None = None
    bottom_cards: list[CardInstance] = field(default_factory=list)
    played_cards: list[CardInstance] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    event_queue: list[dict[str, Any]] = field(default_factory=list)
    pending_interaction: dict[str, Any] | None = None
    interaction_queue: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "game_id": self.game_id,
            "seed": self.seed,
            "players": {key: value.to_dict(True) for key, value in self.players.items()},
            "current_player": self.current_player,
            "landlord": self.landlord,
            "target_ranks": list(self.target_ranks),
            "target_card_ids": list(self.target_card_ids),
            "target_action_type": self.target_action_type,
            "trick_owner": self.trick_owner,
            "consecutive_passes": self.consecutive_passes,
            "turn_index": self.turn_index,
            "next_card_sequence": self.next_card_sequence,
            "terminal": self.terminal,
            "winner": self.winner,
            "bottom_cards": [card.to_dict() for card in self.bottom_cards],
            "played_cards": [card.to_dict() for card in self.played_cards],
            "history": list(self.history),
            "event_queue": list(self.event_queue),
            "pending_interaction": self.pending_interaction,
            "interaction_queue": list(self.interaction_queue),
        }


@dataclass(frozen=True)
class Observation:
    game_id: str
    observer: str
    hand: tuple[dict[str, Any], ...]
    current_player: str
    landlord: str
    target_ranks: tuple[str, ...]
    target_action_type: str
    trick_owner: str | None
    opponent_card_counts: tuple[int, int]
    history: tuple[dict[str, Any], ...]
    hero: str | None
    hero_state: dict[str, Any]
    pending_interaction: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["hand"] = list(self.hand)
        value["target_ranks"] = list(self.target_ranks)
        value["opponent_card_counts"] = list(self.opponent_card_counts)
        value["history"] = list(self.history)
        return value

    def to_legacy_model_state(self) -> dict[str, Any]:
        return {
            "hand_cards": [card["rank"] for card in self.hand],
            "table_cards": list(self.target_ranks),
            "position": self.observer,
            "opponent_card_counts": list(self.opponent_card_counts),
            "history": [list(event.get("ranks", [])) for event in self.history if event.get("kind") == "play"],
            "hero": self.hero,
            "hero_state": dict(self.hero_state),
            "round_id": f"{self.game_id}_t{len(self.history) + 1}",
        }


@dataclass
class StepResult:
    state: FullGameState
    action: LegalAction
    events: list[dict[str, Any]]
    rewards: dict[str, float]
    terminal: bool


@dataclass(frozen=True)
class HeroSkillSpec:
    hero: str
    name: str
    trigger: str
    limit: int | None
    effect: str
    category: str = "passive"
    interactive: bool = False
    verified: bool = True
    live_verified: bool = False
    rule_id: str = ""
    rules_version: str = "3p.1"
    supported_modes: tuple[str, ...] = ("landlord_3p",)
    choice_kind: str = "automatic"
    optional: bool = False
    documented: bool = True
    projection_verified: bool = False
    sim_verified: bool = False
    ui_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryEvent:
    game_id: str
    sequence: int
    event_type: str
    actor: str | None = None
    observation: dict[str, Any] | None = None
    legal_actions: tuple[dict[str, Any], ...] = ()
    chosen_action: dict[str, Any] | None = None
    rewards: dict[str, float] = field(default_factory=dict)
    terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["legal_actions"] = list(self.legal_actions)
        return value
