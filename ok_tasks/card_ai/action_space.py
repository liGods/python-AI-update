"""The canonical dynamic action space for simulation and policy scoring."""

from __future__ import annotations

from typing import Any

from ok_tasks.card_ai.rlcard_adapter import action_type, legal_actions, to_internal, to_rlcard
from ok_tasks.card_ai.rules import select_card_ids
from ok_tasks.card_ai.schema import CardInstance, LegalAction


class UnifiedActionSpace:
    """Build all decisions from one schema: plays, pass, skills, and choices."""

    @staticmethod
    def enumerate_plays(actor: str, hand: list[CardInstance], target_ranks: list[str]) -> list[LegalAction]:
        result: list[LegalAction] = []
        for effective_action in legal_actions(to_rlcard([card.rank for card in hand]), to_rlcard(target_ranks)):
            ranks = tuple(to_internal(effective_action))
            card_ids = select_card_ids(hand, ranks)
            if card_ids is None:
                continue
            physical_ranks = tuple(next(card.rank for card in hand if card.card_id == card_id) for card_id in card_ids)
            result.append(
                LegalAction(
                    action_id=f"play:{actor}:{','.join(card_ids)}:{','.join(ranks)}",
                    kind="play",
                    actor=actor,
                    card_ids=card_ids,
                    ranks=ranks,
                    action_type=action_type(effective_action),
                    parameters={"physical_ranks": list(physical_ranks)},
                )
            )
        return result

    @staticmethod
    def enumerate_interactions(pending: dict[str, Any]) -> list[LegalAction]:
        actor, skill = pending["actor"], pending["skill"]
        actions = []
        for index, option in enumerate(pending.get("options", [])):
            card_ids = tuple(option.get("card_ids", []))
            ranks = tuple(option.get("ranks", [option["rank"]] if option.get("rank") else []))
            actions.append(LegalAction(
                f"interaction:{actor}:{skill}:{index}", "interaction", actor,
                card_ids=card_ids, ranks=ranks, target=option.get("target"), skill=skill,
                parameters={"option_index": index, **option},
            ))
        if pending.get("optional"):
            actions.append(LegalAction(
                f"interaction:{actor}:{skill}:skip", "interaction", actor, skill=skill,
                parameters={"skip": True},
            ))
        return actions
