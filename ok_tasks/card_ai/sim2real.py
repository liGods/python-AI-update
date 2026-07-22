from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ok_tasks.card_ai.trajectory import atomic_json


@dataclass(frozen=True)
class TransitionComparison:
    game_id: str
    round_id: str
    matched: bool
    expected_count: int | None
    observed_count: int | None
    expected_cards: tuple[str, ...]
    observed_cards: tuple[str, ...]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "round_id": self.round_id,
            "matched": self.matched,
            "expected_count": self.expected_count,
            "observed_count": self.observed_count,
            "expected_cards": list(self.expected_cards),
            "observed_cards": list(self.observed_cards),
            "reason": self.reason,
        }


def compare_visible_transition(
    game_id: str,
    round_id: str,
    before_hand: list[str],
    action: list[str],
    observed_hand: list[str],
    ledger_events: list[dict[str, Any]] | None = None,
) -> TransitionComparison:
    expected = Counter(before_hand)
    expected.subtract(action)
    for event in ledger_events or []:
        cards = list(event.get("cards", []))
        event_type = event.get("ledger_event_type") or event.get("event_type")
        if event_type in {"gain", "recover"}:
            expected.update(cards)
        elif event_type == "discard":
            expected.subtract(cards)
        elif event_type == "transform":
            expected.subtract(event.get("old_cards", []))
            expected.update(cards)
    expected_cards = sorted([rank for rank, count in expected.items() for _ in range(max(0, count))])
    observed_cards = sorted(observed_hand)
    matched = expected_cards == observed_cards
    reason = "" if matched else "手牌变化包含技能、识别误差或模拟规则差异"
    return TransitionComparison(
        game_id,
        round_id,
        matched,
        len(expected_cards),
        len(observed_cards),
        tuple(expected_cards),
        tuple(observed_cards),
        reason,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            values.append(json.loads(line))
        except ValueError:
            values.append({"event_type": "corrupt_json"})
    return values


class Sim2RealCalibrator:
    def analyze(self, runs_root: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
        comparisons: list[TransitionComparison] = []
        corrupt_games = []
        for events_path in sorted(Path(runs_root).glob("*/games/game_*/events.jsonl")):
            events = _read_jsonl(events_path)
            if any(event.get("event_type") == "corrupt_json" for event in events):
                corrupt_games.append(str(events_path.parent))
                continue
            states = [event for event in events if event.get("event_type") == "decision_state"]
            submitted = {
                event.get("round_id"): event
                for event in events
                if event.get("event_type") == "action_submitted" and event.get("round_id")
            }
            for before, after in zip(states, states[1:]):
                action_event = submitted.get(before.get("round_id"))
                if action_event is None:
                    continue
                action = list(action_event.get("action", []))
                if not action:
                    continue
                before_sequence = int(before.get("sequence", 0))
                after_sequence = int(after.get("sequence", 0))
                ledger_events = [
                    event
                    for event in events
                    if event.get("event_type") == "card_ledger"
                    and before_sequence < int(event.get("sequence", 0)) < after_sequence
                ]
                comparisons.append(
                    compare_visible_transition(
                        events_path.parent.name,
                        str(before.get("round_id", "")),
                        list(before.get("hand_cards", [])),
                        action,
                        list(after.get("hand_cards", [])),
                        ledger_events,
                    )
                )
        matches = sum(comparison.matched for comparison in comparisons)
        report = {
            "transitions": len(comparisons),
            "matches": matches,
            "mismatches": len(comparisons) - matches,
            "agreement": matches / len(comparisons) if comparisons else None,
            "passed": bool(comparisons) and matches / len(comparisons) >= 0.995,
            "corrupt_games": corrupt_games,
            "hard_cases": [comparison.to_dict() for comparison in comparisons if not comparison.matched][:1000],
        }
        if output_path is not None:
            atomic_json(output_path, report)
        return report
