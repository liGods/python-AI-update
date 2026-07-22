from __future__ import annotations

import hashlib
import random
from collections import Counter
from time import perf_counter
from typing import Any

import numpy as np

from ok_tasks.ai_model_adapter import CARD_ORDER
from ok_tasks.card_ai.engine import BaiJiangPaiEngine
from ok_tasks.card_ai.policies import StableRulePolicy
from ok_tasks.card_ai.schema import CardInstance, FullGameState, PlayerState, POSITIONS


def _known_ranks(state: dict[str, Any]) -> list[str]:
    values = list(state.get("hand_cards", []))
    for action in state.get("history", []):
        if isinstance(action, dict):
            values.extend(action.get("ranks", []))
        else:
            values.extend(action or [])
    return values


def _determinize(state: dict[str, Any], sample_index: int) -> BaiJiangPaiEngine:
    observer = state.get("position", "landlord_down")
    if observer not in POSITIONS:
        observer = "landlord_down"
    payload = json_fingerprint(state, sample_index)
    rng = random.Random(payload)
    remaining = Counter({rank: 4 for rank in CARD_ORDER[:13]})
    remaining.update({"X": 1, "D": 1})
    remaining.subtract(_known_ranks(state))
    pool = [rank for rank, count in remaining.items() for _ in range(max(0, count))]
    rng.shuffle(pool)
    opponents = [position for position in POSITIONS if position != observer]
    requested_counts = list(state.get("opponent_card_counts", [17, 17]))[:2]
    requested_counts += [17] * (2 - len(requested_counts))
    skill_estimates = [max(0, int(value)) for value in list(state.get("opponent_skill_card_estimates", [0, 0]))[:2]]
    skill_estimates += [0] * (2 - len(skill_estimates))
    players = {position: PlayerState(position) for position in POSITIONS}
    players[observer].hero = state.get("hero")
    players[observer].skill_uses = dict(state.get("hero_state", {}).get("skill_uses", {}))
    players[observer].marks = dict(state.get("hero_state", {}).get("marks", {}))
    counter = 0

    def make(rank: str, owner: str, source: str = "determinization") -> CardInstance:
        nonlocal counter
        counter += 1
        return CardInstance(f"i{sample_index:04d}_{counter:04d}", rank, rank, owner, source)

    players[observer].hand = [make(rank, observer, "observed") for rank in state.get("hand_cards", [])]
    for position, count, skill_count in zip(opponents, requested_counts, skill_estimates):
        inferred_skill_count = min(int(count), skill_count)
        natural_ranks = []
        while len(natural_ranks) < int(count) - inferred_skill_count:
            natural_ranks.append(pool.pop() if pool else rng.choice(CARD_ORDER))
        skill_ranks = [rng.choice(CARD_ORDER) for _ in range(inferred_skill_count)]
        players[position].hand = [make(rank, position) for rank in natural_ranks] + [make(rank, position, "observed_skill_gain") for rank in skill_ranks]
    target = list(state.get("table_cards", []))
    target_owner = POSITIONS[(POSITIONS.index(observer) - 1) % len(POSITIONS)] if target else None
    target_cards = [make(rank, "table", "public_history") for rank in target]
    full = FullGameState(
        game_id=f"search_{payload}",
        seed=payload,
        players=players,
        current_player=observer,
        target_ranks=target,
        target_card_ids=[card.card_id for card in target_cards],
        target_action_type=state.get("table_action_type", "unknown"),
        trick_owner=target_owner,
        played_cards=target_cards,
        history=[],
    )
    return BaiJiangPaiEngine(full)


def json_fingerprint(state: dict[str, Any], sample_index: int) -> int:
    stable = repr(
        (
            state.get("round_id"),
            tuple(state.get("hand_cards", [])),
            tuple(state.get("table_cards", [])),
            tuple(state.get("opponent_card_counts", [])),
            tuple(state.get("opponent_skill_card_estimates", [])),
            sample_index,
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(stable).digest()[:8], "big")


def information_set_search(
    state: dict[str, Any],
    candidates: list[dict[str, Any]],
    prior_scores: np.ndarray,
    budget_ms: int = 300,
    maximum_rollout_steps: int = 200,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = perf_counter()
    deadline = start + max(0, min(int(budget_ms), 1200)) / 1000.0
    if len(candidates) <= 1 or budget_ms <= 0:
        return np.asarray(prior_scores, dtype=np.float32), {"samples": 0, "elapsed_ms": 0.0, "reason": "search_not_needed"}
    totals = np.zeros(len(candidates), dtype=np.float64)
    visits = np.zeros(len(candidates), dtype=np.int64)
    sample_index = 0
    policy = StableRulePolicy()
    while perf_counter() < deadline:
        candidate_index = sample_index % len(candidates)
        try:
            engine = _determinize(state, sample_index)
            ranks = list(candidates[candidate_index].get("cards", candidates[candidate_index].get("ranks", [])))
            action = next(
                action
                for action in engine.legal_actions()
                if action.kind == "play" and list(action.ranks) == ranks
            )
            observer = engine.state.current_player
            result = engine.step(action)
            for _ in range(maximum_rollout_steps):
                if result.terminal or perf_counter() >= deadline:
                    break
                actions = engine.legal_actions()
                if not actions:
                    break
                result = engine.step(policy.select(engine, actions))
            if result.terminal:
                totals[candidate_index] += result.rewards[observer]
                visits[candidate_index] += 1
        except (StopIteration, RuntimeError, ValueError):
            pass
        sample_index += 1
    values = np.divide(totals, np.maximum(visits, 1), dtype=np.float64)
    normalized_prior = np.asarray(prior_scores, dtype=np.float64)
    if normalized_prior.size:
        normalized_prior = (normalized_prior - normalized_prior.mean()) / (normalized_prior.std() + 1e-6)
    combined = normalized_prior + values * 0.75
    elapsed = (perf_counter() - start) * 1000.0
    return combined.astype(np.float32), {
        "samples": int(visits.sum()),
        "attempts": sample_index,
        "visits": visits.tolist(),
        "values": values.tolist(),
        "elapsed_ms": round(elapsed, 3),
    }
