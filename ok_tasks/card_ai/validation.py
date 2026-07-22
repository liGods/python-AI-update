from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path
from typing import Any

from ok_tasks.card_ai.engine import BaiJiangPaiEngine
from ok_tasks.card_ai.heroes import SIMULATED_HEROES
from ok_tasks.card_ai.policies import RandomLegalPolicy, StableRulePolicy
from ok_tasks.card_ai.schema import POSITIONS
from ok_tasks.card_ai.trajectory import atomic_json


def run_property_validation(target_steps: int = 10_000_000, seed: int = 20260718) -> dict[str, Any]:
    completed_steps = 0
    completed_games = 0
    failures = []
    while completed_steps < target_steps:
        game_seed = seed + completed_games
        heroes = {
            position: SIMULATED_HEROES[(completed_games * len(POSITIONS) + index) % len(SIMULATED_HEROES)]
            for index, position in enumerate(POSITIONS)
        }
        engine = BaiJiangPaiEngine.create(game_seed, heroes)
        policy = RandomLegalPolicy(game_seed) if completed_games % 5 == 0 else StableRulePolicy()
        try:
            for _ in range(1000):
                actions = engine.legal_actions()
                if not actions:
                    raise AssertionError("非终局没有合法动作")
                engine.step(policy.select(engine, actions))
                completed_steps += 1
                if engine.state.terminal or completed_steps >= target_steps:
                    break
            if not engine.state.terminal and completed_steps < target_steps:
                raise AssertionError("牌局超过1000步")
            completed_games += int(engine.state.terminal)
        except Exception as error:
            failures.append(
                {
                    "seed": game_seed,
                    "heroes": heroes,
                    "error": str(error),
                    "current_player": engine.state.current_player,
                    "pending_interaction": engine.state.pending_interaction,
                    "target_ranks": list(engine.state.target_ranks),
                    "hands": {
                        position: [card.rank for card in player.hand]
                        for position, player in engine.state.players.items()
                    },
                    "history_tail": engine.state.history[-5:],
                }
            )
            break
    return {
        "requested_steps": target_steps,
        "completed_steps": completed_steps,
        "completed_games": completed_games,
        "passed": not failures and completed_steps >= target_steps,
        "failures": failures,
    }


def run_parallel_property_validation(
    target_steps: int = 10_000_000, seed: int = 20260718, workers: int = 12
) -> dict[str, Any]:
    worker_count = max(1, int(workers))
    if worker_count == 1:
        return run_property_validation(target_steps, seed)
    steps_per_worker = ceil(target_steps / worker_count)
    seeds = [seed + index * 10_000_000 for index in range(worker_count)]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(run_property_validation, [steps_per_worker] * worker_count, seeds))
    failures = [failure for result in results for failure in result.get("failures", [])]
    completed_steps = sum(int(result.get("completed_steps", 0)) for result in results)
    return {
        "requested_steps": target_steps,
        "completed_steps": completed_steps,
        "completed_games": sum(int(result.get("completed_games", 0)) for result in results),
        "workers": worker_count,
        "passed": not failures and completed_steps >= target_steps,
        "failures": failures,
    }


def run_resumable_property_validation(
    target_steps: int = 10_000_000,
    seed: int = 20260718,
    workers: int = 12,
    checkpoint_path: str | Path = "data/card_ai/training/property_validation.json",
    chunk_steps: int = 5000,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    state = {
        "target_steps": int(target_steps),
        "seed": int(seed),
        "completed_steps": 0,
        "completed_games": 0,
        "chunks": 0,
        "passed": False,
        "failures": [],
    }
    if checkpoint.is_file():
        try:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        except ValueError:
            saved = {}
        if int(saved.get("seed", -1)) == int(seed):
            state.update(saved)
            state["target_steps"] = max(int(state.get("target_steps", 0)), int(target_steps))
    while int(state["completed_steps"]) < int(target_steps) and not state.get("failures"):
        remaining = int(target_steps) - int(state["completed_steps"])
        requested = min(max(1, int(chunk_steps)), remaining)
        chunk_seed = int(seed) + int(state["completed_steps"]) * 10
        result = run_parallel_property_validation(requested, chunk_seed, workers)
        completed = min(requested, int(result.get("completed_steps", 0)))
        state["completed_steps"] = int(state["completed_steps"]) + completed
        state["completed_games"] = int(state.get("completed_games", 0)) + int(result.get("completed_games", 0))
        state["chunks"] = int(state.get("chunks", 0)) + 1
        state["last_chunk"] = result
        state["failures"] = list(result.get("failures", []))
        state["passed"] = not state["failures"] and int(state["completed_steps"]) >= int(target_steps)
        atomic_json(checkpoint, state)
        if completed == 0:
            break
    return state
