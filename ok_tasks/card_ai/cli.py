from __future__ import annotations

import argparse
import json

from ok_tasks.card_ai.heroes import HERO_REGISTRY, PASSIVE_OWNED_HEROES, SIMULATED_HEROES, iter_unverified_skills
from ok_tasks.card_ai.quality import RuntimeQualityGate, collect_runtime_quality
from ok_tasks.card_ai.self_play import SelfPlayConfig, SelfPlayRunner
from ok_tasks.card_ai.sim2real import Sim2RealCalibrator
from ok_tasks.card_ai.validation import run_resumable_property_validation


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="百将牌规则模拟与验证工具")
    commands = parser.add_subparsers(dest="command", required=True)
    heroes = commands.add_parser("heroes", help="显示英雄注册与验证状态")
    heroes.set_defaults(handler=_heroes)
    simulate = commands.add_parser("simulate", help="生成确定性规则自对弈轨迹")
    simulate.add_argument("--games", type=int, default=1000)
    simulate.add_argument("--seed", type=int, default=20260718)
    simulate.add_argument("--workers", type=int, default=12)
    simulate.add_argument("--output", default="data/card_ai/rule_simulation")
    simulate.set_defaults(handler=_simulate)
    validate = commands.add_parser("validate", help="运行模拟器随机属性测试")
    validate.add_argument("--steps", type=int, default=10_000_000)
    validate.add_argument("--seed", type=int, default=20260718)
    validate.add_argument("--workers", type=int, default=12)
    validate.add_argument("--chunk-steps", type=int, default=5000)
    validate.add_argument("--checkpoint", default="data/card_ai/validation/property_validation.json")
    validate.set_defaults(handler=_validate)
    quality = commands.add_parser("quality", help="检查自动化运行质量")
    quality.add_argument("--runs", default="data/card_ai/runs")
    quality.set_defaults(handler=_quality)
    sim2real = commands.add_parser("sim2real", help="分析真实手牌变化与模拟预测差异")
    sim2real.add_argument("--runs", default="data/card_ai/runs")
    sim2real.add_argument("--output", default="data/card_ai/sim2real_report.json")
    sim2real.set_defaults(handler=_sim2real)
    return parser


def _heroes(_) -> None:
    _print({
        "registered_heroes": len(HERO_REGISTRY),
        "simulated_heroes": list(SIMULATED_HEROES),
        "passive_curriculum_heroes": list(PASSIVE_OWNED_HEROES),
        "unverified_skills": [skill.to_dict() for skill in iter_unverified_skills()],
    })


def _simulate(args) -> None:
    _print(SelfPlayRunner().run_parallel(SelfPlayConfig(args.games, args.seed), args.output, args.workers))


def _validate(args) -> None:
    _print(run_resumable_property_validation(
        args.steps, args.seed, args.workers, checkpoint_path=args.checkpoint, chunk_steps=args.chunk_steps
    ))


def _quality(args) -> None:
    _print(RuntimeQualityGate().evaluate(collect_runtime_quality(args.runs)).to_dict())


def _sim2real(args) -> None:
    _print(Sim2RealCalibrator().analyze(args.runs, args.output))


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
