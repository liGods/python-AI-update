from __future__ import annotations

import hashlib
import random
from collections import Counter
from itertools import combinations
from typing import Any

from ok_tasks.ai_model_adapter import CARD_ORDER
from ok_tasks.card_ai.heroes import HERO_REGISTRY, normalize_hero_name
from ok_tasks.card_ai.action_space import UnifiedActionSpace
from ok_tasks.card_ai.rules import (
    WILDCARD,
    estimate_route_turns,
    rank_key,
    sorted_ranks,
    validate_card_selection,
)
from ok_tasks.card_ai.schema import (
    POSITIONS,
    CardInstance,
    FullGameState,
    LegalAction,
    Observation,
    PlayerState,
    StepResult,
)


class SimulationError(RuntimeError):
    pass


class BaiJiangPaiEngine:
    """Deterministic three-player engine with explicit skill-card provenance."""

    def __init__(self, state: FullGameState):
        self.state = state

    @classmethod
    def create(
        cls,
        seed: int,
        heroes: dict[str, str | None] | None = None,
        game_id: str | None = None,
    ) -> "BaiJiangPaiEngine":
        hero_map = {position: normalize_hero_name((heroes or {}).get(position)) for position in POSITIONS}
        deck: list[CardInstance] = []
        ranks = [rank for rank in CARD_ORDER[:13] for _ in range(4)] + ["X", "D"]
        for index, rank in enumerate(ranks):
            deck.append(CardInstance(f"d{index:03d}", rank, rank, "deck"))
        random.Random(seed).shuffle(deck)
        players = {position: PlayerState(position, hero_map[position]) for position in POSITIONS}
        deal_order = ("landlord", "landlord_down", "landlord_up")
        for index, card in enumerate(deck[:51]):
            owner = deal_order[index % 3]
            card.owner = owner
            players[owner].hand.append(card)
        bottom_cards = deck[51:]
        for card in bottom_cards:
            card.owner = "landlord"
            players["landlord"].hand.append(card)
        for player in players.values():
            player.hand.sort(key=lambda card: (rank_key(card.rank), card.card_id))
        state = FullGameState(
            game_id=game_id or f"sim_{seed}",
            seed=seed,
            players=players,
            bottom_cards=bottom_cards,
        )
        engine = cls(state)
        engine._schedule_game_start_interactions()
        engine.validate_invariants()
        return engine

    def observe(self, observer: str) -> Observation:
        if observer not in self.state.players:
            raise SimulationError(f"未知观察者: {observer}")
        player = self.state.players[observer]
        ordered_opponents = [position for position in POSITIONS if position != observer]
        hero_state = {
            "skill_uses": dict(player.skill_uses),
            "marks": dict(player.marks),
            "last_action_type": player.extra.get("last_action_type"),
            "extra": dict(player.extra),
        }
        visible_interaction = None
        if self.state.pending_interaction and self.state.pending_interaction.get("actor") == observer:
            visible_interaction = dict(self.state.pending_interaction)
        return Observation(
            game_id=self.state.game_id,
            observer=observer,
            hand=tuple(card.to_dict() for card in player.hand),
            current_player=self.state.current_player,
            landlord=self.state.landlord,
            target_ranks=tuple(self.state.target_ranks),
            target_action_type=self.state.target_action_type,
            trick_owner=self.state.trick_owner,
            opponent_card_counts=tuple(len(self.state.players[position].hand) for position in ordered_opponents),
            history=tuple(dict(event) for event in self.state.history),
            hero=player.hero,
            hero_state=hero_state,
            pending_interaction=visible_interaction,
        )

    def project_action(self, action: LegalAction) -> Any:
        """Project a legal action through the shared hero policy without mutating state."""

        from ok_tasks.card_ai.hero_policy import HeroDecisionContext, evaluate_legal_action

        if action.action_id not in {candidate.action_id for candidate in self.legal_actions()}:
            raise SimulationError(f"动作不在当前合法集合中: {action.action_id}")
        context = HeroDecisionContext.from_engine(self, action.actor)
        return evaluate_legal_action(context, action, route_evaluator=estimate_route_turns)

    def skill_choices(self, skill: str, action: LegalAction | None = None) -> tuple[Any, ...]:
        """Enumerate documented simulator choices, including cancel when allowed."""

        from ok_tasks.card_ai.hero_policy import HeroDecisionContext, enumerate_skill_choices

        context = HeroDecisionContext.from_engine(self)
        return enumerate_skill_choices(context, skill, action)

    def project_skill_choice(self, choice: Any, action: LegalAction | None = None) -> Any:
        """Resolve a simulator skill choice with the same pure handler used by policies."""

        from ok_tasks.card_ai.hero_policy import HeroDecisionContext, apply_skill_choice

        context = HeroDecisionContext.from_engine(self)
        return apply_skill_choice(context, choice, action=action, route_evaluator=estimate_route_turns)

    def legal_actions(self, actor: str | None = None) -> list[LegalAction]:
        if self.state.terminal:
            return []
        if self.state.pending_interaction is not None:
            return UnifiedActionSpace.enumerate_interactions(self.state.pending_interaction)
        actor = actor or self.state.current_player
        if actor != self.state.current_player:
            return []
        player = self.state.players[actor]
        result = UnifiedActionSpace.enumerate_plays(actor, player.hand, self.state.target_ranks)
        if not self.state.target_ranks and player.hero == "大乔" and not player.extra.get("贤助本轮取消") and self._skill_available(player, "贤助", 1):
            result.append(LegalAction(f"skill:{actor}:贤助", "skill", actor, skill="贤助"))
        if self.state.target_ranks:
            result.append(LegalAction(f"pass:{actor}", "pass", actor))
            if not any(action.kind == "play" for action in result) and self._skill_available(player, "疑城", 3):
                result.append(
                    LegalAction(
                        f"skill:{actor}:疑城",
                        "skill",
                        actor,
                        skill="疑城",
                        parameters={"reason": "no_legal_response"},
                    )
                )
        return result

    def step(self, action: LegalAction | str) -> StepResult:
        action_id = action if isinstance(action, str) else action.action_id
        legal = {candidate.action_id: candidate for candidate in self.legal_actions()}
        if action_id not in legal:
            raise SimulationError(f"动作不在当前合法集合中: {action_id}")
        selected = legal[action_id]
        events: list[dict[str, Any]] = []
        if selected.kind == "play":
            self._execute_play(selected, events)
        elif selected.kind == "pass":
            self._execute_pass(selected.actor, events)
        elif selected.kind == "skill":
            self._execute_skill(selected, events)
        else:
            self._resolve_interaction(selected, events)
        self.state.turn_index += 1
        rewards = self._terminal_rewards()
        self.validate_invariants()
        return StepResult(self.state, selected, events, rewards, self.state.terminal)

    def _execute_play(self, action: LegalAction, events: list[dict[str, Any]]) -> None:
        actor = action.actor
        player = self.state.players[actor]
        if not validate_card_selection(player.hand, action.card_ids):
            raise SimulationError("出牌引用了不在手牌中的实例")
        previous = self._target_record()
        pre_hand = list(player.hand)
        selected_cards = [next(card for card in player.hand if card.card_id == card_id) for card_id in action.card_ids]
        for card in selected_cards:
            player.hand.remove(card)
            card.owner = "table"
            self.state.played_cards.append(card)
        event = {
            "kind": "play",
            "turn": self.state.turn_index,
            "actor": actor,
            "card_ids": list(action.card_ids),
            "ranks": list(action.ranks),
            "action_type": action.action_type,
            "was_largest": max((rank_key(card.rank) for card in pre_hand), default=-1)
            in {rank_key(rank) for rank in action.ranks},
        }
        self.state.history.append(event)
        self._emit(events, "play", **event)
        if previous is not None:
            self._on_play_beaten(previous, actor, events)
        self._after_play(actor, action, selected_cards, previous is not None, events)
        self.state.target_ranks = list(action.ranks)
        self.state.target_card_ids = list(action.card_ids)
        self.state.target_action_type = action.action_type
        self.state.trick_owner = actor
        self.state.consecutive_passes = 0
        if not player.hand:
            self._finish_game(actor, events)
            return
        resume_player = self._next_player(actor)
        self.state.current_player = resume_player
        self._activate_next_interaction(resume_player)

    def _execute_pass(self, actor: str, events: list[dict[str, Any]]) -> None:
        self._apply_pass_skills(actor, events)
        self._complete_pass(actor, events)

    def _complete_pass(self, actor: str, events: list[dict[str, Any]]) -> None:
        self.state.history.append({"kind": "pass", "turn": self.state.turn_index, "actor": actor, "ranks": []})
        self._emit(events, "pass", actor=actor)
        self.state.consecutive_passes += 1
        if self.state.consecutive_passes >= 2 and self.state.trick_owner is not None:
            self._apply_daqiao_jieyuan(events)
            self.state.target_ranks = []
            self.state.target_card_ids = []
            self.state.target_action_type = "none"
            self.state.consecutive_passes = 0
            self.state.current_player = self.state.trick_owner
            self.state.trick_owner = None
        else:
            self.state.current_player = self._next_player(actor)
        self._activate_next_interaction(self.state.current_player)

    def _execute_skill(self, action: LegalAction, events: list[dict[str, Any]]) -> None:
        if action.skill == "贤助":
            player = self.state.players[action.actor]
            if player.hero != "大乔" or not self._skill_available(player, "贤助", 1):
                raise SimulationError("贤助次数已经耗尽或当前武将不匹配")
            self._queue_interaction(
                {
                    "actor": action.actor,
                    "skill": "贤助",
                    "effect": "fill_both_largest",
                    "options": [{"target": position} for position in POSITIONS if position != action.actor],
                    "optional": True,
                }
            )
            self._emit(events, "skill", actor=action.actor, skill="贤助")
            self._activate_next_interaction(action.actor)
            return
        if action.skill != "疑城":
            raise SimulationError(f"尚未实现主动技能动作: {action.skill}")
        player = self.state.players[action.actor]
        if not self._skill_available(player, "疑城", 3):
            raise SimulationError("疑城次数已经耗尽")
        self._use_skill(player, "疑城")
        for index in range(2):
            self._gain_rank(player, self._random_rank(tuple(CARD_ORDER), f"疑城-{index}"), "徐盛:疑城", events)
        options = [{"card_ids": [card.card_id], "rank": card.rank} for card in player.hand]
        self._queue_interaction(
            {
                "actor": action.actor,
                "skill": "疑城",
                "effect": "discard_one",
                "options": options,
                "optional": False,
                "after_action": "pass",
            }
        )
        self._emit(events, "skill", actor=action.actor, skill="疑城")
        self._activate_next_interaction(self._next_player(action.actor))

    def _interaction_actions(self, pending: dict[str, Any]) -> list[LegalAction]:
        return UnifiedActionSpace.enumerate_interactions(pending)

    def _resolve_interaction(self, action: LegalAction, events: list[dict[str, Any]]) -> None:
        pending = self.state.pending_interaction
        if pending is None or pending.get("actor") != action.actor or pending.get("skill") != action.skill:
            raise SimulationError("二级技能动作与当前待处理交互不一致")
        skip = bool(action.parameters.get("skip"))
        effect = pending["effect"]
        player = self.state.players[action.actor]
        if skip and action.skill == "贤助":
            player.extra["贤助本轮取消"] = True
        if not skip:
            if effect == "copy_bottom":
                rank = action.parameters["rank"]
                card = self._gain_rank(player, rank, "诸葛均:耕读", events, {"gengdu_copy"})
                player.extra["gengdu_copy_id"] = card.card_id
                self._use_skill(player, "耕读")
            elif effect == "gain_rank":
                self._gain_rank(player, action.parameters["rank"], f"{action.actor}:{action.skill}", events)
                self._use_skill(player, action.skill or "")
            elif effect == "discard_group":
                self._discard_ids(player, tuple(action.parameters["card_ids"]), f"{action.actor}:{action.skill}", events)
                self._use_skill(player, action.skill or "")
            elif effect == "discard_one":
                self._discard_ids(player, tuple(action.parameters["card_ids"]), f"{action.actor}:{action.skill}", events)
            elif effect == "convert_group":
                operation = action.parameters["operation"]
                rank = action.parameters["rank"]
                if operation == "solo_to_pair":
                    self._gain_rank(player, rank, "卢植:儒宗", events)
                else:
                    card_id = action.parameters["card_ids"][0]
                    self._discard_ids(player, (card_id,), "卢植:儒宗", events)
                self._use_skill(player, "儒宗")
            elif effect == "ganglie":
                self._discard_ids(player, tuple(action.card_ids), "夏侯惇:刚烈", events)
                self._gain_rank(player, action.parameters["rank"], "夏侯惇:刚烈", events)
                self._use_skill(player, "刚烈")
            elif effect == "take_cards":
                target = self.state.players.get(action.target or str(pending.get("target", "")))
                if target is None or not validate_card_selection(target.hand, action.card_ids):
                    raise SimulationError("游侠选择的目标牌已经失效")
                for card_id in action.card_ids:
                    card = next(card for card in target.hand if card.card_id == card_id)
                    target.hand.remove(card)
                    card.owner = player.position
                    player.hand.append(card)
                self._sort_hand(player)
                self._sort_hand(target)
                self._use_skill(player, "游侠")
            elif effect == "fill_both_largest":
                target = self.state.players.get(action.target or "")
                if target is None or target.position == player.position:
                    raise SimulationError("贤助目标无效")
                self._fill_largest_to_three(player, "大乔:贤助", events)
                self._fill_largest_to_three(target, "大乔:贤助", events)
                self._use_skill(player, "贤助")
            else:
                raise SimulationError(f"未知二级技能效果: {effect}")
            self._emit(events, "interaction", actor=action.actor, skill=action.skill, effect=effect)
        after_action = pending.get("after_action")
        resume_player = pending.get("resume_player", self._next_player(action.actor))
        self.state.pending_interaction = None
        if not player.hand:
            self._finish_game(action.actor, events)
            return
        if self.state.interaction_queue:
            self._activate_next_interaction(resume_player)
        elif after_action == "pass":
            self._complete_pass(action.actor, events)
        else:
            self.state.current_player = resume_player

    def _after_play(
        self,
        actor: str,
        action: LegalAction,
        played_cards: list[CardInstance],
        beat_previous: bool,
        events: list[dict[str, Any]],
    ) -> None:
        player = self.state.players[actor]
        hero = player.hero
        if hero == "大乔":
            player.extra.pop("贤助本轮取消", None)
        if hero == "典韦":
            self._dianwei_after_play(player, events)
        if hero == "张飞":
            previous_type = player.extra.get("last_action_type")
            if previous_type == action.action_type:
                for card in self._lowest_cards(player, 2):
                    self._increase_card(card, "张飞:咆哮", events)
            player.extra["last_action_type"] = action.action_type
        if hero == "关羽" and action.action_type == "straight" and self._skill_available(player, "单骑", 1):
            self._use_skill(player, "单骑")
            self._gain_rank(player, WILDCARD, "关羽:单骑", events, {"wildcard"})
        if hero == "陆逊" and len(player.hand) != 1 and self._skill_available(player, "破蜀", 20):
            self._use_skill(player, "破蜀")
            self._gain_rank(player, self._random_rank(tuple(CARD_ORDER[8:]), "陆逊:破蜀"), "陆逊:破蜀", events)
        copied_id = player.extra.get("gengdu_copy_id")
        if hero == "诸葛均" and copied_id in action.card_ids and player.hand:
            self._queue_interaction(
                {
                    "actor": actor,
                    "skill": "耕读",
                    "effect": "discard_one",
                    "options": [{"card_ids": [card.card_id], "rank": card.rank} for card in player.hand],
                    "optional": False,
                }
            )
            player.extra.pop("gengdu_copy_id", None)
        if hero == "凌统" and action.action_type in {"solo", "pair"} and self._skill_available(player, "勇进", 2):
            options = self._lingtong_discard_options(player, action.action_type)
            if options:
                self._queue_interaction(
                    {"actor": actor, "skill": "勇进", "effect": "discard_group", "options": options, "optional": True}
                )
        for position, observer in self.state.players.items():
            if observer.hero == "关银屏" and len(action.ranks) >= 4 and self._skill_available(observer, "花武", 5):
                self._use_skill(observer, "花武")
                self._gain_rank(observer, self._random_rank(("J", "Q", "K"), "关银屏:花武"), "关银屏:花武", events)
            if observer.hero == "皇甫嵩" and action.ranks and all(rank in {"3", "4"} for rank in action.ranks) and self._skill_available(observer, "平乱", 3):
                self._use_skill(observer, "平乱")
                for index in range(len(action.ranks)):
                    self._gain_rank(observer, self._random_rank(tuple(CARD_ORDER[4:]), f"皇甫嵩:平乱:{index}"), "皇甫嵩:平乱", events)
            if position != actor and observer.hero == "曹洪" and action.action_type == "pair" and self._skill_available(observer, "敛财", 3):
                self._use_skill(observer, "敛财")
                for rank in action.ranks:
                    self._gain_rank(observer, rank, "曹洪:敛财", events)
            if position != actor and observer.hero == "关羽" and action.action_type == "straight" and self._skill_available(observer, "武圣", 2):
                options = [{"rank": rank} for rank in dict.fromkeys(action.ranks)]
                self._queue_interaction(
                    {"actor": position, "skill": "武圣", "effect": "gain_rank", "options": options, "optional": True}
                )
        if beat_previous and hero == "卢植" and self._skill_available(player, "儒宗", 3):
            options = self._luzhi_convert_options(player)
            if options:
                self._queue_interaction(
                    {"actor": actor, "skill": "儒宗", "effect": "convert_group", "options": options, "optional": True}
                )

    def _on_play_beaten(self, previous: dict[str, Any], new_actor: str, events: list[dict[str, Any]]) -> None:
        owner = self.state.players[previous["actor"]]
        attacker = self.state.players[new_actor]
        if attacker.hero == "甘宁" and owner.hand and self._skill_available(attacker, "游侠", 3):
            lowest = tuple(sorted(owner.hand, key=lambda card: (rank_key(card.rank), card.card_id))[:2])
            self._queue_interaction(
                {
                    "actor": new_actor,
                    "skill": "游侠",
                    "effect": "take_cards",
                    "target": owner.position,
                    "options": [
                        {
                            "target": owner.position,
                            "card_ids": [card.card_id for card in lowest],
                            "ranks": [card.rank for card in lowest],
                        }
                    ],
                    "optional": True,
                }
            )
        hero = owner.hero
        if hero == "夏侯惇" and previous.get("was_largest") and self._skill_available(owner, "刚烈", 1) and len(owner.hand) >= 3:
            highest = max(previous["ranks"], key=rank_key)
            options = []
            seen_ranks = set()
            for selected in combinations(owner.hand, 3):
                discard_ranks = tuple(card.rank for card in selected)
                if discard_ranks in seen_ranks:
                    continue
                seen_ranks.add(discard_ranks)
                options.append(
                    {
                        "rank": highest,
                        "ranks": [highest],
                        "card_ids": [card.card_id for card in selected],
                        "discard_ranks": list(discard_ranks),
                        "recover_rank": highest,
                    }
                )
            self._queue_interaction(
                {
                    "actor": owner.position,
                    "skill": "刚烈",
                    "effect": "ganglie",
                    "options": options,
                    "optional": True,
                }
            )
        if hero == "赵云" and owner.marks.get("冲阵回收", 0) < 7 and previous.get("action_type") in {"solo", "pair"} and all(rank_key(rank) < rank_key("A") for rank in previous["ranks"]):
            recovered = 0
            for card_id in previous.get("card_ids", []):
                card = next((value for value in self.state.played_cards if value.card_id == card_id), None)
                if card is None:
                    continue
                self.state.played_cards.remove(card)
                card.owner = owner.position
                self._increase_card(card, "赵云:冲阵", events)
                owner.hand.append(card)
                recovered += 1
            owner.marks["冲阵回收"] = min(7, owner.marks.get("冲阵回收", 0) + recovered)
            self._sort_hand(owner)
        if hero == "陆逊" and len(owner.hand) != 1 and self._skill_available(owner, "御魏", 20):
            self._use_skill(owner, "御魏")
            self._discard_lowest(owner, 1, "陆逊:御魏", events)

    def _apply_pass_skills(self, actor: str, events: list[dict[str, Any]]) -> None:
        player = self.state.players[actor]
        if player.hero == "典韦":
            player.marks["不屈"] = 0
        if player.hero == "姜维":
            if len(player.hand) == 1:
                card = player.hand[0]
                self._increase_card(card, "姜维:绝计", events)
                if card.rank == "D":
                    player.hand.remove(card)
                    for index in range(4):
                        self._gain_rank(player, "3", "姜维:绝计", events, {f"split-{index}"})
            elif len(player.hand) > 1:
                self._discard_lowest(player, 1, "姜维:北伐", events)

    def _dianwei_after_play(self, player: PlayerState, events: list[dict[str, Any]]) -> None:
        if len(player.hand) < 3 or player.extra.get("不屈失效"):
            return
        old = player.marks.get("不屈", 0)
        new = min(4, old + 1)
        player.marks["不屈"] = new
        self._emit(events, "mark", actor=player.position, skill="不屈", value=new)
        if new == old:
            return
        if new == 2:
            counts = Counter(card.rank for card in player.hand)
            singles = [rank for rank, count in counts.items() if count == 1 and rank not in {"X", "D", WILDCARD}]
            if singles:
                self._gain_rank(player, max(singles, key=rank_key), "典韦:血战", events)
        elif new == 3:
            self._gain_rank(player, "2", "典韦:血战", events)
        elif new == 4:
            self._gain_rank(player, WILDCARD, "典韦:血战", events, {"wildcard"})
            count = player.marks.get("血战万能牌", 0) + 1
            player.marks["血战万能牌"] = count
            if count >= 2:
                player.extra["不屈失效"] = True

    def _apply_daqiao_jieyuan(self, events: list[dict[str, Any]]) -> None:
        if self.state.target_action_type != "solo" or len(self.state.target_card_ids) != 1:
            return
        card_id = self.state.target_card_ids[0]
        card = next((value for value in self.state.played_cards if value.card_id == card_id), None)
        if card is None:
            return
        for position in POSITIONS:
            player = self.state.players[position]
            if position == self.state.trick_owner or player.hero != "大乔" or not self._skill_available(player, "结缘", 1):
                continue
            self.state.played_cards.remove(card)
            card.owner = position
            player.hand.append(card)
            self._sort_hand(player)
            self._use_skill(player, "结缘")
            self._emit(events, "gain", actor=position, skill="结缘", cards=[card.rank], card_ids=[card.card_id])
            return

    def _fill_largest_to_three(self, player: PlayerState, source: str, events: list[dict[str, Any]]) -> None:
        if not player.hand:
            return
        largest = max((card.rank for card in player.hand), key=rank_key)
        missing = max(0, 3 - sum(card.rank == largest for card in player.hand))
        for _ in range(missing):
            self._gain_rank(player, largest, source, events)

    def _schedule_game_start_interactions(self) -> None:
        for position in POSITIONS:
            player = self.state.players[position]
            if player.hero == "诸葛均" and self._skill_available(player, "耕读", 1):
                options = [{"rank": card.rank} for card in self.state.bottom_cards]
                self._queue_interaction(
                    {"actor": position, "skill": "耕读", "effect": "copy_bottom", "options": options, "optional": True}
                )
        self._activate_next_interaction("landlord")

    def _queue_interaction(self, interaction: dict[str, Any]) -> None:
        self.state.interaction_queue.append(interaction)

    def _activate_next_interaction(self, resume_player: str) -> None:
        if self.state.pending_interaction is not None or not self.state.interaction_queue:
            return
        pending = self.state.interaction_queue.pop(0)
        pending.setdefault("resume_player", resume_player)
        self.state.pending_interaction = pending
        self.state.current_player = pending["actor"]

    def _target_record(self) -> dict[str, Any] | None:
        if not self.state.target_ranks or self.state.trick_owner is None:
            return None
        return next(
            (
                event
                for event in reversed(self.state.history)
                if event.get("kind") == "play" and event.get("actor") == self.state.trick_owner
            ),
            None,
        )

    def _lingtong_discard_options(self, player: PlayerState, action_kind: str) -> list[dict[str, Any]]:
        counts = Counter(card.rank for card in player.hand)
        options = []
        if action_kind == "solo":
            for rank, count in counts.items():
                if count >= 2:
                    ids = [card.card_id for card in player.hand if card.rank == rank][:2]
                    options.append({"card_ids": ids, "ranks": [rank, rank]})
        else:
            for rank, count in counts.items():
                if count == 1:
                    card = next(card for card in player.hand if card.rank == rank)
                    options.append({"card_ids": [card.card_id], "ranks": [rank]})
        return options

    def _luzhi_convert_options(self, player: PlayerState) -> list[dict[str, Any]]:
        counts = Counter(card.rank for card in player.hand)
        options = []
        for rank, count in counts.items():
            if rank in {"X", "D", WILDCARD}:
                continue
            cards = [card.card_id for card in player.hand if card.rank == rank]
            if count == 1:
                options.append({"operation": "solo_to_pair", "rank": rank, "card_ids": cards})
            elif count == 2:
                options.append({"operation": "pair_to_solo", "rank": rank, "card_ids": cards[:1]})
        return options

    def _gain_rank(
        self,
        player: PlayerState,
        rank: str,
        source: str,
        events: list[dict[str, Any]],
        tags: set[str] | None = None,
    ) -> CardInstance:
        card_id = f"s{self.state.next_card_sequence:06d}"
        self.state.next_card_sequence += 1
        card = CardInstance(card_id, rank, rank, player.position, source, set(tags or ()))
        player.hand.append(card)
        self._sort_hand(player)
        self._emit(events, "gain", actor=player.position, card_id=card_id, rank=rank, source=source)
        return card

    def _discard_ids(self, player: PlayerState, card_ids: tuple[str, ...], source: str, events: list[dict[str, Any]]) -> None:
        for card_id in card_ids:
            card = next((value for value in player.hand if value.card_id == card_id), None)
            if card is None:
                raise SimulationError(f"弃牌实例不存在: {card_id}")
            player.hand.remove(card)
            card.owner = "discard"
            self.state.played_cards.append(card)
            self._emit(events, "discard", actor=player.position, card_id=card_id, rank=card.rank, source=source)

    def _discard_lowest(self, player: PlayerState, count: int, source: str, events: list[dict[str, Any]]) -> None:
        self._discard_ids(player, tuple(card.card_id for card in self._lowest_cards(player, count)), source, events)

    def _lowest_cards(self, player: PlayerState, count: int) -> list[CardInstance]:
        return sorted(player.hand, key=lambda card: (rank_key(card.rank), card.card_id))[:count]

    def _increase_card(self, card: CardInstance, source: str, events: list[dict[str, Any]]) -> None:
        if card.rank == WILDCARD:
            return
        current = rank_key(card.rank)
        choices = tuple(rank for rank in CARD_ORDER if rank_key(rank) > current)
        if not choices:
            return
        old_rank = card.rank
        card.rank = self._random_rank(choices, f"{source}:{card.card_id}")
        self._emit(events, "transform", actor=card.owner, card_id=card.card_id, old_rank=old_rank, rank=card.rank, source=source)

    def _random_rank(self, choices: tuple[str, ...], label: str) -> str:
        if not choices:
            raise SimulationError("随机点数候选为空")
        payload = f"{self.state.seed}|{self.state.game_id}|{self.state.turn_index}|{len(self.state.history)}|{label}".encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return random.Random(seed).choice(choices)

    def _skill_available(self, player: PlayerState, skill: str, limit: int | None) -> bool:
        if player.hero not in HERO_REGISTRY:
            return False
        if not any(spec.name == skill and spec.verified for spec in HERO_REGISTRY[player.hero]):
            return False
        return limit is None or player.skill_uses.get(skill, 0) < limit

    @staticmethod
    def _use_skill(player: PlayerState, skill: str) -> None:
        player.skill_uses[skill] = player.skill_uses.get(skill, 0) + 1

    @staticmethod
    def _sort_hand(player: PlayerState) -> None:
        player.hand.sort(key=lambda card: (rank_key(card.rank), card.card_id))

    @staticmethod
    def _next_player(position: str) -> str:
        index = POSITIONS.index(position)
        return POSITIONS[(index + 1) % len(POSITIONS)]

    def _finish_game(self, winner: str, events: list[dict[str, Any]]) -> None:
        self.state.terminal = True
        self.state.winner = winner
        self.state.pending_interaction = None
        self.state.interaction_queue.clear()
        self._emit(events, "game_end", winner=winner)

    def _terminal_rewards(self) -> dict[str, float]:
        if not self.state.terminal or self.state.winner is None:
            return {position: 0.0 for position in POSITIONS}
        landlord_won = self.state.winner == self.state.landlord
        return {
            position: (1.0 if landlord_won else -1.0)
            if position == self.state.landlord
            else (-1.0 if landlord_won else 1.0)
            for position in POSITIONS
        }

    def _emit(self, events: list[dict[str, Any]], event_type: str, **payload: Any) -> None:
        event = {"event_type": event_type, "turn": self.state.turn_index, **payload}
        events.append(event)
        self.state.event_queue.append(event)

    def validate_invariants(self) -> None:
        locations: dict[str, str] = {}
        for position, player in self.state.players.items():
            for card in player.hand:
                if card.card_id in locations:
                    raise SimulationError(f"牌实例重复出现在多个区域: {card.card_id}")
                locations[card.card_id] = position
                if card.owner != position:
                    raise SimulationError(f"牌实例持有者不一致: {card.card_id}")
                if card.rank not in set(CARD_ORDER) | {WILDCARD}:
                    raise SimulationError(f"未知牌点数: {card.rank}")
        for card in self.state.played_cards:
            if card.card_id in locations:
                raise SimulationError(f"已出牌仍存在于手牌: {card.card_id}")
            locations[card.card_id] = card.owner
        if self.state.current_player not in self.state.players:
            raise SimulationError("当前玩家位置无效")
        if self.state.terminal and self.state.winner is None:
            raise SimulationError("终局缺少胜者")
        if self.state.winner is not None and self.state.players[self.state.winner].hand:
            raise SimulationError("胜者仍有手牌")
