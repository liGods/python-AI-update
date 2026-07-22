from collections import Counter  # 统计候选动作执行后各点数的剩余数量。
from functools import lru_cache  # 缓存相同手牌和桌面牌对应的合法动作集合。

from rlcard.games.doudizhu.utils import CARD_TYPE, INDEX, contains_cards  # 复用 RLCard 官方斗地主动作空间与牌型定义。
from ok_tasks.card_ai.rlcard_adapter import (
    RLCARD_TO_INTERNAL,
    action_beats as _adapter_action_beats,
    action_order as _adapter_action_order,
    action_type as _adapter_action_type,
    contains_with_wildcards as _adapter_contains_with_wildcards,
    is_five_bomb as _adapter_is_five_bomb,
    legal_action_variants as _adapter_legal_action_variants,
    legal_actions as _adapter_legal_actions,
    physical_action as _adapter_physical_action,
    to_internal as _adapter_to_internal,
    to_rlcard as _adapter_to_rlcard,
)
from ok_tasks.card_ai.decision.candidate import CandidateDecision
from ok_tasks.card_ai.decision.core import DecisionPolicyCore
from ok_tasks.card_ai.decision.stage import StageContext, classify_game_stage, stage_score_components
from ok_tasks.card_ai.decision.context import DecisionContext
from ok_tasks.card_ai.decision.explanation import structured_score
from ok_tasks.card_ai.decision.hero_strategy import evaluate_hand_expansion, evaluate_luxun_collection
from ok_tasks.PolicyOptimizer import POLICY_PROFILES  # 导入经过边界限制的三套策略配置。

_WILDCARD = "W"
_MAX_LOGGED_RANDOM_BRANCHES = 8
_DECISION_CORE = DecisionPolicyCore()


def _hero_context(state):  # 延迟导入避免 RlCardRuleModel -> card_ai.__init__ -> engine -> rules 的初始化环。
    from ok_tasks.card_ai.hero_policy import HeroDecisionContext

    return HeroDecisionContext.from_legacy_state(state)


def _evaluate_hero_play(context, action, route_evaluator, action_type=None, effective_action=None):  # 生产规则与离线模拟共享完整技能结算投影。
    from ok_tasks.card_ai.hero_policy import evaluate_play

    if action_type is not None:  # RLCard 已经给出精确牌型时，不再让统一策略用简化规则重复猜测。
        from ok_tasks.card_ai.schema import LegalAction

        physical_ranks = tuple(action)  # 统一策略按真实实体牌扣除手牌，万能牌不会被当作不存在的自然牌。
        effective_ranks = tuple(effective_action if effective_action is not None else action)  # 生效点数仅作为可解释元数据保留。
        action = LegalAction(
            action_id=f"rule:{context.position}:{','.join(physical_ranks)}:{action_type}",
            kind="play",
            actor=context.position,
            ranks=physical_ranks,
            action_type=action_type,
            parameters={"effective_ranks": list(effective_ranks)},
        )
    return evaluate_play(context, action, route_evaluator=route_evaluator)


def _estimate_hero_turns(cards):  # 万能牌分支使用策略层的守恒估算。
    from ok_tasks.card_ai.hero_policy import estimate_remaining_turns

    return estimate_remaining_turns(tuple(cards))


def _to_rlcard(cards):  # 将项目牌组转换成 RLCard 要求的有序字符串。
    return _adapter_to_rlcard(cards)


def _to_internal(action):  # 将 RLCard 动作字符串转换回项目牌组列表。
    return _adapter_to_internal(action)


def _contains_with_wildcards(hand, action):  # 判断实体手牌能否通过万能牌补齐一个 RLCard 生效动作。
    return _adapter_contains_with_wildcards(hand, action)


def _physical_action(hand, effective_action):  # 把生效动作还原成屏幕上实际需要点击的实体牌。
    return _adapter_physical_action(hand, effective_action)


def _legal_action_variants(hand, target=""):  # 同时返回 RLCard 生效动作与可点击实体动作。
    return _adapter_legal_action_variants(hand, target)


def _is_five_bomb(action):  # 判断百将牌技能增牌后形成的五张同点数超级炸弹。
    return _adapter_is_five_bomb(action)


def _action_beats(candidate, target):  # 使用 RLCard 牌型和权重判断候选动作能否压过桌面动作。
    return _adapter_action_beats(candidate, target)


def _legal_actions(hand, target=""):  # 从 RLCard 完整动作空间中生成当前屏幕状态的合法动作。
    return _adapter_legal_actions(hand, target)


def _remaining_hand(hand, action):  # 计算执行候选动作后的规范剩余手牌字符串。
    remaining = Counter(hand)  # 统计当前手牌每种点数的数量。
    remaining.subtract(action)  # 扣除候选动作使用的所有牌。
    cards = [card for card in tuple(INDEX) + (_WILDCARD,) for _ in range(max(0, remaining[card]))]  # 同时保留尚未使用的万能牌实体。
    return "".join(cards)  # 返回 RLCard 组合分解器可直接读取的字符串。


def _longest_run(counts, minimum_count, minimum_length):  # 估算顺子、连对或飞机能够合并多少点数组。
    best_length = 0  # 初始化尚未找到有效连续结构。
    current_length = 0  # 记录当前从三到 A 的连续点数长度。
    for card in tuple(INDEX)[:12]:  # 连续牌型不能包含二和大小王。
        if counts[card] >= minimum_count:  # 当前点数具备指定的单张、对子或三张数量。
            current_length += 1  # 将当前点数加入连续结构。
            best_length = max(best_length, current_length)  # 更新目前找到的最长连续长度。
        else:  # 当前点数中断了连续结构。
            current_length = 0  # 从下一个点数重新开始统计。
    return best_length if best_length >= minimum_length else 0  # 未达到合法最短长度时不计算合并收益。


def _estimated_turns(hand):  # 根据标准牌型结构估算清空剩余手牌所需手数。
    if not hand:  # 已经没有剩余手牌表示当前动作可以直接获胜。
        return 0  # 将直接出完作为最优评分。
    if hand in CARD_TYPE[0]:  # 整副剩余手牌本身已经是一手合法牌型时无需再按点数组拆散估算。
        return 1  # 精确识别两手走完残局，避免为了保炸弹错过确定性短牌路。
    counts = Counter(hand)  # 统计剩余手牌的各点数数量结构。
    groups = sum(1 for count in counts.values() if count > 0)  # 将每种相同点数先视为一手基础组合。
    solo_run = _longest_run(counts, 1, 5)  # 查找能够组成顺子的最长连续点数。
    pair_run = _longest_run(counts, 2, 3)  # 查找能够组成连对的最长连续点数。
    trio_run = _longest_run(counts, 3, 2)  # 查找能够组成飞机的最长连续点数。
    sequence_saving = max(0, solo_run - 1, pair_run - 1, trio_run - 1)  # 选择收益最大的连续牌型避免重复计算同一批牌。
    trio_groups = sum(1 for count in counts.values() if count == 3)  # 统计可以携带单牌或对子的自然三张数量。
    attachment_groups = sum(1 for count in counts.values() if count in {1, 2})  # 统计可用于三带牌或飞机翅膀的自然附件。
    attachment_saving = min(trio_groups, attachment_groups)  # 每组三张最多合并一个附件组合。
    return max(1, groups - sequence_saving - attachment_saving)  # 返回至少一手的保守剩余手数估计。


def _card_count_vector(cards):  # 将规范手牌编码成可缓存、可相减的十五维点数计数。
    counts = Counter(cards)  # 统计技能加牌后可能超过标准牌库数量的各点数张数。
    return tuple(counts[card] for card in INDEX)  # 固定按三到大王顺序返回不可变状态。


@lru_cache(maxsize=48)  # 缓存最近完整手牌的精确分解器，连续候选共享同一批中间状态。
def _turn_planner(full_hand):  # 使用当前完整手牌的合法动作建立精确最少手数动态规划。
    if _WILDCARD in full_hand:  # 十五维 RLCard 状态向量不能无损表示万能牌实体及其多种替代值。
        return None  # 统一降级到支持万能牌的守恒估算，避免错误丢牌或指数搜索。
    if len(full_hand) > 24:  # 大量英雄技能加牌时状态空间可能快速膨胀。
        return None  # 超过实战安全上限时使用保守快速估算，避免阻塞界面操作。
    legal_actions = _legal_actions(full_hand, "")  # 一次性枚举所有可由当前手牌组成的主动合法牌型。
    if len(legal_actions) > 600:  # 极端重复技能牌可能产生过大的组合空间。
        return None  # 保持实时性优先，交由旧估算函数安全降级。
    actions_by_rank = [[] for _ in INDEX]  # 按动作包含的点数索引分桶以减少每个子状态的扫描数量。
    for action in legal_actions:  # 将全部合法动作转换成固定计数向量。
        vector = _card_count_vector(action)  # 编码动作消耗的每种点数数量。
        solo_cost = int(len(action) == 1)  # 记录该动作是否会占用一次难以取得牌权的单牌回合。
        solo_risk = max(0, INDEX["A"] - INDEX[action]) if solo_cost else 0  # 三到 K 的低位单牌越小越难在残局自然打出，A 以上视为控制牌。
        for rank_index, count in enumerate(vector):  # 找出该动作包含的所有点数。
            if count:  # 只把动作加入实际涉及的点数桶。
                actions_by_rank[rank_index].append((vector, solo_cost, solo_risk))  # 同时保存未来单出次数和低位孤张风险。

    @lru_cache(maxsize=None)  # 同一完整手牌下精确缓存每个剩余计数状态。
    def solve(state):  # 计算当前剩余牌的最少手数，并在手数相同后最小化单出次数和低位孤张风险。
        if not any(state):  # 没有牌时已经完成全部分解。
            return 0, 0, 0  # 终局不需要额外动作、单牌回合或孤张风险。
        first_rank = next(index for index, count in enumerate(state) if count)  # 选择最低剩余点数破除动作排列对称性。
        card_count = sum(state)  # 统计最坏情况下需要逐张打出的牌数。
        fallback_risk = sum(count * max(0, INDEX["A"] - rank_index) for rank_index, count in enumerate(state))  # 计算全部逐张出牌时的低位风险。
        best = card_count, card_count, fallback_risk  # 最坏情况始终可以逐张打出所有手牌。
        for vector, solo_cost, solo_risk in actions_by_rank[first_rank]:  # 任一完整分解必有一手包含当前最低点数。
            if all(used <= available for used, available in zip(vector, state)):  # 仅尝试当前剩余牌确实能够组成的动作。
                remaining = tuple(available - used for used, available in zip(vector, state))  # 扣除该合法动作得到子状态。
                tail_turns, tail_solos, tail_risk = solve(remaining)  # 读取余牌的完整残局路线质量。
                best = min(best, (1 + tail_turns, solo_cost + tail_solos, solo_risk + tail_risk))  # 依次最小化手数、单出次数和低位孤张。
        return best  # 返回确定性的精确残局路线质量。

    return solve  # 返回带有局内状态缓存的求解函数。


def _minimum_turns(full_hand, remaining):  # 优先精确计算剩余手数，并为超大技能手牌提供快速安全降级。
    if not remaining:  # 当前动作已经清空手牌。
        return 0  # 立即胜利是零手剩余。
    planner = _turn_planner(full_hand)  # 获取以当前完整手牌合法动作构建的共享分解器。
    if planner is None:  # 超出实时搜索安全上限。
        return _estimated_turns(remaining)  # 使用不会阻塞操作的保守结构估算。
    return planner(_card_count_vector(remaining))[0]  # 对候选动作后的真实剩余牌执行精确最少手数计算。


def _minimum_route_quality(full_hand, remaining):  # 在相同最少手数路线中继续评价未来单牌回合和低位孤张风险。
    if not remaining:  # 当前动作已经清空手牌。
        return 0, 0, 0  # 直接获胜没有后续残局负担。
    planner = _turn_planner(full_hand)  # 复用与最少手数完全相同的合法动作分解缓存。
    if planner is not None:  # 普通残局可以执行精确路线分析。
        return planner(_card_count_vector(remaining))  # 返回最少手数、单牌回合数和低位风险。
    counts = Counter(remaining)  # 超大技能手牌使用快速结构降级，避免阻塞实时点击。
    natural_singles = [card for card, count in counts.items() if count == 1 and card in INDEX]  # 万能牌不是固定点数孤张，不能用 RLCard 索引评价风险。
    risk = sum(max(0, INDEX["A"] - INDEX[card]) for card in natural_singles)  # 为低位自然孤张赋予更高残局风险。
    return _estimated_turns(remaining), len(natural_singles), risk  # 保留原估算手数并提供保守孤张指标。


def minimum_turns_after_removal(hand_cards, removed_cards):  # 为英雄弃牌交互计算移除指定牌后的精确剩余手数。
    full_hand = _to_rlcard(hand_cards)  # 将屏幕内部牌面转换成 RLCard 规范顺序。
    removed = _to_rlcard(removed_cards)  # 将技能计划弃置的牌组转换成规范动作字符串。
    if not full_hand or not removed or not contains_cards(full_hand, removed):  # 缺少手牌或移除数量超过实际持有数量时拒绝伪造结果。
        raise ValueError("技能弃牌不属于当前完整手牌")  # 将识牌或调用错误交给上层安全暂停。
    remaining = _remaining_hand(full_hand, removed)  # 不要求弃牌本身属于斗地主合法出牌牌型，只按点数扣除。
    return _minimum_turns(full_hand, remaining)  # 使用与正常出牌相同的精确牌路规划器评价剩余结构。


def evaluate_bid_strength(hand_cards):  # 根据完整开局手牌的控制牌和可组合结构计算可解释叫分强度。
    hand = _to_rlcard(hand_cards)  # 统一转换大小王并规范点数顺序。
    if not hand:  # 未收到完整手牌时不能安全叫地主。
        return {"score": 0.0, "recommended_bid": 0, "reason": "未识别到完整手牌"}  # 使用不叫作为安全结果。
    counts = Counter(hand)  # 统计王、二、炸弹、三张和对子结构。
    turns, future_solos, solo_risk = _minimum_route_quality(hand, hand)  # 计算当前整手牌的最少出牌手数及难出孤张负担。
    has_rocket = bool(counts["B"] and counts["R"])  # 大小王齐全时具备最高控制力。
    bomb_count = sum(int(count >= 4) for card, count in counts.items() if card not in {"B", "R"})  # 技能牌可能超过四张，同点数仍只计一个炸弹组。
    triple_count = sum(int(count >= 3) for count in counts.values())  # 三张既能带牌也能组成飞机。
    pair_count = sum(int(count >= 2) for count in counts.values())  # 对子和连对能够减少牌权需求。
    control_score = (24.0 if has_rocket else 10.0 * counts["R"] + 8.0 * counts["B"])  # 王炸按组合价值计分，单王按控制力分别计分。
    control_score += 5.0 * counts["2"] + 2.0 * counts["A"] + 0.8 * counts["K"]  # 二、A、K依次提供不同强度的接牌能力。
    structure_score = 16.0 * bomb_count + 2.0 * triple_count + 0.75 * pair_count  # 奖励炸弹、三张和成组手牌。
    structure_score += max(0, 10 - turns) * 4.0  # 最少手数越少，成为地主后连续掌控牌局的能力越强。
    weakness_penalty = 1.5 * future_solos + 0.15 * solo_risk  # 扣除未来必须单出的次数及低位孤张风险。
    score = round(max(0.0, control_score + structure_score - weakness_penalty), 2)  # 生成稳定且便于日志解释的总分。
    if score >= 50.0:  # 王炸、炸弹和紧凑牌路兼具时承担三倍底分。
        recommended_bid = 3  # 极强牌叫三分。
    elif score >= 34.0:  # 有明显控制力或非常紧凑的多牌型结构。
        recommended_bid = 2  # 强牌叫二分。
    elif score >= 20.0:  # 牌力略高于平均且孤张负担可控。
        recommended_bid = 1  # 中等牌叫一分。
    else:  # 控制牌少、手数多或低位孤张过多。
        recommended_bid = 0  # 弱牌不叫。
    if has_rocket:  # 王炸即使其余手牌较散也至少值得二分竞争地主。
        recommended_bid = max(2, recommended_bid)  # 防止孤张惩罚掩盖绝对牌权价值。
    if has_rocket and bomb_count:  # 王炸加普通炸弹属于明确的三分牌力下限。
        recommended_bid = 3  # 直接提升到三分。
    reason = f"牌力{score:.2f}，预计{turns}手，王炸{int(has_rocket)}，炸弹{bomb_count}，二{counts['2']}张，未来单牌{future_solos}手"  # 生成中文决策解释。
    return {"score": score, "recommended_bid": recommended_bid, "turns": turns, "future_solos": future_solos, "solo_risk": solo_risk, "rocket": has_rocket, "bombs": bomb_count, "twos": counts["2"], "reason": reason}  # 返回日志和界面可共同使用的完整指标。


def choose_bid(hand_cards, available_bids=(1, 2, 3)):  # 根据牌力和当前仍可点击的分数选择不叫或一至三分。
    evaluation = evaluate_bid_strength(hand_cards)  # 先计算不受界面状态影响的自身牌力上限。
    desired = int(evaluation["recommended_bid"])  # 读取自身愿意承担的最高分。
    available = sorted({int(value) for value in available_bids if int(value) in {1, 2, 3}})  # 清理模板实际识别到的可用叫分按钮。
    affordable = [value for value in available if value <= desired]  # 只选择不超过自身牌力上限且仍可点击的分数。
    chosen = max(affordable) if affordable else 0  # 对手叫分已超过承受能力时选择不叫。
    evaluation["available_bids"] = available  # 记录当前界面允许的分数供回放判断已有最高叫分。
    evaluation["bid"] = chosen  # 保存最终考虑对手叫分后的选择。
    return chosen, evaluation  # 返回语义叫分和完整解释。


def choose_skill_card_pick(hand_cards, option_cards, hero=None):  # 模拟获得每张展示牌后的完整牌路并决定获取或取消技能。
    valid_ranks = set("3456789TJQKA2XD")  # 获取牌库只评价当前规则模型能够确定处理的标准点数。
    hand = [card for card in hand_cards if card in valid_ranks]  # 过滤万能牌等尚不能由标准动作空间精确模拟的实体。
    options = [card for card in option_cards if card in valid_ranks]  # 过滤OCR噪声并保留屏幕顺序。
    if not hand or not options:  # 缺少完整手牌或展示牌时禁止猜测获取。
        return None, {"decision": "cancel", "reason": "技能获取牌库信息不完整", "candidates": []}  # 安全选择取消并保留诊断原因。
    baseline = _to_rlcard(hand)  # 转换当前手牌供合法牌型分解器使用。
    before_turns, before_solos, before_risk = _minimum_route_quality(baseline, baseline)  # 计算不获取任何牌的基准路线。
    counts = Counter(hand)  # 统计每个选项能够补成对子、三张或炸弹的程度。
    candidates = []  # 保存每张展示牌的可解释收益。
    control_values = {"K": 0.8, "A": 2.0, "2": 4.0, "X": 5.0, "D": 6.0}  # 高牌即使不成组也保留有限控制收益。
    for option in options:  # 逐张模拟获得展示牌后的完整手牌。
        after = _to_rlcard(hand + [option])  # 构造获得当前牌后的规范状态。
        after_turns, after_solos, after_risk = _minimum_route_quality(after, after)  # 重新计算最少手数和孤张负担。
        existing = counts[option]  # 读取当前手中已有同点数牌数。
        group_bonus = {1: 2.0, 2: 3.0, 3: 5.0}.get(existing, 0.0)  # 补成对子、三张和炸弹分别获得结构奖励。
        route_gain = (before_turns - after_turns) * 8.0  # 减少一次未来牌权需求是最主要收益。
        solo_gain = (before_solos - after_solos) * 3.0  # 减少未来单出次数能够提高真实走牌概率。
        risk_gain = (before_risk - after_risk) * 0.15  # 清除低位孤张获得额外小幅奖励。
        hero_bonus = 1.0 if hero == "诸葛均" else 0.0  # 耕读复制牌打出后还能弃牌，给予保守的后续收益补偿。
        utility = round(route_gain + solo_gain + risk_gain + group_bonus + control_values.get(option, 0.0) + hero_bonus, 2)  # 汇总当前选项的净收益。
        candidates.append({"card": option, "utility": utility, "after_turns": after_turns, "after_solos": after_solos, "after_risk": after_risk, "group_bonus": group_bonus})  # 保存回放所需评分分项。
    best = max(candidates, key=lambda item: (item["utility"], -item["after_turns"], -item["after_solos"], INDEX[_to_rlcard([item["card"]])]))  # 选择净收益最高且牌路更紧凑的展示牌。
    if best["utility"] <= 0.0:  # 所有选项都会增加未来牌权负担且没有足够结构或控制收益。
        return None, {"decision": "cancel", "reason": "所有展示牌都会令当前牌路变差", "before_turns": before_turns, "before_solos": before_solos, "candidates": candidates}  # 取消仍会消耗技能，但避免主动恶化手牌。
    return best["card"], {"decision": "pick", "selected": best["card"], "reason": f"选择{best['card']}，技能牌收益{best['utility']:.2f}", "before_turns": before_turns, "before_solos": before_solos, "candidates": candidates}  # 返回最佳点数和完整解释。


def _is_bomb_action(action):  # 判断动作是否消耗炸弹或王炸控制牌。
    return _is_five_bomb(action) or any(card_type in {"bomb", "rocket"} for card_type, _ in CARD_TYPE[0].get(action, []))  # 五炸属于最高炸弹；四带二仍按普通带牌组合评分。


def _action_type(action):  # 将 RLCard 的带长度牌型名称归一成武将技能使用的稳定分类。
    if _is_five_bomb(action):  # RLCard 标准动作表没有技能产生的五张同点数。
        return "five_bomb"  # 返回稳定扩展牌型供比较、日志和训练使用。
    types = [card_type for card_type, _ in CARD_TYPE[0].get(action, [])]  # 读取当前动作全部可能牌型解释。
    if not types:  # 未知动作不能参与英雄牌型偏好。
        return "unknown"  # 返回明确未知分类避免字符串猜测。
    card_type = types[0]  # RLCard 对规范动作的首个解释足够用于偏好评分。
    if card_type.startswith("solo_chain_"):  # 合并所有长度的顺子。
        return "straight"  # 关羽单骑只关心顺子而不区分长度。
    if card_type.startswith("pair_chain_"):  # 合并所有长度的连对。
        return "pair_chain"  # 返回统一连对分类。
    if "chain" in card_type:  # 剩余带 chain 的牌型均属于飞机及其带牌形式。
        return "airplane"  # 返回统一飞机分类。
    return card_type  # 单牌、对子、三张和炸弹直接沿用 RLCard 名称。


# Keep legacy callers stable while sharing the canonical standard-rule adapter
# with the simulator.  The surrounding rule model still owns policy scoring.
_action_type = _adapter_action_type


def evaluate_guan_yinping_action(hand, action, hero_state=None):  # 模拟花武随机获得 J、Q、K 后的真实残局，而不是把四张以上动作一律当成收益。
    hero_state = hero_state if isinstance(hero_state, dict) else {}  # 兼容旧回放没有英雄状态或传入空值的情况。
    skill_uses = hero_state.get("skill_uses", {}) if isinstance(hero_state.get("skill_uses", {}), dict) else {}  # 容错读取花武已使用次数。
    uses = max(0, int(skill_uses.get("花武", 0) or 0))  # 将日志或运行状态中的次数规范为非负整数。
    remaining = _remaining_hand(hand, action)  # 先计算正常出牌后的牌面，随后再叠加花武获得牌。
    base_turns, base_solos, base_risk = _minimum_route_quality(hand, remaining)  # 保存不触发技能时的基准牌路供解释和比较。
    active = len(action) >= 4 and uses < 5  # 花武在四张以上动作且五次额度尚未耗尽时触发。
    evaluation = {"active": active, "uses_before": uses, "uses_after": min(5, uses + int(active)), "net_cards_removed": len(action) - int(active), "base_turns": base_turns, "base_solos": base_solos, "base_solo_risk": base_risk}  # 返回日志可直接序列化的公共分项。
    if not active:  # 未触发时保持普通斗地主动作结果，不凭英雄名称修改评分。
        evaluation.update({"expected_turns": float(base_turns), "expected_solos": float(base_solos), "expected_solo_risk": float(base_risk), "worst_turns": base_turns, "group_completion_chance": 0.0, "outcomes": {}})  # 补齐稳定字段供候选日志和测试使用。
        return evaluation  # 立即返回普通牌路结果。
    outcomes = {}  # 保存随机获得每个点数后的精确残局质量。
    remaining_counts = Counter(remaining)  # 判断随机牌能否补成对子、三张或更高结构。
    for gained_rank in ("J", "Q", "K"):  # 花武只会从三个指定人头牌中随机获得一张。
        augmented_hand = _to_rlcard(hand + gained_rank)  # 同一获得点数的完整动作空间可被所有候选共享缓存，兼容手中尚未使用的万能牌。
        outcome_hand = _to_rlcard(remaining + gained_rank)  # 构造该随机分支下实际留下的完整手牌，避免万能牌进入 RLCard 索引。
        turns, solos, solo_risk = _minimum_route_quality(augmented_hand, outcome_hand)  # 精确计算获得牌后还需要多少手及孤张风险。
        outcomes[gained_rank] = {"turns": turns, "solos": solos, "solo_risk": solo_risk, "completes_group": remaining_counts[gained_rank] > 0}  # 记录该牌是否补成现有同点数组。
    evaluation.update({"expected_turns": sum(item["turns"] for item in outcomes.values()) / 3.0, "expected_solos": sum(item["solos"] for item in outcomes.values()) / 3.0, "expected_solo_risk": sum(item["solo_risk"] for item in outcomes.values()) / 3.0, "worst_turns": max(item["turns"] for item in outcomes.values()), "group_completion_chance": sum(int(item["completes_group"]) for item in outcomes.values()) / 3.0, "outcomes": outcomes})  # 使用三种等概率结果评价随机技能的期望与最坏情况。
    return evaluation  # 返回花武完整随机分支，供评分、日志和离线训练共同使用。


def evaluate_zhao_yun_action(hand, action, hero_state=None, pressure_context=None):  # 评价赵云用低单牌或对子主动创造冲阵回收机会的收益。
    hero_state = hero_state if isinstance(hero_state, dict) else {}  # 兼容旧日志缺少英雄状态。
    marks = hero_state.get("marks", {}) if isinstance(hero_state.get("marks", {}), dict) else {}  # 读取实战确认的冲阵累计回收张数。
    recovered = max(0, min(7, int(marks.get("冲阵回收", 0) or 0)))  # 将异常日志值约束到技能有效范围。
    action_type = _action_type(action)  # 冲阵只接受单牌或对子。
    rank = max(action, key=INDEX.__getitem__)  # 单牌和对子点数一致，读取其唯一主点数。
    context = pressure_context if isinstance(pressure_context, dict) else {}  # 缺少牌桌上下文时使用普通前中期默认值。
    nearest_enemy = int(context.get("nearest_enemy", 17))  # 敌方五张内必须优先阻断，不能故意送出可被压低牌。
    eligible = action_type in {"solo", "pair"} and INDEX[rank] <= INDEX["A"]  # 技能点数边界按小于等于 A 处理。
    safe_window = nearest_enemy > 5 and len(hand) > 7  # 仅在前中期且手牌充足时主动诱导，残局不为技能破坏胜利路线。
    active = eligible and recovered < 7 and safe_window  # 达到七张或进入收尾阶段后恢复公共策略。
    progress = min(7 - recovered, len(action)) if active else 0  # 对子被压可一次推进两张，但不超过剩余额度。
    low_rank_bonus = 1 if active and INDEX[rank] <= INDEX["9"] else 0  # 更低的牌更容易被对手正常压制，触发概率更高。
    completion_bonus = 2 if active and recovered + progress >= 7 else 0  # 临近七张时优先完成最后一次有效冲阵。
    opportunity = progress * 2 + low_rank_bonus + completion_bonus  # 形成有界机会分，不能覆盖直接胜利和最少手数。
    reason = "前中期用低牌创造冲阵回收机会" if active else "冲阵已满或牌局进入收尾阶段，按通用胜利策略出牌" if eligible else "当前牌型不能触发冲阵"  # 为逐局日志生成明确原因。
    return {"active": active, "eligible": eligible, "recovered": recovered, "remaining": 7 - recovered, "expected_progress": progress, "opportunity": opportunity, "rank": RLCARD_TO_INTERNAL.get(rank, rank), "reason": reason}  # 返回候选日志和评分共用字段。


def _hero_action_preference(hero, action, last_action_type, hero_state=None, skill_evaluation=None, pressure_context=None, hand=None):  # 为英雄技能相关动作提供不破坏基础残局评分的次级偏好。
    action_type = _action_type(action)  # 获取当前动作的统一牌型。
    if hero == "关羽" and action_type == "straight":  # 单骑通过顺子获得万能牌。
        return -4  # 在同等剩余手数动作中明显优先顺子。
    if hero == "张飞" and last_action_type and action_type == last_action_type:  # 咆哮要求连续两次相同牌型。
        return -3  # 优先延续上一手牌型但不覆盖直接获胜和剩余手数。
    if hero == "关银屏" and len(action) >= 4:  # 花武在四张以上动作后必定随机获得 J、Q 或 K，不能把加牌一律当成正收益。
        evaluation = skill_evaluation if isinstance(skill_evaluation, dict) else None  # 优先采用调用方已计算的完整随机牌路避免重复搜索。
        if evaluation is None:  # 兼容直接调用此辅助函数的旧测试和扩展脚本。
            skill_uses = hero_state.get("skill_uses", {}) if isinstance(hero_state, dict) and isinstance(hero_state.get("skill_uses", {}), dict) else {}  # 读取可选花武次数。
            if int(skill_uses.get("花武", 0) or 0) >= 5:  # 五次额度耗尽后长牌不再产生技能收益或负担。
                return 0  # 完全恢复公共出牌评分。
            return -1  # 信息不足时只保留很弱的同分触发偏好，不再强推长牌。
        if not evaluation.get("active"):  # 实际额度已经耗尽时不得继续奖励花武。
            return 0  # 交还普通牌路决定。
        route_delta = float(evaluation["expected_turns"]) - float(evaluation["base_turns"])  # 比较随机加牌前后的期望牌权需求。
        if route_delta < 0:  # 随机牌能够整体缩短路线时才把花武视为明显收益。
            return -4  # 在同等主牌路中优先获取真正有结构价值的技能牌。
        if route_delta == 0 and evaluation.get("group_completion_chance", 0.0) > 0:  # 不增加手数且可能补成对子或三张时仍有温和价值。
            return -2  # 作为次级同分项选择结构更好的触发动作。
        return 1  # 会增加未来手数或留下新高位孤张时轻微避开触发。
    if hero == "凌统" and action_type in {"solo", "pair"}:  # 勇进只由单牌或对子触发弃牌。
        return -2  # 同等剩余手数时优先保留技能交互机会。
    if hero == "卢植" and action_type in {"solo", "pair"}:  # 儒宗可在压牌后转换单牌和对子。
        return -1  # 使用较弱偏好避免为了技能牺牲更高效组合。
    if hero == "赵云":  # 冲阵需要主动打出容易被压的低单牌或对子。
        evaluation = skill_evaluation if isinstance(skill_evaluation, dict) else evaluate_zhao_yun_action(hand or action, action, hero_state, pressure_context)  # 使用已计算结果或为直接调用补算。
        return -int(evaluation["opportunity"]) if evaluation.get("active") else 0  # 只在安全前中期提高机会，残局和七张后完全停用诱导。
    if hero == "陆逊" and len(action) > 1:  # 破蜀每次出牌都会补牌，多张动作更容易抵消手牌增长。
        return -1  # 轻微偏好多张动作而不强行改变标准牌路。
    return 0  # 其他武将或无关牌型保持标准算法评分。


def build_table_pressure_context(state):  # 根据敌方剩余牌、桌面牌和已见控制牌决定本回合使用最大牌、中位牌或常规保留策略。
    enemy_counts = _enemy_counts(state)  # 只使用当前身份下真正敌方玩家的牌数，不能把农民队友当作压制目标。
    nearest_enemy = min(enemy_counts or [17])  # 敌方最少手牌者决定当前牌桌的紧迫程度。
    history = state.get("history", []) if isinstance(state.get("history", []), list) else []  # 读取可见出牌历史估计高控制牌已经消耗的程度。
    exposed_controls = sum(card in {"A", "2", "X", "D", "B", "R"} for played in history if isinstance(played, (list, tuple)) for card in played)  # 统计已经公开的 A、二和大小王。
    position = state.get("position", "landlord_down")  # 地主需要同时封锁两名农民，农民只对地主施加压力。
    teammate_count = state.get("teammate_card_count") if position != "landlord" else None  # 农民保存队友牌数供送牌修正使用。
    if state.get("table_is_teammate", False):  # 桌面来自队友时不能套用消耗地主的中位牌策略。
        mode = "team_support"  # 由后续安全接管规则决定放行或使用最优成型牌。
        reason = f"队友剩余{teammate_count if teammate_count is not None else '未知'}张，按团队牌路决定放行或成型牌接管"  # 保存明确的协作上下文。
    elif nearest_enemy <= 5:  # 任一真正敌方进入五张以内即视为收尾威胁，不再等待到只剩一两张。
        mode = "maximum_control"  # 使用当前不破坏最短牌路的最大牌，尽量让敌方无法接牌并把牌权留在我方。
        reason = f"{'农民' if position == 'landlord' else '地主'}最少{nearest_enemy}张，已见控制牌{exposed_controls}张，胜利优先执行最大强度封锁"  # 生成带身份的中文原因。
    elif nearest_enemy <= 10:  # 敌方仍需数手但已进入可以主动消耗其控制牌的阶段。
        mode = "medium_attrition"  # 使用十到Q附近的中位牌试探，迫使敌方交出更高点数。
        reason = f"敌方最少{nearest_enemy}张，使用中位牌试探并消耗敌方高牌"  # 记录本回合消耗策略。
    else:  # 敌方手牌很多时过早消耗高牌无法形成可靠封锁。
        mode = "conserve_control"  # 继续优先清理低牌和完整组合，保留后期控制资源。
        reason = f"敌方最少{nearest_enemy}张，继续保留高控制牌"  # 记录常规发展阶段。
    skill_card_estimates = [max(0, int(value)) for value in state.get("opponent_skill_card_estimates", [0, 0]) if isinstance(value, int)]  # 读取从牌数回升确认的其他玩家技能牌数量。
    return {"mode": mode, "nearest_enemy": nearest_enemy, "enemy_counts": enemy_counts, "exposed_controls": exposed_controls, "position": position, "teammate_count": teammate_count, "history_cards": [card for played in history if isinstance(played, (list, tuple)) for card in played], "skill_card_estimates": skill_card_estimates, "skill_uncertainty": min(0.60, 0.15 + 0.10 * sum(skill_card_estimates)), "reason": reason}  # 返回候选评分和带技能增牌风险的稳定上下文。


def evaluate_table_pressure(action, pressure_context):  # 将牌桌压力模式转换成单个合法动作的可解释战术代价。
    context = pressure_context if isinstance(pressure_context, dict) else {"mode": "conserve_control"}  # 旧调用缺少上下文时安全保持原有低牌策略。
    rank_index = max(INDEX[card] for card in action)  # 使用动作最高主点数表示该手牌的控制强度。
    mode = context.get("mode", "conserve_control")  # 读取本回合最大控制、中位消耗或常规保留模式。
    if mode == "maximum_control":  # 敌方即将走完时，在牌路同样短的动作中直接选择更难被压的牌。
        cost = -rank_index  # 点数越大代价越小，从而优先二、王或更大的同牌型。
        desired_rank = "maximum"  # 日志标明目标不是固定点数而是当前最大控制力。
    elif mode == "medium_attrition":  # 中残局用中等强度逼迫敌方交出 A、二、王或炸弹。
        if INDEX["T"] <= rank_index <= INDEX["Q"]:  # 十、J、Q均属于可用于消耗敌方高牌的中位区间。
            cost = 0  # 区间内部保持同等，再由手牌结构和动作长度决定。
        elif rank_index < INDEX["T"]:  # 低牌无法有效逼出敌方高控制牌。
            cost = INDEX["T"] - rank_index  # 越接近十越适合作为试探牌。
        else:  # A、二和王属于后期绝对控制资源，不能在中盘当普通试探牌浪费。
            cost = 3 + (rank_index - INDEX["Q"]) * 2  # 对超过Q的牌施加更高代价，明确保留顶级牌。
        desired_rank = "T-Q"  # 使用区间名称方便中文日志解释。
    else:  # 发展阶段保持原有从低到高走牌逻辑。
        cost = 0  # 不提前介入主评分，继续由原有孤张、动作长度和高牌成本决定正常牌路。
        desired_rank = "low"  # 标记当前目标为保留高牌。
    return {"mode": mode, "cost": cost, "rank_index": rank_index, "desired_rank": desired_rank, "cards_committed": len(action), "reason": context.get("reason", "按常规牌路评分")}  # 返回可训练的动作压力分项。


def evaluate_tactical_utility(hand, action, target, pressure_context):  # 按用户给出的手牌压力、牌权、身份和炸弹四部分计算量化收益。
    context = pressure_context if isinstance(pressure_context, dict) else {}  # 旧状态缺少身份上下文时使用空对象安全降级。
    counts = Counter(hand)  # 统计动作打出前每个点数的自然组合数量。
    action_type = _action_type(action)  # 使用统一牌型判断孤张、对子、炸弹和连续成型牌。
    main_rank = max(action, key=INDEX.__getitem__)  # 读取动作最高点数用于估算外部更大同牌型数量。
    relief = 5 if action_type == "solo" and counts[main_rank] == 1 else 3 if action_type == "pair" and counts[main_rank] == 2 else 2 if action_type == "trio" and counts[main_rank] == 3 else 0  # 清除孤单、孤对和零散三张分别获得5、3、2分。
    known = Counter(hand) + Counter(context.get("history_cards", []))  # 将我方手牌和已见出牌合并为最小可见牌账本。
    deck_limits = {card: (1 if card in {"B", "R"} else 4) for card in INDEX}  # 使用标准牌库估算外部牌；英雄增牌只会令结果偏保守。
    required_count = 1 if action_type == "solo" else 2 if action_type == "pair" else 3 if action_type == "trio" else 4 if action_type == "bomb" else 5 if action_type == "five_bomb" else None  # 仅对可可靠估算的同点数牌型计算收回牌权。
    higher_options = None  # 顺子、连对和带牌需要完整组合，保持未知而不伪造确定控制。
    if required_count is not None:  # 单、对、三张和普通炸弹可以按剩余同点数数量估计。
        higher_options = sum(max(0, deck_limits[rank] - known[rank]) >= required_count for rank in INDEX if INDEX[rank] > INDEX[main_rank])  # 统计外部仍可能组成的更大同牌型点数。
    skill_uncertainty = float(context.get("skill_uncertainty", 0.15))  # 即使尚未观察到净增牌，未知英雄技能仍保留基础生成风险。
    if action_type == "five_bomb" and main_rank == "R":  # 五张大王不存在更高五炸，是技能牌环境中唯一绝对控制动作。
        control_probability = 1.0  # 明确标记稳定收权。
    elif action_type == "rocket":  # 王炸会被任意五炸压制，不能再按标准斗地主视为百分百最大。
        control_probability = max(0.20, 0.92 - skill_uncertainty)  # 根据已观察技能牌数量下调收权概率。
    elif higher_options is not None:  # 对可计数牌型按更大组合数量生成基础概率。
        base_probability = 0.90 if higher_options == 0 else 0.72 if higher_options == 1 else max(0.20, 0.55 - 0.08 * (higher_options - 2))  # 标准牌库越少更大组合，基础收权概率越高。
        control_probability = max(0.05, base_probability - skill_uncertainty)  # 技能回收、复制和生成牌统一降低确定性。
    else:  # 复杂连续牌型无法只靠点数计数准确判断。
        control_probability = max(0.10, 0.45 - skill_uncertainty / 2.0)  # 保守估计而不声称稳定收权。
    control_gain = 10 if control_probability >= 0.90 else 5 if control_probability >= 0.60 else -3  # 概率足够高才给予稳定或大概率收权收益。
    remaining_count = len(hand) - len(action)  # 计算正常出牌后的我方剩余张数供身份残局修正。
    position = context.get("position", "landlord_down")  # 读取地主或农民身份。
    nearest_enemy = int(context.get("nearest_enemy", 17))  # 读取真正敌方最少牌数。
    is_top_control = INDEX[main_rank] >= INDEX["2"] or action_type in {"bomb", "rocket", "five_bomb"}  # 二、王和所有炸弹视为压制型动作。
    role_adjustment = 15 if remaining_count == 0 else 0  # 一次出完获得明确终局奖励，虽已由硬优先级保证仍写入量化日志。
    critical_enemy_count = 3 if position == "landlord" else 5  # 地主在任一农民三张内封锁；农民在地主五张内进入团队阻断。
    if nearest_enemy <= critical_enemy_count:  # 按身份使用用户给出的收尾威胁阈值。
        control_bonus = 8 if position == "landlord" else 10  # 地主封锁双农民加八分，农民阻止地主收尾加十分。
        role_adjustment += control_bonus if is_top_control else (-5 if position == "landlord" else -6)  # 顶级控制牌获奖，容易送权的小牌受罚。
    if position != "landlord" and context.get("teammate_count") is not None and int(context["teammate_count"]) <= 5 and not target:  # 农民先手且队友进入收尾时尝试提供可接的小牌型。
        role_adjustment += 12 if INDEX[main_rank] <= INDEX["9"] and len(action) <= int(context["teammate_count"]) else 0  # 小牌型匹配队友剩余张数时按团队送牌收益加十二分。
    bomb_adjustment = 0  # 初始化炸弹特殊收益。
    if action_type == "rocket":  # 火箭只在敌方收尾时值得使用。
        bomb_adjustment = 20 if nearest_enemy <= 3 else -15  # 紧急阻断+20，闲时乱出-15。
    elif action_type in {"bomb", "five_bomb"}:  # 普通炸弹和最高五炸均只在阻断或明确抢权时使用。
        bomb_adjustment = (20 if action_type == "five_bomb" else 12) if nearest_enemy <= 3 else 2 if target else -10  # 五炸紧急价值等同火箭且可压火箭，闲时仍避免空炸。
    total = relief + control_gain + role_adjustment + bomb_adjustment  # 合并四部分得到正数出牌、负数不出的可解释收益。
    return {"total": total, "relief": relief, "control_gain": control_gain, "control_probability": round(control_probability, 4), "higher_same_type_options": higher_options, "skill_uncertainty": skill_uncertainty, "role_adjustment": role_adjustment, "bomb_adjustment": bomb_adjustment, "action_type": action_type}  # 返回候选日志和pass阈值共用的全部分项。


@lru_cache(maxsize=2048)  # 不同候选和随机技能分支会重复产生相同的结算后手牌。
def _post_skill_route_turns(cards):  # 对技能完整结算后的真实整手牌重新建立合法牌路。
    internal = tuple(cards)  # 缓存键使用内部点数的稳定元组。
    if "W" in internal:  # RLCard 不认识百将牌万能牌。
        return _estimate_hero_turns(internal)  # 保守保留万能牌实体，不把它从牌数中漏掉。
    canonical = _to_rlcard(internal)  # 技能可能新增原手牌没有的点数，因此不能复用原手牌动作表。
    return _minimum_route_quality(canonical, canonical)[0]  # 重新评价完整结算后的最少手数。


def _projection_route_evaluator(full_hand):  # 为统一武将投影提供与生产规则一致的牌路入口。
    full_counts = Counter(full_hand)  # 普通出牌分支仍可复用原整手牌的共享精确规划器。
    def evaluate(cards):  # 技能新增牌仍由同一入口评价，万能牌使用保守结构估算避免被标准动作空间丢弃。
        internal = tuple(cards)
        if _WILDCARD in full_hand:
            return _post_skill_route_turns(internal)  # 万能实体可能已打出，始终按结算后手牌重新选择精确或保守规划。
        if "W" in internal:
            return _post_skill_route_turns(internal)
        remaining = _to_rlcard(internal)
        if all(count <= full_counts[rank] for rank, count in Counter(remaining).items()):
            return _minimum_route_quality(full_hand, remaining)[0]
        return _post_skill_route_turns(internal)  # 技能新增原手牌外点数时按完整新手牌重建。

    return evaluate  # 每个牌局候选共享完整手牌规划缓存。


def _context_for_score(hand, target, hero, hero_state, enemy_counts, pressure_context):  # 兼容直接调用评分函数时构造最小公开决策上下文。
    pressure = pressure_context if isinstance(pressure_context, dict) else {}  # 容错旧测试和历史回放。
    state = {
        "hand_cards": _to_internal(hand),
        "table_cards": _to_internal(target),
        "hero": hero,
        "hero_state": hero_state if isinstance(hero_state, dict) else {},
        "position": pressure.get("position", "landlord_down"),
        "enemy_card_counts": list(enemy_counts or ()),
        "table_is_teammate": bool(pressure.get("table_is_teammate", False)),
        "pressure_context": pressure,
    }
    return _hero_context(state)  # 上下文不包含任何敌方暗牌。


def _score_action(hand, action, urgent, hero=None, last_action_type=None, policy_id="balanced", target="", enemy_counts=None, hero_state=None, pressure_context=None, skill_projection=None, hero_context=None, physical_action=None, action_type=None, hand_expansion=None, hero_skill_evaluation=None):  # 为一个合法生效动作生成完整技能结算后的确定性评分，同时兼容旧调用签名。
    profile = POLICY_PROFILES.get(policy_id, POLICY_PROFILES["balanced"])  # 未知策略编号安全回退到均衡策略。
    physical = physical_action if physical_action is not None else action  # 普通牌两种表示相同；万能牌动作必须按真实屏幕实体扣牌。
    exact_action_type = action_type or _action_type(action)  # 优先采用 RLCard 已确认的精确牌型，禁止统一策略再次用实体万能牌猜测。
    remaining = _remaining_hand(hand, physical)  # 只从真实手牌中扣除实际需要点击的实体牌。
    context = hero_context or _context_for_score(hand, target, hero, hero_state, enemy_counts, pressure_context)  # 全部入口统一为公开信息上下文。
    projection = skill_projection or _evaluate_hero_play(
        context,
        _to_internal(physical),
        _projection_route_evaluator(hand),
        action_type=exact_action_type,
        effective_action=_to_internal(action),
    )  # 先按实体牌扣除，再把 RLCard 精确牌型和生效点数交给统一英雄策略。
    expansion = hand_expansion or evaluate_hand_expansion(
        context.hand,
        _to_internal(physical),
        projection,
    )
    route_origin = remaining if _WILDCARD in hand and _WILDCARD not in remaining else hand
    route_solos, route_solo_risk = _minimum_route_quality(route_origin, remaining)[1:]  # 万能牌已打出时恢复自然剩余手牌的精确残局诊断。
    pressure = evaluate_table_pressure(action, pressure_context)  # 牌桌压力始终使用生效点数，避免把万能牌本身误当作固定牌面。
    tactical = evaluate_tactical_utility(hand, action, target, pressure_context)  # 战术收益同样按该动作在牌桌上的真实生效牌型计算。
    endgame_solos = route_solos if len(hand) <= 10 else 0  # 十张以内进入残局，优先清理以后难以取得牌权打出的孤张。
    endgame_solo_risk = route_solo_risk if len(hand) <= 10 else 0  # 同样手数和单出次数下优先先打低位孤张，保留控制牌。
    bomb_penalty = 0 if urgent or not _is_bomb_action(action) else 1  # 牌路相同时才尽量保留炸弹和王炸，不能压过确定性短牌路线。
    hero_preference = -projection.skill_resource_value * float(profile["hero_scale"])  # 策略配置只作为统一资源契约之后的次级倍率。
    control_cost = projection.control_card_cost  # 实体万能牌与二、王的消耗直接复用统一投影定义。
    length_preference = -len(physical) * int(profile["length_bonus"])  # 在主要评分完全相同时偏好多张动作。
    high_card_cost = projection.high_card_cost * int(profile["high_card_bias"])  # 使用统一投影对万能牌和高控制牌的稳定排序。
    critical_finish_risk = projection.enemy_finish_risk if pressure["mode"] == "maximum_control" else 0  # 强封锁阶段优先避开与敌方剩余张数相同的送牌型。
    critical_pressure = pressure["cost"] if pressure["mode"] == "maximum_control" else 0  # 敌方五张以内允许最大控制力优先于普通结构细分。
    medium_pressure = pressure["cost"] if pressure["mode"] == "medium_attrition" else 0  # 中盘消耗只在统一技能契约之后比较。
    luxun_collection = -float(hero_skill_evaluation.get("expected_total", 0.0)) if hero == "陆逊" and isinstance(hero_skill_evaluation, dict) else 0.0  # 同等完整牌路下优先保留可被破蜀补成炸弹、三张、对子或顺子的骨架。
    formed_lead_types = {"straight", "pair_chain", "airplane", "trio_solo", "trio_pair", "trio"}  # 定义应在前中期优先走掉的顺子、连对、飞机和三带结构。
    lead_shape_priority = -len(action) if not target and len(hand) > 10 and exact_action_type in formed_lead_types else 0  # 仅在前中期优先走成型牌，十张内仍先清难出的孤张。
    lead_shape_rank = max(INDEX[card] for card in action) if lead_shape_priority else len(INDEX)  # 生效动作不含万能牌，可安全按主点数稳定排序。
    projection_prefix = tuple(projection.score_key[1:-3])  # 先读取终局、技能分支、敌友与资源评分。
    flower_control_waste = projection.control_card_cost if hero == "关银屏" and "花武" in projection.triggered_skills else 0  # 花武不能靠消耗二、王或万能牌换取少打一张普通牌。
    projection_priority = projection_prefix[:2] + (flower_control_waste,) + projection_prefix[2:]  # 直接胜利和紧急阻断优先，其余花武路线先保留控制牌再比较预计手数。
    legacy_refinements = (
        critical_finish_risk,
        critical_pressure,
        luxun_collection,
        bomb_penalty,
        lead_shape_priority,
        lead_shape_rank,
        endgame_solos,
        medium_pressure,
        endgame_solo_risk,
        length_preference,
        -tactical["total"],
        hero_preference,
        -expansion.expected_total,
        -expansion.worst_total,
        control_cost,
        high_card_cost,
    )  # 旧牌桌压力与残局规则只能在统一英雄策略契约完全相同后作为稳定细分项。
    return projection_priority + legacy_refinements + (action,)  # 最后一项保留生效动作字符串，确保完全同分时结果可复现。


def _enemy_counts(state):  # 从左右座位牌数中只提取真正敌方玩家，避免农民把队友误判为威胁。
    explicit = state.get("enemy_card_counts")  # 新版观察状态会直接给出按身份解析后的敌方牌数。
    if isinstance(explicit, (list, tuple)) and explicit:  # 接受真实任务和历史回放中的列表或元组。
        return [int(count) for count in explicit if isinstance(count, int) and count > 0]  # 丢弃无效 OCR 数量。
    return [int(count) for count in state.get("opponent_card_counts", [17, 17]) if isinstance(count, int) and count > 0]  # 旧日志缺少队友字段时保持兼容。


def _bounded_projection_log(projection, detail_limit=_MAX_LOGGED_RANDOM_BRANCHES):  # 保留完整内部投影，只限制写入逐局日志的随机分支明细。
    branches = tuple(projection.random_branches)
    limit = max(0, int(detail_limit))
    if len(branches) <= limit:
        return projection.to_dict()
    value = {name: getattr(projection, name) for name in projection.__dataclass_fields__ if name != "random_branches"}  # 大分支时避免先深复制全部明细。
    value["choice"] = projection.choice.to_dict() if projection.choice else None
    value["score_key"] = list(projection.score_key)
    value["random_branches"] = [branch.to_dict() for branch in branches[:limit]]
    value["random_branch_summary"] = {
        "total_count": len(branches),
        "logged_count": min(limit, len(branches)),
        "omitted_count": max(0, len(branches) - limit),
        "probability_sum": round(sum(branch.probability for branch in branches), 6),
        "expected_remaining_turns": projection.expected_remaining_turns,
        "worst_remaining_turns": projection.worst_remaining_turns,
        "min_remaining_cards": min((len(branch.hand) for branch in branches), default=0),
        "max_remaining_cards": max((len(branch.hand) for branch in branches), default=0),
        "max_risk": max((branch.risk for branch in branches), default=0.0),
    }
    return value


def _stage_context(hand, target, enemy_counts, pressure_context, hero_context):
    """Build a stage context from public state already visible to this player."""

    pressure = pressure_context if isinstance(pressure_context, dict) else {}
    seen = [rank for event in hero_context.history for rank in event.get("ranks", ())]
    seen.extend(target)
    counts = Counter(seen)
    nearest_enemy = min(enemy_counts or [17, 17])
    return StageContext(
        own_card_count=len(hand),
        position=str(pressure.get("position", "landlord_down")),
        enemy_card_counts=tuple(int(count) for count in enemy_counts or ()),
        teammate_card_count=pressure.get("teammate_count") if isinstance(pressure.get("teammate_count"), int) else None,
        table_has_cards=bool(target),
        table_is_teammate=bool(pressure.get("table_is_teammate", False)),
        seen_bombs=sum(count >= 4 for rank, count in counts.items() if rank not in {"B", "R", "X", "D"}),
        seen_jokers=sum(counts[rank] for rank in ("B", "R", "X", "D")),
        seen_twos=counts["2"],
        one_turn_finish_risk=nearest_enemy <= 1 or (bool(target) and nearest_enemy == len(target)),
    )


def _score_action_with_stage(
    hand, action, urgent, hero=None, last_action_type=None, policy_id="balanced", target="",
    enemy_counts=None, hero_state=None, pressure_context=None, skill_projection=None,
    hero_context=None, physical_action=None, action_type=None, game_stage=None, hand_expansion=None,
    hero_skill_evaluation=None,
):
    """Append public-stage refinements after all existing legacy score fields."""

    legacy_score = _score_action(
        hand, action, urgent, hero, last_action_type, policy_id, target, enemy_counts,
        hero_state, pressure_context, skill_projection, hero_context, physical_action,
        action_type, hand_expansion, hero_skill_evaluation,
    )
    context = hero_context or _context_for_score(hand, target, hero, hero_state, enemy_counts, pressure_context)
    stage = game_stage or classify_game_stage(_stage_context(hand, target, enemy_counts, pressure_context, context))
    projection = skill_projection or _evaluate_hero_play(context, _to_internal(physical_action or action), _projection_route_evaluator(hand))
    return legacy_score[:-1] + stage_score_components(stage, action_type or _action_type(action), action, projection) + legacy_score[-1:]


def _build_candidate_records(hand, target, enemy_counts, hero=None, last_action_type=None, policy_id="balanced", hero_state=None, pressure_context=None, decision_context=None):  # 一次构建候选、英雄投影和评分，供解释与最终选牌共同消费。
    hero_state = hero_state if isinstance(hero_state, dict) else {}
    pressure_context = pressure_context if isinstance(pressure_context, dict) else {}
    position = pressure_context.get("position", "landlord_down")
    urgent = min(enemy_counts or [17, 17]) <= (3 if position == "landlord" else 5)
    context = decision_context or _context_for_score(hand, target, hero, hero_state, enemy_counts, pressure_context)
    game_stage = classify_game_stage(_stage_context(hand, target, enemy_counts, pressure_context, context))
    route_evaluator = _projection_route_evaluator(hand)
    projection_cache = {}
    records_by_physical = {}
    for effective_action, physical_action in _legal_action_variants(hand, target):
        exact_action_type = _action_type(effective_action)
        projection_key = ("play", physical_action, exact_action_type, effective_action)
        projection = projection_cache.get(projection_key)
        if projection is None:
            projection = _evaluate_hero_play(
                context,
                _to_internal(physical_action),
                route_evaluator,
                action_type=exact_action_type,
                effective_action=_to_internal(effective_action),
            )
            projection_cache[projection_key] = projection
        if not projection.legal:
            continue
        hand_expansion = evaluate_hand_expansion(
            context.hand,
            _to_internal(physical_action),
            projection,
            game_stage.value,
        )
        skill_evaluation = evaluate_guan_yinping_action(hand, physical_action, hero_state) if hero == "关银屏" else evaluate_zhao_yun_action(hand, effective_action, hero_state, pressure_context) if hero == "赵云" else evaluate_luxun_collection(hand, physical_action, projection) if hero == "陆逊" else None
        score = _score_action_with_stage(
            hand,
            effective_action,
            urgent,
            hero,
            last_action_type,
            policy_id,
            target,
            enemy_counts,
            hero_state,
            pressure_context,
            projection,
            context,
            physical_action=physical_action,
            action_type=exact_action_type,
            game_stage=game_stage,
            hand_expansion=hand_expansion,
            hero_skill_evaluation=skill_evaluation,
        )
        record = CandidateDecision(
            effective_action=effective_action,
            physical_action=physical_action,
            action_type=exact_action_type,
            projection=projection,
            score=score,
            hero_skill_evaluation=skill_evaluation,
            table_pressure=evaluate_table_pressure(effective_action, pressure_context),
            tactical_utility=evaluate_tactical_utility(hand, effective_action, target, pressure_context),
            game_stage=game_stage.value,
            hand_expansion=hand_expansion.to_dict(),
        )
        previous = records_by_physical.get(physical_action)
        if previous is None or (record.score, _adapter_action_order(effective_action), effective_action) < (previous.score, _adapter_action_order(previous.effective_action), previous.effective_action):
            records_by_physical[physical_action] = record  # 同一实体点击动作只保留评分最优的万能牌生效解释。
    return tuple(records_by_physical.values()), projection_cache, route_evaluator


def _serialise_candidate(record):  # 将内部投影对象转换成保持旧字段兼容的有界日志对象。
    projection = record.projection
    effective_action = record.effective_action
    physical_action = record.physical_action
    return {
        "cards": _to_internal(physical_action),
        "physical_cards": _to_internal(physical_action),
        "effective_cards": _to_internal(effective_action),
        "physical_action": physical_action,
        "action": effective_action,
        "action_type": record.action_type,
        "game_stage": record.game_stage,
        "score": list(record.score[:-1]),
        "score_components": structured_score(record, _SCORE_REASON_LABELS),
        "is_bomb": _is_bomb_action(effective_action),
        "remaining_turns": projection.expected_remaining_turns,
        "worst_remaining_turns": projection.worst_remaining_turns,
        "opponent_finish_risk": projection.enemy_finish_risk,
        "hero_skill_evaluation": dict(record.hero_skill_evaluation) if record.hero_skill_evaluation is not None else None,
        "skill_projection": _bounded_projection_log(projection),
        "triggered_rule_ids": list(projection.triggered_rules),
        "table_pressure": dict(record.table_pressure),
        "tactical_utility": dict(record.tactical_utility),
        "hand_expansion": dict(record.hand_expansion),
    }


def enumerate_action_candidates(state, decision_context=None, candidate_records=None):  # 枚举并解释当前状态的全部合法候选供日志和训练使用。
    hand = _to_rlcard(state.get("hand_cards", []))  # 转换并排序当前完整实体手牌，万能牌保留为 W。
    target = _to_rlcard(state.get("table_cards", []))  # 转换需要压过的最近桌面生效动作。
    enemy_counts = _enemy_counts(state)  # 只读取地主或两名农民敌方的真实剩余牌数。
    hero_state = state.get("hero_state", {}) if isinstance(state.get("hero_state", {}), dict) else {}  # 容错读取英雄局内状态。
    pressure_context = build_table_pressure_context(state)  # 为全部候选使用同一个牌桌压力阶段，保证评分可比较。
    context = decision_context or _hero_context({**state, "pressure_context": pressure_context})  # 每轮只构造一次公开技能上下文。
    records = candidate_records
    if records is None:
        records, _, _ = _build_candidate_records(hand, target, enemy_counts, state.get("hero"), hero_state.get("last_action_type"), state.get("policy_id", "balanced"), hero_state, pressure_context, context)
    return [_serialise_candidate(record) for record in records]  # 日志序列化不再触发第二次英雄技能投影。


def _choose_action_legacy(hand, target, enemy_counts, hero=None, last_action_type=None, policy_id="balanced", protect_teammate_play=False, hero_state=None, pressure_context=None, decision_context=None, candidate_records=None, projection_cache=None, route_evaluator=None):  # 旧选择器保留到本次重构验证完成。
    context = decision_context or _context_for_score(hand, target, hero, hero_state, enemy_counts, pressure_context)
    records = candidate_records
    if records is None:
        records, built_cache, built_route_evaluator = _build_candidate_records(hand, target, enemy_counts, hero, last_action_type, policy_id, hero_state, pressure_context, context)
        projection_cache = built_cache if projection_cache is None else projection_cache
        route_evaluator = built_route_evaluator if route_evaluator is None else route_evaluator
    if not records:
        return ""
    projection_cache = projection_cache if projection_cache is not None else {}
    route_evaluator = route_evaluator or _projection_route_evaluator(hand)
    position = pressure_context.get("position", "landlord_down") if isinstance(pressure_context, dict) else "landlord_down"
    urgent = min(enemy_counts or [17, 17]) <= (3 if position == "landlord" else 5)
    winning_records = [record for record in records if record["projection"].terminal]
    teammate_route_takeover = None
    teammate_count = pressure_context.get("teammate_count") if isinstance(pressure_context, dict) else None
    nearest_enemy = min(enemy_counts or [17])
    # 非紧急先手若最优候选只是单张二或王，不应为规避一次随机技能增牌而先耗控制牌。
    # 允许普通单牌的期望手数差 0.5、最坏手数差 1，再按最低点数清理。
    if not target and not urgent and not winning_records:
        preliminary_best = min(records, key=lambda record: record["score"])
        if preliminary_best["action_type"] == "solo" and any(card in {"2", "B", "R"} for card in preliminary_best["action"]):
            ordinary_solos = [
                record for record in records
                if record["action_type"] == "solo"
                and not any(card in {"2", "B", "R"} for card in record["action"])
                and record["projection"].expected_remaining_turns <= preliminary_best["projection"].expected_remaining_turns + 0.5
                and record["projection"].worst_remaining_turns <= preliminary_best["projection"].worst_remaining_turns + 1
            ]
            if ordinary_solos:
                return min(ordinary_solos, key=lambda record: max(INDEX.get(card, len(INDEX)) for card in record["action"]))["physical_action"]
    # 敌方进入收尾窗口时，队友牌权不再是绝对保护条件。若队友仍有超过两张牌，
    # 放行可能直接把回合交给即将结束的敌方；优先选择最低代价的合法接管动作。
    if target and protect_teammate_play and urgent and (teammate_count is None or int(teammate_count) > 2):
        emergency_records = [record for record in records if record["action"]]
        if emergency_records:
            # 紧急阻断只要求夺回牌权；能用普通牌时不消耗炸弹、二或王。
            economical = [
                record for record in emergency_records
                if not _is_bomb_action(record["action"])
                and not any(card in {"2", "B", "R"} for card in record["action"])
            ]
            best_record = min(economical or emergency_records, key=lambda record: record["score"])
            return best_record["physical_action"]
    # 敌方 8~10 张、队友仍明显较多时进入中度收尾压力：普通成型牌应主动接管，
    # 但不为此消耗炸弹、王或二。只有敌方不超过八张且没有普通牌时才允许用一对二。
    medium_team_pressure = bool(
        target and protect_teammate_play and teammate_count is not None
        and nearest_enemy <= 10 and int(teammate_count) >= 12
        and int(teammate_count) - nearest_enemy >= 4
    )
    if medium_team_pressure:
        economical = [
            record for record in records
            if record["action"]
            and not _is_bomb_action(record["action"])
            and not any(card in {"2", "B", "R"} for card in record["action"])
        ]
        if economical:
            return min(economical, key=lambda record: record["score"])["physical_action"]
        emergency_pair_twos = [
            record for record in records
            if nearest_enemy <= 8 and (record["action"] == "22" or record["action"] == ("2", "2"))
        ]
        if emergency_pair_twos:
            return emergency_pair_twos[0]["physical_action"]
    if target and protect_teammate_play and not winning_records and (teammate_count is None or int(teammate_count) > 5):
        baseline_turns = _post_skill_route_turns(tuple(_to_internal(hand)))
        takeover_types = {"straight", "pair_chain", "airplane", "trio_solo", "trio_pair"}
        takeover_records = [
            record for record in records
            if record["action_type"] in takeover_types
            and not _is_bomb_action(record["action"])
            and not any(card in {"2", "B", "R"} for card in record["action"])
            and record["projection"].expected_remaining_turns < baseline_turns
        ]
        if takeover_records:
            teammate_route_takeover = min(takeover_records, key=lambda record: record["score"])
    if target and protect_teammate_play and not winning_records and teammate_route_takeover is None:
        return ""
    # 地主面对普通单牌/对子时，若有不消耗二、王、炸弹的合法动作，主动夺回牌权，
    # 避免安全层连续多轮放行导致两名农民轮流控场。
    if target and position == "landlord" and not urgent:
        landlord_economical = [
            record for record in records
            if record["action"]
            and not _is_bomb_action(record["action"])
            and not any(card in {"2", "B", "R"} for card in record["action"])
        ]
        if landlord_economical:
            return min(landlord_economical, key=lambda record: record["score"])["physical_action"]
    best_record = teammate_route_takeover or min(records, key=lambda record: record["score"])
    farmer_route_press = None
    if target and position != "landlord" and not protect_teammate_play and not winning_records:
        baseline_turns = _post_skill_route_turns(tuple(_to_internal(hand)))
        route_records = [
            record for record in records
            if not _is_bomb_action(record["action"])
            and not any(card in {"2", "B", "R"} for card in record["action"])
            and record["projection"].expected_remaining_turns < baseline_turns
        ]
        if route_records:
            farmer_route_press = min(route_records, key=lambda record: record["score"])
            best_record = farmer_route_press
    best_tactical = best_record["tactical_utility"]
    best_remaining_turns = best_record["projection"].worst_remaining_turns
    if (
        target and position != "landlord" and not protect_teammate_play and not urgent
        and not winning_records and nearest_enemy > 10 and best_remaining_turns > 1
        and any(card in {"2", "B", "R"} for card in best_record["action"])
    ):
        return ""  # 地主仍有大量牌时，不用二或王换取数手之后才可能兑现的牌路收益。
    reserved_control_only = bool(target and position != "landlord" and all(_is_bomb_action(record["action"]) or any(card in {"2", "B", "R"} for card in record["action"]) for record in records))
    if not urgent and not winning_records and farmer_route_press is None and reserved_control_only and best_remaining_turns > 1:
        return ""
    if target and position != "landlord" and not urgent and not winning_records and farmer_route_press is None and teammate_route_takeover is None and best_remaining_turns > 1 and best_tactical["total"] < 0:
        return ""
    if target and not urgent and all(_is_bomb_action(record["action"]) for record in records) and best_remaining_turns > 1:
        return ""
    if target and not urgent:
        pass_key = ("pass", "", "none")
        pass_projection = projection_cache.get(pass_key)
        if pass_projection is None:
            pass_projection = _evaluate_hero_play(context, "pass", route_evaluator)
            projection_cache[pass_key] = pass_projection
        if pass_projection.triggered_rules and (pass_projection.expected_remaining_turns, pass_projection.worst_remaining_turns) < (best_record["projection"].expected_remaining_turns, best_record["projection"].worst_remaining_turns):
            return ""
    return best_record["physical_action"]  # 最终只返回真实可点击实体，生效动作仅保留在候选解释中。


def _choose_action(hand, target, enemy_counts, hero=None, last_action_type=None, policy_id="balanced", protect_teammate_play=False, hero_state=None, pressure_context=None, decision_context=None, candidate_records=None, projection_cache=None, route_evaluator=None):
    """Select one physical action through immutable candidates and soft scoring."""

    hero_state = hero_state if isinstance(hero_state, dict) else {}
    pressure_context = pressure_context if isinstance(pressure_context, dict) else {}
    hero_context = decision_context or _context_for_score(hand, target, hero, hero_state, enemy_counts, pressure_context)
    records = candidate_records
    if records is None:
        records, built_cache, built_route_evaluator = _build_candidate_records(
            hand, target, enemy_counts, hero, last_action_type, policy_id, hero_state, pressure_context, hero_context
        )
        projection_cache = built_cache if projection_cache is None else projection_cache
        route_evaluator = built_route_evaluator if route_evaluator is None else route_evaluator
    selection_context = DecisionContext(
        hand=hand,
        target=target,
        enemy_counts=tuple(enemy_counts or (17, 17)),
        hero=hero,
        last_action_type=last_action_type,
        policy_id=policy_id,
        protect_teammate_play=protect_teammate_play,
        hero_state=hero_state,
        pressure=pressure_context,
        hero_context=hero_context,
    )
    projection_cache = projection_cache if projection_cache is not None else {}
    route_evaluator = route_evaluator or _projection_route_evaluator(hand)

    def baseline_turns() -> int:
        return _post_skill_route_turns(tuple(_to_internal(hand)))

    def project_pass():
        pass_key = ("pass", "", "none")
        projection = projection_cache.get(pass_key)
        if projection is None:
            projection = _evaluate_hero_play(hero_context, "pass", route_evaluator)
            projection_cache[pass_key] = projection
        return projection

    decision = _DECISION_CORE.choose(
        selection_context,
        records,
        is_bomb=_is_bomb_action,
        rank_index=lambda card: INDEX.get(card, len(INDEX)),
        baseline_turns=baseline_turns,
        pass_projection=project_pass,
    )
    _choose_action.last_search_result = decision.search
    if decision.candidate is None:
        return ""
    return decision.candidate.physical_action


def load_model(weights_path):  # 创建无需外部权重的 RLCard 官方规则模型。
    return {"engine": "rlcard-action-space-v3", "last_decision": None}  # 返回可保存最近候选解释的轻量模型对象。


_SCORE_REASON_LABELS = (
    "技能结算后的直接胜利",
    "敌方紧急阻断",
    "技能结算后预计剩余手数",
    "随机技能最坏剩余手数",
    "技能分支期望风险",
    "技能分支最坏风险",
    "队友牌权成本",
    "敌方收尾风险",
    "技能目标敌友成本",
    "其他公开武将触发成本",
    "技能资源与标记价值",
    "敌方关键收尾牌型风险",
    "紧急控牌强度",
    "陆逊破蜀集炸弹与顺子收益",
    "炸弹成本",
    "主动成型牌优先级",
    "成型牌点数",
    "残局孤张手数",
    "中盘控牌强度",
    "残局孤张风险",
    "动作长度",
    "牌桌战术收益",
    "策略配置下技能资源价值",
    "技能扩大出牌能力期望收益",
    "技能扩大出牌能力最坏收益",
    "二、王和万能牌控制成本",
    "高牌成本",
)


def _rejected_candidates(candidates, chosen_physical_action):  # 为每个未选候选记录第一个决定性排序差异。
    chosen = next((candidate for candidate in candidates if candidate.get("physical_action") == chosen_physical_action), None)
    if chosen is None:
        return [
            {"cards": candidate.get("cards", []), "reason": "统一不出投影或团队安全层优于该出牌候选"}
            for candidate in candidates
        ]
    records = []
    chosen_score = chosen.get("score", [])
    for candidate in candidates:
        if candidate is chosen:
            continue
        candidate_score = candidate.get("score", [])
        differing = next(
            (index for index, (best, other) in enumerate(zip(chosen_score, candidate_score)) if best != other),
            None,
        )
        label = _SCORE_REASON_LABELS[differing] if differing is not None and differing < len(_SCORE_REASON_LABELS) else "确定性平局项"
        records.append({"cards": candidate.get("cards", []), "reason": f"{label}不如最终候选", "triggered_rule_ids": candidate.get("triggered_rule_ids", [])})
    return records


def predict(model, state):  # 根据屏幕识别状态选择一手合法且尽量减少剩余手数的动作。
    hand = _to_rlcard(state.get("hand_cards", []))  # 转换并排序当前完整手牌。
    target = _to_rlcard(state.get("table_cards", []))  # 转换需要压过的最近桌面动作。
    if not hand:  # 没有完整手牌时拒绝生成猜测动作。
        raise ValueError("RLCard 规则模型没有收到完整手牌")  # 报告识牌状态错误供任务安全兜底。
    if target and target not in CARD_TYPE[0] and not _is_five_bomb(target):  # 桌面 OCR 结果既非标准牌型也非五炸时拒绝错误跟牌。
        raise ValueError(f"桌面牌不是 RLCard 支持的合法牌型: {target}")  # 保留具体错误动作供素材排查。
    hero_state = state.get("hero_state", {}) if isinstance(state.get("hero_state", {}), dict) else {}  # 容错读取可选英雄局内状态。
    enemy_counts = _enemy_counts(state)  # 解析当前身份下真正需要阻断的敌方牌数。
    pressure_context = build_table_pressure_context(state)  # 根据真实敌方牌数和已见控制牌确定本回合压制阶段。
    decision_context = _hero_context({**state, "pressure_context": pressure_context})  # 生产入口只构造公开信息技能上下文。
    candidate_records, projection_cache, route_evaluator = _build_candidate_records(
        hand,
        target,
        enemy_counts,
        state.get("hero"),
        hero_state.get("last_action_type"),
        state.get("policy_id", "balanced"),
        hero_state,
        pressure_context,
        decision_context,
    )  # 普通牌和万能牌共享同一批实体/生效候选与技能投影。
    candidates = enumerate_action_candidates(state, decision_context, candidate_records=candidate_records)  # 日志复用候选投影，不重复执行随机技能分支。
    protect_teammate_play = bool(state.get("table_is_teammate", False)) or bool(state.get("teammate_should_finish", False))  # 只要上一手来自队友就保护其牌权，旧回放字段继续兼容。
    action = _choose_action(
        hand,
        target,
        enemy_counts,
        state.get("hero"),
        hero_state.get("last_action_type"),
        state.get("policy_id", "balanced"),
        protect_teammate_play,
        hero_state,
        pressure_context,
        decision_context,
        candidate_records=candidate_records,
        projection_cache=projection_cache,
        route_evaluator=route_evaluator,
    )  # 选择阶段返回真实可点击实体牌，牌型与技能触发仍使用缓存中的生效动作。
    chosen_record = next((record for record in candidate_records if record.physical_action == action), None)
    if isinstance(model, dict):  # 规则模型字典允许保存本次可解释决策。
        farmer_pressed_landlord = bool(target and action and state.get("position", "landlord_down") != "landlord" and not protect_teammate_play)  # 标记农民本回合确实使用合法牌压制地主。
        if chosen_record is not None:
            chosen_projection = _bounded_projection_log(chosen_record.projection)
            effective_action = chosen_record.effective_action
        else:
            pass_key = ("pass", "", "none")
            pass_projection = projection_cache.get(pass_key)
            if pass_projection is None:
                pass_projection = _evaluate_hero_play(decision_context, "pass", route_evaluator)
                projection_cache[pass_key] = pass_projection
            chosen_projection = _bounded_projection_log(pass_projection)  # 不出也记录完整技能结算结果。
            effective_action = ""
        action_finishes = bool(action and chosen_projection.get("terminal"))  # 强制技能结算后才标记真正结束。
        teammate_safe_takeover = bool(protect_teammate_play and action and not action_finishes)  # 标记队友牌较多时由我方成型牌安全接管的情况。
        reason = "队友出的牌仍在桌面，当前没有安全成型牌接管，选择不出保护队友牌权" if protect_teammate_play and not action else "压过队友后可直接出完，立即结束牌局不会损害队友" if protect_teammate_play and action_finishes else "队友牌较多，我方用最优牌路中的三带、顺子或连对成型牌继续压制" if teammate_safe_takeover else "普通压牌属于当前最优牌路，主动压制地主并减少其连续出牌机会" if farmer_pressed_landlord else pressure_context["reason"]  # 生成可学习的队友接管、主动压地主和牌桌压力决策原因。
        model["last_decision"] = {"round_id": state.get("round_id"), "policy_id": state.get("policy_id", "balanced"), "candidates": candidates, "chosen": _to_internal(action), "final_choice": _to_internal(action), "effective_choice": _to_internal(effective_action), "reason": reason, "table_is_teammate": protect_teammate_play, "table_pressure": pressure_context, "rule_ids": chosen_projection.get("triggered_rules", []), "triggered_skills": chosen_projection.get("triggered_skills", []), "skill_before_cards": len(decision_context.hand), "skill_after_cards": chosen_projection.get("expected_remaining_cards"), "random_branches": chosen_projection.get("random_branches", []), "chosen_projection": chosen_projection, "rejected_candidates": _rejected_candidates(candidates, action)}  # 保存实体选择、生效牌型、规则编号、技能前后牌数、随机分支和拒绝其他候选的原因。
    return _to_internal(action)  # 返回项目内部点数列表供现有屏幕映射与提交逻辑使用。
