import re  # 使用正则表达式过滤 OCR 返回的牌面字符。
from collections import Counter  # 按点数匹配模型动作和屏幕手牌位置。
from itertools import combinations  # 枚举技能要求选择的全部合法实体牌组合。
from pathlib import Path  # 解析项目内模型适配脚本和训练权重路径。

import cv2  # 计算手牌边缘以逐张分割重叠卡牌。
import numpy as np  # 统计卡牌边缘强度并生成等距牌位。
from ok.feature.Box import Box  # 为分割后的每张手牌创建可点击屏幕区域。

from ok_tasks.GameRuntime import CardLedger, GameState, HeroRuntimeState  # 导入统一牌局状态、牌账本和英雄局内状态。
from ok_tasks.MaterialCollectorTask import MaterialCollectorTask  # 复用已经验证的进场、选英雄和结算状态机。
from ok_tasks.HeroStrategy import HeroStatistics, classify_action, normalize_hero_name  # 接入账号武将识别、策略状态和胜率统计。
from ok_tasks.card_ai.hero_policy import HeroDecisionContext, apply_skill_choice, select_skill_choice  # 实战二级交互与离线模拟共享同一技能选择和完整结算投影。
from ok_tasks.card_ai.heroes import HERO_REGISTRY, PASSIVE_OWNED_HEROES, T0_HEROES, T1_HEROES, normalize_hero_name as normalize_registered_hero_name  # 选将仍限账号列表，牌局内识别覆盖全65武将。
from ok_tasks.card_ai.rules import estimate_route_turns  # 使用技能模拟器相同的稳定剩余牌路估算器。
from ok_tasks.PolicyOptimizer import PolicyOptimizer  # 导入整局策略探索、晋升和回滚管理器。
from ok_tasks.StrategyLearning import StrategyLearningPipeline  # 导入逐局失败复盘、相似局对照和策略知识入库流水线。
from ok_tasks.RlCardRuleModel import choose_bid  # 复用开局叫分评估；技能交互统一交给 hero_policy。
from ok_tasks.ai_model_adapter import CARD_ORDER, AiModelError, TrainedModelAdapter, normalize_card  # 加载训练模型并校验动作。


LEAD_SEQUENCE_RANKS = CARD_ORDER[:12]  # 主动顺子和连对只允许使用三到 A，不包含二和大小王。
IDENTITY_SEAT_FEATURES = {"self": "Identity Mark No. 1", "next_player": "Identity Mark No. 2", "next_next_player": "Identity Mark No. 3"}  # 使用图片中已经配置的自己、下家、下下家座位关系。
MAX_SKILL_HAND_CARDS = 40  # 英雄复制、回收和补牌会突破标准斗地主二十张上限，因此保留充足的技能手牌容量。


def is_live_skill_interaction_verified(hero, skill=None):  # 统一读取注册表的 UI 验证状态，未完成实战验证仍由逐次结果校验保护。
    specs = tuple(spec for spec in HERO_REGISTRY.get(hero or "", ()) if spec.interactive)
    if skill:
        specs = tuple(spec for spec in specs if spec.name == skill)
    return bool(specs) and all(spec.ui_verified for spec in specs)


def sync_observed_passive_skill_uses(hero, runtime_state, ledger_changes):  # 根据真实新增牌同步无需按钮确认的被动技能次数。
    if runtime_state is None:  # 缺少跨回合状态时不能安全累计任何被动技能。
        return 0  # 保持旧会话兼容并避免空对象异常。
    if hero == "赵云":  # 赵云唯一的技能加牌来源就是被压后收回并变大的单牌或对子。
        observed = sum(len(change.cards) for change in ledger_changes if change.event_type == "gain")  # 每张重新回到手中的牌累计一次冲阵回收进度。
        if not observed:  # 本次没有确认到新增手牌。
            return 0  # 不根据动画或牌数猜测技能触发。
        before = int(runtime_state.marks.get("冲阵回收", 0) or 0)  # 读取此前真实确认的累计回收张数。
        runtime_state.marks["冲阵回收"] = min(7, before + observed)  # 达到七张后封顶，供策略停止继续诱导低牌被压。
        return runtime_state.marks["冲阵回收"] - before  # 返回本次新增进度供结构化日志展示。
    if hero != "关银屏":  # 其余英雄的增牌不能仅凭点数变化唯一反推出技能次数。
        return 0  # 保持各自已有的按钮或事件状态逻辑。
    observed = sum(1 for change in ledger_changes if change.event_type == "gain" for card in change.cards if card in {"J", "Q", "K"})  # 每张新增人头牌对应一次花武触发，允许两名玩家连续长牌造成多次获得。
    if not observed:  # 本次没有能够确认的花武获得牌。
        return 0  # 不改变运行状态，避免普通手牌识别抖动消耗技能次数。
    before = int(runtime_state.skill_uses.get("花武", 0) or 0)  # 读取此前已经确认的真实次数。
    runtime_state.skill_uses["花武"] = min(5, before + observed)  # 按技能五次上限累计，供下一次动作评分判断是否还会加牌。
    return runtime_state.skill_uses["花武"] - before  # 返回实际新增次数供结构化日志记录。


def update_opponent_skill_card_estimates(previous_counts, current_counts, estimates):  # 用对手牌数的可确认回升追踪敌方和队友技能增牌。
    current = [int(value) for value in list(current_counts or [])[:2]]  # 固定按左、右两个座位读取当前牌数。
    current += [17] * (2 - len(current))  # 缺失座位使用保守默认值但不主动推断技能牌。
    tracked = [max(0, int(value)) for value in list(estimates or [])[:2]]  # 复制上轮仍可能保留的技能牌估计。
    tracked += [0] * (2 - len(tracked))  # 为旧会话补齐两个座位。
    if not isinstance(previous_counts, (list, tuple)) or len(previous_counts) < 2:  # 第一帧没有可靠差值，不能把开局牌数当成技能获得。
        return tracked, []  # 仅建立基线，不产生虚假事件。
    changes = []  # 保存本轮明确的技能增牌或估计消耗事件。
    for index, (before, after) in enumerate(zip(previous_counts[:2], current)):  # 分座位比较相邻两次我方决策看到的手牌数。
        delta = int(after) - int(before)  # 正数表示在两次观察之间净增加手牌。
        old_estimate = tracked[index]  # 保存变化前估计供日志记录。
        if 0 < delta <= 5:  # 一至五张净回升可以可靠视为技能获得；更大跳变更可能是OCR回退到默认17。
            tracked[index] += delta  # 将明确新增牌计入仍可能存在的技能牌数量。
        elif delta < 0:  # 对手出牌后无法知道打出的是自然牌还是技能牌。
            tracked[index] = max(0, tracked[index] + delta)  # 保守假设优先消耗技能牌，避免永久高估不确定性。
        if tracked[index] != old_estimate:  # 只记录真正改变估计的座位。
            changes.append({"seat_index": index, "count_before": int(before), "count_after": int(after), "delta": delta, "estimate_before": old_estimate, "estimate_after": tracked[index]})  # 返回完整可回放变化。
    return tracked, changes  # 返回本轮更新后的两个座位估计及日志事件。


def stabilize_opponent_card_counts(previous_counts, observed_counts, maximum_single_gain=5):  # 过滤牌数 OCR 在十一等窄数字上回跳到十七的异常结果。
    observed = [int(value) for value in list(observed_counts or [])[:2]]  # 固定按左、右座位读取本帧原始观察值。
    observed += [17] * (2 - len(observed))  # 旧调用缺少座位时保持兼容默认值。
    if not isinstance(previous_counts, (list, tuple)) or len(previous_counts) < 2:  # 首帧没有时序基线，不能拒绝合法的技能初始牌数。
        return observed, []  # 直接建立本局第一组可靠基线。
    stable = []  # 保存提供给身份策略和搜索器的稳定牌数。
    anomalies = []  # 保存被拒绝的 OCR 回跳供逐局日志诊断。
    for seat_index, (before, after) in enumerate(zip(previous_counts[:2], observed)):  # 分座位执行单向异常校验。
        before_value = int(before)  # 规范上一可靠牌数。
        gain = int(after) - before_value  # 正数表示技能增牌或 OCR 错读，负数可由正常出牌产生。
        if gain > int(maximum_single_gain):  # 当前实战英雄一次结算最多净增五张，超过上限视为十七回退或错读。
            stable.append(before_value)  # 保留上一可靠值，避免算法错误退出紧急封锁阶段。
            anomalies.append({"seat_index": seat_index, "previous_count": before_value, "observed_count": int(after), "accepted_count": before_value, "reason": "single_gain_exceeds_skill_limit"})  # 记录完整纠错证据。
        else:  # 正常出牌减少或一至五张技能增加均保留。
            stable.append(int(after))  # 让英雄技能获得牌仍能进入牌账本和不确定性估计。
    return stable, anomalies  # 同时返回决策值和可观察异常。


def choose_hero_candidate(candidates, statistics, exploration_games=10, preferred=None):  # 从三个OCR候选中先执行用户偏好，再按安全可自动化的T0、T1顺序选将。
    recognized = [hero for hero in candidates if hero is not None]  # 保留左中右顺序并过滤无法确认的名称。
    automation_ready = [  # 被动英雄和已完成二级交互验证的英雄均可进入强度榜优先池。
        hero for hero in recognized
        if hero in PASSIVE_OWNED_HEROES
        or (HERO_REGISTRY.get(hero) and all(not spec.interactive or spec.ui_verified for spec in HERO_REGISTRY[hero]))
    ]
    selection_pool = automation_ready or recognized  # 有可靠自动化候选时排除尚未验证的交互英雄，否则维持旧统计兜底。
    if preferred in automation_ready:  # 指定偏好只有同时属于本轮可安全自动化候选时才具有最高优先级。
        return preferred  # 返回用户偏好的可靠英雄。
    t0_candidates = [hero for hero in T0_HEROES if hero in selection_pool]  # 按用户强度榜顺序提取当前可安全自动化的T0候选。
    if t0_candidates:  # T0强度先验高于普通武将的冷启动探索，避免为了补样本主动放弃顶级角色。
        return statistics.choose(t0_candidates, exploration_games)  # 多名T0同时出现时仍用真实局数和胜率挑战同档内部顺序。
    t1_candidates = [hero for hero in T1_HEROES if hero in selection_pool]  # T0缺席时只在可靠候选中挑战用户给出的T1顺序。
    if t1_candidates:
        return statistics.choose(t1_candidates, exploration_games)
    return statistics.choose(selection_pool, exploration_games) if selection_pool else None  # 没有安全T0/T1时保持原有候选探索与平滑胜率选择。


def detect_skill_option_card_boxes(frame, region=None):  # 从技能获取牌库区域定位完整展示牌，避免点击 OCR 文字框的偏移坐标。
    if frame is None or frame.size == 0:  # 空画面无法执行卡片几何检测。
        return []  # 返回空列表交给安全暂停分支。
    height, width = frame.shape[:2]  # 读取当前游戏画面尺寸以按比例适配不同分辨率。
    if region is not None:  # 优先使用用户明确标注的技能获取牌库区域。
        top = int(round(region.y + region.height * 0.24))  # 从弹窗标题下方开始扫描展示牌。
        bottom = int(round(region.y + region.height * 0.70))  # 在进度条和取消/确定按钮上方结束。
        left = int(round(region.x + region.width * 0.08))  # 保留弹窗内部左侧第一张牌。
        right = int(round(region.x + region.width * 0.92))  # 保留弹窗内部右侧最后一张牌。
    else:  # 兼容旧素材尚未加载新区域的会话。
        top = int(round(height * 0.235))  # 覆盖真实弹窗卡牌从顶部边框到下边框的主要浅色区域。
        bottom = int(round(height * 0.445))  # 排除下方取消和确定按钮，防止把按钮误当成卡牌。
        left = int(round(width * 0.24))  # 限制在居中的技能弹窗内部。
        right = int(round(width * 0.76))  # 保留三张底牌并排显示的完整横向范围。
    if bottom <= top or right <= left:  # 极小或异常画面不能形成有效检测区域。
        return []  # 拒绝生成猜测坐标。
    gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)  # 将卡片区域转为灰度以提取浅色牌面。
    bright_columns = (gray > 170).mean(axis=0) >= 0.50  # 完整牌面纵列大部分为浅色，而弹窗背景与卡间空隙较暗。
    runs = []  # 保存连续浅色纵列形成的候选牌框。
    run_start = None  # 记录当前连续区间的起点。
    for offset, active in enumerate(bright_columns):  # 从左到右扫描浅色列。
        if active and run_start is None:  # 首次进入浅色卡片区域。
            run_start = offset  # 保存卡片左边缘。
        elif not active and run_start is not None:  # 离开当前浅色卡片区域。
            runs.append((run_start, offset))  # 保存当前候选区间。
            run_start = None  # 准备检测下一张卡片。
    if run_start is not None:  # 最后一张卡片可能延伸到检测区域右边界。
        runs.append((run_start, len(bright_columns)))  # 补记未闭合区间。
    min_width = max(36, int(round(width * 0.025)))  # 同时兼容三张大牌和顺子技能中更多张较窄的展示牌。
    max_width = int(round(width * 0.105))  # 排除连成大块的弹窗浅色背景或按钮。
    card_boxes = []  # 保存通过尺寸校验的完整可点击卡片框。
    for start, end in runs:  # 检查每个连续浅色区间。
        card_width = end - start  # 计算当前候选宽度。
        if min_width <= card_width <= max_width:  # 只接受符合技能大卡片比例的区间。
            card_boxes.append(Box(left + start, top, card_width, bottom - top, name="skill_option_card"))  # 使用绝对坐标创建整张牌的安全点击框。
    return card_boxes if 1 <= len(card_boxes) <= 12 else []  # 接受一至十二张合理展示牌，异常布局保持暂停。


def classify_identity_regions(frame, regions):  # 根据三个已标注身份区的蓝色“农”和金色“主”判断我方位置。
    if frame is None or frame.size == 0:  # 空画面不能进行颜色识别。
        return None  # 返回未知交给手牌数安全回退。

    def icon_kind(box):  # 对单个身份图标计算主色比例。
        left = max(0, int(box.x))  # 限制裁剪左边界。
        top = max(0, int(box.y))  # 限制裁剪上边界。
        right = min(frame.shape[1], int(box.x + box.width))  # 限制裁剪右边界。
        bottom = min(frame.shape[0], int(box.y + box.height))  # 限制裁剪下边界。
        if right <= left or bottom <= top:  # 无效矩形没有可分类像素。
            return None  # 返回未知。
        hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)  # 转换到HSV稳定区分蓝底农民和金底地主。
        blue = cv2.inRange(hsv, (90, 60, 40), (130, 255, 255))  # 提取蓝色农民图标背景。
        gold = cv2.inRange(hsv, (8, 60, 40), (45, 255, 255))  # 提取金色地主图标背景与文字。
        blue_ratio = cv2.countNonZero(blue) / float(blue.size)  # 计算农民色占比。
        gold_ratio = cv2.countNonZero(gold) / float(gold.size)  # 计算地主色占比。
        if gold_ratio >= 0.30 and gold_ratio > blue_ratio:  # 金色明显占优时识别为地主。
            return "landlord"  # 返回地主图标。
        if blue_ratio >= 0.30 and blue_ratio > gold_ratio:  # 蓝色明显占优时识别为农民。
            return "farmer"  # 返回农民图标。
        return None  # 动画、遮挡或颜色不足时不猜测。

    self_feature = IDENTITY_SEAT_FEATURES["self"]  # 读取图片中配置的自己身份框名称。
    next_feature = IDENTITY_SEAT_FEATURES["next_player"]  # 读取图片中配置的下家身份框名称。
    next_next_feature = IDENTITY_SEAT_FEATURES["next_next_player"]  # 读取图片中配置的下下家身份框名称。
    self_kind = icon_kind(regions.get(self_feature)) if regions.get(self_feature) else None  # 识别自己座位的主农图标。
    next_kind = icon_kind(regions.get(next_feature)) if regions.get(next_feature) else None  # 识别下家座位的主农图标。
    next_next_kind = icon_kind(regions.get(next_next_feature)) if regions.get(next_next_feature) else None  # 识别下下家座位的主农图标。
    if self_kind == "landlord":  # 我方图标直接为主。
        return "landlord"  # 锁定地主位置。
    if next_kind == "landlord":  # 自己的下家是地主，因此自己位于地主行动顺序之前。
        return "landlord_up"  # 我方为地主上家。
    if next_next_kind == "landlord":  # 自己的下下家也是自己的上家，因此自己位于地主行动顺序之后。
        return "landlord_down"  # 我方为地主下家。
    return None  # 三个区域没有可靠地主图标时交给旧逻辑。


def resolve_team_context(position, opponent_counts, table_player=None):  # 根据自己、下家、下下家的固定座位关系解析敌友信息。
    left_count, right_count = (list(opponent_counts) + [17, 17])[:2]  # 标注区域按左侧下下家、右侧下家的顺序保存。
    if position == "landlord":  # 地主的左右两家都是敌方农民。
        return [left_count, right_count], None, False  # 地主没有队友，桌面任一对手动作都属于敌方。
    if position == "landlord_up":  # 下家是地主时，左侧下下家是我方农民队友。
        return [right_count], left_count, table_player == "next_next_player"  # 右侧地主为敌方并识别左侧队友动作。
    return [left_count], right_count, table_player == "next_player"  # 地主下家以左侧地主为敌方、右侧下家为队友。


def choose_terminal_wildcard_action(hand_cards, table_cards):  # 处理最后一张万能牌与一张自然单牌能够直接组成对子的残局。
    cards = [str(card) for card in hand_cards]  # 保留实体万能牌编码供后续点击真实牌框。
    if len(cards) != 2 or cards.count("W") != 1:  # 只接管用户反馈的两张终局边界，其他万能组合继续由通用模型处理。
        return []  # 非目标残局不改变现有策略。
    natural = next((card for card in cards if card != "W"), None)  # 读取万能牌需要复制点数的唯一自然单牌。
    if natural not in CARD_ORDER[:-2]:  # 大小王不能作为普通对子补全目标。
        return []  # 保持单王与万能牌的原有出牌规则。
    target = [str(card) for card in table_cards]  # 复制桌面动作避免修改调用方状态。
    if not target:  # 我方拥有牌权时万能牌可直接复制自然单牌。
        return cards  # 一次选择两张实体牌组成对子并直接结束牌局。
    if len(target) == 2 and target[0] == target[1] and target[0] in CARD_ORDER and CARD_ORDER.index(natural) > CARD_ORDER.index(target[0]):  # 跟牌时只在补出的对子确实更大时使用。
        return cards  # 返回实体万能牌和自然牌，游戏负责按自然点数解释对子。
    return []  # 单牌、炸弹、较大对子等桌面动作不能由该终局对子压制。


def resolve_effective_action(physical_action, model_decision):  # 只采纳与本次真实点击动作匹配的生效牌面，避免读取旧决策日志。
    physical = [str(card) for card in physical_action]
    if not isinstance(model_decision, dict) or model_decision.get("chosen") != physical:
        return physical
    effective = model_decision.get("effective_choice")
    if not isinstance(effective, (list, tuple)) or len(effective) != len(physical):
        return physical
    effective_cards = [str(card) for card in effective]
    if any(card not in CARD_ORDER and card != "W" for card in effective_cards):
        return physical
    return effective_cards


def _longest_consecutive_group(ranks, minimum_length):  # 从候选点数中选择最长且起点最低的连续序列。
    indexes = sorted(LEAD_SEQUENCE_RANKS.index(rank) for rank in set(ranks) if rank in LEAD_SEQUENCE_RANKS)  # 将候选点数转换为可比较的连续序号。
    best = []  # 初始化尚未找到的最佳连续序列。
    current = []  # 初始化当前正在扫描的连续序列。
    for index in indexes:  # 按点数从小到大扫描所有候选序号。
        if current and index != current[-1] + 1:  # 当前点数与上一点数不连续时结束本段。
            if len(current) > len(best):  # 只在当前段更长时替换最佳结果。
                best = list(current)  # 保存长度更优且起点更低的连续段。
            current = []  # 开始记录新的连续段。
        current.append(index)  # 将当前点数加入正在扫描的连续段。
    if len(current) > len(best):  # 扫描结束后检查最后一个连续段。
        best = list(current)  # 保存最后一段更长的结果。
    if len(best) < minimum_length:  # 斗地主顺子或连对没有达到最低组数时判定不可用。
        return []  # 返回空序列让策略继续选择其他合法牌型。
    return [LEAD_SEQUENCE_RANKS[index] for index in best]  # 将最佳连续序号转换回内部点数。


def is_basic_legal_lead(action):  # 判断常见主动牌型是否可以作为一手合法打出。
    if choose_terminal_wildcard_action(action, []):  # 万能牌与唯一自然单牌可以直接补成对子。
        return True  # 接受两张实体牌组成的万能对子。
    cards = [normalize_card(card) for card in action]  # 将动作牌面统一成内部点数。
    if not cards or any(card is None for card in cards):  # 空动作和未知点数不能用于主动出牌。
        return False  # 拒绝不完整或无法识别的牌型。
    counts = Counter(cards)  # 统计动作中各点数的张数。
    groups = sorted(counts.values())  # 获取与具体点数无关的牌型数量结构。
    if len(cards) == 1:  # 任意一张牌都是合法单牌。
        return True  # 接受单牌主动动作。
    if len(cards) == 2 and (groups == [2] or set(cards) == {"X", "D"}):  # 检查对子或王炸。
        return True  # 接受普通对子和大小王组合。
    if len(cards) == 3 and groups == [3]:  # 检查三张相同点数。
        return True  # 接受不带牌的三张。
    if len(cards) == 4 and (groups == [1, 3] or groups == [4]):  # 检查三带一或四张炸弹。
        return True  # 接受三带一和普通炸弹。
    if len(cards) == 5 and groups in ([2, 3], [5]):  # 检查三带二或技能牌形成的五炸。
        return True  # 接受三张带一个对子以及高于王炸的五张同点数。
    if len(cards) >= 5 and groups == [1] * len(cards):  # 检查所有点数只出现一次的顺子候选。
        sequence = _longest_consecutive_group(cards, len(cards))  # 验证全部点数是否组成无二无王的连续序列。
        return len(sequence) == len(cards)  # 只有整手连续时接受顺子。
    if len(cards) >= 6 and len(cards) % 2 == 0 and groups == [2] * (len(cards) // 2):  # 检查三组以上连对候选。
        sequence = _longest_consecutive_group(counts.keys(), len(counts))  # 验证所有对子点数是否连续。
        return len(sequence) == len(counts)  # 只有整手连续时接受连对。
    if len(cards) >= 6 and len(cards) % 3 == 0 and groups == [3] * (len(cards) // 3):  # 检查不带翅膀的连续三张。
        sequence = _longest_consecutive_group(counts.keys(), len(counts))  # 验证所有三张点数是否连续。
        return len(sequence) == len(counts)  # 只有整手连续时接受纯飞机。
    return False  # 其他复杂牌型交给训练模型而不由兜底策略猜测。


def choose_lead_action(hand_cards):  # 在没有训练模型时选择确定且合法的主动出牌组合。
    wildcard_finish = choose_terminal_wildcard_action(hand_cards, [])  # 优先检查无需模型的万能牌两张终局。
    if wildcard_finish:  # 两张牌可以组成对子时不允许拆成两个单牌。
        return wildcard_finish  # 一次选中万能牌和自然单牌直接出完。
    hand = [normalize_card(card) for card in hand_cards]  # 统一当前完整手牌点数。
    hand = [card for card in hand if card is not None]  # 丢弃调用方意外传入的未知点数。
    if not hand:  # 没有有效手牌时无法主动出牌。
        return []  # 返回空动作让任务安全停止点击。
    if is_basic_legal_lead(hand):  # 当前全部手牌本身就是一手合法牌型时优先结束牌局。
        return list(hand)  # 一次打出完整手牌而不是拆成多手。
    counts = Counter(hand)  # 统计每种点数在当前手牌中的数量。
    singleton_sequence = _longest_consecutive_group([rank for rank, count in counts.items() if count == 1], 5)  # 查找无需拆对子或三张的最长顺子。
    if singleton_sequence:  # 找到五张以上自然顺子时优先减少手牌数量。
        return singleton_sequence  # 返回起点最低的最长自然顺子。
    pair_sequence = _longest_consecutive_group([rank for rank, count in counts.items() if count == 2], 3)  # 查找无需拆三张或炸弹的最长连对。
    if pair_sequence:  # 找到三组以上自然连对时优先整组打出。
        return [rank for rank in pair_sequence for _ in range(2)]  # 将连续点数展开成合法连对动作。
    triples = [rank for rank in CARD_ORDER if counts[rank] == 3]  # 按从小到大顺序查找自然三张。
    if triples:  # 手牌包含不需要拆炸弹的三张时尝试带牌。
        triple = triples[0]  # 选择最低三张以保留高点数控制牌。
        pairs = [rank for rank in CARD_ORDER if rank != triple and counts[rank] == 2]  # 查找可作为三带二的自然对子。
        if pairs:  # 有自然对子时优先组成三带二减少五张手牌。
            return [triple] * 3 + [pairs[0]] * 2  # 返回最低三张带最低对子。
        singles = [rank for rank in CARD_ORDER if rank != triple and counts[rank] == 1]  # 查找可作为三带一的自然单牌。
        if singles:  # 有自然单牌时组成三带一避免拆其他组合。
            return [triple] * 3 + [singles[0]]  # 返回最低三张带最低单牌。
        return [triple] * 3  # 没有合适带牌时直接打出最低三张。
    pairs = [rank for rank in CARD_ORDER if counts[rank] == 2]  # 按从小到大顺序查找自然对子。
    if pairs:  # 没有更高效组合时打出最低对子。
        return [pairs[0], pairs[0]]  # 返回两张相同点数组成合法对子。
    singles = [rank for rank in CARD_ORDER if counts[rank] == 1]  # 按从小到大顺序查找自然单牌。
    if singles:  # 保留炸弹并优先清理最低孤张。
        return [singles[0]]  # 返回最低自然单牌而不是固定点击屏幕第一张。
    rocket = [rank for rank in ("X", "D") if counts[rank] == 1]  # 检查只剩大小王组合的异常边界。
    if len(rocket) == 2:  # 大小王同时存在时可以合法组成王炸。
        return rocket  # 返回王炸避免无法主动出牌。
    bombs = [rank for rank in CARD_ORDER if counts[rank] == 4]  # 最后查找不得不打出的普通炸弹。
    return [bombs[0]] * 4 if bombs else [min(hand, key=CARD_ORDER.index)]  # 仅在没有其他牌型时打最低炸弹或最低可识别牌。


def choose_skill_interaction_action(hero, hand_cards, option_cards, last_action_type=None, *, pending_skill=None, skill_uses=None, marks=None):  # 使用统一技能策略为六名已验证交互英雄选择完整二级动作。
    hero = normalize_registered_hero_name(hero)  # 统一英雄别名，确保注册表规则编号稳定命中。
    hand = sorted((card for card in (normalize_card(value) for value in hand_cards) if card is not None), key=CARD_ORDER.index)  # 规范化并稳定排列完整手牌。
    options = sorted((card for card in (normalize_card(value) for value in option_cards) if card is not None), key=CARD_ORDER.index)  # 规范化桌面技能选项。
    skill_name = pending_skill or {"夏侯惇": "刚烈", "关羽": "武圣", "徐盛": "疑城", "诸葛均": "耕读", "凌统": "勇进", "卢植": "儒宗"}.get(hero)  # 将当前已验证 UI 映射到注册技能。
    if hero not in HERO_REGISTRY or not skill_name or not hand:  # 缺少英雄、技能或完整手牌时不能构造守恒选择。
        return None, [], "当前界面没有可验证的语义选项"  # 保持安全暂停，不执行猜测点击。

    card_ids = [f"ui{index}" for index in range(len(hand))]  # 为屏幕实体牌生成本次纯决策内稳定标识。
    pending_options = []  # 保存与权威模拟器相同格式的二级合法选项。
    effect = ""  # 标记当前交互结算类型供策略投影。
    optional = False  # 进入强制选牌阶段后默认必须完成，只有明确取消按钮的取牌技能允许跳过。
    if hero == "夏侯惇" and len(hand) >= 3:  # 刚烈应枚举全部三张弃牌组合，而不是固定弃最小牌。
        effect = "ganglie"
        seen = set()
        for indexes in combinations(range(len(hand)), 3):
            ranks = tuple(hand[index] for index in indexes)
            if ranks in seen:  # 相同点数组合对应同一屏幕语义，避免重复评分。
                continue
            seen.add(ranks)
            pending_options.append({"card_ids": [card_ids[index] for index in indexes], "ranks": list(ranks), "discard_ranks": list(ranks)})
    elif hero == "关羽" and options:  # 武圣枚举顺子中每个公开可取点数以及不发动。
        effect = "gain_rank"
        optional = True
        pending_options = [{"rank": rank, "ranks": [rank]} for rank in dict.fromkeys(options)]
    elif hero == "徐盛":  # 疑城获得两张后的弃牌必须枚举当前完整手牌。
        effect = "discard_one"
        pending_options = [{"card_ids": [card_ids[index]], "rank": rank, "ranks": [rank]} for index, rank in enumerate(hand)]
    elif hero == "诸葛均" and options:  # 耕读开局复制枚举三张公开底牌并允许取消。
        effect = "copy_bottom"
        optional = True
        pending_options = [{"rank": rank, "ranks": [rank]} for rank in dict.fromkeys(options)]
    elif hero == "诸葛均":  # 打出耕读复制牌后的弃牌属于强制完整结算。
        effect = "discard_one"
        pending_options = [{"card_ids": [card_ids[index]], "rank": rank, "ranks": [rank]} for index, rank in enumerate(hand)]
    elif hero == "凌统":  # 勇进按上一手真实牌型枚举相反数量组。
        effect = "discard_group"
        counts = Counter(hand)
        if last_action_type == "solo":
            for rank in CARD_ORDER:
                indexes = [index for index, value in enumerate(hand) if value == rank][:2]
                if len(indexes) == 2:
                    pending_options.append({"card_ids": [card_ids[index] for index in indexes], "ranks": [rank, rank]})
        elif last_action_type == "pair":
            for index, rank in enumerate(hand):
                if counts[rank] == 1:
                    pending_options.append({"card_ids": [card_ids[index]], "ranks": [rank]})
    elif hero == "卢植":  # 儒宗同时枚举单牌补对和对子减一两类合法转换。
        effect = "convert_group"
        counts = Counter(hand)
        for rank in CARD_ORDER[:-2]:
            indexes = [index for index, value in enumerate(hand) if value == rank]
            if counts[rank] == 1:
                pending_options.append({"operation": "solo_to_pair", "rank": rank, "ranks": [rank], "card_ids": [card_ids[indexes[0]]]})
            elif counts[rank] == 2:
                pending_options.append({"operation": "pair_to_solo", "rank": rank, "ranks": [rank], "card_ids": [card_ids[indexes[0]]]})
    if not pending_options:  # 当前牌面没有满足技能形状的完整合法选择。
        return None, [], "统一技能策略未找到合法二级选项"  # 保持暂停并等待更完整画面。

    context = HeroDecisionContext(  # 仅使用当前公开且已识别的信息构造共享纯逻辑上下文。
        hand=tuple(hand),
        hero=hero,
        hand_card_ids=tuple(card_ids),
        skill_uses=dict(skill_uses or {}),
        marks=dict(marks or {}),
        extra={"pending_interaction": {"actor": "landlord_down", "skill": skill_name, "effect": effect, "options": pending_options, "optional": optional}},
    )
    choice = select_skill_choice(context, skill_name, route_evaluator=estimate_route_turns)  # 按完整技能结算后的预计与最坏牌路选择。
    if choice is None:  # 次数上限或规则窗口不允许继续发动。
        return None, [], "统一技能策略判定当前技能不可用"  # 不点击任何牌。
    projection = apply_skill_choice(context, choice, route_evaluator=estimate_route_turns)  # 生成与离线模拟一致的日志理由。
    if choice.skip:  # 可取消技能只有完整结算不优于普通路线时才跳过。
        return "skip", [], f"统一技能策略选择不发动；{projection.reason}"  # 由调用方仅在可靠识别取消按钮时执行。
    ui_ranks = list(choice.ranks)  # 默认屏幕选择与纯逻辑点数一一对应。
    if hero == "卢植" and choice.parameters.get("operation") == "pair_to_solo" and ui_ranks:  # 儒宗减对界面需要选中完整对子。
        ui_ranks = [ui_ranks[0], ui_ranks[0]]  # 只改变屏幕点击表示，不改变技能投影的单张减少语义。
    source = "options" if effect in {"gain_rank", "copy_bottom"} else "hand"  # 点击来源只消费策略选择，不参与再次决策。
    return source, ui_ranks, f"统一技能策略：{projection.reason}"  # 返回可解释且数量明确的最终动作。


def is_active_skill_confirm_button(frame, box):  # 使用按钮周围的金色像素判断二级技能确认按钮是否已经可用。
    if frame is None or frame.size == 0 or box is None:  # 空画面或空按钮不能作为可提交依据。
        return False  # 拒绝在未验证状态下点击。
    horizontal_padding = max(8, int(box.width * 0.75))  # 将 OCR 文字框向左右扩展到按钮填充区域。
    vertical_padding = max(8, int(box.height * 0.65))  # 将 OCR 文字框向上下扩展到按钮背景区域。
    left = max(0, int(box.x) - horizontal_padding)  # 限制扩展后的左边界不越过画面。
    top = max(0, int(box.y) - vertical_padding)  # 限制扩展后的上边界不越过画面。
    right = min(frame.shape[1], int(box.x + box.width) + horizontal_padding)  # 限制扩展后的右边界不越过画面。
    bottom = min(frame.shape[0], int(box.y + box.height) + vertical_padding)  # 限制扩展后的下边界不越过画面。
    if right <= left or bottom <= top:  # 无效裁切范围无法验证按钮状态。
        return False  # 返回不可提交。
    hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)  # 转换到 HSV 以适应亮度动画变化。
    gold = cv2.inRange(hsv, (8, 55, 70), (45, 255, 255))  # 提取选中后按钮常见的黄色和金色填充。
    return cv2.countNonZero(gold) / float(gold.size) >= 0.08  # 金色占比不足时视为灰色禁用按钮。


def is_skill_selection_auto_resolved(hero, before_cards, selected_cards, after_cards):  # 根据完整手牌差值判断无确认按钮的技能是否已经自动提交。
    if hero not in {"凌统", "卢植"}:  # 当前只对能够用严格手牌差值验证的勇进和儒宗检查自动提交。
        return False  # 其他英雄仍必须验证各自确认按钮。
    before = [card for card in (normalize_card(value) for value in before_cards) if card is not None]  # 规范化选择前完整手牌。
    selected = [card for card in (normalize_card(value) for value in selected_cards) if card is not None]  # 规范化计划弃置牌组。
    after = [card for card in (normalize_card(value) for value in after_cards) if card is not None]  # 规范化点击动画后的完整手牌。
    if hero == "卢植":  # 儒宗可能在点击目标后立即完成单牌与对子的数量转换。
        if not selected or len(set(selected)) != 1:  # 目标必须是同一点数的一张单牌或完整对子。
            return False  # 拒绝部分选择和混合点数。
        rank = selected[0]  # 读取被转换的唯一点数。
        before_counts = Counter(before)  # 保存转换前完整手牌数量。
        after_counts = Counter(after)  # 保存转换后完整手牌数量。
        expected = Counter(before_counts)  # 构造严格预期结果。
        if len(selected) == 1 and before_counts[rank] == 1:  # 单牌补成对子会增加一张同点数牌。
            expected[rank] += 1  # 只允许目标点数增加一张。
        elif len(selected) == 2 and before_counts[rank] == 2:  # 对子变成单牌会减少一张同点数牌。
            expected[rank] -= 1  # 保留一张并删除零值前无需额外处理。
        else:  # 选择数量与当前自然单牌或对子结构不符。
            return False  # 禁止将 OCR 抖动当成技能完成。
        return expected == after_counts  # 其他所有点数必须完全不变才能确认儒宗已经自动结算。
    if len(before) != len(after) + len(selected):  # 勇进张数必须恰好减少技能选择数量。
        return False  # 部分 OCR 或其他技能变化不能误判为提交成功。
    expected = Counter(before)  # 复制选择前点数数量用于精确扣除。
    expected.subtract(selected)  # 扣除本次语义动作计划弃置的牌。
    if any(count < 0 for count in expected.values()):  # 计划弃牌超过当前实际持有数量表示识别或映射错误。
        return False  # 禁止把异常差值标记为成功。
    expected = Counter({card: count for card, count in expected.items() if count > 0})  # 清除扣除后的零数量点数。
    return expected == Counter(after)  # 只有剩余每个点数完全一致才确认游戏已经自动提交。


def detect_hand_slots(hand_image, estimated_card_width=None, allow_center_fallback=True, expected_count=None):  # 根据卡牌竖边识别一到四十张重叠卡牌的位置，并允许残局按牌账本张数消除王字内竖边歧义。
    height, width = hand_image.shape[:2]  # 读取当前分辨率下的手牌区域大小。
    if width < 40 or height < 40:  # 过滤无法包含有效卡牌的异常小区域。
        return []  # 返回空列表交由安全识牌失败分支处理。
    card_width = max(40, int(round(estimated_card_width if estimated_card_width is not None else width * 208 / 1621)))  # 手牌按区域比例估算，桌面牌可传入独立缩放宽度。
    scan_height = min(height, max(80, int(round(height * 0.55))))  # 只扫描包含点数和竖边的牌面上半部分。
    gray = cv2.cvtColor(hand_image[:scan_height], cv2.COLOR_BGR2GRAY)  # 将手牌上半部分转换为灰度图。
    gradient = np.abs(np.diff(gray.astype(np.int16), axis=1)).sum(axis=0)  # 累加每个横坐标的竖向边缘强度。
    peak_limit = max(3500.0, float(np.max(gradient)) * 0.38)  # 使用自适应阈值过滤牌面文字与花色产生的弱竖边。
    raw_peaks = [index for index in range(1, gradient.size - 1) if gradient[index] >= peak_limit and gradient[index] >= gradient[index - 1] and gradient[index] >= gradient[index + 1]]  # 收集所有可能的卡牌边框峰值。
    ranked_peaks = sorted(raw_peaks, key=lambda index: float(gradient[index]), reverse=True)  # 按边缘强度排序以便合并粗边框的相邻峰值。
    selected_peaks = []  # 保存经过横向非极大值抑制后的独立竖边。
    for peak in ranked_peaks:  # 依次检查由强到弱的候选竖边。
        if all(abs(peak - existing) > 10 for existing in selected_peaks):  # 同一粗边框十一像素内只保留最强峰值。
            selected_peaks.append(peak)  # 记录新的独立卡牌竖边。
    peaks = sorted(selected_peaks)  # 恢复从左到右顺序用于寻找等距卡牌起点链。
    best_chain = []  # 初始化最长等距竖边链。
    best_step = card_width  # 初始化单牌可见宽度所需的默认间距。
    candidate_chains = []  # 保留全部等距链，供常规识别失败后的已知张数残局消歧。
    for first_index, first in enumerate(peaks):  # 尝试每个强竖边作为首张牌左边缘。
        for second in peaks[first_index + 1:]:  # 使用后续竖边估算相邻重叠牌间距。
            step = second - first  # 计算当前假设的手牌重叠间距。
            if step < 24 or step > 110:  # 允许技能加牌后更密集的牌框间距，同时排除绝大多数牌内笔画。
                continue  # 继续评估下一个可能的相邻卡牌竖边。
            chain = [first, second]  # 用前两个等距竖边建立候选卡牌链。
            tolerance = max(7, int(round(step * 0.10)))  # 允许缩放、圆角和抗锯齿造成的小量位置误差。
            while len(chain) < MAX_SKILL_HAND_CARDS:  # 技能手牌可以超过二十张，但仍使用硬上限防止异常峰值无限扩展。
                expected = chain[-1] + step  # 预测下一张重叠牌的左边缘位置。
                matches = [peak for peak in peaks if peak > chain[-1] and abs(peak - expected) <= tolerance]  # 查找预测位置附近的实际强竖边。
                if not matches:  # 当前等距链已经到达最后一张牌。
                    break  # 结束该候选链的扩展。
                chain.append(min(matches, key=lambda peak: abs(peak - expected)))  # 加入最接近预测位置的竖边。
            candidate_chains.append((chain, step))  # 保存当前链，不能只保留被王字内部笔画拉长的最多峰值方案。
            if len(chain) > len(best_chain) or len(chain) == len(best_chain) and sum(float(gradient[peak]) for peak in chain) > sum(float(gradient[peak]) for peak in best_chain):  # 优先保留张数最多且总体边缘更强的链。
                best_chain = chain  # 保存当前最可靠的全部卡牌起点。
                best_step = step  # 保存实际重叠间距供裁切牌面使用。
    if isinstance(expected_count, int) and expected_count >= 2:  # 只有牌账本提供明确残局张数时才覆盖常规最长链选择。
        exact_chains = [(chain, step) for chain, step in candidate_chains if len(chain) == expected_count]  # 禁止截取长链，以免把技能新增牌静默隐藏。
        if exact_chains:  # 找到与真实剩余张数一致的完整等距链。
            def exact_chain_score(item):  # 使用末张完整右边框优先排除 JOKER 文字内部的假竖边。
                chain, step = item  # 读取当前候选链及其相邻牌距。
                expected_right = chain[-1] + card_width  # 最后一张牌应在一个完整牌宽后出现右边框。
                right_error = min((abs(peak - expected_right) for peak in peaks), default=card_width)  # 计算最近强竖边与理论右边框的偏差。
                strength = sum(float(gradient[peak]) for peak in chain)  # 同几何误差下优先保留真实边框强度更高的链。
                return right_error, -step, -strength  # 小残局的真实重叠间距通常大于王字内部笔画间距。

            best_chain, best_step = min(exact_chains, key=exact_chain_score)  # 选择与牌账本、末牌右边框共同一致的三张残局链。
    if len(best_chain) >= 2:  # 两个以上等距左边缘已经足以确认少量居中手牌。
        visible_width = min(card_width, max(28, best_step + 4))  # 密集牌局缩窄 OCR 框，避免把相邻新牌点数一起读入。
        return [(start, visible_width) for start in best_chain]  # 返回检测到的实际卡牌位置而不再依赖区域左右边界。
    full_card_pairs = [(left, right) for left in peaks for right in peaks if int(card_width * 0.75) <= right - left <= int(card_width * 1.25)]  # 查找单张完整卡牌的左右边缘组合。
    if full_card_pairs:  # 少量牌只剩一张时使用最强完整牌框定位。
        left, _ = max(full_card_pairs, key=lambda pair: float(gradient[pair[0]]) + float(gradient[pair[1]]))  # 选择总边缘强度最高的完整牌框。
        return [(left, card_width)]  # 返回居中单牌的真实横坐标。
    if not allow_center_fallback:  # 桌面空白区域不能凭中心位置猜测存在一张牌。
        return []  # 返回空结果，阻止把背景文字误识别成桌面动作。
    center_start = max(0, (width - card_width) // 2)  # 动画帧缺少稳定边缘时假设唯一手牌在区域中央显示。
    return [(center_start, card_width)]  # 返回一张居中卡牌区域供独立 OCR。


def _is_wildcard_card(card_image):  # 根据万能牌特有的橙色点数与图标区分同点数普通牌。
    if card_image is None or card_image.size == 0:  # 缺少有效彩色牌面时不能判断技能牌标记。
        return False  # 保持普通 OCR 点数结果。
    hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)  # 使用 HSV 分离橙色万能牌与红黑普通花色。
    saturation = hsv[:, :, 1]  # 读取饱和度过滤白色卡面背景。
    value = hsv[:, :, 2]  # 读取亮度排除阴影和黑色点数。
    colored = (saturation >= 80) & (value >= 80)  # 统计当前牌面所有明显彩色像素。
    orange = (hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 28) & (saturation >= 100) & (value >= 100)  # 提取截图中橙色万能点数和旋涡图标。
    colored_count = int(np.count_nonzero(colored))  # 计算彩色内容总量供比例判断。
    orange_count = int(np.count_nonzero(orange))  # 计算万能牌特征像素数量。
    return orange_count >= 80 and orange_count / float(max(1, colored_count)) >= 0.70  # 同时要求绝对数量和占比，避免把红桃边缘误判成万能牌。


def parse_card_text(text, card_image=None, detect_wildcard=False):  # 从单张卡牌 OCR 文本中提取点数并区分大小王和万能牌。
    cleaned = re.sub(r"[^0-9A-Za-z]", "", str(text)).upper()  # 去掉花色和 OCR 附带的符号噪声。
    if detect_wildcard and cleaned and _is_wildcard_card(card_image):  # 手牌仅剩两张时识别橙色技能牌实体。
        return "W"  # 使用模拟器统一的万能牌编码，保留其可替代点数语义。
    if cleaned.startswith("JOK"):  # 将竖排 JOKER 识别结果归类为王。
        if card_image is None:  # 缺少颜色画面时无法可靠区分大小王。
            return "X"  # 使用小王作为无颜色信息时的保守默认值。
        blue, green, red = cv2.split(card_image)  # 拆分牌面颜色通道以判断王字颜色。
        red_pixels = (red.astype(np.int16) > blue.astype(np.int16) + 35) & (red.astype(np.int16) > green.astype(np.int16) + 25)  # 找出明显红色文字像素。
        return "D" if int(np.count_nonzero(red_pixels)) > 30 else "X"  # 红色王判为大王，黑色王判为小王。
    if cleaned.startswith("10"):  # 兼容牌面显示数字 10 的样式。
        return "T"  # 将数字十转换为模型内部 T。
    if cleaned and cleaned[0] in "23456789TJQKA":  # 读取普通牌 OCR 结果的首个有效点数字符。
        return normalize_card(cleaned[0])  # 使用公共归一化逻辑返回内部点数。
    return None  # 无法确定点数时拒绝猜测。


class AiCardPlayingTask(MaterialCollectorTask):  # 定义由训练模型决策的自动打牌任务。
    def __init__(self, *args, **kwargs):  # 初始化任务名称和 AI 专用配置。
        super().__init__(*args, **kwargs)  # 先加载现有自动导航和模板配置。
        self.name = "AI 自动打牌"  # 在 ok-script 任务列表显示独立入口。
        self.description = "只在 RLCard 合法动作中使用规则和英雄策略选择出牌。"
        self.default_config.update({
            "AI Adapter": "ok_tasks/RlCardRuleModel.py",
            "AI Position": "auto",  # 默认根据手牌张数推断地主，农民位置可在界面手动指定。
            "AI Fallback Hint": False,  # AI 任务始终自行决策，不再调用游戏提示；保留键仅兼容旧配置文件。
            "Card OCR Threshold": 0.25,  # 设置牌面 OCR 的最低置信度。
            "Auto Select Hero": True,  # 默认只从账号已拥有且已实现的武将中自动选择。
            "Auto Use Hero Skills": True,  # 默认处理已有可靠确认素材覆盖的英雄技能。
            "Hero Exploration Games": 10,  # 每名候选武将先完成十局冷启动统计。
            "Hero Statistics File": "data/hero_stats.json",  # 将逐武将及身份胜率保存到项目数据目录。
            "Policy State File": "data/card_ai/policy_state.json",  # 保存三套策略累计表现和稳定版本指针。
            "Policy Minimum Games": 20,  # 候选和当前策略至少各完成二十局才允许比较晋升。
            "Preferred Hero": "",  # 可选指定优先英雄，候选中不存在时仍按统计选择。
            "Search Budget Ms": 300,  # 困难局面在合法动作上执行信息集搜索的默认时间预算。
        })  # 完成 AI 默认配置注册。
        self.config_description.update({  # 为 GUI 配置补充中文说明。
            "AI Adapter": "固定使用 RLCard 规则策略适配器。",
            "AI Position": "auto、landlord、landlord_up 或 landlord_down。",  # 说明三种 DouZero 身份位置。
            "AI Fallback Hint": "兼容旧配置；AI 自动打牌任务不会调用游戏提示。",  # 明确完全由本项目规则或模型选择动作。
            "Card OCR Threshold": "手牌和桌面牌文字识别置信度。",  # 说明牌面 OCR 阈值用途。
            "Auto Select Hero": "识别三个候选武将，优先在全被动技能角色中按统计选择；没有被动角色时再选择其他已拥有角色。",  # 说明被动武将优先及无候选时的安全回退。
            "Auto Use Hero Skills": "自动确认已有可靠素材覆盖的技能；缺少二级交互素材时保守等待。",  # 说明技能自动化安全边界。
            "Hero Exploration Games": "每名武将在按平滑胜率选择前至少采集的有效结算局数。",  # 说明冷启动探索规则。
            "Hero Statistics File": "逐武将总体、地主和农民胜负统计 JSON 路径。",  # 说明统计文件内容。
            "Policy State File": "保守、均衡和技能优先策略的统计及回滚状态路径。",  # 说明策略状态文件用途。
            "Policy Minimum Games": "候选策略自动晋升前双方至少需要完成的有效对局数。",  # 说明安全验证样本门槛。
            "Preferred Hero": "候选列表中优先选择的已拥有英雄，留空时按探索胜率选择。",  # 说明一键预设的英雄偏好。
            "Search Budget Ms": "困难残局的规则搜索预算，范围0到1200毫秒。",
        })  # 完成 AI 配置说明注册。

    def _prepare_session(self):  # 初始化采集状态和训练模型会话。
        super()._prepare_session()  # 保留原任务的局数、素材和状态初始化。
        self.ai_model = None  # 默认将模型标记为不可用。
        self.ai_history = []  # 重置本次牌局的可见动作历史。
        self.expected_hand_count = None  # 初始化技能回收检测所需的预期手牌数量。
        self.hand_change_pending = False  # 初始化尚无需要多帧确认的技能手牌变化。
        self.last_opponent_card_counts = None  # 初始化尚无可比较的对手牌数基线。
        self.opponent_skill_card_estimates = [0, 0]  # 初始化左右座位均未确认技能增牌。
        self.resolved_ai_position = None  # 锁定本局首次确认的身份避免技能加牌后误判地主。
        self.last_ai_error = None  # 重置最近一次模型或识牌错误。
        self.ai_unavailable_reported = False  # 重置模型不可用警告的单次显示锁。
        self.current_hero = None  # 初始化本局尚未识别或选择武将。
        self.hero_last_action_type = None  # 初始化张飞等连续牌型技能使用的上一手类型。
        self.hero_result_recorded = False  # 防止同一结算画面重复累计武将胜负。
        self.hero_statistics = HeroStatistics(self.config.get("Hero Statistics File", "data/hero_stats.json"))  # 加载可恢复的逐武将统计。
        self.policy_optimizer = PolicyOptimizer(self.config.get("Policy State File", "data/card_ai/policy_state.json"), self.config.get("Policy Minimum Games", 20))  # 加载可回滚的整局策略管理器。
        self.current_policy_id = self.policy_optimizer.choose_policy()  # 为断点恢复或第一局选择稳定策略或受限候选。
        self.card_ledger = CardLedger()  # 创建允许技能复制和回收重复牌的本局牌账本。
        self.hero_runtime_state = HeroRuntimeState()  # 创建英雄技能跨回合状态对象。
        self.info_set("Policy", self.current_policy_id)  # 在任务面板显示当前整局策略编号。
        try:
            adapter_path = Path("ok_tasks/RlCardRuleModel.py")
            self.ai_model = TrainedModelAdapter(adapter_path).load()
            source_name = adapter_path.name
            self.log_info(f"AI 算法已加载: {source_name}", notify=True)  # 在界面明确显示当前使用的规则算法或权重。
            self.info_set("Decision Source", source_name)  # 在任务状态中持续显示实际决策来源。
        except AiModelError as error:  # 处理高容量适配器配置路径或接口错误。
            self.last_ai_error = str(error)  # 保存错误供界面状态和测试读取。
            try:  # 高容量入口失败时仍使用项目自己的稳定合法动作规则，不依赖游戏提示。
                fallback_path = Path("ok_tasks/RlCardRuleModel.py")  # 指向始终随项目发布的 RLCard 完整动作空间规则模型。
                self.ai_model = TrainedModelAdapter(fallback_path, None).load()  # 加载只会返回合法动作的本地规则模型。
                self.log_warning(f"高容量 AI 未启用，已切换本地稳定规则: {error}", notify=True)  # 告知用户当前仍是自主选牌。
                self.info_set("Decision Source", fallback_path.name)  # 在状态面板显示真实决策来源。
            except AiModelError as fallback_error:  # 两层模型都无法加载时停止猜测点击。
                self.last_ai_error = f"{error}; {fallback_error}"  # 合并两个可诊断错误。
                self.log_warning(f"AI 规则加载失败，本轮不会调用游戏提示: {fallback_error}", notify=True)  # 明确遵守完全自主决策约束。
                self.info_set("Decision Source", "AI 未加载")  # 显示当前不会自动猜牌。

    def _play_with_hint(self, frame):  # 将普通跟牌回合改由训练模型处理。
        self._play_with_ai(frame)  # 完全由 AI 识牌和合法动作枚举决策，失败时保留现场或安全不出，绝不点击游戏提示。

    def _handle_state(self, state, matched_box, frame):  # 在原状态机动作前维护每一局独立的 AI 状态。
        self.current_play_state = state  # 保存当前是主动出牌还是跟牌，禁止桌面 OCR 失败时误按主动回合枚举动作。
        if state == "bidding":  # AI任务根据完整手牌结构自主选择不叫或一至三分。
            if not self.config.get("Auto Navigate", True) or not self._action_ready(state):  # 尊重导航开关并防止同一叫分画面重复点击。
                return  # 等待下一帧或用户手动操作。
            self._handle_ai_bidding(frame)  # 执行完整识牌、牌力计算和可靠按钮点击。
            return  # 禁止父任务继续固定点击一分。
        if state == "skill_prompt" and not self.config.get("Auto Use Hero Skills", True):  # 用户关闭英雄技能自动化时统一拒绝确认弹窗。
            self._click_skill_prompt_action("Cancel the use of the skill", after_sleep=1.0)  # 优先使用取消模板，漂移时使用用户标注的完整技能询问区域。
            return  # 不再交给父类的 Confirm Skills 配置重复处理。
        if state == "hero_select":  # 新一局进入选将界面时旧牌局历史已经全部失效。
            self.ai_history = []  # 清空上一局可见动作历史。
            self.expected_hand_count = None  # 清空上一局预期手牌数量。
            self.hand_change_pending = False  # 清除上一局遗留的技能动画确认标记。
            self.last_opponent_card_counts = None  # 清除上一局对手牌数基线。
            self.opponent_skill_card_estimates = [0, 0]  # 清除上一局敌方和队友技能牌估计。
            self.resolved_ai_position = None  # 允许新一局重新判断地主或农民身份。
            self.current_hero = None  # 清除上一局武将，等待本轮候选 OCR。
            self.hero_last_action_type = None  # 清除上一局连续牌型状态。
            self.hero_result_recorded = False  # 允许新一局真实结算写入统计。
            self.current_policy_id = self.policy_optimizer.choose_policy()  # 为整局固定选择稳定策略或百分之二十探索候选。
            self.card_ledger = CardLedger()  # 为新牌局创建独立牌账本。
            self.hero_runtime_state = HeroRuntimeState()  # 为新牌局创建独立英雄状态。
            self.info_set("Policy", self.current_policy_id)  # 在状态面板显示本局策略编号。
        elif state in {"result_win", "result_loss"} and not self.hero_result_recorded:  # 在父状态机锁存结算前记录武将结果。
            self._record_hero_result(state == "result_win")  # 将胜负同时写入总体和身份分项。
            self.policy_optimizer.record_game(self.current_policy_id, state == "result_win", getattr(self, "game_submit_failures", 0))  # 将本局胜负和操作失败归因到固定策略。
            if self.ai_model is not None:  # 高容量模型加载成功时记录候选小流量真实结果。
                self.ai_model.record_game(state == "result_win", getattr(self, "game_submit_failures", 0))  # 只有适配器实现回调时才更新候选统计。
            self.hero_result_recorded = True  # 锁定本局已经写入统计。
        elif self.in_match and self.current_hero is None:  # 选将 OCR 失败时从牌局左下角补识别当前武将。
            self._ensure_current_hero(frame)  # 只在尚未识别时执行一次小区域 OCR。
        super()._handle_state(state, matched_box, frame)  # 继续执行原有导航、选将、叫分和打牌状态机。

    def _handle_ai_bidding(self, frame):  # 根据开局手牌结构选择不叫、一分、二分或三分。
        self.in_match = True  # 叫分界面出现后确认已经进入正式牌局。
        recorder = getattr(self, "run_recorder", None)  # 读取逐局记录器支持中途恢复。
        if recorder is not None:  # 正常会话确保本局日志已经建立。
            recorder.ensure_game(hero=getattr(self, "current_hero", None), position=None, policy_id=getattr(self, "current_policy_id", "balanced"))  # 叫分前身份尚未确定。
        hand_boxes, stable_frame = self._recognize_stable_hand(frame)  # 叫分必须使用完整且稳定的十七张开局手牌。
        if not hand_boxes:  # 任意牌漏识别时不能用残缺手牌高估牌力。
            self._record_event("bidding_deferred", reason="开局手牌未完整稳定识别")  # 写入诊断日志等待下一轮重试。
            return False  # 不执行猜测点击。
        feature_names = {1: "1 point", 2: "two points", 3: "three points"}  # 映射语义分数到已有模板名称。
        available_boxes = {}  # 保存当前仍为金色可点击状态的叫分按钮。
        for value, feature_name in feature_names.items():  # 逐一检查一至三分按钮是否仍可用。
            if self.feature_exists(feature_name):  # 跳过旧素材缺失的可选分数。
                box = self.find_one(feature_name, threshold=self.config.get("Template Threshold", 0.8))  # 模板颜色同时区分可用和灰色禁用按钮。
                if box is not None:  # 当前分数仍可点击。
                    available_boxes[value] = box  # 保存真实按钮框供最终点击。
        bid, evaluation = choose_bid([card for card, _ in hand_boxes], available_boxes.keys())  # 结合自身牌力和对手已叫分情况选择最终动作。
        clicked = False  # 初始化尚未成功提交叫分。
        if bid > 0 and bid in available_boxes:  # 目标分数存在可靠模板框。
            self.click_box(available_boxes[bid], after_sleep=1.0)  # 点击所选一至三分按钮并等待状态变化。
            clicked = True  # 标记已执行叫分。
        elif bid == 0:  # 弱牌或对手叫分超过自身承受范围时选择不叫。
            height, width = stable_frame.shape[:2]  # 按当前游戏分辨率创建不叫文字识别区域。
            no_bid_region = Box(int(width * 0.14), int(height * 0.50), int(width * 0.18), int(height * 0.15), name="no_bid_region")  # 覆盖截图中左侧不叫按钮并排除倒计时。
            no_bid_boxes = self.ocr(box=no_bid_region, frame=stable_frame, threshold=0.20, log=False)  # 使用清晰中文按钮文字确认点击目标。
            no_bid_box = next((box for box in no_bid_boxes if re.search(r"不叫|不抢|放弃", str(box.name or ""))), None)  # 兼容不同叫地主文案。
            if no_bid_box is not None:  # 只有OCR明确确认不叫文字时才点击。
                self.click_box(no_bid_region, after_sleep=1.0)  # OCR只用于确认语义，点击稳定的完整按钮中心避免文字框坐标漂移。
                clicked = True  # 标记动作已经提交。
        self.info_set("Bid Decision", f"{bid}分 / {evaluation['score']:.2f}")  # 在状态面板显示本局叫分和牌力。
        self._record_event("bidding", clicked=clicked, hand_cards=[card for card, _ in hand_boxes], **evaluation)  # 保存手牌、结构评分、可用按钮和最终结果供胜率校准。
        if not clicked:  # 模板或不叫OCR未能可靠定位时保留现场重试。
            self._capture_state("bidding_button_unavailable", stable_frame, force=True)  # 保存按钮状态以补充不同叫分阶段素材。
            self.log_warning(f"叫分决策为{bid}分，但对应按钮未可靠识别，将等待重试。")  # 明确说明没有执行随机坐标点击。
        return clicked  # 返回本次叫分是否实际提交。

    def _finalize_session(self, status):  # 在任务结束后分析策略表现并生成包含优化结果的报告。
        optimizer = getattr(self, "policy_optimizer", None)  # 读取可能已经成功初始化的策略优化器。
        optimization = optimizer.optimize() if optimizer is not None else {"changed": False, "message": "策略优化器未初始化"}  # 每次结束都执行验证但不在样本不足时强制切换。
        recorder = getattr(self, "run_recorder", None)  # 读取父任务创建的持久运行记录器。
        if recorder is None:  # 记录器初始化失败时仅返回优化结果。
            return optimization  # 保留原异常并避免二次错误。
        if recorder.game_id is not None:  # 手动停止或异常退出时先封存当前未结算牌局。
            recorder.end_game(None, status="incomplete")  # 不完整牌局保留全部现场但不参与失败策略验证。
        policy_path = Path(self.config.get("Policy State File", "data/card_ai/policy_state.json"))  # 复用现有策略状态目录避免增加迁移配置项。
        learning_pipeline = StrategyLearningPipeline(self.config.get("Data Root", "data/card_ai/runs"), policy_path.with_name("strategy_library.json"), self.config.get("Policy Minimum Games", 20))  # 创建使用同一最少样本门槛的复盘流水线。
        learning = learning_pipeline.process(recorder.root, baseline_policy=optimization.get("active", "balanced"))  # 对全部完整历史牌局生成失败假设并执行相似局策略对照。
        optimization["learning"] = learning  # 将复盘摘要同时写入优化对象方便旧版报告读取。
        optimization["message"] = f"{optimization['message']}；{learning['message']}"  # 在任务面板给出完整闭环结果。
        summary = recorder.finalize(status=status, optimization=optimization, missing_resources=getattr(self, "missing_resources", []), learning=learning)  # 写入带策略优化和复盘入库结果的机器总结与中文报告。
        self.info_set("Optimization", optimization.get("message", "未执行"))  # 在任务状态面板显示晋升、回滚或样本不足信息。
        self.info_set("Run Report", str(recorder.root / "报告.txt"))  # 在任务状态面板显示中文报告路径。
        return summary  # 返回完整总结供测试和调用方检查。

    def _select_middle_hero(self):  # 识别三名候选并优先在全被动技能武将中按统计选择。
        if not self.config.get("Auto Select Hero", True):  # 用户关闭智能选将时保留原固定中间位行为。
            return super()._select_middle_hero()  # 调用已验证的中间英雄选择流程。
        frame = self.next_frame()  # 获取稳定的选将画面用于三个卡槽 OCR。
        candidates, slots = self._recognize_hero_candidates(frame)  # 首次识别当前三个候选武将。
        if not any(hero in PASSIVE_OWNED_HEROES for hero in candidates):  # 当前候选没有全被动技能角色时才消耗换将次数。
            candidates, slots = self._swap_until_passive_candidate(candidates, slots)  # 依次使用两个已标注换将按钮寻找被动角色。
        preferred = normalize_hero_name(self.config.get("Preferred Hero", ""))  # 读取并规范化用户一键预设的偏好英雄。
        chosen = choose_hero_candidate(candidates, self.hero_statistics, self.config.get("Hero Exploration Games", 10), preferred)  # 换将后再次按被动角色、局数和胜率选择。
        chosen_index = candidates.index(chosen) if chosen in candidates else 1  # 没有可靠候选时回退中间卡槽。
        chosen_slot = slots[chosen_index]  # 获取选中武将的实际卡槽区域。
        if chosen_slot is not None:  # 当前素材存在对应卡槽标注时执行精确点击。
            self.click_box(chosen_slot, after_sleep=0.4)  # 点击完整武将卡片完成选中。
        else:  # 极旧素材缺少卡槽时使用原中间坐标兜底。
            self.click_relative(0.50, 0.48, after_sleep=0.4)  # 点击选将画面中央武将。
        self.current_hero = chosen  # 保存可靠识别结果，回退中间位时等待牌局内补识别。
        self.hero_runtime_state.hero = chosen  # 同步统一英雄状态供牌账本和模型使用。
        if chosen is not None:  # 只有识别成功时展示具体英雄状态。
            self.info_set("Current Hero", chosen)  # 在任务面板显示本局使用武将。
            skill_mode = "全被动优先" if chosen in PASSIVE_OWNED_HEROES else "换将后无被动候选兜底"  # 记录本次选择是否命中被动英雄优先规则。
            self.log_info(f"自动选择武将: {chosen}（{skill_mode}）")  # 将选将结果和选择来源写入日志便于核对 OCR。
        selected_frame = self.next_frame()  # 获取武将被选中后的高亮画面。
        self._capture_state("hero_auto_selected", selected_frame, force=True)  # 保存自动选将结果供后续修正 OCR。
        self.click_relative(0.15, 0.82, after_sleep=0.4)  # 点击空白处关闭技能说明浮层。
        self._click_feature("select", after_sleep=1.5)  # 点击选定进入牌局。

    def _recognize_hero_candidates(self, frame):  # 从当前选将画面读取左中右三个候选及卡槽。
        candidates = []  # 保存按左中右顺序识别出的规范武将名。
        slots = []  # 保存与候选顺序一致的可点击卡槽。
        for index in (1, 2, 3):  # 依次读取三个未选中布局的武将卡片。
            feature_name = f"hero_slot_{index}_layout_a"  # 生成现有 COCO 标注中的卡槽名称。
            if not self.feature_exists(feature_name):  # 缺少卡槽标注时保留空候选并继续。
                candidates.append(None)  # 使用空值维持左中右索引稳定。
                slots.append(None)  # 同步保存不可点击卡槽占位。
                continue  # 继续读取下一个候选位置。
            slot = self.get_box_by_name(feature_name)  # 读取当前分辨率下的整张武将卡区域。
            boxes = self.ocr(box=slot, frame=frame, threshold=0.18, target_height=1080, log=False)  # 在单个卡槽内识别武将名文字。
            text = "".join(str(box.name or "") for box in sorted(boxes, key=lambda box: (box.y, box.x)))  # 合并横排或竖排 OCR 文本。
            candidates.append(normalize_hero_name(text))  # 只接受账号支持的完整规范武将名。
            slots.append(slot)  # 保存当前候选对应的点击区域。
        return candidates, slots  # 返回保持屏幕位置对应关系的候选与卡槽。

    def _swap_until_passive_candidate(self, candidates, slots):  # 使用左右两个换将按钮直到出现全被动技能角色或次数耗尽。
        swap_features = ((0, "Hero Slot 1 - Swap Player"), (1, "Hero Slot 2 - Swap Player"))  # 对应图片3中左侧和中间武将的换将按钮。
        for slot_index, feature_name in swap_features:  # 每个真实按钮最多点击一次，防止同一选将界面无限消耗换将次数。
            if any(hero in PASSIVE_OWNED_HEROES for hero in candidates):  # 前一次换将已经获得被动角色。
                break  # 立即保留被动候选，不继续浪费换将次数。
            swap_button = self._find_first_feature([feature_name], self.config.get("Template Threshold", 0.8))  # 仅点击当前画面实际存在的对应换将按钮。
            if swap_button is None:  # 按钮不可用、次数耗尽或模板未匹配时跳过该位置。
                continue  # 尝试另一个已标注换将按钮。
            previous_hero = candidates[slot_index] if slot_index < len(candidates) else None  # 保存换将前角色用于结构化日志。
            self.click_box(swap_button, after_sleep=0.9)  # 点击完整换将按钮并等待新武将卡片动画稳定。
            refreshed_frame = self.next_frame()  # 获取换将完成后的新画面。
            candidates, slots = self._recognize_hero_candidates(refreshed_frame)  # 重新识别全部卡槽，避免只读到动画残帧。
            self._record_event("hero_swapped", slot=slot_index + 1, previous_hero=previous_hero, current_hero=candidates[slot_index] if slot_index < len(candidates) else None, candidates=candidates, passive_found=any(hero in PASSIVE_OWNED_HEROES for hero in candidates))  # 保存换将结果供胜率和OCR诊断。
        return candidates, slots  # 返回最终候选供被动优先选择器使用。

    def _ensure_current_hero(self, frame):  # 从牌局左下角竖排名称补识别当前武将。
        if not self.feature_exists("Our hero"):  # 当前素材没有我方武将名称区域时无法补识别。
            return None  # 保持未知武将并继续使用标准打牌算法。
        region = self.get_box_by_name("Our hero")  # 获取左下角我方竖排武将名区域。
        boxes = self.ocr(box=region, frame=frame, threshold=0.15, target_height=1080, log=False)  # 只对很小区域执行低成本 OCR。
        text = "".join(str(box.name or "") for box in sorted(boxes, key=lambda box: (box.y, box.x)))  # 按竖排顺序合并文字。
        hero = normalize_registered_hero_name(text)  # 牌局内允许识别注册表全部65名，自动选将仍由账号拥有列表过滤。
        if hero is not None:  # 完整识别成功时锁定本局武将。
            self.current_hero = hero  # 保存英雄供模型策略和结算统计使用。
            self.hero_runtime_state.hero = hero  # 同步统一英雄局内状态中的英雄名称。
            self.info_set("Current Hero", hero)  # 在任务面板持续显示识别结果。
            self.log_info(f"牌局内识别武将: {hero}")  # 记录补识别来源便于排查选将 OCR。
        return hero  # 返回识别结果供测试和调用方检查。

    def _record_hero_result(self, won):  # 只在胜利或失败结算画面记录一局有效统计。
        hero = self.current_hero or "未知英雄"  # 识别失败时保留独立未知桶而不污染具体武将胜率。
        position = self.resolved_ai_position or "landlord_down"  # 尚未锁定身份时保守归入农民分项。
        try:  # 捕获磁盘只读或统计文件异常以保证再来一局仍可继续。
            self.hero_statistics.record(hero, won, position)  # 写入总体和本局身份分项。
        except OSError as error:  # 文件系统写入失败不应中断牌局状态机。
            self.log_warning(f"武将胜率统计保存失败: {error}")  # 报告可操作的统计路径错误。
            return False  # 返回失败供测试验证安全行为。
        self.info_set("Hero Result", f"{hero}: {'胜' if won else '负'}")  # 在任务面板显示最近一局结果。
        return True  # 返回统计已经成功保存。

    def _recognize_stable_hand(self, frame):  # 统一要求最多五帧中的连续三帧完全一致，宁可慢一点也不把抖动牌面交给模型。
        first_boxes = self._recognize_cards(frame, "Deck of cards")  # 先读取状态机提供的当前帧。
        if not first_boxes:  # 首帧已经无法完整识别时无需继续使用残缺状态。
            observed = [[]]  # 保存逐帧补漏过程供日志诊断。
            previous_signature = None  # 尚未取得一份完整手牌，不能立即接受单帧结果。
            stable_streak = 0  # 记录连续相同完整识别结果的帧数。
            latest_frame = frame  # 初始化最终现场为调用方提供的当前帧。
            for attempt in range(5):  # 首帧任意单牌 OCR 抖动时再等待最多五帧，并要求连续三帧完整一致。
                if hasattr(self, "_executor"):  # 正常任务使用可响应停止信号的短等待。
                    self.sleep(0.12)  # 等待牌面动画和 OCR 输入稳定。
                latest_frame = self.next_frame()  # 获取新的真实游戏画面。
                current_boxes = self._recognize_cards(latest_frame, "Deck of cards")  # 重新分割并读取整手牌。
                current_signature = tuple(card for card, _ in current_boxes)  # 保留点数和屏幕顺序的一致性签名。
                observed.append(list(current_signature))  # 写入诊断序列。
                if current_boxes and previous_signature == current_signature:  # 当前完整结果与上一帧一致时增加稳定计数。
                    stable_streak += 1  # 连续稳定帧数增加一。
                elif current_boxes:  # 新的完整牌面签名从第一帧重新开始计数。
                    stable_streak = 1  # 当前帧作为新候选签名的第一份确认。
                else:  # 空结果不能参与稳定确认。
                    stable_streak = 0  # 清除被空帧打断的稳定计数。
                if current_boxes and stable_streak >= 3:  # 只有连续三份完整结果一致才接受。
                    self.hand_change_pending = False  # 清除可能残留的技能动画锁。
                    self._record_event("hand_recognition_recovered", attempts=attempt + 2, hand_cards=list(current_signature))  # 记录从首帧失败恢复成功。
                    return current_boxes, latest_frame  # 使用最新牌框继续决策和点击。
                previous_signature = current_signature if current_boxes else None  # 空结果不能参与连续一致判断。
            self._record_event("hand_recognition_unstable", expected_count=getattr(self, "expected_hand_count", None), observations=observed)  # 保存连续失败详情。
            self._capture_state("hand_recognition_unstable", latest_frame, force=True)  # 保存最终现场供继续改进 OCR。
            return [], latest_frame  # 禁止将残缺手牌交给模型。
        first_cards = [card for card, _ in first_boxes]  # 提取第一帧按屏幕顺序排列的点数。
        expected_count = getattr(self, "expected_hand_count", None)  # 读取正常出牌后预计应剩的手牌数。
        previous_boxes = first_boxes  # 保存上一帧结果用于连续一致性比较。
        previous_signature = tuple(first_cards)  # 使用带顺序的完整点数签名防止排序动画造成点击框错位。
        stable_streak = 1  # 首帧完整识别结果只计作一份确认，不能单帧直接提交。
        latest_frame = frame  # 保存最后检查的画面供失败截图。
        observed = [list(first_cards)]  # 收集最多三帧识别结果写入诊断日志。
        for attempt in range(4):  # 再读取最多四帧，使所有场景都达到连续三帧稳定确认。
            test_sleeper = self.__dict__.get("sleep")  # 纯逻辑测试可能提供实例级短等待桩而没有执行器。
            if callable(test_sleeper):  # 优先使用测试对象明确提供的等待函数。
                test_sleeper(0.12)  # 跳过或模拟短等待。
            elif hasattr(self, "_executor"):  # 正常 ok-script 任务存在执行器时使用可响应停止信号的等待。
                self.sleep(0.12)  # 等待加牌飞行动画和手牌重新排版趋于稳定。
            latest_frame = self.next_frame()  # 获取下一张真实游戏画面。
            current_boxes = self._recognize_cards(latest_frame, "Deck of cards")  # 对新排版重新执行完整分割和 OCR。
            current_cards = [card for card, _ in current_boxes]  # 提取当前帧完整点数签名。
            observed.append(list(current_cards))  # 保存本帧结果供日志定位首个分歧。
            current_signature = tuple(current_cards)  # 保留顺序以保证稍后点击框和点数一一对应。
            if current_boxes and current_signature == previous_signature:  # 当前完整结果与上一帧一致时增加稳定计数。
                stable_streak += 1  # 连续稳定帧数增加一。
            elif current_boxes:  # OCR 或动画产生新签名时重新开始稳定计数。
                stable_streak = 1  # 当前帧作为新候选签名的第一份确认。
            else:  # 空结果打断当前候选签名。
                stable_streak = 0  # 清除连续稳定计数。
            if current_boxes and stable_streak >= 3:  # 连续三帧完整一致后才接受技能变化或普通手牌。
                self.hand_change_pending = False  # 连续两帧稳定后解除技能动画锁。
                self._record_event("hand_recognition_stabilized", expected_count=expected_count, actual_count=len(current_cards), attempts=attempt + 2, hand_cards=current_cards)  # 记录技能变化后的稳定识牌结果。
                return current_boxes, latest_frame  # 使用最新稳定帧继续桌面识别和模型决策。
            previous_boxes = current_boxes  # 将当前结果作为下一次连续一致性比较基准。
            previous_signature = current_signature  # 更新完整点数签名。
        self._record_event("hand_recognition_unstable", expected_count=expected_count, observations=observed)  # 三帧仍不一致时保留每帧结果供诊断。
        self._capture_state("hand_recognition_unstable", latest_frame, force=True)  # 保存最终现场以便继续调整素材和阈值。
        return [], latest_frame  # 禁止将动画或残缺手牌交给牌账本和 AI。

    def _handle_unknown_skill_interaction(self, frame):  # 处理六名当前账号英雄已经明确规则的二级选牌界面。
        hero = getattr(self, "current_hero", None)  # 读取当前已确认的规范英雄名称。
        runtime_state = getattr(self, "hero_runtime_state", None)  # 读取已识别的待处理技能名称，避免同一武将多个技能混用验证状态。
        pending_skill = getattr(runtime_state, "pending_interaction", None)
        if isinstance(pending_skill, dict):
            pending_skill = pending_skill.get("skill")
        if not self.config.get("Auto Use Hero Skills", True) or not is_live_skill_interaction_verified(hero, pending_skill):  # 用户关闭技能或注册表未完成 UI 验证时禁止尝试。
            self._record_event("skill_interaction_deferred", hero=hero, skill=pending_skill, reason="技能 UI 状态尚未验证")
            return False  # 交给父任务保存现场并暂停。
        hand_boxes, frame = self._recognize_stable_hand(frame)  # 技能界面恰是加牌和重排高发阶段，先完成多帧一致性确认。
        if hero == "凌统" and not hand_boxes:  # 勇进首次发动会播放遮挡大部分手牌的武将动画。
            for retry in range(5):  # 额外等待最多五轮，避免把正常技能动画立即判为未知界面。
                test_sleeper = self.__dict__.get("sleep")  # 纯逻辑测试可以提供实例级等待桩。
                if callable(test_sleeper):  # 优先使用测试对象明确提供的短等待函数。
                    test_sleeper(0.40)  # 等待武将动画逐步退出手牌区域。
                elif hasattr(self, "_executor"):  # 正常 ok-script 任务存在执行器时使用可中断等待。
                    self.sleep(0.40)  # 防止连续读取同一张动画帧。
                retry_frame = self.next_frame()  # 获取技能动画后的新画面。
                hand_boxes, frame = self._recognize_stable_hand(retry_frame)  # 重新执行完整多帧稳定识别。
                if hand_boxes:  # 已恢复完整手牌时立即进入语义选牌。
                    self._record_event("skill_interaction_animation_recovered", hero=hero, retries=retry + 1, hand_cards=[card for card, _ in hand_boxes])  # 记录首次技能动画恢复耗时。
                    break  # 停止额外等待。
        option_boxes = []  # 默认当前交互没有桌面可获取牌选项。
        skill_pool_region = self.get_box_by_name("skill_card_pool") if hero in {"关羽", "诸葛均"} and self.feature_exists("skill_card_pool") else None  # 读取用户明确标注的技能获取牌库区域。
        skill_option_slots = detect_skill_option_card_boxes(frame, skill_pool_region) if hero in {"关羽", "诸葛均"} else []  # 获取牌技能优先定位完整实体卡片而不是复用桌面 OCR 文字框。
        pool_evaluation = None  # 默认当前交互不是可取消的获取牌库决策。

        def choose_interaction(option_values):  # 将屏幕识别结果转换成共享技能策略上下文，点击层不再自行决定牌点。
            return choose_skill_interaction_action(
                hero,
                [card for card, _ in hand_boxes],
                option_values,
                getattr(self, "hero_last_action_type", None),
                pending_skill=pending_skill,
                skill_uses=getattr(runtime_state, "skill_uses", {}),
                marks=getattr(runtime_state, "marks", {}),
            )

        if skill_option_slots:  # 已检测到技能获取牌库中的并排展示牌。
            if not hand_boxes:  # 没有完整手牌时无法比较获取前后的真实牌路。
                self._record_event("skill_interaction_deferred", hero=hero, reason="技能获取牌库已识别，但当前手牌不完整")  # 保存拒绝猜测的原因。
                return False  # 等待下一帧重新识牌而不是盲目获取或取消。
            option_boxes = self._recognize_skill_option_cards(frame, skill_option_slots)  # 只使用 OCR 判断点数，点击始终使用实体卡片中心。
            if len(option_boxes) != len(skill_option_slots):  # 任意一张展示牌漏识别都会改变最优获取结果。
                self._record_event("skill_interaction_deferred", hero=hero, reason="技能获取牌库点数未完整识别", detected_slots=len(skill_option_slots), recognized_cards=[card for card, _ in option_boxes])  # 保存完整诊断信息。
                return False  # 不把残缺选项交给策略，也不点击错误牌。
            source, ranks, reason = choose_interaction([card for card, _ in option_boxes])  # 枚举取牌和跳过的完整技能结算。
            pool_evaluation = {"decision": "cancel" if source == "skip" else "pick", "chosen": list(ranks), "reason": reason, "policy": "hero_policy_v3"}  # 保存统一规则选择供回放。
            if source == "skip":  # 所有完整取牌路线都不优于不发动。
                return self._cancel_skill_card_pool(frame, skill_pool_region, hero, pool_evaluation)  # 明确取消且记录本次技能已消耗。
        elif hero in {"关羽", "诸葛均"} and self.feature_exists("Playing card area"):  # 武圣及旧版耕读布局继续读取已有桌面牌区域。
            option_region = self.get_box_by_name("Playing card area")  # 复用已有桌面牌区域，避免在未知界面使用坐标猜测。
            option_boxes = self._ocr_card_group(frame, option_region)  # 读取所有明确识别出的可选牌点数及其真实框。
            source, ranks, reason = choose_interaction([card for card, _ in option_boxes])  # 旧布局同样使用共享技能选择，仅保留原点击验证。
        else:  # 当前界面是弃牌、转换等非获取牌库交互。
            source, ranks, reason = choose_interaction([])  # 根据完整手牌枚举技能动作并比较完整结算。
        if source == "skip":  # 当前布局允许跳过但没有已确认的取牌库取消按钮。
            self._record_event("skill_interaction_deferred", hero=hero, skill=pending_skill, reason=f"{reason}；当前布局没有可靠取消按钮")  # 明确记录策略结论和 UI 阻塞。
            return False  # 保持安全暂停，禁止猜测取消坐标。
        if source is None or not ranks:  # 当前画面没有足够信息生成完整技能动作。
            self._record_event("skill_interaction_deferred", hero=hero, reason=reason)  # 记录暂停原因供后续补素材。
            return False  # 不执行任何猜测点击。
        available_boxes = hand_boxes if source == "hand" else option_boxes  # 选择与语义动作来源一致的可点击牌框。
        try:  # 捕获 OCR 只识别到部分重复牌的情况。
            selected_boxes = self._boxes_for_action(ranks, available_boxes)  # 要求屏幕牌框数量与技能动作完全一致。
        except AiModelError as error:  # 映射不完整时不得执行部分选择。
            self._record_event("skill_interaction_deferred", hero=hero, reason=str(error), action=ranks)  # 保存缺失牌框诊断信息。
            return False  # 保持画面原状并暂停。
        for box in selected_boxes:  # 逐张选择规则确定的目标牌。
            self.click_box(box, after_sleep=0.12)  # 使用 OCR 或手牌分割得到的真实牌框点击。
        selected_frame = self.next_frame()  # 获取选择完成后的稳定画面验证按钮是否启用。
        self._capture_state("skill_interaction_selected", selected_frame, force=True)  # 保存选择结果供逐局复核。
        if hero == "卢植":  # 儒宗不同界面版本可能点击后自动转换，也可能继续显示确定按钮。
            before_cards = [card for card, _ in hand_boxes]  # 保存转换前完整手牌用于严格差值验证。
            verification_frame = selected_frame  # 从选择后的第一帧开始检查是否已经自动结算。
            for attempt in range(3):  # 短暂等待变牌动画，未自动结算再进入通用确认按钮流程。
                after_boxes = self._recognize_cards(verification_frame, "Deck of cards")  # 重新读取整手牌，禁止依赖旧点击坐标。
                after_cards = [card for card, _ in after_boxes]  # 提取完整点数签名。
                if is_skill_selection_auto_resolved(hero, before_cards, ranks, after_cards):  # 只有目标点数精确增减一张才确认成功。
                    runtime_state = getattr(self, "hero_runtime_state", None)  # 读取统一英雄状态。
                    if runtime_state is not None:  # 兼容旧会话和最小测试对象。
                        runtime_state.pending_interaction = None  # 清除儒宗待处理交互。
                        runtime_state.skill_uses["儒宗"] = min(3, runtime_state.skill_uses.get("儒宗", 0) + 1)  # 按三次上限累计真实成功次数。
                    self.hand_change_pending = True  # 下一次决策必须等待新牌序稳定并写入牌账本。
                    self.turn_action_completed = True  # 锁住技能后的残留出牌按钮，防止动画期间再次选牌。
                    self._record_event("skill_interaction_resolved", hero=hero, source=source, action=ranks, reason=reason, auto_submit=True, attempts=attempt + 1, after_hand_cards=after_cards)  # 保存自动转换证据。
                    return True  # 重新进入状态识别而不继续寻找或点击按钮。
                if attempt < 2:  # 尚有等待机会时获取新帧。
                    test_sleeper = self.__dict__.get("sleep")  # 测试可注入无等待桩。
                    if callable(test_sleeper):
                        test_sleeper(0.25)
                    elif hasattr(self, "_executor"):
                        self.sleep(0.25)
                    verification_frame = self.next_frame()  # 获取变牌动画后的最新画面。
            selected_frame = verification_frame  # 未自动结算时使用最新稳定画面继续寻找确认按钮。
        if hero == "凌统":  # 勇进真实二级界面点击目标牌后自动弃置，不存在额外确定按钮。
            verification_frame = selected_frame  # 从刚点击后的首帧开始检查手牌是否完成重排。
            before_cards = [card for card, _ in hand_boxes]  # 保存点击前完整手牌签名用于严格差值验证。
            for attempt in range(7):  # 最多等待约一秒八，覆盖弃牌飞行动画和手牌重排。
                after_boxes = self._recognize_cards(verification_frame, "Deck of cards")  # 只读取底部手牌区，不把桌面弃牌标记当成手牌。
                after_cards = [card for card, _ in after_boxes]  # 提取当前帧完整手牌点数。
                if is_skill_selection_auto_resolved(hero, before_cards, ranks, after_cards):  # 要求点击前后只减少计划弃置牌。
                    runtime_state = getattr(self, "hero_runtime_state", None)  # 读取统一英雄状态供记录次数和清除交互。
                    if runtime_state is not None:  # 兼容旧会话和最小测试对象。
                        runtime_state.pending_interaction = None  # 标记勇进二级交互已经自动完成。
                        runtime_state.skill_uses["勇进"] = min(2, runtime_state.skill_uses.get("勇进", 0) + 1)  # 按技能上限累计真实成功次数。
                    self.hand_change_pending = True  # 下一次决策仍执行稳定手牌确认并让牌账本记录技能弃牌。
                    self._capture_state("skill_interaction_auto_resolved", verification_frame, force=True)  # 保存带桌面弃牌标记的成功素材。
                    self._record_event("skill_interaction_resolved", hero=hero, source=source, action=ranks, reason=reason, auto_submit=True, attempts=attempt + 1, after_hand_cards=after_cards, click_targets=[{"x": box.x, "y": box.y, "width": box.width, "height": box.height} for box in selected_boxes])  # 写入可回放的自动提交证据。
                    return True  # 通知状态机重新识别已完成技能后的牌局状态。
                if attempt < 6:  # 尚有等待次数时获取下一帧继续验证。
                    test_sleeper = self.__dict__.get("sleep")  # 纯逻辑测试可以提供实例级等待桩。
                    if callable(test_sleeper):  # 优先使用测试对象明确提供的短等待函数。
                        test_sleeper(0.30)  # 等待弃牌飞行动画继续播放。
                    elif hasattr(self, "_executor"):  # 正常 ok-script 任务存在执行器时使用可中断等待。
                        self.sleep(0.30)  # 避免无间隔抓取多个相同动画帧。
                    verification_frame = self.next_frame()  # 获取下一张游戏画面重新检查完整手牌。
            self._capture_state("skill_interaction_auto_resolve_unverified", verification_frame, force=True)  # 保存超时现场继续补充异常素材。
            self._record_event("skill_interaction_deferred", hero=hero, reason="勇进点击后手牌差值未稳定，未按旧坐标回滚", action=ranks)  # 明确记录不回滚原因。
            return False  # 保留安全暂停，禁止点击已经重排后可能对应其他牌的旧坐标。
        confirm_buttons = self.wait_ocr(0.25, 0.45, 0.78, 0.72, match=re.compile("确定|确认|选定|弃置|获得|转换"), time_out=1.2, raise_if_not_found=False, log=False) or []  # 覆盖诸葛均等技能弹窗中部按钮，并把 OCR 超时的 None 安全归一为空列表。
        verification_frame = self.next_frame()  # 获取与确认按钮最接近的画面用于颜色状态验证。
        confirm_box = next((box for box in confirm_buttons if is_active_skill_confirm_button(verification_frame, box)), None)  # 过滤灰色禁用或动画中的按钮。
        if confirm_box is None:  # 没有同时满足文字和可用颜色的确认按钮。
            if hero != "卢植":  # 儒宗选择后可能已经开始重排，旧牌框坐标不能用于撤销。
                for box in reversed(selected_boxes):  # 其他确认型技能仍按反序完整取消选中牌。
                    self.click_box(box, after_sleep=0.08)  # 再次点击同一真实牌框撤销选择。
            self._capture_state("skill_interaction_confirm_unavailable", verification_frame, force=True)  # 保存按钮未就绪现场供补素材。
            self._record_event("skill_interaction_deferred", hero=hero, reason="确认按钮未识别或仍为禁用状态", action=ranks)  # 记录安全回退原因。
            return False  # 父任务随后暂停，不继续随机尝试。
        self.click_box(confirm_box, after_sleep=1.0)  # 点击已通过文字和颜色双重验证的确认按钮。
        runtime_state = getattr(self, "hero_runtime_state", None)  # 读取统一英雄状态供清除待处理交互。
        if runtime_state is not None:  # 兼容旧会话和最小测试对象。
            runtime_state.pending_interaction = None  # 标记本次二级技能选择已经完成。
            if hero == "卢植":  # 确认按钮版本的儒宗同样需要累计次数。
                runtime_state.skill_uses["儒宗"] = min(3, runtime_state.skill_uses.get("儒宗", 0) + 1)  # 防止后续重复触发超过技能上限。
        if pool_evaluation is not None:  # 本次确定属于获取牌库技能而非普通弃牌交互。
            self._mark_skill_pool_consumed(hero)  # 获取成功后累计技能次数并避免同一弹窗重复处理。
        self.hand_change_pending = True  # 技能确认后下一次使用手牌前必须等待加牌、弃牌或变点动画稳定。
        self.turn_action_completed = True  # 锁定技能弹窗后的残留操作状态，直到状态机真正离开并进入下一回合。
        self._record_event("skill_interaction_resolved", hero=hero, source=source, action=ranks, reason=reason, pool_evaluation=pool_evaluation, click_targets=[{"x": box.x, "y": box.y, "width": box.width, "height": box.height} for box in selected_boxes])  # 写入可回放的完整技能决策、收益和点击位置。
        return True  # 通知采集状态机重新识别技能完成后的新界面。

    def _mark_skill_pool_consumed(self, hero):  # 获取或取消后统一记录技能已经消耗一次。
        runtime_state = getattr(self, "hero_runtime_state", None)  # 读取统一英雄局内状态。
        if runtime_state is None:  # 兼容绕过会话初始化的纯逻辑测试。
            return  # 没有状态对象时只保留事件日志。
        runtime_state.pending_interaction = None  # 清除本次获取牌库交互。
        skill_name, limit = {"诸葛均": ("耕读", 1), "关羽": ("武圣", 2)}.get(hero, ("获取牌", 99))  # 映射当前已实现的获取牌技能次数上限。
        runtime_state.skill_uses[skill_name] = min(limit, runtime_state.skill_uses.get(skill_name, 0) + 1)  # 获取和取消均消耗本次技能次数。

    def _cancel_skill_card_pool(self, frame, region, hero, evaluation):  # 所有展示牌收益为负时可靠点击取消并消耗技能。
        if region is None:  # 缺少用户标注区域时不能生成安全取消按钮框。
            self._record_event("skill_interaction_deferred", hero=hero, reason="缺少技能获取牌库区域，无法安全取消", pool_evaluation=evaluation)  # 保存资源缺口。
            return False  # 禁止使用随机相对坐标。
        cancel_region = Box(int(region.x + region.width * 0.06), int(region.y + region.height * 0.76), int(region.width * 0.36), int(region.height * 0.17), name="skill_pool_cancel")  # 按标注弹窗构造完整取消按钮区域。
        cancel_text = self.ocr(box=cancel_region, frame=frame, threshold=0.20, log=False)  # OCR只用于确认当前按钮语义。
        confirmed = any(re.search(r"取消|放弃", str(box.name or "")) for box in cancel_text)  # 兼容技能界面的取消文案。
        if not confirmed:  # 没有明确文字证据时不点击。
            self._capture_state("skill_pool_cancel_unavailable", frame, force=True)  # 保存当前不同按钮皮肤供补素材。
            self._record_event("skill_interaction_deferred", hero=hero, reason="技能牌收益为负，但取消按钮未可靠识别", pool_evaluation=evaluation)  # 写入策略和识别分歧。
            return False  # 保留界面等待安全恢复。
        self.click_box(cancel_region, after_sleep=1.0)  # 点击完整取消按钮中心，不使用可能漂移的OCR文字框。
        self._mark_skill_pool_consumed(hero)  # 用户明确说明取消同样消耗技能，因此同步次数状态。
        self._record_event("skill_interaction_cancelled", hero=hero, reason=evaluation.get("reason"), pool_evaluation=evaluation, skill_consumed=True)  # 保存取消原因和技能消耗事实。
        return True  # 通知状态机重新识别取消后的牌局。

    def _play_lowest_single(self, frame):  # 将主动出牌回合改由训练模型处理。
        if not self._play_with_ai(frame):  # 高容量模型未完成动作时仍只使用项目自己的合法组合规则。
            self._play_lead_heuristic(frame)  # 根据完整手牌选择顺子、连对、三带等动作，不点击提示或固定第一张。

    def _play_lead_heuristic(self, frame):  # 没有训练模型时根据完整手牌选择确定性的合法主动牌型。
        card_boxes, frame = self._recognize_stable_hand(frame)  # 技能变化后先取得连续两帧一致的完整手牌及最新画面。
        if not card_boxes:  # 识牌不完整时不能安全组合主动牌型。
            self._capture_state("lead_hand_recognition_failed", frame, force=True)  # 保存识牌失败现场供继续改进素材。
            self.log_warning("主动出牌时未完整识别手牌，已停止本次选牌以避免乱出。")  # 明确说明没有执行随机点击。
            return False  # 返回失败等待下一轮重新识别。
        hand_cards = [card for card, _ in card_boxes]  # 提取策略需要的完整手牌点数。
        action = choose_lead_action(hand_cards)  # 使用确定性策略选择顺子、连对、三带、对子或最小单牌。
        if not action or not is_basic_legal_lead(action):  # 对兜底策略结果再做一次合法性检查。
            self._capture_state("lead_strategy_failed", frame, force=True)  # 保存无法生成合法主动动作的现场。
            self.log_warning(f"主动出牌策略未生成合法牌型: {action}")  # 报告具体异常动作供排查。
            return False  # 拒绝点击任何未经合法性验证的牌组。
        try:  # 捕获屏幕牌框数量与策略动作意外不一致的情况。
            selected_boxes = self._boxes_for_action(action, card_boxes)  # 将策略点数组合精确映射到屏幕手牌。
        except AiModelError as error:  # 处理无法定位完整动作的安全错误。
            self._capture_state("lead_action_mapping_failed", frame, force=True)  # 保存点数映射失败现场。
            self.log_warning(f"主动牌型无法映射到屏幕: {error}")  # 记录不执行残缺点击的原因。
            return False  # 拒绝只选中动作的一部分。
        for box in selected_boxes:  # 逐张点击策略决定的合法牌型。
            self.click_box(box, after_sleep=0.12)  # 点击牌面顶部区域完成精确选牌。
        selected_frame = self.next_frame()  # 获取全部目标牌抬起后的新画面。
        self._capture_state("lead_strategy_selected", selected_frame, force=True)  # 保存本次主动策略选牌结果供复核。
        submitted = self._submit_selected_cards(selected_frame, "lead_strategy_without_play_button")  # 使用现有可靠流程点击出牌并检查提交。
        if submitted:  # 只有按钮提交成功或安全不出恢复后才更新回合状态。
            if getattr(self, "last_submit_used_pass", False):  # 出牌按钮失败后实际执行的是不出，不能扣除计划牌组。
                self.expected_hand_count = len(hand_cards)  # 实际未打出任何牌，保持当前完整手牌数量。
                self.hero_last_action_type = None  # 不出会中断连续牌型技能状态。
                self.hand_change_pending = True  # 不出后的英雄弃牌或变点仍需多帧确认。
                self.turn_action_completed = True  # 父恢复流程已经结束当前回合。
                self._record_event("planned_action_cancelled", round_id=getattr(self, "current_round_id", None), planned_action=list(action), actual_action=[], reason="submit_failed_then_pass")  # 明确记录计划动作未发生。
                return True  # 返回回合已安全完成但不再写入虚假出牌账本。
            action_type = classify_action(action)  # 计算本次主动动作的英雄牌型分类。
            self.card_ledger.record_play(action, getattr(self, "current_hero", None), action_type)  # 将成功主动动作写入允许技能变化的牌账本。
            self.hand_change_pending = True  # 英雄被动可能在出牌后加牌或变点，下一次识牌必须多帧确认。
            self.turn_action_completed = True  # 锁定当前我方回合避免再次选择同一牌型。
            self._record_event("action_submitted", round_id=getattr(self, "current_round_id", None), action=list(action), action_type=action_type, source="lead_heuristic")  # 写入完整主动动作和来源。
        self.log_info(f"主动出牌策略选择: {' '.join(action)}")  # 在日志中显示实际选择的牌型和点数。
        return bool(submitted)  # 返回主动牌型是否已经成功提交。

    def _play_with_ai(self, frame):  # 识别牌局状态、请求模型动作并映射为屏幕点击。
        if self.ai_model is None:  # 检查训练模型是否已经成功加载。
            if not self.ai_unavailable_reported:  # 避免每个回合重复保存同一模型缺失现场。
                self.ai_unavailable_reported = True  # 锁定本次运行已经报告过模型缺失。
                self._capture_state("ai_model_unavailable", frame, force=True)  # 只保存一张模型缺失现场供排查。
                self.log_warning("训练模型不可用，已切换到游戏提示自动选牌。", notify=True)  # 明确说明任务仍会继续自动操作。
            return False  # 不执行任何可能改变牌局的猜测点击。
        self.last_ai_error = None  # 每个新回合开始时清除上一回合可恢复的动作错误。
        card_boxes, frame = self._recognize_stable_hand(frame)  # 技能获得、回收或变牌后使用连续两帧一致的完整手牌和最新画面。
        if not card_boxes:  # OCR 没有得到任何有效手牌时停止当前回合。
            self._capture_state("ai_hand_recognition_failed", frame, force=True)  # 保存识牌失败素材便于继续标注。
            self._record_event("ocr_failed", round_id=getattr(self, "current_round_id", None), region="hand")  # 记录完整手牌识别失败供质量统计。
            self.log_warning("AI 未识别到手牌，本回合未执行点击。")  # 告知用户具体失败阶段。
            return False  # 避免根据错误手牌调用模型。
        hand_cards = [card for card, _ in card_boxes]  # 提取按屏幕顺序识别的完整手牌点数。
        if not hasattr(self, "card_ledger"):  # 兼容绕过正常初始化的纯逻辑测试对象。
            self.card_ledger = CardLedger()  # 为测试或断点恢复现场补建空牌账本。
            self.card_ledger.last_hand = list(hand_cards[:getattr(self, "expected_hand_count", len(hand_cards))])  # 使用可用牌数建立保守的上一手快照。
        if not hasattr(self, "hero_runtime_state"):  # 兼容旧会话和纯逻辑测试对象。
            self.hero_runtime_state = HeroRuntimeState(hero=getattr(self, "current_hero", None), last_action_type=getattr(self, "hero_last_action_type", None))  # 根据现有字段恢复英雄状态。
        ledger_changes = self.card_ledger.observe_hand(hand_cards, self.expected_hand_count, getattr(self, "current_hero", None))  # 比较牌面变化并保留技能加牌、弃牌和重复点数来源。
        if ledger_changes:  # 当前完整手牌与上一观察存在技能级变化。
            for change in ledger_changes:  # 逐项写入技能变化事件便于牌局回放。
                change_payload = change.to_dict()  # 将账本事件转换成 JSON 可写对象。
                change_payload["ledger_event_type"] = change_payload.pop("event_type")  # 避免账本类型与外层日志事件类型字段冲突。
                self._record_event("card_ledger", **change_payload)  # 保存事件类型、牌点数、来源和附加信息。
            gained_count = sum(len(change.cards) for change in ledger_changes if change.event_type == "gain")  # 统计本次技能实际新增牌数。
            if gained_count:  # 只有真实新增牌时更新任务面板提示。
                self.log_info(f"检测到技能获得 {gained_count} 张牌，已写入牌账本并保留可见历史。")  # 说明不再简单清空全部动作历史。
                self.info_set("Skill Cards Acquired", gained_count)  # 在任务面板显示最近一次技能加牌数量。
            observed_passive = sync_observed_passive_skill_uses(getattr(self, "current_hero", None), self.hero_runtime_state, ledger_changes)  # 将可由牌面唯一确认的被动技能进度同步到算法状态。
            if observed_passive and getattr(self, "current_hero", None) == "关银屏":  # 新增 J、Q、K 可以确认花武触发。
                self._record_event("hero_skill_observed", hero="关银屏", skill="花武", observed_uses=observed_passive, total_uses=self.hero_runtime_state.skill_uses["花武"], limit=5)  # 保存跨回合可回放的花武次数。
                self.log_info(f"关银屏花武已确认触发 {self.hero_runtime_state.skill_uses['花武']}/5 次，后续出牌已按剩余次数重新评分。")  # 让运行日志明确显示专项策略状态。
            elif observed_passive and getattr(self, "current_hero", None) == "赵云":  # 赵云回收牌重新出现时确认冲阵进度。
                self._record_event("hero_skill_observed", hero="赵云", skill="冲阵", recovered_cards=observed_passive, total_recovered=self.hero_runtime_state.marks["冲阵回收"], limit=7)  # 保存每次真实回收及累计进度。
                self.log_info(f"赵云冲阵已确认回收 {self.hero_runtime_state.marks['冲阵回收']}/7 张，出牌策略已更新触发机会。")  # 在运行日志显示策略依据。
        self.expected_hand_count = len(hand_cards)  # 将当前完整识牌数量作为下一次状态检查基准。
        table_boxes = self._recognize_cards(frame, "Playing card area")  # 识别当前需要压过的桌面牌组。
        table_cards = [card for card, _ in table_boxes]  # 提取桌面牌点数作为模型状态。
        if getattr(self, "current_play_state", None) == "play_follow" and not table_cards:  # 跟牌界面不允许把桌面识别失败误当成主动牌权。
            observed = []  # 保存补识别结果供日志诊断。
            for _ in range(2):  # 再读取两帧处理上一手牌落桌动画和临时 OCR 抖动。
                if hasattr(self, "_executor"):  # 正常任务使用短等待避免阻塞停止信号。
                    self.sleep(0.12)  # 等待桌面牌稳定。
                retry_frame = self.next_frame()  # 获取最新画面重新识别桌面牌。
                retry_boxes = self._recognize_cards(retry_frame, "Playing card area")  # 使用逐张桌面牌 OCR 补漏。
                observed.append([card for card, _ in retry_boxes])  # 保存本次结果。
                if retry_boxes:  # 找到完整桌面牌组时继续正常合法动作枚举。
                    table_boxes = retry_boxes  # 使用补识别后的牌框。
                    table_cards = observed[-1]  # 同步模型输入点数。
                    frame = retry_frame  # 后续按钮与身份识别使用同一最新画面。
                    break  # 停止额外等待。
            if not table_cards:  # 多帧仍无法确定上一手牌型时禁止生成主动动作。
                self._capture_state("follow_table_recognition_failed", frame, force=True)  # 保存缺失桌面牌现场。
                self._record_event("ocr_failed", round_id=getattr(self, "current_round_id", None), region="table", observations=observed)  # 记录桌面 OCR 失败。
                pass_box = self._find_first_feature(["Not"], self.config.get("Template Threshold", 0.8))  # 在信息不足时选择规则允许的安全不出。
                if pass_box is None:  # 当前帧没有可验证的不出按钮时不执行任何坐标猜测。
                    self.log_warning("跟牌时未完整识别桌面牌，已停止本轮点击。")  # 报告安全等待原因。
                    return False  # 等待下一轮重新识别。
                self.click_box(pass_box, after_sleep=1.0)  # AI 在不完整信息下明确选择不出，而不是调用提示。
                self.ai_history.append([])  # 按真实动作写入可见历史。
                self.hero_last_action_type = None  # 不出中断连续牌型技能状态。
                self.hand_change_pending = True  # 姜维等技能可能在不出后改变手牌。
                self.expected_hand_count = len(hand_cards)  # 不预先扣减当前手牌。
                self.turn_action_completed = True  # 锁定当前回合避免重复点击。
                self._record_event("action_submitted", round_id=getattr(self, "current_round_id", None), action=[], source="ai_unknown_table_pass")  # 记录自主安全决策来源。
                return True  # 本回合已由 AI 完成合法动作。
        if table_cards and (not self.ai_history or self.ai_history[-1] != table_cards):  # 检查当前对手动作是否尚未写入历史。
            self.ai_history.append(list(table_cards))  # 将最近一名对手的有效出牌加入模型历史。
        self.hero_runtime_state.hero = getattr(self, "current_hero", None)  # 同步当前英雄规范名称。
        self.hero_runtime_state.last_action_type = getattr(self, "hero_last_action_type", None)  # 同步张飞等连续牌型状态。
        position = self._resolve_position(len(hand_cards), frame)  # 先锁定身份，技能加牌后仍沿用开局识别结果。
        observed_opponent_counts = self._read_opponent_counts(frame)  # 按左侧下下家、右侧下家读取本帧原始剩余牌数。
        opponent_counts, opponent_count_anomalies = stabilize_opponent_card_counts(getattr(self, "last_opponent_card_counts", None), observed_opponent_counts)  # 用时序约束阻止十一等数字被误识别成十七后改变战术阶段。
        for anomaly in opponent_count_anomalies:  # 将每个被拒绝的牌数回跳写入逐局日志。
            self._record_event("opponent_count_rejected", round_id=getattr(self, "current_round_id", None), **anomaly)  # 为后续补素材和 OCR 校准保存座位及原始读数。
        self.opponent_skill_card_estimates, opponent_skill_changes = update_opponent_skill_card_estimates(getattr(self, "last_opponent_card_counts", None), opponent_counts, getattr(self, "opponent_skill_card_estimates", [0, 0]))  # 用净回升识别其他玩家技能获得牌。
        self.last_opponent_card_counts = list(opponent_counts)  # 保存当前可靠牌数供下一次我方决策比较。
        for skill_change in opponent_skill_changes:  # 将每个座位的技能牌估计变化写入逐局日志。
            self._record_event("opponent_skill_cards", round_id=getattr(self, "current_round_id", None), **skill_change)  # 支持后续回放校准技能获得与实际出牌。
        enemy_counts, teammate_count, table_is_teammate = resolve_team_context(position, opponent_counts, getattr(self, "last_table_player", None))  # 将屏幕座位转换为模型所需敌友关系。
        game_state = GameState(hand_cards=list(hand_cards), table_cards=list(table_cards), position=position, opponent_card_counts=opponent_counts, opponent_skill_card_estimates=list(self.opponent_skill_card_estimates), enemy_card_counts=enemy_counts, teammate_card_count=teammate_count, table_player=getattr(self, "last_table_player", None), table_is_teammate=bool(table_cards and table_is_teammate), history=list(self.ai_history), hero_state=self.hero_runtime_state, round_id=getattr(self, "current_round_id", "") or "", policy_id=getattr(self, "current_policy_id", "balanced"))  # 构造统一且可回放的牌局状态。
        state = game_state.to_model_state()  # 转换成兼容现有规则和训练模型的字典接口。
        state["search_budget_ms"] = max(0, min(1200, int(self.config.get("Search Budget Ms", 300))))  # 传递受严格上限保护的搜索时间预算。
        self.info_set("Hand Cards", " ".join(hand_cards))  # 在任务面板显示本回合完整识别手牌。
        self.info_set("Position", state["position"])  # 在任务面板显示地主或农民位置。
        self.info_set("Team Play", "保护队友牌权" if state["table_is_teammate"] else "正常对地主压牌")  # 实时显示本轮敌友判断，方便发现座位或桌面归属误识别。
        self._record_event("decision_state", round_id=state["round_id"], hand_cards=state["hand_cards"], table_cards=state["table_cards"], position=state["position"], opponent_card_counts=state["opponent_card_counts"], opponent_skill_card_estimates=state["opponent_skill_card_estimates"], enemy_card_counts=state["enemy_card_counts"], teammate_card_count=state["teammate_card_count"], table_player=state["table_player"], table_is_teammate=state["table_is_teammate"], history=state["history"], hero=state["hero"], hero_state=state["hero_state"], policy_id=state["policy_id"])  # 保存模型实际收到的完整可见状态、技能增牌和敌友关系。
        effective_action = []  # 历史与技能连续牌型使用生效牌面；点击和牌账本始终保留实体牌。
        wildcard_finish = choose_terminal_wildcard_action(hand_cards, table_cards)  # 在标准模型前处理万能牌加单牌的确定性终局。
        if wildcard_finish:  # 该动作由实体万能牌规则明确保证合法。
            action = wildcard_finish  # 一次选择万能牌与自然单牌组成对子。
            natural = next(card for card in action if card != "W")  # 终局特判已经保证恰有一张可复制的自然牌。
            effective_action = [natural, natural]  # 张飞等连续牌型与公开历史应看到游戏实际结算的对子。
            self._record_event("decision", round_id=state.get("round_id"), policy_id="wildcard_terminal_v1", candidates=[{"cards": list(action), "physical_cards": list(action), "effective_cards": list(effective_action), "action_type": "pair", "remaining_turns": 0}], chosen=list(action), effective_choice=list(effective_action), reason="万能牌复制最后一张自然单牌组成对子，一次出完")  # 保存独立且可回放的终局决策。
        elif len(hand_cards) == 2 and hand_cards.count("W") == 1:  # 跟牌动作不能组成终局对子时也不能把万能实体交给标准 RLCard。
            natural = next(card for card in hand_cards if card != "W")  # 读取仍可按普通单牌使用的自然牌。
            can_follow_solo = len(table_cards) == 1 and natural in CARD_ORDER and table_cards[0] in CARD_ORDER and CARD_ORDER.index(natural) > CARD_ORDER.index(table_cards[0])  # 判断自然单牌能否独立压过桌面单牌。
            action = [natural] if can_follow_solo else []  # 能安全压牌时先出自然牌，否则合法不出等待下一次主动牌权。
            effective_action = list(action)  # 该分支没有使用万能实体，生效牌面与点击牌完全一致。
            self._record_event("decision", round_id=state.get("round_id"), policy_id="wildcard_terminal_v1", candidates=[{"cards": list(action), "action_type": "solo" if action else "pass"}], chosen=list(action), reason="万能牌两张残局无法组成可压对子，使用自然单牌或安全不出")  # 避免未知万能编码导致模型异常或随机点击。
        else:  # 普通牌局继续使用规则或神经模型对合法动作排序。
            try:  # 捕获训练模型推理和动作校验错误。
                action = self.ai_model.predict(state)  # 请求训练模型选择本回合动作。
            except (AiModelError, LookupError, ValueError, RuntimeError) as error:  # 处理模型输出非法、点数映射缺失或推理运行失败。
                self.last_ai_error = str(error)  # 保存最近错误供状态面板查看。
                self._capture_state("ai_prediction_failed", frame, force=True)  # 保存发生模型错误时的完整画面。
                self._record_event("prediction_failed", round_id=state.get("round_id"), error=str(error), policy_id=state.get("policy_id"))  # 保存模型异常和策略版本。
                self.log_warning(f"AI 推理失败: {error}")  # 在日志中报告可操作的错误原因。
                return False  # 不点击未经验证的模型动作。
            model_data = getattr(getattr(self.ai_model, "model", None), "get", lambda key, default=None: default)("last_decision", None)  # 读取规则模型保存的可解释候选和选择理由。
            effective_action = resolve_effective_action(action, model_data)  # 仅使用本轮匹配日志中的万能赋值结果更新公开历史。
            if model_data is not None:  # 当前适配器支持可解释决策时写入完整候选列表。
                self._record_event("decision", **model_data)  # 保存所有合法候选、评分分项和最终动作。
                pressure_data = model_data.get("table_pressure") if isinstance(model_data, dict) else None  # 读取本回合地主/农民压制阶段。
                if isinstance(pressure_data, dict):  # 新策略提供最大封锁、中位消耗和保留控制三种模式。
                    self.info_set("Table Pressure", pressure_data.get("mode", "unknown"))  # 在任务面板实时显示实际启用的战术层。
        self.info_set("Chosen Action", "不出" if not action else " ".join(action))  # 在任务面板显示模型最终动作。
        if not action:  # 空动作按照斗地主规则表示不出。
            pass_box = self._find_first_feature(["Not"], self.config.get("Template Threshold", 0.8))  # 查找当前回合的不出按钮。
            if pass_box is None:  # 主动出牌回合不允许不出时拒绝空动作。
                self.log_warning("AI 返回不出，但当前回合没有不出按钮。")  # 报告模型动作与界面规则冲突。
                return False  # 等待下一次识别而不误点其他按钮。
            self.click_box(pass_box, after_sleep=1.0)  # 点击不出并结束当前回合。
            self.ai_history.append([])  # 将不出动作写入模型历史。
            self.hero_last_action_type = None  # 不出会中断张飞连续牌型并重置典韦连续状态。
            self.hand_change_pending = True  # 姜维等英雄可能在不出后弃牌或变点，下一次识牌必须多帧确认。
            self.expected_hand_count = len(hand_cards)  # 不出不会改变我方当前手牌数量。
            self.turn_action_completed = True  # 锁定当前回合已经完成不出动作。
            self._record_event("action_submitted", round_id=state.get("round_id"), action=[], source="ai", policy_id=state.get("policy_id"))  # 记录 AI 主动不出和策略版本。
            return True  # 返回本回合动作已成功执行。
        selected_boxes = self._boxes_for_action(action, card_boxes)  # 将模型点数组合映射到真实手牌框。
        for box in selected_boxes:  # 逐张选择模型决定打出的手牌。
            self.click_box(box, after_sleep=0.12)  # 点击牌面上方可见区域完成选牌。
        selected_frame = self.next_frame()  # 获取全部目标牌被选中后的新画面。
        self._capture_state("ai_hand_selected", selected_frame, force=True)  # 保存 AI 选牌结果用于复核训练效果。
        submitted = self._submit_selected_cards(selected_frame, "ai_without_play_button")  # 使用两种按钮布局的可靠流程提交动作。
        if not submitted:  # 百将牌技能规则拒绝标准斗地主动作或按钮仍未就绪时执行恢复。
            for box in selected_boxes:  # 逐张取消本次模型已经抬起但未能提交的牌。
                self.click_box(box, after_sleep=0.08)  # 再次点击原牌面区域恢复未选中状态。
            self.log_warning("模型动作未形成可提交牌型，已取消选牌并等待重新识别决策。")  # 避免下一轮重复点击导致选中状态混乱，且不调用游戏提示。
            self.last_ai_error = "action_rejected"  # 标记主动回合需要退回安全单牌而不是继续组合。
            return False  # 允许父类提示流程处理英雄技能产生的非标准规则。
        if getattr(self, "last_submit_used_pass", False):  # 黄色出牌按钮最终失败且恢复逻辑实际选择了不出。
            self.ai_history.append([])  # 按真实结果把本回合记录为不出而不是计划动作。
            self.hero_last_action_type = None  # 不出会中断张飞等连续牌型状态。
            self.hero_runtime_state.last_action_type = None  # 同步统一英雄状态。
            self.expected_hand_count = len(hand_cards)  # 未打出计划牌组，因此手牌基准不能错误扣减。
            self.hand_change_pending = True  # 姜维等不出技能可能继续改变手牌。
            self.turn_action_completed = True  # 恢复逻辑已经通过不出完成当前回合。
            self._record_event("planned_action_cancelled", round_id=state.get("round_id"), planned_action=list(action), actual_action=[], reason="submit_failed_then_pass", policy_id=state.get("policy_id"))  # 保存模型计划与实际动作分歧供训练过滤。
            return True  # 不再把未发生的计划动作写入牌账本。
        self.ai_history.append(list(effective_action))  # 历史记录游戏实际结算牌面，避免万能牌型退化成 other。
        self.hero_last_action_type = classify_action(effective_action)  # 保存生效牌型供下一回合英雄策略使用。
        self.hero_runtime_state.last_action_type = self.hero_last_action_type  # 同步统一英雄状态的上一手牌型。
        self.card_ledger.record_play(action, getattr(self, "current_hero", None), self.hero_last_action_type)  # 将成功动作写入本局牌账本而不假设标准牌库数量。
        self.hand_change_pending = True  # 出牌后英雄可能获得、回收或改变手牌，下一回合先等待连续稳定帧。
        self.expected_hand_count = len(hand_cards) - len(action)  # 记录正常出牌后的预期剩余手牌数量。
        self.turn_action_completed = True  # 锁定当前回合防止按钮动画期间重复选牌。
        self._record_event("action_submitted", round_id=state.get("round_id"), action=list(action), physical_action=list(action), effective_action=list(effective_action), action_type=self.hero_last_action_type, source="ai", policy_id=state.get("policy_id"), click_targets=[{"x": box.x, "y": box.y, "width": box.width, "height": box.height} for box in selected_boxes])  # 同时记录真实点击实体与技能看到的生效牌型。
        return True  # 返回 AI 选牌和提交已经执行。

    def _recognize_cards(self, frame, region_name):  # 在标注牌区内识别点数并保留点击框。
        if not self.feature_exists(region_name):  # 检查 COCO 标注是否包含目标牌区。
            return []  # 缺少牌区时返回空结果交由安全分支处理。
        region = self.get_box_by_name(region_name)  # 读取适配当前分辨率的牌区坐标。
        if region_name == "Deck of cards":  # 手牌高度重叠时不能直接使用整区通用 OCR。
            if self.feature_exists("Selected area"):  # 优先使用向上扩展的选中牌区兼容技能或残留选牌状态。
                region = self.get_box_by_name("Selected area")  # 读取能够完整覆盖抬起卡牌点数的区域。
            return self._recognize_hand_cards(frame, region)  # 先按牌边缘切成单张卡牌再逐张识别。
        if region_name == "Playing card area":  # 桌面区域可能同时保留左右两名对手的上一手牌。
            return self._recognize_latest_table_cards(frame, region)  # 分侧读取并只返回轮到我方前最近的有效动作。
        boxes = self.ocr(box=region, frame=frame, threshold=self.config.get("Card OCR Threshold", 0.25), target_height=720, log=False)  # 对牌区执行一次 OCR。
        recognized = []  # 初始化有效牌面和点击框列表。
        for box in sorted(boxes, key=lambda item: item.x):  # 按手牌从左到右的显示顺序处理 OCR 结果。
            card = parse_card_text(box.name or "")  # 从可能带花色噪声的 OCR 文本提取首个点数。
            if card is not None:  # 只保留能够确定点数的文字框。
                recognized.append((card, box))  # 同时保存点数和可以直接点击的屏幕框。
        return recognized  # 返回按屏幕顺序排列的可点击牌组。

    def _recognize_latest_table_cards(self, frame, region):  # 分开识别左、右对手牌堆并选择最近动作。
        half_width = region.width // 2  # 将宽桌面区按玩家左右位置分成两半。
        left_region = Box(region.x, region.y, half_width, region.height, name="left_table_cards")  # 创建左侧对手出牌识别区。
        right_region = Box(region.x + half_width, region.y, region.width - half_width, region.height, name="right_table_cards")  # 创建右侧对手出牌识别区。
        left_cards = self._ocr_card_group(frame, left_region)  # 读取行动顺序中紧邻我方的左侧玩家牌组。
        if left_cards:  # 左侧玩家有有效出牌时它就是我方需要响应的最近动作。
            self.last_table_player = "next_next_player"  # 图片标注中的左侧座位是我方下下家。
            return left_cards  # 不把右侧更早的牌组错误合并进当前牌型。
        right_cards = self._ocr_card_group(frame, right_region)  # 左侧不出时读取右侧玩家最后的有效牌组。
        self.last_table_player = "next_player" if right_cards else None  # 图片标注中的右侧座位是我方下家，无牌时清除旧归属。
        return right_cards  # 返回最近有效牌组并保持其座位归属。

    def _ocr_card_group(self, frame, region):  # 对单个玩家的桌面牌堆执行 OCR 并提取所有点数。
        boxes = self.ocr(box=region, frame=frame, threshold=self.config.get("Card OCR Threshold", 0.25), log=False)  # 在单侧窄区域识别牌面文字。
        recognized = []  # 初始化当前玩家的桌面牌组。
        for box in sorted(boxes, key=lambda item: item.x):  # 按牌堆从左到右顺序处理识别结果。
            card = parse_card_text(box.name or "")  # 从可能附带花色的文本中提取点数。
            if card is not None:  # 过滤背景标题和装饰文字。
                recognized.append((card, box))  # 保存牌面点数及其屏幕框。
        group_image = frame[region.y:region.y + region.height, region.x:region.x + region.width]  # 裁出单侧桌面区域执行卡牌竖边分割。
        table_card_width = max(60, int(round(frame.shape[1] * 118 / 1920)))  # 按真实一九二零截图中的桌面单牌宽度缩放。
        slots = detect_hand_slots(group_image, estimated_card_width=table_card_width, allow_center_fallback=False)  # 空白桌面不生成虚假居中单牌。
        if not slots:  # 技能大卡片等非标准布局继续使用通用 OCR 结果。
            return recognized  # 返回已识别的明确文本。
        slot_results = []  # 保存逐张桌面牌识别结果。
        scan_height = min(region.height, max(120, int(round(region.height * 0.58))))  # 覆盖桌面牌点数与花色上半部。
        for relative_x, slot_width in slots:  # 按检测到的真实桌面牌左边缘逐张 OCR。
            slot = Box(region.x + relative_x, region.y, slot_width, scan_height, name="table_card")  # 创建不跨相邻卡牌的窄框。
            patch = frame[slot.y:slot.y + slot.height, slot.x:slot.x + slot.width]  # 保留王字颜色信息。
            text_boxes = self.ocr(box=slot, frame=frame, threshold=max(0.12, self.config.get("Card OCR Threshold", 0.25) * 0.65), log=False)  # 对小尺寸桌面牌适当降低单牌阈值。
            card = next((parse_card_text(text_box.name or "", patch) for text_box in sorted(text_boxes, key=lambda item: (item.x, item.y)) if parse_card_text(text_box.name or "", patch) is not None), None)  # 取最左侧有效点数。
            if card is None:  # 窄体 J 等牌面在重叠可见宽度内可能不足以形成稳定 OCR 文本框。
                retry_widths = (slot.width + 20, max(table_card_width, slot.width * 2))  # 先小幅扩展，再使用完整桌面牌宽处理 OCR 对裁切宽度的敏感性。
                for requested_width in retry_widths:  # 最多执行两次局部补识别。
                    retry_width = min(region.x + region.width - slot.x, requested_width)  # 限制补识别框不越过单侧桌面区域。
                    retry_slot = Box(slot.x, slot.y, retry_width, scan_height, name="table_card_retry")  # 创建只用于失败牌的扩展框。
                    retry_patch = frame[retry_slot.y:retry_slot.y + retry_slot.height, retry_slot.x:retry_slot.x + retry_slot.width]  # 裁出补识别画面。
                    retry_boxes = self.ocr(box=retry_slot, frame=frame, threshold=0.10, log=False)  # 使用更低阈值补识别窄体牌面。
                    for text_box in sorted(retry_boxes, key=lambda item: (item.x, item.y)):  # 扩框可能包含下一张牌，因此只接受最左侧有效点数。
                        card = parse_card_text(text_box.name or "", retry_patch)  # 解析当前牌点数。
                        if card is not None:  # 成功后不读取右侧相邻牌。
                            break  # 保留当前左侧牌结果。
                    if card is not None:  # 当前宽度已经补识别成功。
                        break  # 不再继续扩大区域。
            if card is None:  # 任意一张缺失都会改变牌型长度和合法响应集合。
                return recognized if len(recognized) == len(slots) else []  # 只有通用 OCR 数量也完整时才保留，否则判整组失败。
            slot_results.append((card, slot))  # 保存完整逐张牌组。
        return slot_results  # 使用数量经过竖边校验的桌面牌组，避免部分 OCR 被当成单牌。

    def _recognize_skill_option_cards(self, frame, slots):  # 逐张识别诸葛均技能弹窗中的大卡片并保留实体卡片点击框。
        recognized = []  # 保存点数与完整卡片框的对应关系。
        for slot in slots:  # 每张底牌独立 OCR，避免三张牌文字框坐标相互污染。
            patch = frame[slot.y:slot.y + slot.height, slot.x:slot.x + slot.width]  # 裁出当前完整卡片供王字颜色解析。
            text_boxes = self.ocr(box=slot, frame=frame, threshold=max(0.10, self.config.get("Card OCR Threshold", 0.25) * 0.60), log=False)  # 大卡片点数清晰，使用适度低阈值兼容窄体 J。
            card = next((parse_card_text(text_box.name or "", patch) for text_box in sorted(text_boxes, key=lambda item: (item.x, item.y)) if parse_card_text(text_box.name or "", patch) is not None), None)  # 只读取当前卡片最靠左上的合法点数。
            if card is None:  # 单张漏识别时不能安全判断三张底牌中的最大牌。
                continue  # 保留其他识别结果用于日志，但禁止后续提交。
            recognized.append((card, slot))  # 点数用于决策，完整实体卡片框用于可靠点击中心。
        return recognized  # 返回按屏幕从左到右排列的技能选项。

    def _recognize_hand_cards(self, frame, region):  # 对重叠手牌逐张裁切并识别左上角点数。
        hand_image = frame[region.y:region.y + region.height, region.x:region.x + region.width]  # 从当前帧裁出完整手牌区域。
        slots = detect_hand_slots(hand_image)  # 根据竖边估算当前实际手牌张数和每张牌位置。
        recognized = self._ocr_detected_hand_slots(frame, region, slots)  # 先执行不依赖历史张数的常规完整识别。
        expected_count = getattr(self, "expected_hand_count", None)  # 读取上一手成功提交后可证明的正常剩余张数。
        expected_cards = None  # 初始化扣除上一手后的牌面组成基线。
        ledger = getattr(self, "card_ledger", None)  # 读取本局牌账本，技能新增牌不能被残局消歧静默丢弃。
        if ledger is not None and ledger.last_hand is not None:  # 有上一份完整手牌时才能建立严格点数基线。
            expected_cards = Counter(ledger.last_hand)  # 复制上一手完整牌面组成。
            expected_cards.subtract(getattr(ledger, "pending_play", []))  # 扣除已经确认提交的普通出牌。
            expected_cards = Counter({card: count for card, count in expected_cards.items() if count > 0})  # 清除扣牌后的零数量点数。
        if recognized and not (isinstance(expected_count, int) and 1 <= expected_count <= 4 and len(recognized) != expected_count):  # 数量符合预期时保留常规识别及技能增牌结果。
            return recognized  # 不使用旧牌账本强行截断已经完整识别的新手牌。
        if isinstance(expected_count, int) and 1 <= expected_count <= 4:  # 对四张以内残局启用张数消歧，覆盖王炸加两张单牌。
            fallback_slots = detect_hand_slots(hand_image, expected_count=expected_count)  # 从全部候选边链中选择张数和末牌右边框一致的方案。
            if fallback_slots != slots:  # 只有消歧产生不同牌位时才值得再次调用 OCR。
                recovered = self._ocr_detected_hand_slots(frame, region, fallback_slots)  # 按真实三张牌框重新识别大王、小王和单牌。
                recovered_cards = Counter(card for card, _ in recovered)  # 统计消歧后的实体点数，防止技能牌被静默截断。
                baseline_matches = expected_cards is None or recovered_cards == expected_cards  # 有牌账本时还必须与扣除普通出牌后的组成完全一致。
                if len(recovered) == expected_count and baseline_matches:  # 必须同时满足张数和组成才能交给出牌算法。
                    return recovered  # 返回可安全点击的全部残局实体牌。
        return []  # 常规和受限残局消歧均失败时继续安全等待。

    def _ocr_detected_hand_slots(self, frame, region, slots):  # 对已经确定的手牌槽逐张 OCR，供常规分割与残局消歧共用。
        recognized = []  # 初始化逐张识别结果。
        scan_height = min(region.height, max(140, int(round(region.height * 0.82))))  # 覆盖竖排 JOKER 全文并提高边缘牌 OCR 稳定性。
        for relative_x, slot_width in slots:  # 逐个处理分割后的卡牌位置。
            slot = Box(region.x + relative_x, region.y, slot_width, scan_height, name="hand_card")  # 创建当前单牌的绝对屏幕 OCR 区域。
            patch = frame[slot.y:slot.y + slot.height, slot.x:slot.x + slot.width]  # 裁出当前牌面用于大小王颜色判断。
            boxes = self.ocr(box=slot, frame=frame, threshold=self.config.get("Card OCR Threshold", 0.25), log=False)  # 对单张卡牌独立执行 OCR 避免相邻牌合并。
            card = None  # 初始化当前单牌尚未识别的点数。
            for text_box in sorted(boxes, key=lambda item: (item.x, item.y)):  # 从左到右检查当前单牌 OCR 返回的所有文本块。
                card = parse_card_text(text_box.name or "", patch, detect_wildcard=True)  # 所有手牌张数都识别橙色万能牌实体，避免普通阶段误当作十。
                if card is not None:  # 找到首个合法点数后停止读取花色文本。
                    break  # 避免花色误识别覆盖已经确定的点数。
            if card is None:  # 首次 OCR 可能因窄框、边缘牌或短暂低置信度漏字。
                full_card_width = max(slot.width, int(round(region.width * 208 / 1621)))  # 估算当前分辨率下一张完整手牌宽度。
                retry_widths = (slot.width + 12, slot.width + 32, full_card_width)  # OCR 对十等宽字符的裁切宽度敏感，按小、中、完整牌宽依次补识别。
                for requested_width in retry_widths:  # 最多执行三次局部补识别。
                    retry_width = min(region.x + region.width - slot.x, requested_width)  # 限制不越过手牌区域右边界。
                    retry_slot = Box(slot.x, slot.y, retry_width, scan_height, name="hand_card_retry")  # 创建低阈值补识别框。
                    retry_patch = frame[retry_slot.y:retry_slot.y + retry_slot.height, retry_slot.x:retry_slot.x + retry_slot.width]  # 裁出补识别画面。
                    retry_boxes = self.ocr(box=retry_slot, frame=frame, threshold=max(0.10, self.config.get("Card OCR Threshold", 0.25) * 0.60), log=False)  # 仅对失败牌降低阈值。
                    for text_box in sorted(retry_boxes, key=lambda item: (item.x, item.y)):  # 优先读取当前牌左侧点数，忽略扩框后可能出现的相邻牌。
                        card = parse_card_text(text_box.name or "", retry_patch, detect_wildcard=True)  # 补识别同样保留普通阶段万能牌实体编码。
                        if card is not None:  # 补识别成功后停止。
                            break  # 保留当前牌点数。
                    if card is not None:  # 当前宽度已经成功补出点数。
                        break  # 不再继续扩展 OCR 区域。
            if card is None:  # 任意一张手牌无法识别都会使模型状态不完整。
                return []  # 整体判定识牌失败，禁止使用残缺手牌推理。
            recognized.append((card, slot))  # 保存点数和以牌面顶部为中心的安全点击区域。
        return recognized  # 返回数量完整且按屏幕顺序排列的手牌。

    def _boxes_for_action(self, action, card_boxes):  # 为模型动作选择数量完全一致的屏幕牌框。
        remaining = Counter(action)  # 统计每种点数还需要选择的张数。
        selected = []  # 初始化最终点击框列表。
        for card, box in card_boxes:  # 按屏幕顺序扫描当前完整手牌。
            if remaining[card] <= 0:  # 当前点数已经选够时跳过重复牌。
                continue  # 继续检查下一张手牌。
            selected.append(box)  # 保存本张目标牌的点击框。
            remaining[card] -= 1  # 扣除一张已经映射成功的目标牌。
        missing = {card: count for card, count in remaining.items() if count > 0}  # 汇总未能映射到屏幕的模型动作。
        if missing:  # OCR 框与已校验手牌意外不一致时停止执行。
            raise AiModelError(f"无法在屏幕上定位 AI 动作: {missing}")  # 防止只选择模型动作的一部分。
        return selected  # 返回可以依次点击的完整动作框。

    def _resolve_position(self, hand_count, frame=None):  # 解析当前玩家在模型中的三人位置。
        configured = self.config.get("AI Position", "auto")  # 读取用户明确指定的位置。
        if configured in {"landlord", "landlord_up", "landlord_down"}:  # 检查配置是否为合法 DouZero 位置。
            return configured  # 优先使用用户确定的身份方向。
        if self.resolved_ai_position is None:  # 自动身份只允许在本局第一次完整识牌时判断。
            regions = {}  # 收集当前素材中三个固定身份区域。
            if frame is not None:  # 只有提供当前帧时才能执行颜色身份识别。
                for name in IDENTITY_SEAT_FEATURES.values():  # 按图片已经配置的自己、下家和下下家读取身份区。
                    if self.feature_exists(name):  # 跳过旧素材中不存在的区域。
                        try:  # 防止类别存在但坐标尚未加载。
                            regions[name] = self.get_box_by_name(name)  # 获取当前分辨率适配后的身份框。
                        except ValueError:  # 单个区域失败不影响其他区域。
                            continue  # 继续检查其余身份框。
            detected = classify_identity_regions(frame, regions) if regions else None  # 优先按“主/农”颜色与座位关系识别完整位置。
            self.resolved_ai_position = detected or ("landlord" if hand_count >= 20 else "landlord_down")  # 动画遮挡时才使用旧手牌数回退。
            self._record_event("identity_detected", position=self.resolved_ai_position, source="identity_regions" if detected else "hand_count_fallback")  # 记录身份来源供胜率和识别诊断。
        return self.resolved_ai_position  # 技能回收导致手牌超过二十张时仍保持原有身份。

    def _read_opponent_counts(self, frame):  # OCR 读取左右两名对手的剩余牌数。
        counts = []  # 初始化左右对手牌数列表。
        previous = list(getattr(self, "last_opponent_card_counts", None) or [])  # OCR 暂时缺失时优先沿用上一可靠观察而不是重置到十七。
        for seat_index, region_name in enumerate(("opponent_left_card_count", "opponent_right_card_count")):  # 按左、右固定顺序读取标注区域。
            fallback = int(previous[seat_index]) if seat_index < len(previous) else 17  # 本局首帧才使用开局默认值。
            if not self.feature_exists(region_name):  # 缺少对应标注时使用安全默认值。
                counts.append(fallback)  # 已进入牌局时保持上一可靠牌数，避免策略紧迫度回跳。
                continue  # 继续读取另一侧对手。
            boxes = self.ocr(box=self.get_box_by_name(region_name), frame=frame, match=re.compile(r"\d{1,2}"), threshold=0.2, log=False)  # OCR 识别一到两位剩余牌数。
            digits = re.sub(r"\D", "", boxes[0].name) if boxes else ""  # 从首个 OCR 结果提取纯数字。
            counts.append(int(digits) if digits and 0 < int(digits) <= MAX_SKILL_HAND_CARDS else fallback)  # 英雄技能可能令对手超过二十张；无效结果保留时序基线。
        return counts  # 返回模型所需的左右对手牌数。
