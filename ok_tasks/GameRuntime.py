from collections import Counter  # 导入计数器以记录牌面数量变化。
from dataclasses import asdict, dataclass, field  # 导入数据类工具以构造可序列化牌局状态。
from typing import Any  # 导入通用类型以保存扩展技能数据。


@dataclass  # 将一次候选动作定义为稳定的数据对象。
class ActionCandidate:  # 描述规则模型评估的一手合法候选牌。
    cards: list[str]  # 保存候选动作包含的完整点数列表。
    action_type: str  # 保存规范化后的牌型名称。
    score: list[Any]  # 保存可解释的逐项评分结果。
    reason: str = ""  # 保存候选动作的简短选择理由。
    legal: bool = True  # 标记动作是否通过合法动作枚举。

    def to_dict(self):  # 将候选动作转换成 JSON 可写对象。
        return asdict(self)  # 返回数据类全部公开字段。


@dataclass  # 将英雄局内变化定义为独立状态。
class HeroRuntimeState:  # 保存英雄技能需要跨回合维护的数据。
    hero: str | None = None  # 保存当前英雄规范名称。
    last_action_type: str | None = None  # 保存上一手成功牌型供连续技能判断。
    skill_uses: dict[str, int] = field(default_factory=dict)  # 保存各技能已经使用的次数。
    marks: dict[str, int] = field(default_factory=dict)  # 保存不屈等可累计技能标记。
    pending_interaction: str | None = None  # 保存尚未完成的二级技能交互名称。
    extra: dict[str, Any] = field(default_factory=dict)  # 保存后续英雄专属扩展状态。

    def to_dict(self):  # 将英雄状态转换成 JSON 可写对象。
        return asdict(self)  # 返回数据类全部公开字段。


@dataclass  # 将一次牌面变化定义为牌账本事件。
class CardLedgerEvent:  # 记录技能和正常出牌造成的牌数量变化。
    sequence: int  # 保存事件在本局中的连续编号。
    event_type: str  # 保存 play、gain、discard、recover 或 transform 等事件类型。
    cards: list[str]  # 保存事件涉及的牌点数。
    source: str  # 保存 normal、hero_skill、opponent 或 unknown 等来源。
    metadata: dict[str, Any] = field(default_factory=dict)  # 保存英雄、动作和数量差等附加信息。

    def to_dict(self):  # 将账本事件转换成 JSON 可写对象。
        return asdict(self)  # 返回数据类全部公开字段。


class CardLedger:  # 维护允许技能复制和回收重复牌的可见牌账本。
    def __init__(self):  # 初始化空牌账本。
        self.events: list[CardLedgerEvent] = []  # 保存按时间顺序排列的全部事件。
        self.last_hand: list[str] | None = None  # 保存上一次完整识别到的我方手牌。
        self.pending_play: list[str] = []  # 保存上次已确认出牌，下一次观察时先从正常预期中扣除。

    def append(self, event_type, cards, source="normal", **metadata):  # 追加一个可追踪的牌面事件。
        event = CardLedgerEvent(len(self.events) + 1, event_type, list(cards), source, dict(metadata))  # 构造带连续编号的新事件。
        self.events.append(event)  # 将新事件写入本局账本。
        return event  # 返回新事件供逐局日志直接记录。

    def observe_hand(self, cards, expected_count=None, hero=None):  # 比较两次完整手牌并推断技能加牌或弃牌。
        current = list(cards)  # 复制当前手牌避免修改识别结果。
        changes = []  # 初始化本次观察产生的账本事件列表。
        if self.last_hand is not None:  # 只有存在上一份完整手牌时才能比较点数差异。
            previous_counts = Counter(self.last_hand)  # 统计上一份手牌点数数量。
            previous_counts.subtract(self.pending_play)  # 正常出牌不应再次被误记为英雄技能弃牌。
            previous_counts = Counter({card: count for card, count in previous_counts.items() if count > 0})  # 清除扣牌后的零值和异常负值。
            current_counts = Counter(current)  # 统计当前手牌点数数量。
            gained = list((current_counts - previous_counts).elements())  # 提取当前新增的具体牌点数。
            lost = list((previous_counts - current_counts).elements())  # 提取当前减少的具体牌点数。
            if gained:  # 技能或回收使手牌出现新增点数时记录事件。
                changes.append(self.append("gain", gained, "hero_skill", hero=hero, expected_count=expected_count))  # 记录技能加牌且不限制标准牌库数量。
            if lost:  # 扣除正常出牌后仍减少的牌均属于技能弃牌或变点数牌，必须保留来源。
                changes.append(self.append("discard", lost, "hero_skill", hero=hero, expected_count=expected_count))  # 记录技能额外弃牌事件。
        self.last_hand = current  # 更新下一回合比较所需的完整手牌快照。
        self.pending_play = []  # 当前观察已经消化上一手正常出牌。
        return changes  # 返回全部变化供调用方写入事件日志。

    def record_play(self, cards, hero=None, action_type=None):  # 记录我方已经确认提交的一手牌。
        self.pending_play = list(cards)  # 记录下一次手牌观察应先扣除的正常出牌。
        return self.append("play", cards, "normal", hero=hero, action_type=action_type)  # 将成功动作作为普通出牌事件追加。

    def to_list(self):  # 将完整账本转换为 JSON 可写列表。
        return [event.to_dict() for event in self.events]  # 按原始顺序输出每个账本事件。


@dataclass  # 将模型可见牌局状态定义为稳定接口。
class GameState:  # 保存一次决策所需的完整可见信息。
    hand_cards: list[str] = field(default_factory=list)  # 保存当前完整手牌。
    table_cards: list[str] = field(default_factory=list)  # 保存需要响应的桌面牌组。
    position: str = "landlord_down"  # 保存地主、地主上家或地主下家位置。
    opponent_card_counts: list[int] = field(default_factory=lambda: [17, 17])  # 保存左右对手剩余牌数。
    opponent_skill_card_estimates: list[int] = field(default_factory=lambda: [0, 0])  # 保存从牌数回升中确认仍可能留在对手手里的技能增牌数量。
    enemy_card_counts: list[int] = field(default_factory=lambda: [17, 17])  # 保存按身份解析后真正敌方玩家的剩余牌数。
    teammate_card_count: int | None = None  # 农民身份下保存队友剩余牌数，地主时为空。
    table_player: str | None = None  # 保存最近有效桌面动作来自下家还是下下家。
    table_is_teammate: bool = False  # 标记最近桌面动作是否来自农民队友。
    history: list[list[str]] = field(default_factory=list)  # 保存本局观察到的有效动作历史。
    hero_state: HeroRuntimeState = field(default_factory=HeroRuntimeState)  # 保存当前英雄跨回合状态。
    round_id: str = ""  # 保存本次我方操作回合的唯一编号。
    policy_id: str = "balanced"  # 保存整局使用的策略版本。

    def to_model_state(self):  # 转换成兼容现有模型适配器的字典。
        return {  # 返回包含旧接口和新增上下文的稳定状态对象。
            "hand_cards": list(self.hand_cards),  # 复制当前完整手牌。
            "table_cards": list(self.table_cards),  # 复制当前桌面牌组。
            "position": self.position,  # 传递当前身份位置。
            "opponent_card_counts": list(self.opponent_card_counts),  # 复制左右对手剩余牌数。
            "opponent_skill_card_estimates": list(self.opponent_skill_card_estimates),  # 传递敌方和队友技能增牌不确定性供记牌与搜索使用。
            "enemy_card_counts": list(self.enemy_card_counts),  # 只传递真正敌方牌数供残局阻断判断。
            "teammate_card_count": self.teammate_card_count,  # 传递队友牌数供农民协作放行。
            "table_player": self.table_player,  # 传递最近动作座位避免从无归属历史猜测。
            "table_is_teammate": self.table_is_teammate,  # 传递桌面动作阵营关系。
            "history": [list(action) for action in self.history],  # 深复制历史动作避免模型意外修改。
            "hero": self.hero_state.hero,  # 保持现有英雄字段兼容性。
            "hero_state": self.hero_state.to_dict(),  # 传递完整英雄技能状态。
            "round_id": self.round_id,  # 传递回合唯一编号供日志和防重使用。
            "policy_id": self.policy_id,  # 传递本局策略编号供规则模型选择评分配置。
        }  # 完成模型状态构造。
