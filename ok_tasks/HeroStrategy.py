import json  # 保存和读取每名武将的持久化胜负统计。
from collections import Counter  # 根据动作点数数量结构识别牌型。
from dataclasses import dataclass  # 使用不可变配置对象描述已支持武将。
from pathlib import Path  # 安全处理用户配置的统计文件路径。


@dataclass(frozen=True)  # 武将配置在运行期间不应被意外修改。
class HeroProfile:  # 描述一个已拥有武将的自动化能力和策略标签。
    name: str  # 保存游戏界面显示的规范武将名称。
    strategy: str  # 保存供规则模型使用的策略标签。
    interactive_skill: bool = False  # 标记技能是否还需要二级选牌或目标交互。


OWNED_HERO_PROFILES = (  # 只注册用户当前账号已经拥有的二十四名武将。
    HeroProfile("甄姬", "global_observer"),  # 被动观察全场牌面并自动结算洛神相关收益。
    HeroProfile("典韦", "keep_playing"),  # 连续出牌可积累不屈层数，普通回合尽量不断档。
    HeroProfile("夏侯惇", "largest_bait", True),  # 刚烈需要在被压后选择弃牌并收回牌。
    HeroProfile("关羽", "prefer_straight", True),  # 顺子触发单骑，武圣需要从对手顺子中选牌。
    HeroProfile("庞统", "global_observer"),  # 自动触发技能按完整结算后的牌路参与决策。
    HeroProfile("姜维", "strategic_pass"),  # 北伐让合理的不出等价于弃置最小牌。
    HeroProfile("张飞", "repeat_type"),  # 连续两次相同牌型可以触发咆哮。
    HeroProfile("赵云", "low_single_pair"),  # 低于 A 的单牌或对子被压后可通过冲阵成长。
    HeroProfile("吕蒙", "skill_card_choice", True),  # 勤学需要在技能界面选择牌或取消发动。
    HeroProfile("孙坚", "special_legal_action", True),  # 主动技能和特殊合法动作需要界面交互确认。
    HeroProfile("小乔", "skill_card_choice", True),  # 星华、巧笑需要选牌或目标交互。
    HeroProfile("徐盛", "no_legal_skill", True),  # 无牌可压时疑城需要抽牌后选择弃牌。
    HeroProfile("陆逊", "prefer_multi_card"),  # 每次出牌后手牌会变化，优先减少更多实体牌。
    HeroProfile("董卓", "target_choice", True),  # 暴政需要根据阵营收益选择目标或安全取消。
    HeroProfile("貂蝉", "target_choice", True),  # 魅惑需要根据敌友关系选择目标或安全取消。
    HeroProfile("曹洪", "pair_observer"),  # 其他玩家打出对子后可能自动获得对子。
    HeroProfile("关银屏", "prefer_large_combo"),  # 四张以上牌型可以触发花武获得人头牌。
    HeroProfile("诸葛均", "copied_card", True),  # 耕读复制牌打出后需要选择弃牌。
    HeroProfile("凌统", "prefer_single_pair", True),  # 单牌和对子可分别联动弃置对子或单牌。
    HeroProfile("卢植", "prefer_single_pair_follow", True),  # 压牌后可以在单牌和对子之间转换。
    HeroProfile("张宝", "skill_card_choice", True),  # 黄符涉及独立牌区和选牌交互。
    HeroProfile("皇甫嵩", "low_card_observer"),  # 全场三四牌型触发平乱并获得新牌。
    HeroProfile("朱儁", "global_observer"),  # 自动触发技能按公开历史投影后续牌路。
    HeroProfile("刘虞", "global_observer"),  # 自动触发技能按完整结算后的牌路参与决策。
)  # 完成当前账号武将配置。


OWNED_HEROES = tuple(dict.fromkeys(profile.name for profile in OWNED_HERO_PROFILES))  # 去重后导出稳定的账号武将顺序。
HERO_PROFILE_BY_NAME = {profile.name: profile for profile in OWNED_HERO_PROFILES}  # 为运行时查询建立名称索引。
HERO_ALIASES = {  # 兼容 OCR 和用户描述中容易出现的同音或形近字。
    "夏侯惊": "夏侯惇",  # 将之前技能说明中的误写统一为夏侯惇。
    "夏侯敦": "夏侯惇",  # 兼容 OCR 把惇识别成敦。
    "皇甫高": "皇甫嵩",  # 兼容嵩字在小尺寸竖排文字中的常见误识别。
    "朱售": "朱儁",  # 兼容儁字在小尺寸武将名中的常见 OCR 结果。
    "诸葛钧": "诸葛均",  # 兼容均字的常见同音 OCR 结果。
}  # 完成武将名称别名表。
CARD_ORDER = ("3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2", "X", "D")  # 定义英雄状态使用的点数顺序。


def normalize_hero_name(value):  # 从 OCR 长文本中提取当前账号支持的规范武将名。
    text = "".join(str(value or "").split())  # 去掉空格和换行以兼容竖排武将名。
    for alias, canonical in HERO_ALIASES.items():  # 优先替换已知 OCR 别名。
        if alias in text:  # 当前文本包含完整别名时确认匹配。
            return canonical  # 返回账号武将的规范名称。
    for hero in OWNED_HEROES:  # 再检查所有规范名称是否直接出现在 OCR 文本中。
        if hero in text:  # 只有完整武将名命中才接受识别。
            return hero  # 返回稳定的规范名称供统计和策略使用。
    return None  # 无法可靠识别时拒绝猜测武将。


def classify_action(cards):  # 将已经成功打出的内部点数列表归一成英雄连续牌型状态。
    values = [str(card) for card in cards]  # 复制动作以避免修改调用方历史。
    if len(values) == 2 and values.count("W") == 1 and any(card in CARD_ORDER[:-2] for card in values):  # 万能牌与一张非王自然牌按游戏规则组成对子。
        return "pair"  # 让张飞、凌统和牌局日志使用真实生效牌型而不是其他类型。
    counts = Counter(values)  # 统计每个点数的张数结构。
    groups = sorted(counts.values())  # 生成与具体点数无关的牌型结构。
    if len(values) == 1:  # 单张动作直接分类。
        return "solo"  # 返回与 RLCard 一致的单牌名称。
    if len(values) == 2 and (groups == [2] or set(values) == {"X", "D"}):  # 检查对子或王炸。
        return "rocket" if set(values) == {"X", "D"} else "pair"  # 区分王炸和普通对子。
    if len(values) == 3 and groups == [3]:  # 检查纯三张。
        return "trio"  # 返回三张分类。
    if len(values) == 4 and groups == [1, 3]:  # 检查三带一。
        return "trio_solo"  # 返回三带一分类。
    if len(values) == 4 and groups == [4]:  # 检查普通炸弹。
        return "bomb"  # 返回炸弹分类。
    if len(values) == 5 and groups == [5]:  # 检查技能增牌形成的五张同点数炸弹。
        return "five_bomb"  # 五炸高于王炸，必须与普通三带二分开记录。
    if len(values) == 5 and groups == [2, 3]:  # 检查三带二。
        return "trio_pair"  # 返回三带二分类。
    indexes = sorted(CARD_ORDER.index(card) for card in counts if card in CARD_ORDER[:12])  # 读取三到 A 的连续序号。
    consecutive = len(indexes) == len(counts) and all(right == left + 1 for left, right in zip(indexes, indexes[1:]))  # 判断所有点数是否连续。
    if len(values) >= 5 and groups == [1] * len(values) and consecutive:  # 检查顺子。
        return "straight"  # 返回关羽和张飞共用的顺子分类。
    if len(values) >= 6 and len(values) % 2 == 0 and groups == [2] * len(counts) and consecutive:  # 检查三连对以上结构。
        return "pair_chain"  # 返回统一连对分类。
    if len(values) >= 6 and all(count >= 3 for count in counts.values()):  # 粗略识别纯飞机和带牌飞机。
        return "airplane"  # 返回统一飞机分类供连续牌型比较。
    return "other"  # 无法稳定细分的扩展牌型使用其他分类。


class HeroStatistics:  # 管理逐武将总胜率及地主、农民分项统计。
    def __init__(self, path):  # 保存统计路径并加载已有数据。
        self.path = Path(path)  # 将配置值转换成可创建父目录的路径对象。
        self.data = self._load()  # 读取现有统计或创建空结构。

    def _load(self):  # 容错读取持久化 JSON。
        if not self.path.is_file():  # 首次运行还没有统计文件时返回空字典。
            return {}  # 不提前写盘，等真实结算后再创建文件。
        try:  # 捕获用户手工编辑或异常退出导致的损坏 JSON。
            value = json.loads(self.path.read_text(encoding="utf-8"))  # 使用 UTF-8 读取中文武将名称。
        except (OSError, ValueError):  # 文件不可读或 JSON 无法解析时安全忽略旧内容。
            return {}  # 保持自动打牌可继续运行而不覆盖损坏文件。
        return value if isinstance(value, dict) else {}  # 只接受以武将名为键的对象结构。

    @staticmethod  # 分项统计不依赖实例状态。
    def _empty_bucket():  # 创建一组新的胜负计数器。
        return {"games": 0, "wins": 0, "losses": 0}  # 同时保存总局数便于直接展示。

    def _hero_record(self, hero):  # 获取并修复指定武将的统计结构。
        record = self.data.setdefault(hero, {})  # 为首次出场武将创建记录。
        for bucket in ("overall", "landlord", "farmer"):  # 确保总计和两种身份分项都存在。
            current = record.get(bucket)  # 读取可能来自旧版本的分项值。
            if not isinstance(current, dict):  # 非对象值不能安全累加。
                record[bucket] = self._empty_bucket()  # 使用完整空结构替换异常值。
            else:  # 已有对象时补齐缺失字段。
                for key, default in self._empty_bucket().items():  # 遍历三个稳定计数字段。
                    current.setdefault(key, default)  # 保留已有数据并补齐新字段。
        return record  # 返回可以直接累加的完整记录。

    def games(self, hero):  # 返回武将当前已经完成的有效对局数。
        return int(self._hero_record(hero)["overall"]["games"])  # 从总计分项读取稳定整数。

    def smoothed_win_rate(self, hero):  # 使用轻量先验避免一两局偶然胜负占据首位。
        bucket = self._hero_record(hero)["overall"]  # 读取该武将总胜负。
        return (float(bucket["wins"]) + 1.0) / (float(bucket["games"]) + 2.0)  # 使用 Beta(1,1) 平滑胜率。

    def choose(self, candidates, exploration_games=10):  # 在三个已识别候选中选择探索不足或胜率最高者。
        available = [hero for hero in candidates if hero in HERO_PROFILE_BY_NAME]  # 过滤未拥有或未实现的武将。
        if not available:  # 没有可靠候选时交由界面中间位回退。
            return None  # 返回空值表示不猜测 OCR 失败的英雄。
        unexplored = [hero for hero in available if self.games(hero) < exploration_games]  # 查找尚未达到冷启动局数的武将。
        if unexplored:  # 冷启动阶段优先补齐样本最少的候选。
            return min(unexplored, key=lambda hero: (self.games(hero), available.index(hero)))  # 同局数按左中右稳定选择。
        return min(available, key=lambda hero: (-self.smoothed_win_rate(hero), self.games(hero), available.index(hero)))  # 成熟阶段选平滑胜率最高者。

    def record(self, hero, won, position):  # 在真实胜负结算时累加一局。
        record = self._hero_record(hero)  # 获取完整的武将统计结构。
        role_bucket = "landlord" if position == "landlord" else "farmer"  # 将两个农民位置合并为农民分项。
        for bucket_name in ("overall", role_bucket):  # 同时更新总计和本局身份分项。
            bucket = record[bucket_name]  # 读取准备更新的计数器。
            bucket["games"] = int(bucket["games"]) + 1  # 有效结算局数增加一。
            result_key = "wins" if won else "losses"  # 根据结算状态选择胜或负字段。
            bucket[result_key] = int(bucket[result_key]) + 1  # 对应结果计数增加一。
        self.path.parent.mkdir(parents=True, exist_ok=True)  # 首次结算时创建统计目录。
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")  # 使用同目录临时文件避免中途写坏主文件。
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")  # 写入可读中文 JSON。
        temporary.replace(self.path)  # 原子替换主统计文件完成持久化。
