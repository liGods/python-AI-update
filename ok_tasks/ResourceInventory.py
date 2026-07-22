from dataclasses import dataclass  # 导入数据类以定义稳定的素材需求记录。

from ok_tasks.card_ai.heroes import iter_unverified_skills  # 将规则文字不完整的英雄技能自动加入素材缺口。


@dataclass(frozen=True)  # 素材需求在运行期间不应被意外修改。
class ResourceRequirement:  # 描述一个必须采集和标注的界面资源。
    key: str  # 保存机器可读的唯一资源键。
    title: str  # 保存用户界面显示的中文名称。
    hero: str | None = None  # 保存资源所属英雄或通用空值。
    feature_names: tuple[str, ...] = ()  # 保存能够证明资源已标注的模板名称。
    trigger: str = "unknown"  # 保存采集器应该关注的触发阶段。
    minimum_samples: int = 3  # 保存建议收集的不同画面数量。


REQUIRED_RESOURCES = (  # 定义当前十四名英雄优先补齐的资源清单。
    ResourceRequirement("xiahou_dun_ganglie_discard", "夏侯惇·刚烈弃三张最小牌", "夏侯惇", ("skill_xiahou_dun_ganglie_discard",), "after_skill_confirm"),  # 定义刚烈弃牌界面需求。
    ResourceRequirement("guan_yu_wusheng_pick", "关羽·武圣选取顺子牌", "关羽", ("skill_card_pool", "skill_guan_yu_wusheng_pick"), "after_skill_confirm"),  # 通用获取牌库区域已经覆盖武圣选牌布局。
    ResourceRequirement("xu_sheng_yicheng_discard", "徐盛·疑城获得两张后弃一张", "徐盛", ("skill_xu_sheng_yicheng_discard",), "no_legal_response"),  # 定义疑城弃牌界面需求。
    ResourceRequirement("zhuge_jun_gengdu_copy", "诸葛均·耕读复制底牌", "诸葛均", ("skill_card_pool", "skill_zhuge_jun_gengdu_copy"), "game_start"),  # 通用获取牌库区域已经覆盖耕读复制布局。
    ResourceRequirement("zhuge_jun_gengdu_discard", "诸葛均·耕读出复制牌后弃牌", "诸葛均", ("skill_zhuge_jun_gengdu_discard",), "after_skill_confirm"),  # 定义耕读弃牌界面需求。
    ResourceRequirement("ling_tong_yongjin_discard", "凌统·勇进弃单牌或对子", "凌统", ("skill_ling_tong_yongjin_discard",), "after_skill_confirm"),  # 定义勇进弃牌界面需求。
    ResourceRequirement("lu_zhi_ruzong_convert", "卢植·儒宗单牌对子转换", "卢植", ("skill_lu_zhi_ruzong_convert",), "after_skill_confirm"),  # 定义儒宗转换界面需求。
    ResourceRequirement("identity_landlord", "地主身份标志", None, ("Identity Mark No. 1",), "in_match", 5),  # 现有第一身份区包含我方金色“主”图标。
    ResourceRequirement("identity_farmer", "农民身份标志", None, ("Identity Mark No. 2", "Identity Mark No. 3"), "in_match", 5),  # 现有左右身份区包含蓝色“农”图标。
    ResourceRequirement("result_loss_replay", "失败结算·再来一局按钮", None, ("Play another round", "result_loss_variant"), "result_loss", 3),  # 复用已标注结算按钮和失败皮肤，运行时另有OCR兜底。
    ResourceRequirement("small_hand_layout", "剩余一至四张手牌布局", None, ("Deck of cards", "Selected area"), "our_turn", 8),  # 少量手牌由动态竖边分割处理，无需固定张数模板。
    ResourceRequirement("active_play_variants", "不同布局的黄色可用出牌按钮", None, ("Confirming the Play", "play_card_layout_a", "play_card_layout_b"), "our_turn", 8),  # 定义出牌按钮多样本需求。
)  # 完成优先素材清单。


def missing_resource_keys(feature_exists):  # 根据当前 FeatureSet 计算尚未标注的资源键。
    missing = []  # 初始化缺失资源键列表。
    for requirement in REQUIRED_RESOURCES:  # 逐项检查所有优先资源。
        if not requirement.feature_names:  # 没有关联模板的资源始终需要人工确认。
            missing.append(requirement.key)  # 将无法自动验证的资源加入缺失列表。
            continue  # 继续检查下一项资源。
        if not any(feature_exists(name) for name in requirement.feature_names):  # 任一关联模板均不存在时判定缺失。
            missing.append(requirement.key)  # 将缺失资源键加入结果。
    for skill in iter_unverified_skills():  # 扫描全部英雄注册表中仍缺权威规则文字的技能。
        key = f"hero_rule_{skill.hero}_{skill.name}"  # 生成可直接关联截图和人工确认的稳定资源键。
        if key not in missing:  # 防止未来显式素材项与自动规则项重复。
            missing.append(key)  # 未验证技能不能进入稳定实战模型。
    return missing  # 返回稳定顺序的资源缺口列表。


def requirement_for_trigger(trigger, hero=None):  # 查找当前界面阶段应该采集的资源需求。
    return [requirement for requirement in REQUIRED_RESOURCES if requirement.trigger == trigger and (requirement.hero is None or requirement.hero == hero)]  # 返回通用和当前英雄匹配项。
