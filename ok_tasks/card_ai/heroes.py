from __future__ import annotations

from collections.abc import Iterable

from ok_tasks.card_ai.schema import HeroSkillSpec


RULES_VERSION = "3p.1"
SUPPORTED_MODES = ("landlord_3p",)
UI_VERIFIED_SKILLS = frozenset(
    {
        ("夏侯惇", "刚烈"),
        ("关羽", "武圣"),
        ("徐盛", "疑城"),
        ("诸葛均", "耕读"),
        ("凌统", "勇进"),
        ("卢植", "儒宗"),
    }
)


OWNED_HEROES = (
    "甄姬",
    "典韦",
    "夏侯惇",
    "关羽",
    "庞统",
    "姜维",
    "张飞",
    "赵云",
    "吕蒙",
    "孙坚",
    "小乔",
    "徐盛",
    "陆逊",
    "董卓",
    "貂蝉",
    "曹洪",
    "关银屏",
    "诸葛均",
    "凌统",
    "卢植",
    "张宝",
    "皇甫嵩",
    "朱儁",
    "刘虞",
)
T0_HEROES = ("陆逊", "甘宁", "大乔")  # 用户确认的实战强度榜顺序；选将时仍需同时满足账号拥有和自动化安全条件。
T1_HEROES = ("刘禅", "诸葛亮", "夏侯惇", "吕蒙", "孙权", "董承", "南华老仙")  # 用户强度榜第二档，供安全候选内的冷启动优先级与专项评测使用。
SIMULATED_HEROES = (  # 仅保留已经接入权威状态机的武将，不能因账号新增拥有角色而自动提升验证状态。
    "典韦",
    "夏侯惇",
    "关羽",
    "张飞",
    "赵云",
    "徐盛",
    "陆逊",
    "甘宁",
    "大乔",
    "姜维",
    "曹洪",
    "关银屏",
    "诸葛均",
    "凌统",
    "卢植",
    "皇甫嵩",
)

HERO_ALIASES = {
    "夏侯惊": "夏侯惇",
    "夏侯敦": "夏侯惇",
    "曹不": "曹丕",
    "羊枯": "羊祜",
    "朱售": "朱儁",
    "诸葛钧": "诸葛均",
    "皇甫高": "皇甫嵩",
}

SKILL_CATEGORY_LABELS = {
    "active": "主动",
    "interactive": "交互",
    "passive": "被动",
}


def classify_skill_category(trigger: str, interactive: bool) -> str:
    """Classify one skill into exactly one UI decision category."""
    if trigger.startswith("active"):
        return "active"
    if interactive:
        return "interactive"
    return "passive"


def _skill(
    hero: str,
    name: str,
    trigger: str,
    limit: int | None,
    effect: str,
    *,
    interactive: bool = False,
    verified: bool = True,
) -> HeroSkillSpec:
    return HeroSkillSpec(
        hero=hero,
        name=name,
        trigger=trigger,
        limit=limit,
        effect=effect,
        category=classify_skill_category(trigger, interactive),
        interactive=interactive,
        verified=verified,
        live_verified=False,
        rule_id=f"{RULES_VERSION}:{hero}:{name}",
        rules_version=RULES_VERSION,
        supported_modes=SUPPORTED_MODES,
        choice_kind="interaction" if interactive else "automatic",
        optional=interactive,
        documented=True,
        projection_verified=True,
        sim_verified=hero in SIMULATED_HEROES,
        ui_verified=hero in SIMULATED_HEROES and (not interactive or (hero, name) in UI_VERIFIED_SKILLS),
    )


HERO_REGISTRY: dict[str, tuple[HeroSkillSpec, ...]] = {
    "典韦": (
        _skill("典韦", "不屈", "after_play", None, "unyielding_mark"),
        _skill("典韦", "血战", "mark_changed", None, "unyielding_rewards"),
    ),
    "夏侯惇": (_skill("夏侯惇", "刚烈", "own_play_beaten", 1, "discard_three_recover_highest", interactive=True),),
    "关羽": (
        _skill("关羽", "单骑", "after_straight", 1, "gain_wildcard"),
        _skill("关羽", "武圣", "other_straight", 2, "take_one_from_straight", interactive=True),
    ),
    "姜维": (
        _skill("姜维", "绝计", "pass_with_one", None, "increase_last_card"),
        _skill("姜维", "北伐", "pass_response", None, "discard_lowest_unless_one"),
    ),
    "张飞": (_skill("张飞", "咆哮", "repeat_action_type", None, "increase_two_lowest"),),
    "赵云": (_skill("赵云", "冲阵", "low_solo_pair_beaten", None, "recover_and_increase_until_seven"),),
    "徐盛": (_skill("徐盛", "疑城", "no_legal_response", 3, "gain_two_discard_one", interactive=True),),
    "陆逊": (
        _skill("陆逊", "破蜀", "after_play", 20, "gain_random_above_ten"),
        _skill("陆逊", "御魏", "own_play_beaten", 20, "discard_lowest_unless_one"),
    ),
    "曹洪": (_skill("曹洪", "敛财", "other_pair", 3, "take_pair"),),
    "关银屏": (_skill("关银屏", "花武", "any_action_over_four", 5, "gain_random_jqk"),),  # 单次出牌数量大于等于四张时触发。
    "诸葛均": (_skill("诸葛均", "耕读", "game_start", 1, "copy_bottom_then_discard_after_play", interactive=True),),
    "凌统": (_skill("凌统", "勇进", "after_solo_or_pair", 2, "discard_opposite_group", interactive=True),),
    "卢植": (_skill("卢植", "儒宗", "after_beating", 3, "convert_solo_pair", interactive=True),),
    "皇甫嵩": (_skill("皇甫嵩", "平乱", "any_all_below_five", 3, "gain_equal_above_six"),),
    "邓艾": (
        _skill("邓艾", "屯田", "pass_under_four", 5, "gain_food"),
        _skill("邓艾", "兵粮", "active", None, "spend_food_for_jokers_or_wildcard", interactive=True),
    ),
    "郭嘉": (
        _skill("郭嘉", "遗计", "after_play", None, "copy_play_to_remaining_hand_once_condition"),
        _skill("郭嘉", "胜论", "new_action_type", 10, "gain_random_card"),
    ),
    "曹操": (
        _skill("曹操", "枭雄", "own_play_beaten", 1, "take_beating_action_except_bomb", interactive=True),
        _skill("曹操", "魏武", "active", 1, "decrease_one_rank_by_two", interactive=True),
    ),
    "甄姬": (
        _skill("甄姬", "镜花", "global_new_action_type", None, "light_petal"),
        _skill("甄姬", "洛神", "petal_three_or_five", 2, "gain_high_pair_then_wildcard"),
    ),
    "曹丕": (
        _skill("曹丕", "世子", "game_start", 1, "fill_twos_to_three"),
        _skill("曹丕", "改制", "play_contains_two", 3, "fill_single_to_pair", interactive=True),
    ),
    "许褚": (
        _skill("许褚", "虎烈", "active", 1, "mark_low_pair_as_tigers", interactive=True),
        _skill("许褚", "裸衣", "after_bomb", None, "reset_hulie"),
    ),
    "夏侯渊": (_skill("夏侯渊", "神速", "after_play", None, "increase_random_card"),),
    "徐晃": (_skill("徐晃", "截粮", "after_beating", 1, "take_beaten_action", interactive=True),),
    "郭女王": (
        _skill("郭女王", "智数", "after_trio_attachment", None, "discard_one", interactive=True),
        _skill("郭女王", "殊宠", "other_trio_attachment", None, "take_attachments"),
    ),
    "曹植": (_skill("曹植", "诗才", "after_response_or_pass", None, "reveal_and_take_between_pairs", interactive=True),),
    "程昱": (
        _skill("程昱", "设伏", "game_start_or_ambush_played", None, "mark_random_ambush"),
        _skill("程昱", "现兵", "ambush_resolution", None, "discard_self_or_other", interactive=True),
    ),
    "羊祜": (_skill("羊祜", "怀柔", "after_beating", 5, "take_one_beaten_card", interactive=True),),
    "荀攸": (
        _skill("荀攸", "奇策", "active", 1, "discard_sum_twelve_gain_above_q", interactive=True),
        _skill("荀攸", "谋佐", "play_qice_card", 1, "reset_qice"),
    ),
    "刘禅": (
        _skill("刘禅", "决政", "game_start", 1, "give_up_to_two", interactive=True),
        _skill("刘禅", "放权", "active_with_lead", 1, "discard_two_largest_transfer_lead", interactive=True),
    ),
    "诸葛亮": (
        _skill("诸葛亮", "妙算", "bidding", None, "see_bottom"),
        _skill("诸葛亮", "神机", "no_legal_response", 3, "transform_above_k_then_pass", interactive=True),
    ),
    "庞统": (
        _skill("庞统", "疑兵", "game_start", 1, "fill_j_to_three"),  # 公开技能说明确认缺失点数为J；仍需拥有英雄后的真实牌局校准。
        _skill("庞统", "连环", "after_trio_attachment", 2, "fill_random_pair_to_trio"),
    ),
    "刘备": (
        _skill("刘备", "仁义", "own_play_beaten", 2, "give_high_card", interactive=True),
        _skill("刘备", "桃园", "after_renyi_twice", 1, "replace_with_guanyu_zhangfei"),
    ),
    "黄忠": (
        _skill("黄忠", "瞄准", "after_beating", 4, "mark_random_enemy_rank"),
        _skill("黄忠", "穿杨", "active_after_two_marks", None, "take_all_marked_cards", interactive=True),
    ),
    "马超": (
        _skill("马超", "铁骑", "active", 2, "protect_next_play", interactive=True),
        _skill("马超", "合围", "response", 2, "combine_observed_history", interactive=True),
    ),
    "魏延": (
        _skill("魏延", "反骨", "after_beating", 3, "increase_below_q_by_two", interactive=True),
        _skill("魏延", "奇谋", "response", 1, "four_above_k_as_four_k", interactive=True),
    ),
    "黄月英": (
        _skill("黄月英", "巧械", "other_non_solo", None, "copy_latest_three_to_wooden_ox"),
        _skill("黄月英", "木牛", "active", 2, "swap_hand_with_wooden_ox", interactive=True),
    ),
    "法正": (
        _skill("法正", "睚眦", "own_play_beaten", None, "add_grudge"),
        _skill("法正", "复仇", "active_grudge_two", None, "take_one_or_two_largest", interactive=True),
    ),
    "糜夫人": (_skill("糜夫人", "存嗣", "first_hand_below_eight", 1, "convert_high_to_wildcard_and_give", interactive=True),),
    "徐庶": (
        _skill("徐庶", "破阵", "exact_plus_one_response_over_four", 1, "gain_wildcard"),
        _skill("徐庶", "无言", "response", 1, "adjust_rank_and_future_pass", interactive=True),
    ),
    "孙策": (
        _skill("孙策", "继志", "game_start", 1, "fill_kings_to_three"),
        _skill("孙策", "借兵", "no_legal_response", 2, "discard_k_gain_high_pair_and_pass", interactive=True),
    ),
    "吕蒙": (
        _skill("吕蒙", "勤学", "response", None, "take_up_to_two_previous_and_pass", interactive=True),
        _skill("吕蒙", "顿悟", "after_bomb", 1, "reset_qinxue"),
    ),
    "大乔": (
        _skill("大乔", "结缘", "unbeaten_other_solo", 1, "take_solo"),
        _skill("大乔", "贤助", "active", 1, "both_fill_largest_to_three", interactive=True),
    ),
    "甘宁": (_skill("甘宁", "游侠", "after_beating", 3, "inspect_and_take_two_lowest", interactive=True),),
    "孙坚": (_skill("孙坚", "得玺", "active_without_trio_or_bomb", 1, "gain_three_kings"),),
    "小乔": (
        _skill("小乔", "星华", "after_play_under_six", 1, "discard_matching_ranks", interactive=True),
        _skill("小乔", "巧笑", "after_xinghua", 1, "reveal_equal_and_take_any", interactive=True),
    ),
    "孙尚香": (
        _skill("孙尚香", "妆营", "game_start", 1, "gain_a_and_j"),
        _skill("孙尚香", "剑舞", "own_play_beaten", 2, "recover_and_swap_a_j"),
    ),
    "孙权": (_skill("孙权", "纵横", "active", 1, "copy_one_and_reset_on_mixed_large_play", interactive=True),),
    "鲁肃": (
        _skill("鲁肃", "豪富", "no_legal_response", None, "reveal_player_count_and_draft", interactive=True),
        _skill("鲁肃", "聚财", "lose_haofu_card", 2, "reset_haofu"),
    ),
    "吴国太": (_skill("吴国太", "懿训", "active", 2, "force_up_to_two_responses", interactive=True),),
    "刘协": (
        _skill("刘协", "上贡", "game_start", 1, "exchange_low_for_largest_non_joker", interactive=True),
        _skill("刘协", "血诏", "third_no_legal_response", 1, "largest_to_random_joker"),
    ),
    "董卓": (
        _skill("董卓", "相国", "bidding", 1, "bid_first"),
        _skill("董卓", "暴政", "after_response_or_pass", None, "discard_after_response_gain_after_pass", interactive=True),
    ),
    "袁术": (
        _skill("袁术", "名门", "game_start_or_unbeaten_play", 2, "increase_chosen_card", interactive=True),
        _skill("袁术", "称帝", "active_with_lead", 1, "swap_identity_with_landlord_gain_two_twos", interactive=True),
    ),
    "左慈": (
        _skill("左慈", "变幻", "game_start", 1, "mirror_ranks_around_nine"),
        _skill("左慈", "神道", "active", 1, "repeat_game_start_skills_gain_twos", interactive=True),
    ),
    "吕布": (
        _skill("吕布", "无双", "beaten_or_pass", 1, "rage_six_take_all_jqk"),
        _skill("吕布", "霸关", "active_with_lead", 1, "restrict_response_to_jqk", interactive=True),
    ),
    "貂蝉": (_skill("貂蝉", "魅惑", "response", None, "discard_same_count_at_least_k_transform_previous_to_3334_or_34567_except_bomb", interactive=True),),  # 规则已补全但尚无账号实战素材。
    "袁绍": (
        _skill("袁绍", "威望", "after_play_and_unbeaten", None, "gain_prestige"),
        _skill("袁绍", "号令", "prestige_three", 2, "take_largest_pair", interactive=True),
    ),
    "公孙瓒": (_skill("公孙瓒", "义从", "active", None, "gain_or_replace_mount", interactive=True),),
    "蔡文姬": (
        _skill("蔡文姬", "和弦", "three_same_action_types", None, "increase_lowest"),
        _skill("蔡文姬", "变奏", "three_distinct_action_types", None, "discard_lowest"),
    ),
    "南华老仙": (
        _skill("南华老仙", "修道", "play_contains_two", None, "gain_three"),
        _skill("南华老仙", "天书", "response_over_two", None, "play_x_minus_one_threes_take_action", interactive=True),
    ),
    "邹氏": (_skill("邹氏", "遗孀", "own_play_beaten", 2, "discard_group_reduce_enemy_max", interactive=True),),
    "吴苋": (_skill("吴苋", "凤兆", "after_beating", 2, "inspect_four_copy_non_joker", interactive=True),),
    "张宝": (_skill("张宝", "黄符", "new_action_type_contains_three", 3, "discard_increasing_count", interactive=True),),
    "张梁": (_skill("张梁", "宣教", "after_beating", 3, "gain_rank_by_action_size_once_each"),),
    "董承": (_skill("董承", "诏命", "game_start_or_joker_play", None, "gain_joker_or_turn_two_solos_to_pairs"),),
    "朱儁": (_skill("朱儁", "合讨", "first_hand_below_twelve", 1, "gain_existing_ranks"),),
    "刘虞": (_skill("刘虞", "保境", "largest_play_resolution", 1, "change_all_solos_by_outcome"),),
}


SKILLS_BY_CATEGORY = {
    category: tuple(
        skill
        for skills in HERO_REGISTRY.values()
        for skill in skills
        if skill.category == category
    )
    for category in SKILL_CATEGORY_LABELS
}
HERO_SKILLS_BY_CATEGORY = {
    hero: {
        category: tuple(skill for skill in skills if skill.category == category)
        for category in SKILL_CATEGORY_LABELS
    }
    for hero, skills in HERO_REGISTRY.items()
}


def has_only_passive_skills(hero: str) -> bool:
    """Return whether an owned hero can play without initiating or resolving skill UI choices."""
    skills = HERO_REGISTRY.get(hero, ())
    return bool(skills) and all(skill.category == "passive" for skill in skills)


PASSIVE_OWNED_HEROES = tuple(hero for hero in OWNED_HEROES if has_only_passive_skills(hero))  # 从权威技能契约生成纯被动英雄库，避免拥有名单扩充后漏同步。


AUTHORITATIVE_SKILL_NAMES = {
    "典韦": frozenset({"不屈", "血战"}),
    "夏侯惇": frozenset({"刚烈"}),
    "关羽": frozenset({"单骑", "武圣"}),
    "张飞": frozenset({"咆哮"}),
    "赵云": frozenset({"冲阵"}),
    "徐盛": frozenset({"疑城"}),
    "陆逊": frozenset({"破蜀", "御魏"}),
    "甘宁": frozenset({"游侠"}),
    "大乔": frozenset({"结缘", "贤助"}),
    "姜维": frozenset({"绝计", "北伐"}),
    "曹洪": frozenset({"敛财"}),
    "关银屏": frozenset({"花武"}),
    "诸葛均": frozenset({"耕读"}),
    "凌统": frozenset({"勇进"}),
    "卢植": frozenset({"儒宗"}),
    "皇甫嵩": frozenset({"平乱"}),
}
AUTHORITATIVE_RULE_IDS = frozenset(
    skill.rule_id
    for hero, skill_names in AUTHORITATIVE_SKILL_NAMES.items()
    for skill in HERO_REGISTRY[hero]
    if skill.name in skill_names
)

# 全量纯策略/投影模拟覆盖；与旧 SIMULATED_HEROES 的权威状态机兼容语义分开。
POLICY_SIMULATED_HEROES = tuple(HERO_REGISTRY)

HERO_MECHANISM_GROUPS = {
    "automatic_history": (
        "典韦", "张飞", "赵云", "陆逊", "曹洪", "关银屏", "皇甫嵩", "郭嘉", "甄姬",
        "夏侯渊", "庞统", "孙尚香", "蔡文姬", "南华老仙", "张梁", "董承", "朱儁", "刘虞",
    ),
    "self_card_choice": (
        "夏侯惇", "关羽", "徐盛", "诸葛均", "凌统", "卢植", "曹操", "曹丕", "许褚", "徐晃",
        "郭女王", "曹植", "羊祜", "荀攸", "魏延", "法正", "糜夫人", "徐庶", "吕蒙", "甘宁",
        "小乔", "孙权", "貂蝉", "吴苋", "张宝",
    ),
    "resource_state": (
        "邓艾", "程昱", "刘备", "黄忠", "马超", "黄月英", "孙策", "鲁肃", "吕布", "袁绍", "公孙瓒", "邹氏",
    ),
    "player_target": ("刘禅", "大乔", "吴国太", "刘协", "董卓", "袁术", "左慈"),
    "special_legality": ("姜维", "诸葛亮", "孙坚"),
}
HERO_PRIMARY_MECHANISM = {
    hero: mechanism for mechanism, heroes in HERO_MECHANISM_GROUPS.items() for hero in heroes
}


def normalize_hero_name(value: object) -> str | None:
    text = "".join(str(value or "").split())
    for alias, canonical in HERO_ALIASES.items():
        if alias in text:
            return canonical
    return next((hero for hero in HERO_REGISTRY if hero in text), None)


def iter_skill_specs() -> Iterable[HeroSkillSpec]:
    """Yield every registered skill in stable hero and skill order."""
    for skills in HERO_REGISTRY.values():
        yield from skills


def skill_by_rule_id(rule_id: str) -> HeroSkillSpec | None:
    """Return the registered skill for an exact stable rule identifier."""
    return next((skill for skill in iter_skill_specs() if skill.rule_id == rule_id), None)


def registry_contract_errors() -> tuple[str, ...]:
    """Return deterministic errors for the complete three-player skill registry."""
    errors: list[str] = []
    skills = tuple(iter_skill_specs())
    if len(HERO_REGISTRY) != 65:
        errors.append(f"expected 65 heroes, found {len(HERO_REGISTRY)}")
    if len(skills) != 102:
        errors.append(f"expected 102 skills, found {len(skills)}")
    grouped_heroes = tuple(hero for heroes in HERO_MECHANISM_GROUPS.values() for hero in heroes)
    if len(grouped_heroes) != len(set(grouped_heroes)):
        errors.append("hero mechanism groups contain duplicates")
    if set(HERO_PRIMARY_MECHANISM) != set(HERO_REGISTRY):
        errors.append("hero mechanism groups do not cover the registry exactly")
    if set(AUTHORITATIVE_SKILL_NAMES) != set(SIMULATED_HEROES):
        errors.append("authoritative hero map does not match SIMULATED_HEROES")
    for hero, names in AUTHORITATIVE_SKILL_NAMES.items():
        registered_names = {skill.name for skill in HERO_REGISTRY.get(hero, ())}
        if names != registered_names:
            errors.append(f"{hero}: authoritative skill map differs from registry")
    if len(AUTHORITATIVE_RULE_IDS) != 21:
        errors.append(f"expected 21 authoritative rules, found {len(AUTHORITATIVE_RULE_IDS)}")

    seen_rule_ids: dict[str, HeroSkillSpec] = {}
    for hero, hero_skills in HERO_REGISTRY.items():
        for skill in hero_skills:
            label = f"{hero}/{skill.name}"
            if skill.hero != hero:
                errors.append(f"{label}: skill hero is {skill.hero!r}")
            if not skill.rule_id.strip():
                errors.append(f"{label}: missing rule_id")
            elif skill.rule_id in seen_rule_ids:
                other = seen_rule_ids[skill.rule_id]
                errors.append(
                    f"{label}: duplicate rule_id {skill.rule_id!r} also used by "
                    f"{other.hero}/{other.name}"
                )
            else:
                seen_rule_ids[skill.rule_id] = skill
            if skill.rules_version != RULES_VERSION:
                errors.append(f"{label}: unsupported rules_version {skill.rules_version!r}")
            if "landlord_3p" not in skill.supported_modes:
                errors.append(f"{label}: landlord_3p is not supported")
            if not skill.trigger.strip():
                errors.append(f"{label}: missing trigger")
            if not skill.effect.strip():
                errors.append(f"{label}: missing effect")
            if not skill.choice_kind.strip():
                errors.append(f"{label}: missing choice_kind")
            expected_category = classify_skill_category(skill.trigger, skill.interactive)
            if skill.category != expected_category:
                errors.append(f"{label}: category {skill.category!r} should be {expected_category!r}")
            if not skill.documented:
                errors.append(f"{label}: not documented")
            if not skill.projection_verified:
                errors.append(f"{label}: policy projection is not verified")
            if skill.sim_verified != (skill.rule_id in AUTHORITATIVE_RULE_IDS):
                errors.append(f"{label}: authoritative simulation status is inconsistent")
            if skill.projection_verified and not skill.documented:
                errors.append(f"{label}: projection_verified requires documented")
            if skill.sim_verified and not skill.documented:
                errors.append(f"{label}: sim_verified requires documented")
            if skill.sim_verified and not skill.projection_verified:
                errors.append(f"{label}: sim_verified requires projection_verified")
            if skill.ui_verified and not skill.sim_verified:
                errors.append(f"{label}: ui_verified requires sim_verified")
            if skill.live_verified and not skill.ui_verified:
                errors.append(f"{label}: live_verified requires ui_verified")
    return tuple(errors)


def iter_unverified_skills() -> Iterable[HeroSkillSpec]:
    for skills in HERO_REGISTRY.values():
        for skill in skills:
            if not skill.verified:
                yield skill


def skills_for_trigger(hero: str | None, trigger: str) -> tuple[HeroSkillSpec, ...]:
    return tuple(skill for skill in HERO_REGISTRY.get(hero or "", ()) if skill.trigger == trigger)
