#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅做评估并输出修改意见：
1) 输入 scene_text + scene_pkl
2) 按 A/B/C/D/E/F 六项标准检查
3) 输出尽量符合 guidance.md 用语风格的修改建议
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import agent_check_pkl_v5 as core
import harness_param_constraints_v1 as harness

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "single": 1,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
}

COUNT_DETERMINERS = ("a", "an", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
ALL_DETERMINERS = COUNT_DETERMINERS + ("the",)
OBJECT_EXTRACTION_STOP_TOKENS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "on",
    "in",
    "at",
    "to",
    "from",
    "with",
    "and",
    "or",
    "but",
    "for",
    "by",
    "as",
    "while",
    "where",
    "which",
    "that",
    "who",
    "whose",
    "whom",
    "placed",
    "positioned",
    "covers",
    "covering",
    "centered",
    "centering",
    "centers",
    "anchor",
    "anchors",
    "anchored",
    "sits",
    "sit",
    "stands",
    "stand",
    "lies",
    "lie",
    "behind",
    "before",
    "after",
    "under",
    "over",
    "above",
    "below",
    "left",
    "right",
    "front",
    "back",
    "center",
    "middle",
    "upper",
    "lower",
    "there",
    "here",
    "near",
    "beside",
}
IRREGULAR_SINGULARS = {
    "men": "man",
    "women": "woman",
    "people": "person",
    "children": "child",
    "teeth": "tooth",
    "feet": "foot",
    "mice": "mouse",
    "geese": "goose",
    "wolves": "wolf",
    "knives": "knife",
    "shelves": "shelf",
    "leaves": "leaf",
}
DIMENSION_LABELS = ("width", "depth", "height")
RELATION_ABOVE_PATTERN = r"(?:above|higher than|高于|上方)"
RELATION_BELOW_PATTERN = r"(?:below|beneath|lower than|低于|下方)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 scene pkl 是否符合文字描述，并输出修改意见文本。",
    )
    parser.add_argument("--scene-text", required=True, help="场景文字描述。")
    parser.add_argument("--scene-pkl", required=True, help="待评估的 pkl 路径。")
    parser.add_argument(
        "--guidance-output",
        default="",
        help="可选，将最终修改意见文本写入该文件。",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="可选，写出包含 A/B/C/D/E/F 结果与问题详情的 JSON。",
    )
    parser.add_argument(
        "--objects",
        default="",
        help="可选，显式指定 prompt 主体及数量，例如 'table:1,bowl:1'。",
    )
    parser.add_argument(
        "--object-manifest",
        default="",
        help="可选，读取 object_manifest.json，并按该主体清单做严格验证。",
    )
    parser.add_argument(
        "--predicates-json",
        default="",
        help="可选，读取 agent_text2pkl_v5.py 输出的 predicates.json。",
    )
    parser.add_argument(
        "--repair-plan-output",
        default="",
        help="可选，写出可被 agent_reverse.py --repair-plan 执行的 JSON 修复计划。",
    )
    return parser.parse_args()


def normalize_scene_pkl_path(scene_pkl: str) -> str:
    candidate = scene_pkl.strip()
    if not candidate:
        raise RuntimeError("scene-pkl 不能为空。")
    if not os.path.isabs(candidate):
        candidate = os.path.join(core.REPO_ROOT, candidate)
    if not os.path.isfile(candidate):
        raise RuntimeError(f"找不到 pkl 文件：{candidate}")
    return candidate


def singularize_word(word: str) -> str:
    lower_word = word.lower().strip()
    if not lower_word:
        return lower_word
    if lower_word in IRREGULAR_SINGULARS:
        return IRREGULAR_SINGULARS[lower_word]
    if lower_word.endswith("ies") and len(lower_word) > 3:
        return lower_word[:-3] + "y"
    if lower_word.endswith("ves") and len(lower_word) > 3:
        stem = lower_word[:-3]
        if stem.endswith("i"):
            return stem[:-1] + "ife"
        return stem + "f"
    if re.search(r"(ches|shes|xes|zes|ses)$", lower_word):
        return lower_word[:-2]
    if lower_word.endswith("s") and not lower_word.endswith("ss"):
        return lower_word[:-1]
    return lower_word


def singularize_phrase(phrase: str) -> str:
    tokens = core.canonicalize_type(phrase).split()
    if not tokens:
        return ""
    tokens[-1] = singularize_word(tokens[-1])
    return " ".join(tokens).strip()


def build_phrase_mapping_candidates(phrase: str) -> List[str]:
    normalized = core.canonicalize_type(phrase)
    if not normalized:
        return []

    tokens = normalized.split()
    candidates: List[str] = []
    seen: Set[str] = set()

    def add(candidate: str) -> None:
        normalized_candidate = core.canonicalize_type(candidate)
        if not normalized_candidate or normalized_candidate in seen:
            return
        seen.add(normalized_candidate)
        candidates.append(normalized_candidate)

    add(normalized)
    add(singularize_phrase(normalized))

    for window in range(min(3, len(tokens)), 0, -1):
        suffix = " ".join(tokens[-window:])
        add(suffix)
        add(singularize_phrase(suffix))

    return candidates


def map_mention_to_asset_type(mention: str, allowed_types: List[str]) -> str:
    for candidate in build_phrase_mapping_candidates(mention):
        mapped_type = core.match_asset_type(None, candidate, allowed_types)
        if mapped_type != "Custom":
            return mapped_type
    return "Custom"


def parse_quantified_object_mentions(scene_text: str) -> List[Tuple[str, int]]:
    normalized_text = core.normalize_free_text(scene_text)
    if not normalized_text:
        return []

    determiner_pattern = re.compile(
        r"\b(?P<det>the|a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
        re.IGNORECASE,
    )
    token_pattern = re.compile(r"[a-zA-Z]+(?:-[a-zA-Z]+)?|\d+")
    quantified_mentions: List[Tuple[str, int]] = []

    for determiner_match in determiner_pattern.finditer(normalized_text):
        determiner = determiner_match.group("det").lower().strip()
        tail = normalized_text[determiner_match.end() :]
        tail_tokens = token_pattern.findall(tail)
        phrase_tokens: List[str] = []

        for token in tail_tokens:
            lower_token = token.lower()
            if lower_token in ALL_DETERMINERS and phrase_tokens:
                break
            if lower_token in OBJECT_EXTRACTION_STOP_TOKENS:
                break
            if lower_token.isdigit() and phrase_tokens:
                break
            phrase_tokens.append(lower_token)
            if len(phrase_tokens) >= 6:
                break

        if not phrase_tokens:
            continue

        phrase = " ".join(phrase_tokens).strip()
        if not phrase:
            continue

        # 用户规则：the 指代前文，不应算作新物体。
        if determiner == "the":
            continue

        if determiner in {"a", "an"}:
            count = 1
        elif determiner.isdigit():
            count = max(1, int(determiner))
        else:
            count = NUMBER_WORDS.get(determiner, 1)
        quantified_mentions.append((phrase, count))

    return quantified_mentions


def infer_expected_type_counts(
    scene_text: str,
    allowed_types: List[str],
) -> Tuple[Counter, List[str], List[Dict[str, Any]]]:
    expected_counts: Counter = Counter()
    parse_warnings: List[str] = []
    parsed_mentions: List[Dict[str, Any]] = []

    quantified_mentions = parse_quantified_object_mentions(scene_text)
    if not quantified_mentions:
        if core.normalize_free_text(scene_text):
            parse_warnings.append(
                "未按 a/an/明确数字 规则解析到任何对象；根据规则，只有 a/an 或明确数字标注的名词会被计入，the 不会被当作新物体。"
            )
        return expected_counts, parse_warnings, parsed_mentions

    for mention, mention_count in quantified_mentions:
        mapped_type = map_mention_to_asset_type(mention, allowed_types)
        parsed_mentions.append(
            {
                "mention": mention,
                "count": mention_count,
                "mapped_type": mapped_type,
            }
        )
        if mapped_type == "Custom":
            parse_warnings.append(f"对象短语 `{mention}` 不能稳定映射到已知资产类型。")
            continue
        expected_counts[mapped_type] += mention_count
    return expected_counts, parse_warnings, parsed_mentions


def expected_type_counts_from_manifest(
    object_manifest: Dict[str, Any],
    source: str,
) -> Tuple[Counter, List[str], List[Dict[str, Any]]]:
    expected_counts: Counter = Counter()
    parse_warnings = list(object_manifest.get("parse_warnings", []) or [])
    parsed_mentions: List[Dict[str, Any]] = []

    for obj in object_manifest.get("objects", []) or []:
        mapped_type = str(obj.get("type", "")).strip()
        mention = str(obj.get("mention") or obj.get("name") or mapped_type).strip()
        if not mapped_type:
            parse_warnings.append(f"{source} 中存在缺少 type 的对象条目。")
            continue
        if mapped_type == harness.CUSTOM_ASSET_TYPE:
            parse_warnings.append(f"{source} 中对象 `{mention}` 不能稳定映射到已知资产类型。")
            continue
        expected_counts[mapped_type] += 1
        parsed_mentions.append(
            {
                "mention": mention,
                "count": 1,
                "mapped_type": mapped_type,
                "source": source,
            }
        )

    return expected_counts, parse_warnings, parsed_mentions


def evaluate_criterion_a(
    scene_text: str,
    subjects: List[Dict[str, Any]],
    allowed_types: List[str],
    expected_manifest: Optional[Dict[str, Any]] = None,
    expected_source: str = "prompt",
) -> Tuple[bool, List[Dict[str, str]], List[str], List[Dict[str, Any]]]:
    issues: List[Dict[str, str]] = []
    if expected_manifest is not None:
        expected_counts, parse_warnings, parsed_mentions = expected_type_counts_from_manifest(
            expected_manifest,
            expected_source,
        )
    else:
        expected_counts, parse_warnings, parsed_mentions = infer_expected_type_counts(scene_text, allowed_types)
    actual_counts = Counter(subject["type"] for subject in subjects)

    all_types = sorted(set(expected_counts.keys()) | set(actual_counts.keys()))
    for asset_type in all_types:
        expected = int(expected_counts.get(asset_type, 0))
        actual = int(actual_counts.get(asset_type, 0))
        if actual < expected:
            issues.append(
                {
                    "category": "missing_object",
                    "object": asset_type,
                    "count": str(expected - actual),
                    "details": f"类型 `{asset_type}` 还缺少 {expected - actual} 个，期望数量={expected}，实际数量={actual}。",
                }
            )
        elif actual > expected:
            issues.append(
                {
                    "category": "extra_object",
                    "object": asset_type,
                    "count": str(actual - expected),
                    "details": f"类型 `{asset_type}` 多出了 {actual - expected} 个，期望数量={expected}，实际数量={actual}。",
                }
            )
    return len(issues) == 0, issues, parse_warnings, parsed_mentions


def compute_support_alignment(subject: Dict[str, Any], support: Dict[str, Any]) -> Dict[str, float]:
    subject_width, subject_depth, subject_height = subject["dims"]
    support_width, support_depth, support_height = support["dims"]

    subject_x_min = subject["scene_x"] - subject_depth / 2.0
    subject_x_max = subject["scene_x"] + subject_depth / 2.0
    subject_y_min = subject["scene_y"] - subject_width / 2.0
    subject_y_max = subject["scene_y"] + subject_width / 2.0

    support_x_min = support["scene_x"] - support_depth / 2.0
    support_x_max = support["scene_x"] + support_depth / 2.0
    support_y_min = support["scene_y"] - support_width / 2.0
    support_y_max = support["scene_y"] + support_width / 2.0

    expected_z = support["scene_z"] + support_height
    xy_tolerance = max(0.05, min(subject_width, subject_depth) * 0.15)
    z_tolerance = max(0.05, subject_height * 0.2)

    return {
        "expected_z": round(expected_z, 6),
        "z_delta": round(expected_z - subject["scene_z"], 6),
        "z_tolerance": round(z_tolerance, 6),
        "x_low_excess": round(support_x_min - subject_x_min, 6),
        "x_high_excess": round(subject_x_max - support_x_max, 6),
        "y_low_excess": round(support_y_min - subject_y_min, 6),
        "y_high_excess": round(subject_y_max - support_y_max, 6),
        "xy_tolerance": round(xy_tolerance, 6),
        "width_oversize": float(subject_width - support_width),
        "depth_oversize": float(subject_depth - support_depth),
    }


def parse_object_pair(value: str) -> Tuple[str, str]:
    first_name, second_name = (value.split("|", 1) + [""])[:2]
    return first_name.strip(), second_name.strip()


def support_map_from_predicates(
    subjects: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    predicates_payload: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    id_to_name = harness.manifest_id_to_name(manifest)
    subject_lookup = {subject["name"]: subject for subject in subjects}
    support_map: Dict[str, Dict[str, Any]] = {}
    for predicate in predicates_payload.get("predicates", []):
        if predicate.get("type") != "support":
            continue
        subject_name = id_to_name.get(str(predicate.get("subject", "")))
        support_name = id_to_name.get(str(predicate.get("object", "")))
        if not subject_name or not support_name:
            continue
        subject = subject_lookup.get(subject_name)
        support = subject_lookup.get(support_name)
        if subject is None or support is None:
            continue
        support_map[subject_name] = support
    return support_map


def evaluate_criterion_b(
    scene_text: str,
    subjects: List[Dict[str, Any]],
    support_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[bool, List[Dict[str, str]]]:
    relation_issues: List[Dict[str, str]] = []
    clauses = core.split_text_clauses(scene_text)
    if not clauses or not subjects:
        return True, relation_issues

    support_map = support_map if support_map is not None else core.build_support_map(subjects, scene_text)
    tolerance = 1e-4

    for subject in subjects:
        for clause in core.subject_relevant_clauses(scene_text, subject):
            if not core.clause_contains_any(clause, core.CENTER_HINT_TOKENS):
                continue
            if abs(subject["scene_x"]) > 0.2 or abs(subject["scene_y"]) > 0.2:
                relation_issues.append(
                    {
                        "category": "relation_center_mismatch",
                        "object": subject["name"],
                        "details": "文字要求居中，但当前不在画面中心附近。",
                    }
                )
                break

    for clause in clauses:
        for first_subject in subjects:
            for second_subject in subjects:
                if first_subject is second_subject:
                    continue

                first_support = support_map.get(first_subject["name"])
                second_support = support_map.get(second_subject["name"])
                same_support_surface = (
                    first_support is not None
                    and second_support is not None
                    and first_support["name"] == second_support["name"]
                )
                lateral = core.lateral_gap(
                    first_subject,
                    second_subject,
                    same_support_surface=same_support_surface,
                )
                longitudinal = core.longitudinal_gap(
                    first_subject,
                    second_subject,
                    same_support_surface=same_support_surface,
                )

                if core.ordered_relation_in_clause(
                    clause,
                    first_subject,
                    second_subject,
                    core.RELATION_LEFT_PATTERN,
                ):
                    if first_subject["scene_y"] > second_subject["scene_y"] - lateral + tolerance:
                        relation_issues.append(
                            {
                                "category": "relation_left_mismatch",
                                "object": f"{first_subject['name']}|{second_subject['name']}",
                                "details": f"文字要求 {first_subject['name']} 在 {second_subject['name']} 左侧。",
                            }
                        )

                if core.ordered_relation_in_clause(
                    clause,
                    first_subject,
                    second_subject,
                    core.RELATION_RIGHT_PATTERN,
                ):
                    if first_subject["scene_y"] < second_subject["scene_y"] + lateral - tolerance:
                        relation_issues.append(
                            {
                                "category": "relation_right_mismatch",
                                "object": f"{first_subject['name']}|{second_subject['name']}",
                                "details": f"文字要求 {first_subject['name']} 在 {second_subject['name']} 右侧。",
                            }
                        )

                if core.ordered_relation_in_clause(
                    clause,
                    first_subject,
                    second_subject,
                    core.RELATION_FRONT_PATTERN,
                ):
                    if first_subject["scene_x"] < second_subject["scene_x"] + longitudinal - tolerance:
                        relation_issues.append(
                            {
                                "category": "relation_front_mismatch",
                                "object": f"{first_subject['name']}|{second_subject['name']}",
                                "details": f"文字要求 {first_subject['name']} 在 {second_subject['name']} 前方。",
                            }
                        )

                if core.ordered_relation_in_clause(
                    clause,
                    first_subject,
                    second_subject,
                    core.RELATION_BEHIND_PATTERN,
                ):
                    if first_subject["scene_x"] > second_subject["scene_x"] - longitudinal + tolerance:
                        relation_issues.append(
                            {
                                "category": "relation_behind_mismatch",
                                "object": f"{first_subject['name']}|{second_subject['name']}",
                                "details": f"文字要求 {first_subject['name']} 在 {second_subject['name']} 后方。",
                            }
                        )

                if core.ordered_relation_in_clause(
                    clause,
                    first_subject,
                    second_subject,
                    RELATION_ABOVE_PATTERN,
                ):
                    if first_subject["scene_z"] <= second_subject["scene_z"] + second_subject["dims"][2] - tolerance:
                        relation_issues.append(
                            {
                                "category": "relation_above_mismatch",
                                "object": f"{first_subject['name']}|{second_subject['name']}",
                                "details": f"文字要求 {first_subject['name']} 在 {second_subject['name']} 上方。",
                            }
                        )

                if core.ordered_relation_in_clause(
                    clause,
                    first_subject,
                    second_subject,
                    RELATION_BELOW_PATTERN,
                ):
                    if first_subject["scene_z"] + first_subject["dims"][2] >= second_subject["scene_z"] + tolerance:
                        relation_issues.append(
                            {
                                "category": "relation_below_mismatch",
                                "object": f"{first_subject['name']}|{second_subject['name']}",
                                "details": f"文字要求 {first_subject['name']} 在 {second_subject['name']} 下方。",
                            }
                        )

    subject_lookup = {subject["name"]: subject for subject in subjects}
    for subject_name, support_subject in support_map.items():
        subject = subject_lookup.get(subject_name)
        if subject is None:
            continue
        alignment = compute_support_alignment(subject, support_subject)
        if (
            abs(alignment["z_delta"]) > alignment["z_tolerance"]
            or alignment["x_low_excess"] > alignment["xy_tolerance"]
            or alignment["x_high_excess"] > alignment["xy_tolerance"]
            or alignment["y_low_excess"] > alignment["xy_tolerance"]
            or alignment["y_high_excess"] > alignment["xy_tolerance"]
        ):
            relation_issues.append(
                {
                    "category": "relation_support_mismatch",
                    "object": f"{subject_name}|{support_subject['name']}",
                    "details": f"文字要求 {subject_name} 放在 {support_subject['name']} 上，但当前没有完整落在 {support_subject['name']} 顶部。",
                }
            )

    return len(core.deduplicate_issues(relation_issues)) == 0, core.deduplicate_issues(relation_issues)


def evaluate_support_specific_issues(
    scene_text: str,
    subjects: List[Dict[str, Any]],
    support_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    physical_issues: List[Dict[str, str]] = []
    size_issues: List[Dict[str, str]] = []
    subject_lookup = {subject["name"]: subject for subject in subjects}
    support_map = support_map if support_map is not None else core.build_support_map(subjects, scene_text)

    for subject_name, support_subject in support_map.items():
        subject = subject_lookup.get(subject_name)
        if subject is None:
            continue

        alignment = compute_support_alignment(subject, support_subject)
        xy_tolerance = alignment["xy_tolerance"]
        z_tolerance = alignment["z_tolerance"]

        width_oversize = alignment["width_oversize"] > xy_tolerance
        depth_oversize = alignment["depth_oversize"] > xy_tolerance

        if width_oversize:
            size_issues.append(
                {
                    "category": "support_width_oversize",
                    "object": f"{subject_name}|{support_subject['name']}",
                    "details": f"{subject_name} 的 width 大于 {support_subject['name']} 的可支撑 width。",
                }
            )
        if depth_oversize:
            size_issues.append(
                {
                    "category": "support_depth_oversize",
                    "object": f"{subject_name}|{support_subject['name']}",
                    "details": f"{subject_name} 的 depth 大于 {support_subject['name']} 的可支撑 depth。",
                }
            )

        if alignment["z_delta"] > z_tolerance:
            physical_issues.append(
                {
                    "category": "support_move_up",
                    "object": subject_name,
                    "details": f"{subject_name} 应更靠近 {support_subject['name']} 顶部，当前需要向上平移。",
                }
            )
        elif alignment["z_delta"] < -z_tolerance:
            physical_issues.append(
                {
                    "category": "support_move_down",
                    "object": subject_name,
                    "details": f"{subject_name} 高于 {support_subject['name']} 顶部，当前需要向下平移。",
                }
            )

        if not depth_oversize:
            if alignment["x_low_excess"] > xy_tolerance and alignment["x_high_excess"] <= xy_tolerance:
                physical_issues.append(
                    {
                        "category": "support_move_forward",
                        "object": subject_name,
                        "details": f"{subject_name} 在 {support_subject['name']} 上偏后，需要向前平移。",
                    }
                )
            elif alignment["x_high_excess"] > xy_tolerance and alignment["x_low_excess"] <= xy_tolerance:
                physical_issues.append(
                    {
                        "category": "support_move_backward",
                        "object": subject_name,
                        "details": f"{subject_name} 在 {support_subject['name']} 上偏前，需要向后平移。",
                    }
                )

        if not width_oversize:
            if alignment["y_low_excess"] > xy_tolerance and alignment["y_high_excess"] <= xy_tolerance:
                physical_issues.append(
                    {
                        "category": "support_move_right",
                        "object": subject_name,
                        "details": f"{subject_name} 在 {support_subject['name']} 上偏左，需要向右平移。",
                    }
                )
            elif alignment["y_high_excess"] > xy_tolerance and alignment["y_low_excess"] <= xy_tolerance:
                physical_issues.append(
                    {
                        "category": "support_move_left",
                        "object": subject_name,
                        "details": f"{subject_name} 在 {support_subject['name']} 上偏右，需要向左平移。",
                    }
                )

    return core.deduplicate_issues(physical_issues), core.deduplicate_issues(size_issues)


def build_object_count_suggestion(asset_type: str, count: int, add: bool) -> Tuple[str, str]:
    if add:
        if count <= 1:
            return (f"增加物体 {asset_type}。", f"Add one {asset_type} object.")
        return (f"增加 {count} 个 {asset_type}。", f"Add {count} {asset_type} objects.")
    if count <= 1:
        return (f"删除物体 {asset_type}。", f"Delete one {asset_type} object.")
    return (f"删除 {count} 个 {asset_type}。", f"Delete {count} {asset_type} objects.")


def build_move_suggestion(subject_name: str, axis_name: str, increase: bool) -> Tuple[str, str]:
    if axis_name == "x":
        if increase:
            return (f"{subject_name} 向前平移。", f"Move {subject_name} forward.")
        return (f"{subject_name} 向后平移。", f"Move {subject_name} backward.")
    if axis_name == "y":
        if increase:
            return (f"{subject_name} 向右平移。", f"Move {subject_name} right.")
        return (f"{subject_name} 向左平移。", f"Move {subject_name} left.")
    if increase:
        return (f"{subject_name} 向上平移。", f"Move {subject_name} up.")
    return (f"{subject_name} 向下平移。", f"Move {subject_name} down.")


def build_all_dimension_resize_suggestion(subject_name: str, enlarge: bool) -> Tuple[str, str]:
    if enlarge:
        return (f"{subject_name} 的所有维度增大。", f"Increase all dimensions of {subject_name}.")
    return (f"{subject_name} 的所有维度减小。", f"Decrease all dimensions of {subject_name}.")


def build_single_dimension_resize_suggestion(
    subject_name: str,
    dimension_index: int,
    enlarge: bool,
) -> Tuple[str, str]:
    dimension_label = DIMENSION_LABELS[dimension_index]
    if enlarge:
        return (
            f"增大 {subject_name} 的 {dimension_label} 维度。",
            f"Increase {subject_name} along the {dimension_label} dimension.",
        )
    return (
        f"减小 {subject_name} 的 {dimension_label} 维度。",
        f"Decrease {subject_name} along the {dimension_label} dimension.",
    )


def build_reference_size_suggestions(
    subject: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
) -> List[Tuple[str, str]]:
    reference_dims = core.pick_reference_dims_for_size_check(subject, reference_dims_map, default_dims_map)
    if not reference_dims:
        return [build_all_dimension_resize_suggestion(subject["name"], enlarge=False)]

    ratios = [
        subject["dims"][idx] / reference_dims[idx] if reference_dims[idx] > 1e-6 else 1.0
        for idx in range(3)
    ]
    enlarge_dims = [idx for idx, ratio in enumerate(ratios) if ratio < 0.92]
    reduce_dims = [idx for idx, ratio in enumerate(ratios) if ratio > 1.08]

    if len(enlarge_dims) == 3:
        return [build_all_dimension_resize_suggestion(subject["name"], enlarge=True)]
    if len(reduce_dims) == 3:
        return [build_all_dimension_resize_suggestion(subject["name"], enlarge=False)]

    suggestions: List[Tuple[str, str]] = []
    for dimension_index in reduce_dims:
        suggestions.append(build_single_dimension_resize_suggestion(subject["name"], dimension_index, enlarge=False))
    for dimension_index in enlarge_dims:
        suggestions.append(build_single_dimension_resize_suggestion(subject["name"], dimension_index, enlarge=True))

    if suggestions:
        return suggestions

    average_ratio = sum(ratios) / len(ratios)
    return [build_all_dimension_resize_suggestion(subject["name"], enlarge=average_ratio < 1.0)]


def build_size_order_suggestions(
    first_subject: Dict[str, Any],
    second_subject: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
) -> List[Tuple[str, str]]:
    first_reference = core.pick_reference_dims_for_size_check(first_subject, reference_dims_map, default_dims_map)
    second_reference = core.pick_reference_dims_for_size_check(second_subject, reference_dims_map, default_dims_map)
    if not first_reference or not second_reference:
        return [build_all_dimension_resize_suggestion(first_subject["name"], enlarge=False)]

    first_ratio = max(first_subject["dims"]) / max(first_reference)
    second_ratio = max(second_subject["dims"]) / max(second_reference)
    suggestions: List[Tuple[str, str]] = []

    if first_ratio > 1.05:
        suggestions.append(build_all_dimension_resize_suggestion(first_subject["name"], enlarge=False))
    if second_ratio < 0.95:
        suggestions.append(build_all_dimension_resize_suggestion(second_subject["name"], enlarge=True))

    if suggestions:
        return suggestions
    return [build_all_dimension_resize_suggestion(first_subject["name"], enlarge=False)]


def build_physical_implausibility_suggestions(
    issue: Dict[str, str],
    subject_lookup: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str]]:
    obj = str(issue.get("object", "")).strip()
    details = str(issue.get("details", "")).strip()
    subject = subject_lookup.get(obj)

    support_match = re.search(r"支撑物\s+(.+?)\s+顶部，合理值应接近\s+(-?\d+(?:\.\d+)?)", details)
    if support_match and subject is not None:
        support_name = support_match.group(1).strip()
        expected_z = float(support_match.group(2))
        suggestions = [
            (
                f"把 {obj} 放在 {support_name} 上。",
                f"Place {obj} on top of {support_name}.",
            )
        ]
        if expected_z > subject["scene_z"]:
            suggestions.append(build_move_suggestion(obj, "z", increase=True))
        elif expected_z < subject["scene_z"]:
            suggestions.append(build_move_suggestion(obj, "z", increase=False))
        return suggestions

    return [
        (
            f"把 {obj} 放到合理支撑面上。",
            f"Place {obj} on a plausible support surface.",
        )
    ]


def suggestions_from_issue(
    issue: Dict[str, str],
    subject_lookup: Dict[str, Dict[str, Any]],
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
) -> List[Tuple[str, str]]:
    category = str(issue.get("category", "")).strip()
    obj = str(issue.get("object", "")).strip()

    if category == "missing_object":
        count = max(1, int(str(issue.get("count", "1"))))
        return [build_object_count_suggestion(obj, count, add=True)]
    if category == "extra_object":
        count = max(1, int(str(issue.get("count", "1"))))
        return [build_object_count_suggestion(obj, count, add=False)]
    if category == "type_count_unverifiable":
        return [
            (
                "请在 prompt 中明确每个对象的类型和数量，例如 a chair 或 two chairs。",
                "Specify each object type and count explicitly in the prompt, for example, a chair or two chairs.",
            )
        ]
    if category == "relation_center_mismatch":
        if obj:
            return [(f"把 {obj} 放在画面中央。", f"Place {obj} at the center of the image.")]
        return []
    if category == "relation_left_mismatch":
        first_name, second_name = parse_object_pair(obj)
        return [(f"把 {first_name} 放到 {second_name} 的左侧。", f"Place {first_name} to the left of {second_name}.")]
    if category == "relation_right_mismatch":
        first_name, second_name = parse_object_pair(obj)
        return [(f"把 {first_name} 放到 {second_name} 的右侧。", f"Place {first_name} to the right of {second_name}.")]
    if category == "relation_front_mismatch":
        first_name, second_name = parse_object_pair(obj)
        return [(f"把 {first_name} 放到 {second_name} 的前方。", f"Place {first_name} in front of {second_name}.")]
    if category == "relation_behind_mismatch":
        first_name, second_name = parse_object_pair(obj)
        return [(f"把 {first_name} 放到 {second_name} 的后方。", f"Place {first_name} behind {second_name}.")]
    if category == "relation_above_mismatch":
        first_name, second_name = parse_object_pair(obj)
        return [(f"把 {first_name} 放到 {second_name} 的上方。", f"Place {first_name} above {second_name}.")]
    if category == "relation_below_mismatch":
        first_name, second_name = parse_object_pair(obj)
        return [(f"把 {first_name} 放到 {second_name} 的下方。", f"Place {first_name} below {second_name}.")]
    if category == "relation_support_mismatch":
        first_name, second_name = parse_object_pair(obj)
        return [(f"把 {first_name} 放在 {second_name} 上。", f"Place {first_name} on top of {second_name}.")]
    if category == "support_move_up":
        return [build_move_suggestion(obj, "z", increase=True)]
    if category == "support_move_down":
        return [build_move_suggestion(obj, "z", increase=False)]
    if category == "support_move_forward":
        return [build_move_suggestion(obj, "x", increase=True)]
    if category == "support_move_backward":
        return [build_move_suggestion(obj, "x", increase=False)]
    if category == "support_move_left":
        return [build_move_suggestion(obj, "y", increase=False)]
    if category == "support_move_right":
        return [build_move_suggestion(obj, "y", increase=True)]
    if category == "frame_clipping":
        return [
            (
                f"将 {obj} 沿 y 轴向图片中心平移；若被 left/right 关系边界卡住，则降低 camera.lens 和 camera.global_scale。",
                f"Move {obj} along the y axis toward the image center; if blocked by left/right relation bounds, decrease camera.lens and camera.global_scale.",
            )
        ]
    if category == "screen_depth_relation":
        return [
            (
                f"用 harness screen-depth 工具调整 {obj} 的前后屏幕位置，直到满足 prompt 分级。",
                f"Use the harness screen-depth tool to adjust {obj}'s front/back screen position until the prompt level is satisfied.",
            )
        ]
    if category == "screen_lateral_gap":
        return [
            (
                f"用 harness screen-lateral 工具调整 {obj} 的左右屏幕间隔，直到满足 prompt 分级。",
                f"Use the harness screen-lateral tool to adjust {obj}'s left/right screen gap until the prompt level is satisfied.",
            )
        ]
    if category == "pairwise_screen_occlusion":
        return [
            (
                f"用 harness pairwise-occlusion 工具分离 {obj}，使任意两个 cube 的屏幕 bbox 遮挡比例不超过 20%。",
                f"Use the harness pairwise-occlusion tool to separate {obj} so any two cubes have no more than 20% projected bbox overlap.",
            )
        ]
    if category == "screen_size_reasonableness":
        return [
            (
                f"用 harness screen-size 工具修正 {obj} 的屏幕 bbox 尺寸比例：先调 x，再尝试 camera.lens/global_scale，必要时补偿 dims。",
                f"Use the harness screen-size tool to correct {obj}'s projected bbox size ratio: adjust x first, then camera.lens/global_scale, and compensate dims only if needed.",
            )
        ]
    if category == "component_compactness":
        return [
            (
                "用 harness component-compactness 工具收紧相机：优先增大 camera.lens，让所有 cube 的 union bbox 达到最低宽高占比，同时保持完整可见和关系约束。",
                "Use the harness component-compactness tool to tighten the camera: increase camera.lens first so the union bbox reaches the minimum width/height ratios while preserving full visibility and relation constraints.",
            )
        ]
    if category == "below_ground":
        return [build_move_suggestion(obj, "z", increase=True)]
    if category == "floating_object":
        return [build_move_suggestion(obj, "z", increase=False)]
    if category == "physical_implausibility":
        return build_physical_implausibility_suggestions(issue, subject_lookup)
    if category in {"size_implausible", "size_suspicious"}:
        subject = subject_lookup.get(obj)
        if subject is not None and subject.get("_screen_size_dim_compensated"):
            return []
        if subject is None:
            return [build_all_dimension_resize_suggestion(obj, enlarge=False)]
        return build_reference_size_suggestions(subject, reference_dims_map, default_dims_map)
    if category == "size_order_conflict":
        first_name, second_name = (obj.split(" vs ", 1) + [""])[:2]
        first_subject = subject_lookup.get(first_name.strip())
        second_subject = subject_lookup.get(second_name.strip())
        if (
            (first_subject is not None and first_subject.get("_screen_size_dim_compensated"))
            or (second_subject is not None and second_subject.get("_screen_size_dim_compensated"))
        ):
            return []
        if first_subject is None or second_subject is None:
            return [build_all_dimension_resize_suggestion(first_name.strip(), enlarge=False)]
        return build_size_order_suggestions(first_subject, second_subject, reference_dims_map, default_dims_map)
    if category == "support_width_oversize":
        subject_name, _ = parse_object_pair(obj)
        return [build_single_dimension_resize_suggestion(subject_name, 0, enlarge=False)]
    if category == "support_depth_oversize":
        subject_name, _ = parse_object_pair(obj)
        return [build_single_dimension_resize_suggestion(subject_name, 1, enlarge=False)]
    if category == "unknown_asset_type":
        return [
            (
                f"请将 {obj} 的类型改为可识别资产类型，并保证数量与文字一致。",
                f"Change {obj} to a recognized asset type and keep counts consistent with the text.",
            )
        ]
    return []


def build_guidance_suggestions(
    issues: List[Dict[str, str]],
    subjects: List[Dict[str, Any]],
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
) -> Tuple[List[str], List[str]]:
    subject_lookup = {subject["name"]: subject for subject in subjects}
    seen: Set[Tuple[str, str]] = set()
    suggestions_cn: List[str] = []
    suggestions_en: List[str] = []

    for issue in core.deduplicate_issues(issues):
        for suggestion_cn, suggestion_en in suggestions_from_issue(
            issue,
            subject_lookup,
            reference_dims_map,
            default_dims_map,
        ):
            normalized_cn = core.normalize_free_text(suggestion_cn)
            normalized_en = core.normalize_free_text(suggestion_en)
            key = (normalized_cn, normalized_en)
            if not normalized_cn or key in seen:
                continue
            seen.add(key)
            suggestions_cn.append(suggestion_cn)
            suggestions_en.append(suggestion_en)
    return suggestions_cn, suggestions_en


def format_inline_numbered(items: List[str]) -> str:
    return " ".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1))


def main() -> None:
    args = parse_args()

    scene_text = args.scene_text.strip()
    if not scene_text:
        raise RuntimeError("scene-text 不能为空。")

    scene_pkl_path = normalize_scene_pkl_path(args.scene_pkl)
    asset_dimensions = core.load_asset_dimensions(core.ASSET_DIMENSIONS_PATH)
    default_dims_map = core.build_default_dims_map(asset_dimensions)
    reference_dims_map = default_dims_map
    allowed_types = list(asset_dimensions.keys())

    scene_dict = core.load_scene_pkl(scene_pkl_path)
    subjects = core.summarize_scene_subjects(scene_dict, allowed_types, reference_dims_map)
    camera = core.summarize_camera(scene_dict)

    if args.object_manifest.strip():
        object_manifest = harness.load_json(args.object_manifest.strip())
        object_manifest = harness.normalize_manifest(object_manifest, allowed_types)
        object_source = "object_manifest"
    else:
        object_manifest = harness.build_object_manifest(
            scene_text=scene_text,
            allowed_types=allowed_types,
            objects_spec=args.objects,
            on_ambiguous="best_effort",
        )
        object_source = "objects_spec" if args.objects.strip() else "prompt"

    local_checks = core.run_local_checks(
        subjects=subjects,
        camera_elevation_deg=camera["camera_elevation_deg"],
        lens_mm=camera["lens_mm"],
        global_scale=camera["global_scale"],
        reference_dims_map=reference_dims_map,
        default_dims_map=default_dims_map,
        scene_text=scene_text,
    )

    criterion_a_pass, criterion_a_issues, parse_warnings, parsed_mentions = evaluate_criterion_a(
        scene_text=scene_text,
        subjects=subjects,
        allowed_types=allowed_types,
        expected_manifest=object_manifest if (args.objects.strip() or args.object_manifest.strip()) else None,
        expected_source=object_source,
    )
    if args.predicates_json.strip():
        predicates_payload = harness.load_json(args.predicates_json.strip())
    else:
        predicates_payload = harness.extract_predicates(scene_text, object_manifest)
    default_azimuths_map = harness.load_asset_default_azimuths(allowed_types)
    camera_facing_payload = harness.build_camera_facing_constraints(scene_text, object_manifest, default_azimuths_map)

    predicate_support_map = support_map_from_predicates(subjects, object_manifest, predicates_payload)
    criterion_b_pass, criterion_b_issues = evaluate_criterion_b(
        scene_text,
        subjects,
        support_map=predicate_support_map,
    )

    support_physical_issues, support_size_issues = evaluate_support_specific_issues(
        scene_text,
        subjects,
        support_map=predicate_support_map,
    )

    physical_categories = {"physical_implausibility", "below_ground", "floating_object"}
    size_categories = {"size_implausible", "size_suspicious", "size_order_conflict"}
    clipping_categories = {"frame_clipping"}

    local_check_issues = core.deduplicate_issues(local_checks.get("hard_issues", []) + local_checks.get("soft_issues", []))
    suppress_predicate_backed_support = bool(predicate_support_map)
    physical_issues = [
        issue
        for issue in local_check_issues
        if issue.get("category") in physical_categories
        and not (
            suppress_predicate_backed_support
            and issue.get("category") in {"physical_implausibility", "floating_object"}
        )
    ]
    size_issues = [
        issue
        for issue in local_check_issues
        if issue.get("category") in size_categories
        and not (
            suppress_predicate_backed_support
            and issue.get("category") in {"size_implausible", "size_suspicious", "size_order_conflict"}
        )
    ]
    clipping_issues = [
        issue
        for issue in local_check_issues
        if issue.get("category") in clipping_categories
    ]

    physical_issues = core.deduplicate_issues(physical_issues + support_physical_issues)
    size_issues = core.deduplicate_issues(size_issues + support_size_issues)

    criterion_c_pass = len(physical_issues) == 0
    criterion_d_pass = len(size_issues) == 0
    criterion_e_pass = len(clipping_issues) == 0
    criterion_f_pass = len(parse_warnings) == 0

    prompt_rule_issues = [
        {
            "category": "type_count_unverifiable",
            "object": "",
            "details": warning,
        }
        for warning in parse_warnings
    ]

    all_issues = core.deduplicate_issues(
        criterion_a_issues
        + criterion_b_issues
        + physical_issues
        + size_issues
        + clipping_issues
        + prompt_rule_issues
    )
    harness_scene, harness_actions = harness.apply_full_harness(
        scene_dict=scene_dict,
        manifest=object_manifest,
        predicates_payload=predicates_payload,
        reference_dims_map=default_dims_map,
        camera_facing_payload=camera_facing_payload,
        default_azimuths_map=default_azimuths_map,
    )
    harness_validation = harness.validate_scene(
        scene_dict=scene_dict,
        manifest=object_manifest,
        predicates_payload=predicates_payload,
        reference_dims_map=default_dims_map,
        camera_facing_payload=camera_facing_payload,
        default_azimuths_map=default_azimuths_map,
    )
    harness_repaired_validation = harness.validate_scene(
        scene_dict=harness_scene,
        manifest=object_manifest,
        predicates_payload=predicates_payload,
        reference_dims_map=default_dims_map,
        camera_facing_payload=camera_facing_payload,
        default_azimuths_map=default_azimuths_map,
    )
    harness_repair_plan = harness.build_repair_plan(
        original_scene=scene_dict,
        repaired_scene=harness_scene,
        actions=harness_actions,
        manifest_path=args.object_manifest.strip(),
        predicates_path=args.predicates_json.strip(),
    )
    all_issues = core.deduplicate_issues(
        all_issues
        + [
            {
                "category": str(issue.get("category", "")),
                "object": str(issue.get("object", "")),
                "details": str(issue.get("details") or issue.get("expected") or ""),
            }
            for issue in harness_validation.get("issues", [])
        ]
    )
    suggestions_cn, suggestions_en = build_guidance_suggestions(
        issues=all_issues,
        subjects=subjects,
        reference_dims_map=default_dims_map,
        default_dims_map=default_dims_map,
    )
    if not suggestions_cn and all_issues:
        suggestions_cn = ["请按 guidance.md 规范，明确对象并给出可执行的尺寸/位置调整语句。"]
        suggestions_en = ["Follow guidance.md and provide explicit, executable size/position directives."]

    feedback_repair_plan = harness.build_feedback_repair_plan(
        scene_dict=scene_dict,
        issues=all_issues,
        manifest=object_manifest,
        reference_dims_map=default_dims_map,
        predicates_payload=predicates_payload,
        camera_facing_payload=camera_facing_payload,
        default_azimuths_map=default_azimuths_map,
    )
    combined_repair_plan = harness.merge_repair_plans(harness_repair_plan, feedback_repair_plan)

    overall_pass = (
        criterion_a_pass
        and criterion_b_pass
        and criterion_c_pass
        and criterion_d_pass
        and criterion_e_pass
        and criterion_f_pass
        and harness_validation["overall_pass"]
    )

    result = {
        "overall_pass": overall_pass,
        "criteria": {
            "A_type_count_match": {"pass": criterion_a_pass, "issues": criterion_a_issues},
            "B_relation_match": {"pass": criterion_b_pass, "issues": criterion_b_issues},
            "C_physical_plausibility": {"pass": criterion_c_pass, "issues": physical_issues},
            "D_size_reasonable": {"pass": criterion_d_pass, "issues": size_issues},
            "E_full_cube_visible": {"pass": criterion_e_pass, "issues": clipping_issues},
            "F_prompt_count_rule": {"pass": criterion_f_pass, "issues": prompt_rule_issues},
        },
        "parsed_mentions": parsed_mentions,
        "parse_warnings": parse_warnings,
        "suggestions": suggestions_cn,
        "suggestions_cn": suggestions_cn,
        "suggestions_en": suggestions_en,
        "harness": {
            "object_manifest": object_manifest,
            "predicates": predicates_payload,
            "camera_facing": camera_facing_payload,
            "validation": harness_validation,
            "repaired_validation": harness_repaired_validation,
            "repair_action_count": combined_repair_plan["action_count"],
            "constraint_repair_action_count": len(harness_actions),
            "feedback_repair_action_count": feedback_repair_plan["action_count"],
            "repair_plan": combined_repair_plan,
            "constraint_repair_plan": harness_repair_plan,
            "feedback_repair_plan": feedback_repair_plan,
        },
    }

    print("评估结论：")
    print(f"- A 物体类型与数量完全匹配: {'通过' if criterion_a_pass else '不通过'}")
    print(f"- B 位置关系与文字完全匹配: {'通过' if criterion_b_pass else '不通过'}")
    print(f"- C 位置物理合理性: {'通过' if criterion_c_pass else '不通过'}")
    print(f"- D 相对尺寸合理性: {'通过' if criterion_d_pass else '不通过'}")
    print(f"- E 每个 cube 完整可见: {'通过' if criterion_e_pass else '不通过'}")
    print(f"- F prompt 计数规则解析稳定: {'通过' if criterion_f_pass else '不通过'}")

    if parse_warnings:
        print("解析提示：")
        for warning in parse_warnings:
            print(f"- {warning}")

    if all_issues:
        print("发现问题：")
        for issue in all_issues:
            print(f"- [{issue.get('category', '')}] {issue.get('object', '')} {issue.get('details', '')}".strip())
    else:
        print("发现问题：无")

    fallback_cn = ["当前场景满足 A/B/C/D/E/F，可保持不变。"]
    fallback_en = ["Current scene already satisfies A/B/C/D/E/F; keep it unchanged."]
    printable_cn = suggestions_cn or fallback_cn
    printable_en = suggestions_en or fallback_en

    print("修改意见（中文）：")
    print(format_inline_numbered(printable_cn))
    print("Modification Guidance (English):")
    print(format_inline_numbered(printable_en))

    guidance_text = " ".join(suggestions_cn).strip()
    if args.guidance_output.strip():
        guidance_output_path = args.guidance_output.strip()
        if not os.path.isabs(guidance_output_path):
            guidance_output_path = os.path.join(core.REPO_ROOT, guidance_output_path)
        guidance_output_dir = os.path.dirname(guidance_output_path)
        if guidance_output_dir:
            os.makedirs(guidance_output_dir, exist_ok=True)
        with open(guidance_output_path, "w", encoding="utf-8") as handle:
            handle.write(guidance_text + ("\n" if guidance_text else ""))
        print(f"已写出修改意见文本：{guidance_output_path}")

    if args.json_output.strip():
        json_output_path = args.json_output.strip()
        if not os.path.isabs(json_output_path):
            json_output_path = os.path.join(core.REPO_ROOT, json_output_path)
        json_output_dir = os.path.dirname(json_output_path)
        if json_output_dir:
            os.makedirs(json_output_dir, exist_ok=True)
        with open(json_output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"已写出评估 JSON：{json_output_path}")

    if args.repair_plan_output.strip():
        repair_plan_output_path = args.repair_plan_output.strip()
        if not os.path.isabs(repair_plan_output_path):
            repair_plan_output_path = os.path.join(core.REPO_ROOT, repair_plan_output_path)
        harness.save_json(combined_repair_plan, repair_plan_output_path)
        print(f"已写出可执行修复计划：{repair_plan_output_path}")


if __name__ == "__main__":
    main()
