#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅做“按 guidance 修正 pkl”：
1) 输入 scene_text + scene_pkl + guidance
2) 严格按 agent_check_pkl_v5.py 的 guidance 规则改动
3) 只修改 guidance 中明确提到的对象和参数
4) 输出前严格核验；若未完全执行到位，则继续迭代修改
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import agent_check_pkl_v5 as core


MAX_APPLY_ATTEMPTS = 4
RELATION_ABOVE_PATTERN = r"(?:above|higher than|高于|上方)"
RELATION_BELOW_PATTERN = r"(?:below|beneath|lower than|低于|下方)"
TIGHT_CONTACT_TOKENS = (
    "紧贴",
    "贴在",
    "贴着",
    "紧挨",
    "紧靠",
    "flush",
    "tightly",
    "touching",
    "in contact with",
)
REFERENCE_WIDTH_TOKENS = ("width", "宽度", "width dimension")
REFERENCE_DEPTH_TOKENS = ("depth", "深度", "depth dimension")
REFERENCE_HEIGHT_TOKENS = ("height", "高度", "height dimension")
ADD_OBJECT_KEYWORDS = ("增加物体", "新增物体", "添加物体", "增加", "新增", "添加", "add")
DELETE_OBJECT_KEYWORDS = ("删除物体", "移除物体", "删除", "移除", "delete", "remove")
OBJECT_NAME_STOP_TOKENS = {
    "on",
    "to",
    "in",
    "at",
    "of",
    "left",
    "right",
    "front",
    "back",
    "behind",
    "above",
    "below",
    "forward",
    "backward",
    "up",
    "down",
    "move",
    "shift",
    "place",
    "put",
    "keep",
    "with",
    "and",
    "or",
    "上",
    "下",
    "左",
    "右",
    "前",
    "后",
    "上方",
    "下方",
    "左边",
    "右边",
    "前面",
    "后面",
    "放",
    "放在",
    "放到",
    "平移",
    "移动",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="严格执行 guidance 并输出修正版 pkl（不做 LLM 评估循环）。",
    )
    parser.add_argument("--scene-text", required=True, help="场景文字描述。")
    parser.add_argument("--scene-pkl", required=True, help="输入 pkl 路径。")
    parser.add_argument("--guidance", required=True, help="修改意见文本。")
    parser.add_argument("-o", "--output", required=True, help="输出 pkl 路径。")
    return parser.parse_args()


def normalize_existing_path(path: str, path_name: str) -> str:
    normalized = path.strip()
    if not normalized:
        raise RuntimeError(f"{path_name} 不能为空。")
    if not os.path.isabs(normalized):
        normalized = os.path.join(core.REPO_ROOT, normalized)
    if not os.path.isfile(normalized):
        raise RuntimeError(f"找不到文件：{normalized}")
    return normalized


def normalize_output_path(path: str) -> str:
    normalized = path.strip()
    if not normalized:
        raise RuntimeError("output 不能为空。")
    if not os.path.isabs(normalized):
        normalized = os.path.join(core.REPO_ROOT, normalized)
    output_dir = os.path.dirname(normalized)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return normalized


def build_runtime_context() -> Dict[str, Any]:
    asset_dimensions = core.load_asset_dimensions(core.ASSET_DIMENSIONS_PATH)
    object_scales = core.load_object_scales(core.OBJECT_SCALES_PATH)
    default_dims_map = core.build_default_dims_map(asset_dimensions)
    reference_dims_map = core.build_reference_dims_map(asset_dimensions, object_scales)
    return {
        "asset_dimensions": asset_dimensions,
        "default_dims_map": default_dims_map,
        "reference_dims_map": reference_dims_map,
        "allowed_types": list(asset_dimensions.keys()),
    }


def strict_collect_guidance_target_names(
    subjects: List[Dict[str, Any]],
    guidance_text: str,
) -> Set[str]:
    guidance_norm = core.normalize_free_text(guidance_text)
    target_names: Set[str] = set()
    for subject in subjects:
        name_norm = core.canonicalize_type(subject["name"])
        if not name_norm:
            continue
        if re.search(r"\d+$", name_norm):
            pattern = rf"(?<![a-z0-9]){re.escape(name_norm)}(?![a-z0-9])"
        else:
            pattern = rf"(?<![a-z0-9]){re.escape(name_norm)}(?![a-z0-9]|\s*\d)"
        if re.search(pattern, guidance_norm):
            target_names.add(subject["name"])
    return target_names


def build_empty_subject_scope() -> Dict[str, Any]:
    return {
        "scene_x": False,
        "scene_y": False,
        "scene_z": False,
        "azimuth_deg": False,
        "dim_indices": set(),
    }


def normalize_guidance_clause_text(clause: str) -> str:
    return re.sub(r"^\s*\d+\s*[\.\):：-]?\s*", "", str(clause or "").strip())


def ordered_relation_matches(
    clause: str,
    first_subject: Dict[str, Any],
    second_subject: Dict[str, Any],
    relation_pattern: str,
) -> bool:
    normalized_clause = core.normalize_free_text(clause)
    for first_alias in core.build_subject_aliases(first_subject):
        if not core.text_contains_alias(normalized_clause, first_alias):
            continue
        for second_alias in core.build_subject_aliases(second_subject):
            if first_alias == second_alias:
                continue
            patterns = (
                rf"{re.escape(first_alias)}.*?{relation_pattern}.*?{re.escape(second_alias)}",
                rf"{re.escape(first_alias)}.*?{re.escape(second_alias)}.*?{relation_pattern}",
            )
            if any(re.search(pattern, normalized_clause) for pattern in patterns):
                return True
    return False


def clause_requests_tight_contact(clause: str) -> bool:
    return core.clause_contains_any(clause, TIGHT_CONTACT_TOKENS)


def guidance_requests_pair_relation(
    guidance_text: str,
    first_subject: Dict[str, Any],
    second_subject: Dict[str, Any],
    relation_pattern: str,
) -> bool:
    for clause in core.split_text_clauses(guidance_text):
        if ordered_relation_matches(clause, first_subject, second_subject, relation_pattern):
            return True
    return False


def extract_object_phrase_from_tail(tail_text: str) -> str:
    token_pattern = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?|\d+|[\u4e00-\u9fff]+")
    raw_tokens = token_pattern.findall(tail_text.strip())
    if not raw_tokens:
        return ""

    tokens: List[str] = []
    skip_leading = {"a", "an", "the", "物体", "object"}
    started = False
    for raw_token in raw_tokens:
        token = raw_token.strip()
        lower_token = token.lower()
        if not started and lower_token in skip_leading:
            continue
        if started and lower_token in OBJECT_NAME_STOP_TOKENS:
            break
        if started and token in OBJECT_NAME_STOP_TOKENS:
            break
        started = True
        tokens.append(token)
        if len(tokens) >= 4:
            break

    if not tokens:
        return ""
    return core.sanitize_name(" ".join(tokens), default_name="")


def extract_subject_name_after_keywords(clause: str, keywords: Tuple[str, ...]) -> str:
    normalized_clause = normalize_guidance_clause_text(clause)
    clause_lower = normalized_clause.lower()
    for keyword in keywords:
        keyword_lower = keyword.lower()
        index = clause_lower.find(keyword_lower)
        if index == -1:
            continue
        tail = normalized_clause[index + len(keyword) :]
        phrase = extract_object_phrase_from_tail(tail)
        if phrase:
            return phrase
    return ""


def make_unique_subject_name(base_name: str, existing_subjects: List[Dict[str, Any]]) -> str:
    candidate = core.sanitize_name(base_name, default_name="object")
    if not candidate:
        candidate = "object"
    existing_names = {subject["name"] for subject in existing_subjects}
    if candidate not in existing_names:
        return candidate
    suffix = 2
    while f"{candidate} {suffix}" in existing_names:
        suffix += 1
    return f"{candidate} {suffix}"


def estimate_new_subject_position(
    current_subjects: List[Dict[str, Any]],
    dims: List[float],
) -> Tuple[float, float, float]:
    if not current_subjects:
        return 0.0, 0.0, 0.0

    mean_x = sum(subject["scene_x"] for subject in current_subjects) / len(current_subjects)
    mean_y = sum(subject["scene_y"] for subject in current_subjects) / len(current_subjects)
    radius = max(max(dims[0], dims[1]), 0.6) + 0.4 * len(current_subjects)
    return round(mean_x + radius, 6), round(mean_y, 6), 0.0


def apply_structural_guidance(
    subjects: List[Dict[str, Any]],
    guidance_text: str,
    allowed_types: List[str],
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    updated_subjects = core.clone_subjects(subjects)
    affected_names: Set[str] = set()

    for clause in core.split_text_clauses(guidance_text):
        normalized_clause = normalize_guidance_clause_text(clause)
        if not normalized_clause:
            continue

        add_name = extract_subject_name_after_keywords(normalized_clause, ADD_OBJECT_KEYWORDS)
        if add_name:
            asset_type = core.match_asset_type(None, add_name, allowed_types)
            reference_dims = reference_dims_map.get(asset_type) or default_dims_map.get(asset_type) or [1.0, 1.0, 1.0]
            scene_x, scene_y, scene_z = estimate_new_subject_position(updated_subjects, reference_dims)
            unique_name = make_unique_subject_name(add_name, updated_subjects)
            updated_subjects.append(
                {
                    "name": unique_name,
                    "type": asset_type,
                    "dims": [round(float(value), 6) for value in reference_dims],
                    "scene_x": scene_x,
                    "scene_y": scene_y,
                    "scene_z": scene_z,
                    "azimuth_deg": 0.0,
                }
            )
            affected_names.add(unique_name)
            continue

        delete_name = extract_subject_name_after_keywords(normalized_clause, DELETE_OBJECT_KEYWORDS)
        if delete_name:
            delete_norm = core.canonicalize_type(delete_name)
            next_subjects: List[Dict[str, Any]] = []
            removed_names: List[str] = []
            for subject in updated_subjects:
                subject_name_norm = core.canonicalize_type(subject["name"])
                subject_base_norm = re.sub(r"\s+\d+$", "", subject_name_norm).strip()
                subject_type_norm = core.canonicalize_type(subject["type"])
                if (
                    delete_norm
                    and (
                        subject_name_norm == delete_norm
                        or subject_base_norm == delete_norm
                        or subject_type_norm == delete_norm
                    )
                ):
                    removed_names.append(subject["name"])
                    continue
                next_subjects.append(subject)
            if removed_names:
                updated_subjects = next_subjects
                affected_names.update(removed_names)

    return updated_subjects, affected_names


def build_camera_scope(
    original_camera: Dict[str, float],
    guidance_target_camera: Dict[str, float],
) -> Dict[str, bool]:
    return {
        "camera_elevation_deg": abs(
            guidance_target_camera["camera_elevation_deg"] - original_camera["camera_elevation_deg"]
        )
        > 1e-6,
        "lens_mm": abs(guidance_target_camera["lens_mm"] - original_camera["lens_mm"]) > 1e-6,
        "global_scale": abs(guidance_target_camera["global_scale"] - original_camera["global_scale"])
        > 1e-6,
    }


def parse_numeric_multiplier(value_text: str) -> Optional[float]:
    normalized = value_text.strip().lower()
    if not normalized:
        return None
    if "/" in normalized:
        parts = normalized.split("/", 1)
        if len(parts) != 2:
            return None
        try:
            numerator = float(parts[0].strip())
            denominator = float(parts[1].strip())
        except ValueError:
            return None
        if abs(denominator) <= 1e-9:
            return None
        return numerator / denominator
    try:
        return float(normalized)
    except ValueError:
        return None


def extract_clause_multiplier(clause: str) -> Optional[float]:
    normalized_clause = core.normalize_free_text(clause)
    patterns = (
        r"(?:现在的|current)\s*([0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+(?:\.[0-9]+)?)?)",
        r"(?:现在的|current)?\s*([0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+(?:\.[0-9]+)?)?)\s*倍",
        r"([0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+(?:\.[0-9]+)?)?)\s*(?:x|times?)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized_clause)
        if not match:
            continue
        multiplier = parse_numeric_multiplier(match.group(1))
        if multiplier is not None and multiplier > 0.0:
            return multiplier
    return None


def extract_numeric_dimension_multipliers(
    subject: Dict[str, Any],
    guidance_text: str,
) -> Dict[int, float]:
    dim_multipliers: Dict[int, float] = {}
    for clause in core.subject_relevant_clauses(guidance_text, subject):
        multiplier = extract_clause_multiplier(clause)
        if multiplier is None:
            continue

        touched = False
        if core.clause_requests_dimension_increase(
            clause,
            core.GUIDANCE_WIDTH_TOKENS,
            core.GUIDANCE_WIDER_TOKENS,
        ) or core.clause_requests_dimension_decrease(
            clause,
            core.GUIDANCE_WIDTH_TOKENS,
            core.GUIDANCE_NARROWER_TOKENS,
        ):
            dim_multipliers[0] = multiplier
            touched = True
        if core.clause_requests_dimension_increase(
            clause,
            core.GUIDANCE_DEPTH_TOKENS,
            core.GUIDANCE_DEEPER_TOKENS,
        ) or core.clause_requests_dimension_decrease(
            clause,
            core.GUIDANCE_DEPTH_TOKENS,
            core.GUIDANCE_SHALLOWER_TOKENS,
        ):
            dim_multipliers[1] = multiplier
            touched = True
        if core.clause_requests_dimension_increase(
            clause,
            core.GUIDANCE_HEIGHT_TOKENS,
            core.GUIDANCE_TALLER_TOKENS,
        ) or core.clause_requests_dimension_decrease(
            clause,
            core.GUIDANCE_HEIGHT_TOKENS,
            core.GUIDANCE_SHORTER_TOKENS,
        ):
            dim_multipliers[2] = multiplier
            touched = True
        if (
            not touched
            and core.clause_contains_any(clause, core.GUIDANCE_ALL_DIMENSION_TOKENS)
            and (
                core.clause_contains_any(clause, core.ENLARGE_HINT_TOKENS)
                or core.clause_contains_any(clause, core.REDUCE_HINT_TOKENS)
            )
        ):
            dim_multipliers[0] = multiplier
            dim_multipliers[1] = multiplier
            dim_multipliers[2] = multiplier

    return dim_multipliers


def extract_reference_dimension_multiplier(
    clause: str,
    reference_subject: Dict[str, Any],
    dimension_tokens: Tuple[str, ...],
) -> Optional[float]:
    normalized_clause = core.normalize_free_text(clause)
    number_pattern = r"([0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+(?:\.[0-9]+)?)?)"
    dimension_pattern = "|".join(re.escape(token) for token in dimension_tokens)

    for alias in core.build_subject_aliases(reference_subject):
        alias_pattern = re.escape(alias)
        patterns = (
            rf"{number_pattern}\s*个?\s*{alias_pattern}\s*的\s*(?:{dimension_pattern})",
            rf"{number_pattern}\s*(?:times?\s*)?(?:the\s*)?(?:{dimension_pattern})\s*of\s*{alias_pattern}",
            rf"{number_pattern}\s*(?:times?\s*)?(?:{alias_pattern}(?:'s)?\s*)?(?:{dimension_pattern})",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized_clause)
            if not match:
                continue
            multiplier = parse_numeric_multiplier(match.group(1))
            if multiplier is not None and multiplier > 0.0:
                return multiplier
    return None


def collect_relative_axis_move_instructions(
    subjects: List[Dict[str, Any]],
    guidance_text: str,
    movable_subject_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    instructions: List[Dict[str, Any]] = []
    clauses = core.split_text_clauses(guidance_text)
    if not clauses:
        return instructions

    for clause in clauses:
        for subject in subjects:
            if movable_subject_names is not None and subject["name"] not in movable_subject_names:
                continue
            if not core.clause_mentions_subject(clause, subject):
                continue

            for reference_subject in subjects:
                if reference_subject is subject:
                    continue
                if not core.clause_mentions_subject(clause, reference_subject):
                    continue

                depth_multiplier = extract_reference_dimension_multiplier(
                    clause,
                    reference_subject,
                    REFERENCE_DEPTH_TOKENS,
                )
                width_multiplier = extract_reference_dimension_multiplier(
                    clause,
                    reference_subject,
                    REFERENCE_WIDTH_TOKENS,
                )
                height_multiplier = extract_reference_dimension_multiplier(
                    clause,
                    reference_subject,
                    REFERENCE_HEIGHT_TOKENS,
                )

                if depth_multiplier is not None:
                    if core.clause_requests_guidance_move(clause, core.GUIDANCE_MOVE_FRONT_TOKENS):
                        instructions.append(
                            {
                                "subject_name": subject["name"],
                                "reference_name": reference_subject["name"],
                                "axis": "scene_x",
                                "dim_index": 1,
                                "sign": 1.0,
                                "multiplier": depth_multiplier,
                            }
                        )
                    if core.clause_requests_guidance_move(clause, core.GUIDANCE_MOVE_BACK_TOKENS):
                        instructions.append(
                            {
                                "subject_name": subject["name"],
                                "reference_name": reference_subject["name"],
                                "axis": "scene_x",
                                "dim_index": 1,
                                "sign": -1.0,
                                "multiplier": depth_multiplier,
                            }
                        )

                if width_multiplier is not None:
                    if core.clause_requests_guidance_move(clause, core.GUIDANCE_MOVE_RIGHT_TOKENS):
                        instructions.append(
                            {
                                "subject_name": subject["name"],
                                "reference_name": reference_subject["name"],
                                "axis": "scene_y",
                                "dim_index": 0,
                                "sign": 1.0,
                                "multiplier": width_multiplier,
                            }
                        )
                    if core.clause_requests_guidance_move(clause, core.GUIDANCE_MOVE_LEFT_TOKENS):
                        instructions.append(
                            {
                                "subject_name": subject["name"],
                                "reference_name": reference_subject["name"],
                                "axis": "scene_y",
                                "dim_index": 0,
                                "sign": -1.0,
                                "multiplier": width_multiplier,
                            }
                        )

                if height_multiplier is not None:
                    if core.clause_requests_guidance_move(clause, core.GUIDANCE_MOVE_UP_TOKENS):
                        instructions.append(
                            {
                                "subject_name": subject["name"],
                                "reference_name": reference_subject["name"],
                                "axis": "scene_z",
                                "dim_index": 2,
                                "sign": 1.0,
                                "multiplier": height_multiplier,
                            }
                        )
                    if core.clause_requests_guidance_move(clause, core.GUIDANCE_MOVE_DOWN_TOKENS):
                        instructions.append(
                            {
                                "subject_name": subject["name"],
                                "reference_name": reference_subject["name"],
                                "axis": "scene_z",
                                "dim_index": 2,
                                "sign": -1.0,
                                "multiplier": height_multiplier,
                            }
                        )

    return instructions


def apply_manual_relative_axis_movement_guidance(
    subjects: List[Dict[str, Any]],
    baseline_subjects: List[Dict[str, Any]],
    guidance_text: str,
    movable_subject_names: Optional[Set[str]] = None,
) -> None:
    instructions = collect_relative_axis_move_instructions(
        subjects,
        guidance_text,
        movable_subject_names=movable_subject_names,
    )
    if not instructions:
        return

    axis_totals: Dict[Tuple[str, str], float] = {}
    for instruction in instructions:
        subject = core.find_matching_subject({"name": instruction["subject_name"]}, subjects)
        reference_subject = core.find_matching_subject({"name": instruction["reference_name"]}, subjects)
        if subject is None or reference_subject is None:
            continue
        delta = reference_subject["dims"][instruction["dim_index"]] * instruction["multiplier"] * instruction["sign"]
        key = (subject["name"], instruction["axis"])
        axis_totals[key] = axis_totals.get(key, 0.0) + delta

    for subject_name, axis_name in axis_totals:
        subject = core.find_matching_subject({"name": subject_name}, subjects)
        baseline_subject = core.find_matching_subject({"name": subject_name}, baseline_subjects)
        if subject is None or baseline_subject is None:
            continue
        subject[axis_name] = round(baseline_subject[axis_name] + axis_totals[(subject_name, axis_name)], 6)


def collect_relative_axis_movement_issues(
    actual_subjects: List[Dict[str, Any]],
    baseline_subjects: List[Dict[str, Any]],
    guidance_text: str,
    movable_subject_names: Optional[Set[str]] = None,
) -> List[Dict[str, str]]:
    expected_subjects = core.clone_subjects(baseline_subjects)
    apply_manual_numeric_dimension_guidance(
        expected_subjects,
        original_subjects=baseline_subjects,
        guidance_text=guidance_text,
    )
    apply_manual_relative_axis_movement_guidance(
        expected_subjects,
        baseline_subjects=baseline_subjects,
        guidance_text=guidance_text,
        movable_subject_names=movable_subject_names,
    )

    issues: List[Dict[str, str]] = []
    tolerance = 1e-4
    instructions = collect_relative_axis_move_instructions(
        expected_subjects,
        guidance_text,
        movable_subject_names=movable_subject_names,
    )
    seen_keys: Set[Tuple[str, str]] = set()

    for instruction in instructions:
        key = (instruction["subject_name"], instruction["axis"])
        if key in seen_keys:
            continue
        seen_keys.add(key)

        actual_subject = core.find_matching_subject({"name": instruction["subject_name"]}, actual_subjects)
        expected_subject = core.find_matching_subject({"name": instruction["subject_name"]}, expected_subjects)
        if actual_subject is None or expected_subject is None:
            continue

        axis_name = instruction["axis"]
        if abs(actual_subject[axis_name] - expected_subject[axis_name]) > tolerance:
            issues.append(
                {
                    "category": "guidance_not_applied",
                    "object": instruction["subject_name"],
                    "details": (
                        f"guidance 要求按参考物体尺寸调整 {axis_name}，"
                        f"预期应为 {expected_subject[axis_name]}，当前为 {actual_subject[axis_name]}。"
                    ),
                }
            )

    return core.deduplicate_issues(issues)


def collect_subject_parameter_scope(
    subjects: List[Dict[str, Any]],
    guidance_text: str,
    guidance_target_names: Set[str],
) -> Dict[str, Dict[str, Any]]:
    scopes = {subject["name"]: build_empty_subject_scope() for subject in subjects}
    if not guidance_text.strip():
        return scopes

    relative_move_instructions = collect_relative_axis_move_instructions(
        subjects,
        guidance_text,
        movable_subject_names=guidance_target_names,
    )
    for instruction in relative_move_instructions:
        scope = scopes.get(instruction["subject_name"])
        if scope is None:
            continue
        scope[instruction["axis"]] = True

    for subject in subjects:
        if subject["name"] not in guidance_target_names:
            continue

        scope = scopes[subject["name"]]
        clauses = core.subject_relevant_clauses(guidance_text, subject)
        if abs(core.subject_size_multiplier_from_guidance(subject, guidance_text) - 1.0) > 1e-6:
            scope["dim_indices"].update({0, 1, 2})
        scope["dim_indices"].update(extract_numeric_dimension_multipliers(subject, guidance_text).keys())

        for clause in clauses:
            clause_other_subjects = [
                other_subject
                for other_subject in subjects
                if other_subject is not subject and core.clause_mentions_subject(clause, other_subject)
            ]
            is_single_subject_move_clause = not clause_other_subjects

            if is_single_subject_move_clause and core.clause_requests_guidance_move(clause, core.GUIDANCE_MOVE_LEFT_TOKENS):
                scope["scene_y"] = True
            if is_single_subject_move_clause and core.clause_requests_guidance_move(clause, core.GUIDANCE_MOVE_RIGHT_TOKENS):
                scope["scene_y"] = True
            if is_single_subject_move_clause and core.clause_requests_guidance_move(clause, core.GUIDANCE_MOVE_FRONT_TOKENS):
                scope["scene_x"] = True
            if is_single_subject_move_clause and core.clause_requests_guidance_move(clause, core.GUIDANCE_MOVE_BACK_TOKENS):
                scope["scene_x"] = True
            if is_single_subject_move_clause and core.clause_requests_guidance_move(clause, core.GUIDANCE_MOVE_UP_TOKENS):
                scope["scene_z"] = True
            if is_single_subject_move_clause and core.clause_requests_guidance_move(clause, core.GUIDANCE_MOVE_DOWN_TOKENS):
                scope["scene_z"] = True
            if core.clause_contains_any(clause, core.CENTER_HINT_TOKENS):
                scope["scene_x"] = True
                scope["scene_y"] = True
            if core.clause_requests_dimension_increase(
                clause,
                core.GUIDANCE_WIDTH_TOKENS,
                core.GUIDANCE_WIDER_TOKENS,
            ) or core.clause_requests_dimension_decrease(
                clause,
                core.GUIDANCE_WIDTH_TOKENS,
                core.GUIDANCE_NARROWER_TOKENS,
            ):
                scope["dim_indices"].add(0)
            if core.clause_requests_dimension_increase(
                clause,
                core.GUIDANCE_DEPTH_TOKENS,
                core.GUIDANCE_DEEPER_TOKENS,
            ) or core.clause_requests_dimension_decrease(
                clause,
                core.GUIDANCE_DEPTH_TOKENS,
                core.GUIDANCE_SHALLOWER_TOKENS,
            ):
                scope["dim_indices"].add(1)
            if core.clause_requests_dimension_increase(
                clause,
                core.GUIDANCE_HEIGHT_TOKENS,
                core.GUIDANCE_TALLER_TOKENS,
            ) or core.clause_requests_dimension_decrease(
                clause,
                core.GUIDANCE_HEIGHT_TOKENS,
                core.GUIDANCE_SHORTER_TOKENS,
            ):
                scope["dim_indices"].add(2)
            if (
                core.clause_contains_any(clause, core.GUIDANCE_ALL_DIMENSION_TOKENS)
                and (
                    core.clause_contains_any(clause, core.ENLARGE_HINT_TOKENS)
                    or core.clause_contains_any(clause, core.REDUCE_HINT_TOKENS)
                )
            ):
                scope["dim_indices"].update({0, 1, 2})
            if abs(core.extract_rotation_delta_from_clause(clause)) > 1e-6:
                scope["azimuth_deg"] = True

            for other_subject in subjects:
                if other_subject is subject:
                    continue
                if not core.clause_mentions_subject(clause, other_subject):
                    continue

                if ordered_relation_matches(
                    clause,
                    subject,
                    other_subject,
                    core.RELATION_LEFT_PATTERN,
                ) or ordered_relation_matches(
                    clause,
                    subject,
                    other_subject,
                    core.RELATION_RIGHT_PATTERN,
                ):
                    scope["scene_y"] = True
                if ordered_relation_matches(
                    clause,
                    subject,
                    other_subject,
                    core.RELATION_FRONT_PATTERN,
                ) or ordered_relation_matches(
                    clause,
                    subject,
                    other_subject,
                    core.RELATION_BEHIND_PATTERN,
                ):
                    scope["scene_x"] = True
                if ordered_relation_matches(
                    clause,
                    subject,
                    other_subject,
                    RELATION_ABOVE_PATTERN,
                ) or ordered_relation_matches(
                    clause,
                    subject,
                    other_subject,
                    RELATION_BELOW_PATTERN,
                ):
                    scope["scene_x"] = True
                    scope["scene_y"] = True
                    scope["scene_z"] = True
                if (
                    other_subject["type"] in core.SUPPORT_SURFACE_TYPES
                    and core.clause_implies_direct_support(clause, subject, other_subject)
                ):
                    scope["scene_x"] = True
                    scope["scene_y"] = True
                    scope["scene_z"] = True

    return scopes


def restrict_subjects_to_scope(
    original_subjects: List[Dict[str, Any]],
    candidate_subjects: List[Dict[str, Any]],
    subject_scopes: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    restricted_subjects: List[Dict[str, Any]] = []

    for original_subject in original_subjects:
        candidate_subject = core.find_matching_subject(original_subject, candidate_subjects) or original_subject
        scope = subject_scopes.get(original_subject["name"], build_empty_subject_scope())
        allowed_dims = set(scope["dim_indices"])

        restricted_subject = {
            "name": original_subject["name"],
            "type": original_subject["type"],
            "dims": list(original_subject["dims"]),
            "scene_x": original_subject["scene_x"],
            "scene_y": original_subject["scene_y"],
            "scene_z": original_subject["scene_z"],
            "azimuth_deg": original_subject["azimuth_deg"],
        }

        if scope["scene_x"]:
            restricted_subject["scene_x"] = candidate_subject["scene_x"]
        if scope["scene_y"]:
            restricted_subject["scene_y"] = candidate_subject["scene_y"]
        if scope["scene_z"]:
            restricted_subject["scene_z"] = candidate_subject["scene_z"]
        if scope["azimuth_deg"]:
            restricted_subject["azimuth_deg"] = candidate_subject["azimuth_deg"]

        for dim_index in allowed_dims:
            restricted_subject["dims"][dim_index] = candidate_subject["dims"][dim_index]

        restricted_subjects.append(restricted_subject)

    return restricted_subjects


def restrict_camera_to_scope(
    original_camera: Dict[str, float],
    candidate_camera: Dict[str, float],
    camera_scope: Dict[str, bool],
) -> Dict[str, float]:
    restricted_camera = dict(original_camera)
    for field_name, allowed in camera_scope.items():
        if allowed:
            restricted_camera[field_name] = candidate_camera[field_name]
    return restricted_camera


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
    }


def apply_manual_relation_guidance(
    subjects: List[Dict[str, Any]],
    guidance_text: str,
    movable_subject_names: Optional[Set[str]] = None,
) -> None:
    clauses = core.split_text_clauses(guidance_text)
    if not clauses or not subjects:
        return

    for _ in range(4):
        changed = False
        support_map = core.build_support_map(subjects, guidance_text)

        for clause in clauses:
            for first_subject in subjects:
                if movable_subject_names is not None and first_subject["name"] not in movable_subject_names:
                    continue
                if not core.clause_mentions_subject(clause, first_subject):
                    continue

                if core.clause_contains_any(clause, core.CENTER_HINT_TOKENS):
                    target_x = 0.0
                    target_y = 0.0
                    if abs(first_subject["scene_x"] - target_x) > 0.2:
                        first_subject["scene_x"] = round((first_subject["scene_x"] + target_x) / 2.0, 6)
                        changed = True
                    if abs(first_subject["scene_y"] - target_y) > 0.2:
                        first_subject["scene_y"] = round((first_subject["scene_y"] + target_y) / 2.0, 6)
                        changed = True

                for second_subject in subjects:
                    if first_subject is second_subject:
                        continue
                    if not core.clause_mentions_subject(clause, second_subject):
                        continue

                    has_front_back_relation = (
                        guidance_requests_pair_relation(guidance_text, first_subject, second_subject, core.RELATION_FRONT_PATTERN)
                        or guidance_requests_pair_relation(guidance_text, first_subject, second_subject, core.RELATION_BEHIND_PATTERN)
                    )
                    has_left_right_relation = (
                        guidance_requests_pair_relation(guidance_text, first_subject, second_subject, core.RELATION_LEFT_PATTERN)
                        or guidance_requests_pair_relation(guidance_text, first_subject, second_subject, core.RELATION_RIGHT_PATTERN)
                    )

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

                    if ordered_relation_matches(
                        clause,
                        first_subject,
                        second_subject,
                        core.RELATION_LEFT_PATTERN,
                    ):
                        if (not has_front_back_relation) and abs(first_subject["scene_x"] - second_subject["scene_x"]) > 1e-6:
                            first_subject["scene_x"] = round(second_subject["scene_x"], 6)
                            changed = True
                        target_y = round(second_subject["scene_y"] - lateral, 6)
                        if clause_requests_tight_contact(clause):
                            target_y = round(
                                second_subject["scene_y"]
                                - (second_subject["dims"][0] / 2.0)
                                - (first_subject["dims"][0] / 2.0),
                                6,
                            )
                        if clause_requests_tight_contact(clause):
                            if abs(first_subject["scene_y"] - target_y) > 1e-6:
                                first_subject["scene_y"] = target_y
                                changed = True
                        elif first_subject["scene_y"] > target_y:
                            first_subject["scene_y"] = target_y
                            changed = True

                    if ordered_relation_matches(
                        clause,
                        first_subject,
                        second_subject,
                        core.RELATION_RIGHT_PATTERN,
                    ):
                        if (not has_front_back_relation) and abs(first_subject["scene_x"] - second_subject["scene_x"]) > 1e-6:
                            first_subject["scene_x"] = round(second_subject["scene_x"], 6)
                            changed = True
                        target_y = round(second_subject["scene_y"] + lateral, 6)
                        if clause_requests_tight_contact(clause):
                            target_y = round(
                                second_subject["scene_y"]
                                + (second_subject["dims"][0] / 2.0)
                                + (first_subject["dims"][0] / 2.0),
                                6,
                            )
                        if clause_requests_tight_contact(clause):
                            if abs(first_subject["scene_y"] - target_y) > 1e-6:
                                first_subject["scene_y"] = target_y
                                changed = True
                        elif first_subject["scene_y"] < target_y:
                            first_subject["scene_y"] = target_y
                            changed = True

                    if ordered_relation_matches(
                        clause,
                        first_subject,
                        second_subject,
                        core.RELATION_FRONT_PATTERN,
                    ):
                        if (not has_left_right_relation) and abs(first_subject["scene_y"] - second_subject["scene_y"]) > 1e-6:
                            first_subject["scene_y"] = round(second_subject["scene_y"], 6)
                            changed = True
                        target_x = round(second_subject["scene_x"] + longitudinal, 6)
                        if clause_requests_tight_contact(clause):
                            target_x = round(
                                second_subject["scene_x"]
                                + (second_subject["dims"][1] / 2.0)
                                + (first_subject["dims"][1] / 2.0),
                                6,
                            )
                        if clause_requests_tight_contact(clause):
                            if abs(first_subject["scene_x"] - target_x) > 1e-6:
                                first_subject["scene_x"] = target_x
                                changed = True
                        elif first_subject["scene_x"] < target_x:
                            first_subject["scene_x"] = target_x
                            changed = True

                    if ordered_relation_matches(
                        clause,
                        first_subject,
                        second_subject,
                        core.RELATION_BEHIND_PATTERN,
                    ):
                        if (not has_left_right_relation) and abs(first_subject["scene_y"] - second_subject["scene_y"]) > 1e-6:
                            first_subject["scene_y"] = round(second_subject["scene_y"], 6)
                            changed = True
                        target_x = round(second_subject["scene_x"] - longitudinal, 6)
                        if clause_requests_tight_contact(clause):
                            target_x = round(
                                second_subject["scene_x"]
                                - (second_subject["dims"][1] / 2.0)
                                - (first_subject["dims"][1] / 2.0),
                                6,
                            )
                        if clause_requests_tight_contact(clause):
                            if abs(first_subject["scene_x"] - target_x) > 1e-6:
                                first_subject["scene_x"] = target_x
                                changed = True
                        elif first_subject["scene_x"] > target_x:
                            first_subject["scene_x"] = target_x
                            changed = True

                    if ordered_relation_matches(
                        clause,
                        first_subject,
                        second_subject,
                        RELATION_ABOVE_PATTERN,
                    ):
                        target_x = round(second_subject["scene_x"], 6)
                        target_y = round(second_subject["scene_y"], 6)
                        target_z = round(second_subject["scene_z"] + second_subject["dims"][2] + 0.05, 6)
                        if clause_requests_tight_contact(clause):
                            target_z = round(second_subject["scene_z"] + second_subject["dims"][2], 6)
                        if abs(first_subject["scene_x"] - target_x) > 1e-6:
                            first_subject["scene_x"] = target_x
                            changed = True
                        if abs(first_subject["scene_y"] - target_y) > 1e-6:
                            first_subject["scene_y"] = target_y
                            changed = True
                        if clause_requests_tight_contact(clause):
                            if abs(first_subject["scene_z"] - target_z) > 1e-6:
                                first_subject["scene_z"] = target_z
                                changed = True
                        elif first_subject["scene_z"] < target_z:
                            first_subject["scene_z"] = target_z
                            changed = True

                    if ordered_relation_matches(
                        clause,
                        first_subject,
                        second_subject,
                        RELATION_BELOW_PATTERN,
                    ):
                        target_x = round(second_subject["scene_x"], 6)
                        target_y = round(second_subject["scene_y"], 6)
                        target_z = round(max(second_subject["scene_z"] - first_subject["dims"][2] - 0.05, 0.0), 6)
                        if clause_requests_tight_contact(clause):
                            target_z = round(max(second_subject["scene_z"] - first_subject["dims"][2], 0.0), 6)
                        if abs(first_subject["scene_x"] - target_x) > 1e-6:
                            first_subject["scene_x"] = target_x
                            changed = True
                        if abs(first_subject["scene_y"] - target_y) > 1e-6:
                            first_subject["scene_y"] = target_y
                            changed = True
                        if clause_requests_tight_contact(clause):
                            if abs(first_subject["scene_z"] - target_z) > 1e-6:
                                first_subject["scene_z"] = target_z
                                changed = True
                        elif first_subject["scene_z"] > target_z:
                            first_subject["scene_z"] = target_z
                            changed = True

                    if (
                        second_subject["type"] in core.SUPPORT_SURFACE_TYPES
                        and core.clause_implies_direct_support(clause, first_subject, second_subject)
                    ):
                        target_z = round(max(second_subject["scene_z"] + second_subject["dims"][2], 0.0), 6)
                        if abs(first_subject["scene_z"] - target_z) > 1e-6:
                            first_subject["scene_z"] = target_z
                            changed = True

                        offset = core.infer_support_surface_offset(clause, second_subject, first_subject)
                        if offset is None:
                            target_x = second_subject["scene_x"]
                            target_y = second_subject["scene_y"]
                        else:
                            target_x = round(second_subject["scene_x"] + offset[0], 6)
                            target_y = round(second_subject["scene_y"] + offset[1], 6)

                        if abs(first_subject["scene_x"] - target_x) > 1e-6:
                            first_subject["scene_x"] = target_x
                            changed = True
                        if abs(first_subject["scene_y"] - target_y) > 1e-6:
                            first_subject["scene_y"] = target_y
                            changed = True
                        core.clamp_subject_within_support(first_subject, second_subject)

        if not changed:
            break


def apply_manual_numeric_dimension_guidance(
    subjects: List[Dict[str, Any]],
    original_subjects: List[Dict[str, Any]],
    guidance_text: str,
) -> None:
    for subject in subjects:
        baseline_subject = core.find_matching_subject(subject, original_subjects) or subject
        multipliers = extract_numeric_dimension_multipliers(baseline_subject, guidance_text)
        if not multipliers:
            continue
        for dim_index, multiplier in multipliers.items():
            subject["dims"][dim_index] = round(max(baseline_subject["dims"][dim_index] * multiplier, 0.01), 6)


def collect_guidance_relation_issues(
    guidance_text: str,
    subjects: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    relation_issues: List[Dict[str, str]] = []
    clauses = core.split_text_clauses(guidance_text)
    if not clauses or not subjects:
        return relation_issues

    support_map = core.build_support_map(subjects, guidance_text)
    tolerance = 1e-4

    for subject in subjects:
        for clause in core.subject_relevant_clauses(guidance_text, subject):
            if not core.clause_contains_any(clause, core.CENTER_HINT_TOKENS):
                continue
            if abs(subject["scene_x"]) > 0.2 or abs(subject["scene_y"]) > 0.2:
                relation_issues.append(
                    {
                        "category": "guidance_not_applied",
                        "object": subject["name"],
                        "details": f"guidance 要求 {subject['name']} 居中，但当前未居中。",
                    }
                )
                break

    for clause in clauses:
        for first_subject in subjects:
            if not core.clause_mentions_subject(clause, first_subject):
                continue
            for second_subject in subjects:
                if first_subject is second_subject:
                    continue
                if not core.clause_mentions_subject(clause, second_subject):
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

                if ordered_relation_matches(
                    clause,
                    first_subject,
                    second_subject,
                    core.RELATION_LEFT_PATTERN,
                ):
                    if clause_requests_tight_contact(clause):
                        expected_y = second_subject["scene_y"] - (second_subject["dims"][0] / 2.0) - (first_subject["dims"][0] / 2.0)
                        if (
                            abs(first_subject["scene_y"] - expected_y) > tolerance
                            or abs(first_subject["scene_x"] - second_subject["scene_x"]) > tolerance
                        ):
                            relation_issues.append(
                                {
                                    "category": "guidance_not_applied",
                                    "object": first_subject["name"],
                                    "details": f"guidance 要求 {first_subject['name']} 紧贴在 {second_subject['name']} 左侧，但当前未满足。",
                                }
                            )
                    elif first_subject["scene_y"] > second_subject["scene_y"] - lateral + tolerance:
                        relation_issues.append(
                            {
                                "category": "guidance_not_applied",
                                "object": first_subject["name"],
                                "details": f"guidance 要求 {first_subject['name']} 在 {second_subject['name']} 左侧，但当前未满足。",
                            }
                        )

                if ordered_relation_matches(
                    clause,
                    first_subject,
                    second_subject,
                    core.RELATION_RIGHT_PATTERN,
                ):
                    if clause_requests_tight_contact(clause):
                        expected_y = second_subject["scene_y"] + (second_subject["dims"][0] / 2.0) + (first_subject["dims"][0] / 2.0)
                        if (
                            abs(first_subject["scene_y"] - expected_y) > tolerance
                            or abs(first_subject["scene_x"] - second_subject["scene_x"]) > tolerance
                        ):
                            relation_issues.append(
                                {
                                    "category": "guidance_not_applied",
                                    "object": first_subject["name"],
                                    "details": f"guidance 要求 {first_subject['name']} 紧贴在 {second_subject['name']} 右侧，但当前未满足。",
                                }
                            )
                    elif first_subject["scene_y"] < second_subject["scene_y"] + lateral - tolerance:
                        relation_issues.append(
                            {
                                "category": "guidance_not_applied",
                                "object": first_subject["name"],
                                "details": f"guidance 要求 {first_subject['name']} 在 {second_subject['name']} 右侧，但当前未满足。",
                            }
                        )

                if ordered_relation_matches(
                    clause,
                    first_subject,
                    second_subject,
                    core.RELATION_FRONT_PATTERN,
                ):
                    if clause_requests_tight_contact(clause):
                        expected_x = second_subject["scene_x"] + (second_subject["dims"][1] / 2.0) + (first_subject["dims"][1] / 2.0)
                        if (
                            abs(first_subject["scene_x"] - expected_x) > tolerance
                            or abs(first_subject["scene_y"] - second_subject["scene_y"]) > tolerance
                        ):
                            relation_issues.append(
                                {
                                    "category": "guidance_not_applied",
                                    "object": first_subject["name"],
                                    "details": f"guidance 要求 {first_subject['name']} 紧贴在 {second_subject['name']} 前方，但当前未满足。",
                                }
                            )
                    elif first_subject["scene_x"] < second_subject["scene_x"] + longitudinal - tolerance:
                        relation_issues.append(
                            {
                                "category": "guidance_not_applied",
                                "object": first_subject["name"],
                                "details": f"guidance 要求 {first_subject['name']} 在 {second_subject['name']} 前方，但当前未满足。",
                            }
                        )

                if ordered_relation_matches(
                    clause,
                    first_subject,
                    second_subject,
                    core.RELATION_BEHIND_PATTERN,
                ):
                    if clause_requests_tight_contact(clause):
                        expected_x = second_subject["scene_x"] - (second_subject["dims"][1] / 2.0) - (first_subject["dims"][1] / 2.0)
                        if (
                            abs(first_subject["scene_x"] - expected_x) > tolerance
                            or abs(first_subject["scene_y"] - second_subject["scene_y"]) > tolerance
                        ):
                            relation_issues.append(
                                {
                                    "category": "guidance_not_applied",
                                    "object": first_subject["name"],
                                    "details": f"guidance 要求 {first_subject['name']} 紧贴在 {second_subject['name']} 后方，但当前未满足。",
                                }
                            )
                    elif first_subject["scene_x"] > second_subject["scene_x"] - longitudinal + tolerance:
                        relation_issues.append(
                            {
                                "category": "guidance_not_applied",
                                "object": first_subject["name"],
                                "details": f"guidance 要求 {first_subject['name']} 在 {second_subject['name']} 后方，但当前未满足。",
                            }
                        )

                if ordered_relation_matches(
                    clause,
                    first_subject,
                    second_subject,
                    RELATION_ABOVE_PATTERN,
                ):
                    if clause_requests_tight_contact(clause):
                        expected_z = second_subject["scene_z"] + second_subject["dims"][2]
                        if (
                            abs(first_subject["scene_z"] - expected_z) > tolerance
                            or abs(first_subject["scene_x"] - second_subject["scene_x"]) > tolerance
                            or abs(first_subject["scene_y"] - second_subject["scene_y"]) > tolerance
                        ):
                            relation_issues.append(
                                {
                                    "category": "guidance_not_applied",
                                    "object": first_subject["name"],
                                    "details": f"guidance 要求 {first_subject['name']} 紧贴在 {second_subject['name']} 上方，但当前未满足。",
                                }
                            )
                    elif (
                        first_subject["scene_z"] <= second_subject["scene_z"] + second_subject["dims"][2] - tolerance
                        or abs(first_subject["scene_x"] - second_subject["scene_x"]) > tolerance
                        or abs(first_subject["scene_y"] - second_subject["scene_y"]) > tolerance
                    ):
                        relation_issues.append(
                            {
                                "category": "guidance_not_applied",
                                "object": first_subject["name"],
                                "details": f"guidance 要求 {first_subject['name']} 在 {second_subject['name']} 上方，但当前未满足。",
                            }
                        )

                if ordered_relation_matches(
                    clause,
                    first_subject,
                    second_subject,
                    RELATION_BELOW_PATTERN,
                ):
                    if clause_requests_tight_contact(clause):
                        expected_z = second_subject["scene_z"] - first_subject["dims"][2]
                        if (
                            abs(first_subject["scene_z"] - expected_z) > tolerance
                            or abs(first_subject["scene_x"] - second_subject["scene_x"]) > tolerance
                            or abs(first_subject["scene_y"] - second_subject["scene_y"]) > tolerance
                        ):
                            relation_issues.append(
                                {
                                    "category": "guidance_not_applied",
                                    "object": first_subject["name"],
                                    "details": f"guidance 要求 {first_subject['name']} 紧贴在 {second_subject['name']} 下方，但当前未满足。",
                                }
                            )
                    elif (
                        first_subject["scene_z"] + first_subject["dims"][2] >= second_subject["scene_z"] + tolerance
                        or abs(first_subject["scene_x"] - second_subject["scene_x"]) > tolerance
                        or abs(first_subject["scene_y"] - second_subject["scene_y"]) > tolerance
                    ):
                        relation_issues.append(
                            {
                                "category": "guidance_not_applied",
                                "object": first_subject["name"],
                                "details": f"guidance 要求 {first_subject['name']} 在 {second_subject['name']} 下方，但当前未满足。",
                            }
                        )

                if (
                    second_subject["type"] in core.SUPPORT_SURFACE_TYPES
                    and core.clause_implies_direct_support(clause, first_subject, second_subject)
                ):
                    alignment = compute_support_alignment(first_subject, second_subject)
                    if (
                        abs(alignment["z_delta"]) > alignment["z_tolerance"]
                        or alignment["x_low_excess"] > alignment["xy_tolerance"]
                        or alignment["x_high_excess"] > alignment["xy_tolerance"]
                        or alignment["y_low_excess"] > alignment["xy_tolerance"]
                        or alignment["y_high_excess"] > alignment["xy_tolerance"]
                    ):
                        relation_issues.append(
                            {
                                "category": "guidance_not_applied",
                                "object": first_subject["name"],
                                "details": f"guidance 要求 {first_subject['name']} 放在 {second_subject['name']} 上，但当前未严格落在其顶部。",
                            }
                        )

    return core.deduplicate_issues(relation_issues)


def collect_scope_issues(
    original_subjects: List[Dict[str, Any]],
    actual_subjects: List[Dict[str, Any]],
    original_camera: Dict[str, float],
    actual_camera: Dict[str, float],
    subject_scopes: Dict[str, Dict[str, Any]],
    camera_scope: Dict[str, bool],
) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    tolerance = 1e-6

    for original_subject in original_subjects:
        actual_subject = core.find_matching_subject(original_subject, actual_subjects)
        if actual_subject is None:
            issues.append(
                {
                    "category": "unmentioned_parameter_changed",
                    "object": original_subject["name"],
                    "details": "输出结果中丢失了原始对象。",
                }
            )
            continue

        scope = subject_scopes.get(original_subject["name"], build_empty_subject_scope())
        if (not scope["scene_x"]) and abs(actual_subject["scene_x"] - original_subject["scene_x"]) > tolerance:
            issues.append(
                {
                    "category": "unmentioned_parameter_changed",
                    "object": original_subject["name"],
                    "details": "scene_x 未在 guidance 中提到，但被修改了。",
                }
            )
        if (not scope["scene_y"]) and abs(actual_subject["scene_y"] - original_subject["scene_y"]) > tolerance:
            issues.append(
                {
                    "category": "unmentioned_parameter_changed",
                    "object": original_subject["name"],
                    "details": "scene_y 未在 guidance 中提到，但被修改了。",
                }
            )
        if (not scope["scene_z"]) and abs(actual_subject["scene_z"] - original_subject["scene_z"]) > tolerance:
            issues.append(
                {
                    "category": "unmentioned_parameter_changed",
                    "object": original_subject["name"],
                    "details": "scene_z 未在 guidance 中提到，但被修改了。",
                }
            )
        if (not scope["azimuth_deg"]) and abs(actual_subject["azimuth_deg"] - original_subject["azimuth_deg"]) > tolerance:
            issues.append(
                {
                    "category": "unmentioned_parameter_changed",
                    "object": original_subject["name"],
                    "details": "azimuth 未在 guidance 中提到，但被修改了。",
                }
            )
        for dim_index in range(3):
            if dim_index in scope["dim_indices"]:
                continue
            if abs(actual_subject["dims"][dim_index] - original_subject["dims"][dim_index]) > tolerance:
                issues.append(
                    {
                        "category": "unmentioned_parameter_changed",
                        "object": original_subject["name"],
                        "details": f"dims[{dim_index}] 未在 guidance 中提到，但被修改了。",
                    }
                )

    for field_name, allowed in camera_scope.items():
        if allowed:
            continue
        if abs(actual_camera[field_name] - original_camera[field_name]) > tolerance:
            issues.append(
                {
                    "category": "unmentioned_parameter_changed",
                    "object": "camera_data",
                    "details": f"{field_name} 未在 guidance 中提到，但被修改了。",
                }
            )

    return core.deduplicate_issues(issues)


def build_scoped_scene_dict(
    base_scene_dict: Dict[str, Any],
    subjects: List[Dict[str, Any]],
    camera_data: Dict[str, float],
    scene_text: str,
) -> Dict[str, Any]:
    return core.build_scene_dict_from_subjects(
        base_scene_dict=base_scene_dict,
        subjects=subjects,
        camera_elevation_deg=camera_data["camera_elevation_deg"],
        lens_mm=camera_data["lens_mm"],
        scene_text=scene_text,
        global_scale=camera_data["global_scale"],
    )


def strict_apply_guidance(
    scene_text: str,
    guidance_text: str,
    scene_dict: Dict[str, Any],
    allowed_types: List[str],
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
) -> Dict[str, Any]:
    original_subjects = core.summarize_scene_subjects(scene_dict, allowed_types, reference_dims_map)
    baseline_subjects, structural_target_names = apply_structural_guidance(
        original_subjects,
        guidance_text=guidance_text,
        allowed_types=allowed_types,
        reference_dims_map=reference_dims_map,
        default_dims_map=default_dims_map,
    )
    original_camera = core.summarize_camera(scene_dict)
    guidance_target_camera = core.apply_guidance_camera_directives(original_camera, guidance_text)
    guidance_target_names: Set[str] = strict_collect_guidance_target_names(baseline_subjects, guidance_text)
    if not guidance_target_names:
        guidance_target_names = core.collect_subject_names_mentioned_in_text(baseline_subjects, guidance_text)
    guidance_target_names.update(structural_target_names)

    subject_scopes = collect_subject_parameter_scope(
        baseline_subjects,
        guidance_text,
        guidance_target_names,
    )
    modifiable_subject_names = {
        subject_name
        for subject_name, scope in subject_scopes.items()
        if scope["scene_x"] or scope["scene_y"] or scope["scene_z"] or scope["azimuth_deg"] or scope["dim_indices"]
    }
    camera_scope = build_camera_scope(original_camera, guidance_target_camera)
    movable_subject_names = {
        subject_name
        for subject_name, scope in subject_scopes.items()
        if scope["scene_x"] or scope["scene_y"] or scope["scene_z"]
    }
    expected_subjects = core.build_guidance_preview_subjects(
        baseline_subjects,
        guidance_text,
        reference_dims_map,
        default_dims_map,
        modifiable_subject_names,
    )
    apply_manual_numeric_dimension_guidance(
        expected_subjects,
        original_subjects=baseline_subjects,
        guidance_text=guidance_text,
    )
    apply_manual_relative_axis_movement_guidance(
        expected_subjects,
        baseline_subjects=baseline_subjects,
        guidance_text=guidance_text,
        movable_subject_names=movable_subject_names or None,
    )

    working_subjects = core.clone_subjects(baseline_subjects)
    working_camera = dict(original_camera)
    last_unresolved: List[Dict[str, str]] = []
    last_scene_dict = scene_dict

    for _ in range(MAX_APPLY_ATTEMPTS):
        forced_scene_dict, _ = core.build_guidance_forced_scene(
            base_scene_dict=scene_dict,
            current_subjects=working_subjects,
            current_camera=working_camera,
            guidance_target_camera=guidance_target_camera,
            scene_text=scene_text,
            guidance_text=guidance_text,
            reference_dims_map=reference_dims_map,
            default_dims_map=default_dims_map,
            guidance_target_names=modifiable_subject_names,
        )

        candidate_subjects = core.summarize_scene_subjects(
            forced_scene_dict,
            allowed_types,
            reference_dims_map,
        )
        candidate_camera = core.summarize_camera(forced_scene_dict)
        scoped_subjects = restrict_subjects_to_scope(
            original_subjects=baseline_subjects,
            candidate_subjects=candidate_subjects,
            subject_scopes=subject_scopes,
        )
        scoped_camera = restrict_camera_to_scope(
            original_camera=original_camera,
            candidate_camera=candidate_camera,
            camera_scope=camera_scope,
        )
        apply_manual_numeric_dimension_guidance(
            scoped_subjects,
            original_subjects=baseline_subjects,
            guidance_text=guidance_text,
        )
        apply_manual_relation_guidance(
            scoped_subjects,
            guidance_text=guidance_text,
            movable_subject_names=movable_subject_names or None,
        )
        apply_manual_relative_axis_movement_guidance(
            scoped_subjects,
            baseline_subjects=baseline_subjects,
            guidance_text=guidance_text,
            movable_subject_names=movable_subject_names or None,
        )
        scoped_scene_dict = build_scoped_scene_dict(
            base_scene_dict=scene_dict,
            subjects=scoped_subjects,
            camera_data=scoped_camera,
            scene_text=scene_text,
        )

        actual_subjects = core.summarize_scene_subjects(
            scoped_scene_dict,
            allowed_types,
            reference_dims_map,
        )
        actual_camera = core.summarize_camera(scoped_scene_dict)

        subject_issues = core.collect_guidance_application_issues(
            original_subjects=baseline_subjects,
            actual_subjects=actual_subjects,
            expected_subjects=expected_subjects,
            guidance_target_names=modifiable_subject_names,
        )
        camera_issues = core.collect_camera_guidance_application_issues(
            original_camera=original_camera,
            actual_camera=actual_camera,
            expected_camera=guidance_target_camera,
        )
        relation_issues = collect_guidance_relation_issues(
            guidance_text=guidance_text,
            subjects=actual_subjects,
        )
        relative_move_issues = collect_relative_axis_movement_issues(
            actual_subjects=actual_subjects,
            baseline_subjects=baseline_subjects,
            guidance_text=guidance_text,
            movable_subject_names=movable_subject_names or None,
        )
        scope_issues = collect_scope_issues(
            original_subjects=baseline_subjects,
            actual_subjects=actual_subjects,
            original_camera=original_camera,
            actual_camera=actual_camera,
            subject_scopes=subject_scopes,
            camera_scope=camera_scope,
        )

        unresolved = core.deduplicate_issues(
            subject_issues + camera_issues + relation_issues + relative_move_issues + scope_issues
        )
        if not unresolved:
            return scoped_scene_dict

        last_unresolved = unresolved
        last_scene_dict = scoped_scene_dict
        working_subjects = actual_subjects
        working_camera = actual_camera

    issue_lines = core.format_issue_lines(last_unresolved)
    lines_preview = "\n".join(issue_lines[:12])
    raise RuntimeError(
        "guidance 经过多轮尝试后仍未被严格执行到位，已拒绝输出结果。"
        + ("\n" + lines_preview if lines_preview else "")
    )


def main() -> None:
    args = parse_args()

    scene_text = args.scene_text.strip()
    guidance_text = args.guidance.strip()
    if not scene_text:
        raise RuntimeError("scene-text 不能为空。")
    if not guidance_text:
        raise RuntimeError("guidance 不能为空。")

    input_pkl = normalize_existing_path(args.scene_pkl, "scene-pkl")
    output_pkl = normalize_output_path(args.output)
    context = build_runtime_context()

    original_scene = core.load_scene_pkl(input_pkl)
    fixed_scene = strict_apply_guidance(
        scene_text=scene_text,
        guidance_text=guidance_text,
        scene_dict=original_scene,
        allowed_types=context["allowed_types"],
        reference_dims_map=context["reference_dims_map"],
        default_dims_map=context["default_dims_map"],
    )

    core.save_scene_pkl(fixed_scene, output_pkl)
    print(f"已保存修正版 pkl：{output_pkl}")
    core.print_scene_parameter_changes(original_scene, fixed_scene)


if __name__ == "__main__":
    main()
