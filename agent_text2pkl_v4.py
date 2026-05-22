#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从英文场景文字生成 scene pkl，并在本地强制满足以下约束：

A. pkl 文件中物体类型、数量和文字完全匹配
B. pkl 文件中的 cube 位置关系和文字完全匹配
C. pkl 文件中的 cube 位置在物理上合理
D. pkl 文件中的 cube 大小合适，物体相对大小符合现实世界常识
E. 图片中必须看到每个 cube 的完整结构
F. prompt 中 a/an 表示 1 个；明确数字表示对应数量；the 不算新物体；prompt 没提到的物体不应出现在 pkl 中
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import agent_check_pkl_v5 as core

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


REPO_ROOT = core.REPO_ROOT
ASSET_DIMENSIONS_PATH = core.ASSET_DIMENSIONS_PATH
OBJECT_SCALES_PATH = core.OBJECT_SCALES_PATH
DEFAULT_OUTPUT_PATH = os.path.join(REPO_ROOT, "inference", "saved_scenes", "example_test.pkl")

API_MODEL = core.API_MODEL
DEFAULT_CAMERA_ELEVATION_DEG = core.DEFAULT_CAMERA_ELEVATION_DEG
DEFAULT_LENS_MM = core.DEFAULT_LENS_MM
DEFAULT_GLOBAL_SCALE = core.DEFAULT_GLOBAL_SCALE
DEFAULT_INFERENCE_PARAMS = dict(core.DEFAULT_INFERENCE_PARAMS)
DEFAULT_TOP_LEVEL_CHECKPOINT = core.DEFAULT_TOP_LEVEL_CHECKPOINT
CUSTOM_ASSET_TYPE = "Custom"

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
    "center",
    "middle",
    "upper",
    "lower",
    "there",
    "here",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="调用大模型，把英文场景文字转换成 pkl，并在本地强制修正为满足 A-F 的布局。",
        epilog=(
            '示例:\n'
            '  python agent_text2pkl_v4.py '
            '--output inference/saved_scenes/example_test.pkl '
            '--scene-text "A sedan is parked on the road. A chair is to the left of the sedan."'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--scene-text",
        required=True,
        help="英文场景描述。直接在运行命令中传入，需用引号包裹。",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"输出 pkl 路径。默认: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--api-key",
        default="REPLACE_WITH_API_KEY",
        help="可选，显式传入 API Key。未提供时会尝试读取环境变量或 chatgpt_api.py。",
    )
    parser.add_argument(
        "--model",
        default=API_MODEL,
        help=f"模型名，默认: {API_MODEL}",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="打印模型返回的原始场景计划 JSON。",
    )
    parser.add_argument(
        "--print-scene-json",
        action="store_true",
        help="打印最终写入 pkl 的 scene_dict（JSON 预览）。",
    )
    return parser.parse_args()


def build_prompt(scene_text: str, allowed_types: List[str]) -> str:
    allowed_types_text = ", ".join(sorted(allowed_types))
    return f"""
You are a strict structured 3D scene planner.
Convert the input English description into a compact JSON scene plan.
Return JSON only. Do not output markdown. Do not add explanations.

Allowed asset types:
[{allowed_types_text}]

Output schema:
{{
  "surrounding_prompt": "a short background/environment prompt",
  "camera_elevation_deg": 12.0,
  "subjects": [
    {{
      "name": "object name; if multiple same-type objects exist, give unique names such as dog 1, dog 2",
      "type": "one allowed asset type or the closest allowed synonym",
      "scene_x": 0.0,
      "scene_y": 0.0,
      "scene_z": 0.0,
      "azimuth_deg": 0.0,
      "size_scale": 1.0
    }}
  ]
}}

Hard rules:
1. Extract every counted object exactly once. If the prompt says "a/an", that means exactly one object. If the prompt uses an explicit number, create exactly that many objects.
2. The word "the" refers to an already mentioned object and does not create a new object.
3. Do not add any extra object that is not mentioned in the prompt.
4. Keep object type mapping exact whenever possible. Example: car -> sedan, person -> man.
5. Make all spatial relations exact: left/right/front/behind/center/on top of must match the prompt.
6. Keep all objects physically plausible. Ground objects use scene_z=0 unless the prompt clearly describes an elevated object. Objects on top of tables or shelves must rest on the support surface.
7. Keep realistic relative sizes. Never make a dog larger than a car, or a chair larger than a truck, unless the prompt explicitly says it is gigantic.
8. Keep the whole layout compact enough that every cube can be fully visible in a single camera view.
9. surrounding_prompt should describe only the environment/background style, not the object list again.

Coordinate system:
- larger scene_x = more in front of / closer to camera
- smaller scene_x = more behind / farther from camera
- smaller scene_y = more to the left
- larger scene_y = more to the right

Example:
Input:
"A sedan is parked on the road. A chair is to the left of the sedan, and a bicycle is in front of it."
Valid output:
{{
  "surrounding_prompt": "highly realistic outdoor lighting.",
  "camera_elevation_deg": 12.0,
  "subjects": [
    {{"name": "sedan", "type": "sedan", "scene_x": 0.0, "scene_y": 0.0, "scene_z": 0.0, "azimuth_deg": -80.0, "size_scale": 1.0}},
    {{"name": "chair", "type": "chair", "scene_x": 0.0, "scene_y": -1.6, "scene_z": 0.0, "azimuth_deg": -90.0, "size_scale": 1.0}},
    {{"name": "bicycle", "type": "bicycle", "scene_x": 2.0, "scene_y": 0.0, "scene_z": 0.0, "azimuth_deg": -140.0, "size_scale": 1.0}}
  ]
}}

Now convert this description:
\"\"\"{scene_text}\"\"\"
    """.strip()


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
        if mapped_type != CUSTOM_ASSET_TYPE:
            return mapped_type
    return CUSTOM_ASSET_TYPE


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


def infer_expected_type_spec(
    scene_text: str,
    allowed_types: List[str],
) -> Tuple[Counter, Dict[str, str], List[Dict[str, Any]], List[str]]:
    expected_counts: Counter = Counter()
    mention_examples: Dict[str, str] = {}
    parsed_mentions: List[Dict[str, Any]] = []
    parse_warnings: List[str] = []

    quantified_mentions = parse_quantified_object_mentions(scene_text)
    if not quantified_mentions:
        parse_warnings.append(
            "未按 a/an/明确数字 规则解析到对象；生成结果将优先依赖模型计划。"
        )
        return expected_counts, mention_examples, parsed_mentions, parse_warnings

    for mention, mention_count in quantified_mentions:
        mapped_type = map_mention_to_asset_type(mention, allowed_types)
        parsed_mentions.append(
            {
                "mention": mention,
                "count": mention_count,
                "mapped_type": mapped_type,
            }
        )
        if mapped_type == CUSTOM_ASSET_TYPE:
            parse_warnings.append(f"对象短语 `{mention}` 不能稳定映射到已知资产类型。")
            continue
        expected_counts[mapped_type] += mention_count
        mention_examples.setdefault(mapped_type, singularize_phrase(mention) or mapped_type)

    return expected_counts, mention_examples, parsed_mentions, parse_warnings


def infer_default_size_scale(name: str, has_reference_dims: bool) -> float:
    lowered_name = core.canonicalize_type(name)
    if "puppy" in lowered_name or "kitten" in lowered_name or "baby" in lowered_name:
        return 0.68
    if "small" in lowered_name or "little" in lowered_name or "tiny" in lowered_name or "mini" in lowered_name:
        return 0.82
    if "large" in lowered_name or "big" in lowered_name or "huge" in lowered_name or "giant" in lowered_name:
        return 1.18
    return 1.0 if has_reference_dims else 0.85


def resolve_subject_dims(
    name: str,
    asset_type: str,
    asset_dimensions: Dict[str, List[float]],
    reference_dims_map: Dict[str, List[float]],
    raw_size_scale: Any = None,
) -> List[float]:
    has_reference_dims = asset_type in reference_dims_map
    if has_reference_dims:
        base_dims = reference_dims_map[asset_type]
    else:
        base_dims = asset_dimensions.get(asset_type, [1.0, 1.0, 1.0])

    default_scale = infer_default_size_scale(name, has_reference_dims=has_reference_dims)
    size_scale = core.to_float(raw_size_scale, default_scale)
    size_scale = core.clamp(size_scale, 0.35, 1.35) if has_reference_dims else core.clamp(size_scale, 0.25, 2.5)
    return [round(float(v) * size_scale, 6) for v in base_dims]


def build_subject_entries_from_plan(
    plan: Dict[str, Any],
    asset_dimensions: Dict[str, List[float]],
    reference_dims_map: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    raw_subjects = plan.get("subjects")
    if not isinstance(raw_subjects, list):
        raw_subjects = []

    allowed_types = list(asset_dimensions.keys())
    entries: List[Dict[str, Any]] = []

    for idx, item in enumerate(raw_subjects):
        if not isinstance(item, dict):
            continue

        name = core.sanitize_name(
            item.get("name") or item.get("description"),
            default_name=f"object_{idx + 1}",
        )
        asset_type = core.match_asset_type(item.get("type"), name, allowed_types)

        entries.append(
            {
                "name": name,
                "type": asset_type,
                "scene_x": round(core.to_float(item.get("scene_x"), 0.0), 6),
                "scene_y": round(core.to_float(item.get("scene_y"), 0.0), 6),
                "scene_z": round(core.to_float(item.get("scene_z"), 0.0), 6),
                "azimuth_deg": round(core.to_float(item.get("azimuth_deg"), 0.0), 6),
                "dims": resolve_subject_dims(
                    name=name,
                    asset_type=asset_type,
                    asset_dimensions=asset_dimensions,
                    reference_dims_map=reference_dims_map,
                    raw_size_scale=item.get("size_scale"),
                ),
            }
        )

    return entries


def estimate_missing_entry_position(
    name: str,
    asset_type: str,
    entries: List[Dict[str, Any]],
    missing_index: int,
) -> Dict[str, float]:
    same_type_entries = [entry for entry in entries if entry["type"] == asset_type]
    base_entry = same_type_entries[0] if same_type_entries else (entries[0] if entries else None)
    if base_entry is None:
        return {
            "scene_x": 0.0,
            "scene_y": 0.0,
            "scene_z": 0.0,
            "azimuth_deg": 0.0,
        }

    radius = 0.9 + 0.25 * (missing_index // 6)
    angle = (missing_index % 6) * (math.pi / 3.0)
    return {
        "scene_x": round(base_entry["scene_x"] + math.cos(angle) * radius, 6),
        "scene_y": round(base_entry["scene_y"] + math.sin(angle) * radius, 6),
        "scene_z": 0.0,
        "azimuth_deg": round(base_entry.get("azimuth_deg", 0.0), 6),
    }


def enforce_expected_object_counts(
    entries: List[Dict[str, Any]],
    scene_text: str,
    asset_dimensions: Dict[str, List[float]],
    reference_dims_map: Dict[str, List[float]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    allowed_types = list(asset_dimensions.keys())
    expected_counts, mention_examples, parsed_mentions, parse_warnings = infer_expected_type_spec(
        scene_text,
        allowed_types,
    )
    if not expected_counts:
        core.ensure_unique_names(entries)
        return entries, parsed_mentions, parse_warnings

    kept_counts: Counter = Counter()
    filtered_entries: List[Dict[str, Any]] = []
    for entry in entries:
        asset_type = entry["type"]
        if expected_counts.get(asset_type, 0) <= 0:
            continue
        if kept_counts[asset_type] >= expected_counts[asset_type]:
            continue
        filtered_entries.append(entry)
        kept_counts[asset_type] += 1

    added_index = 0
    for asset_type, desired_count in expected_counts.items():
        while kept_counts[asset_type] < desired_count:
            display_name = mention_examples.get(asset_type, asset_type)
            position = estimate_missing_entry_position(display_name, asset_type, filtered_entries, added_index)
            filtered_entries.append(
                {
                    "name": core.sanitize_name(display_name, default_name=f"object_{len(filtered_entries) + 1}"),
                    "type": asset_type,
                    "scene_x": position["scene_x"],
                    "scene_y": position["scene_y"],
                    "scene_z": position["scene_z"],
                    "azimuth_deg": position["azimuth_deg"],
                    "dims": resolve_subject_dims(
                        name=display_name,
                        asset_type=asset_type,
                        asset_dimensions=asset_dimensions,
                        reference_dims_map=reference_dims_map,
                    ),
                }
            )
            kept_counts[asset_type] += 1
            added_index += 1

    core.ensure_unique_names(filtered_entries)
    return filtered_entries, parsed_mentions, parse_warnings


def shrink_supported_subjects_to_fit(
    subjects: List[Dict[str, Any]],
    scene_text: str,
    support_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    resolved_support_map = support_map or core.build_support_map(subjects, scene_text)
    for subject in subjects:
        support_subject = resolved_support_map.get(subject["name"])
        if support_subject is None:
            continue

        max_width = max(support_subject["dims"][0] * 0.92, 0.05)
        max_depth = max(support_subject["dims"][1] * 0.92, 0.05)
        if subject["dims"][0] > max_width:
            subject["dims"][0] = round(max_width, 6)
        if subject["dims"][1] > max_depth:
            subject["dims"][1] = round(max_depth, 6)
    return resolved_support_map


def repair_subject_layout(
    subjects: List[Dict[str, Any]],
    scene_text: str,
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
    initial_camera_elevation_deg: float,
    initial_lens_mm: float,
) -> Dict[str, Any]:
    constrained_subjects = core.clone_subjects(subjects)

    for _ in range(2):
        support_map = shrink_supported_subjects_to_fit(constrained_subjects, scene_text)
        constrained_subjects = core.apply_strict_subject_constraints(
            constrained_subjects,
            scene_text=scene_text,
            guidance_text="",
            reference_dims_map=reference_dims_map,
            default_dims_map=default_dims_map,
        )
        shrink_supported_subjects_to_fit(
            constrained_subjects,
            scene_text,
            support_map=core.build_support_map(constrained_subjects, scene_text),
        )

    best_fit = core.fit_scene_to_camera(
        constrained_subjects,
        camera_elevation_deg=initial_camera_elevation_deg,
        lens_mm=initial_lens_mm,
    )

    for factor in (0.94, 0.88):
        if best_fit["passed"]:
            break
        candidate_subjects = core.clone_subjects(constrained_subjects)
        for subject in candidate_subjects:
            subject["dims"] = [round(max(dimension * factor, 0.01), 6) for dimension in subject["dims"]]

        support_map = shrink_supported_subjects_to_fit(candidate_subjects, scene_text)
        candidate_subjects = core.apply_strict_subject_constraints(
            candidate_subjects,
            scene_text=scene_text,
            guidance_text="",
            reference_dims_map=reference_dims_map,
            default_dims_map=default_dims_map,
        )
        shrink_supported_subjects_to_fit(candidate_subjects, scene_text, support_map=support_map)

        candidate_fit = core.fit_scene_to_camera(
            candidate_subjects,
            camera_elevation_deg=initial_camera_elevation_deg,
            lens_mm=initial_lens_mm,
        )
        if candidate_fit["passed"] or candidate_fit.get("score", float("inf")) < best_fit.get("score", float("inf")):
            best_fit = candidate_fit

    return best_fit


def build_base_scene_dict(plan: Dict[str, Any], scene_text: str) -> Dict[str, Any]:
    surrounding_prompt = str(plan.get("surrounding_prompt") or "").strip() or "highly realistic."
    return {
        "subjects_data": [],
        "camera_data": {
            "camera_elevation": math.radians(DEFAULT_CAMERA_ELEVATION_DEG),
            "lens": DEFAULT_LENS_MM,
            "global_scale": DEFAULT_GLOBAL_SCALE,
        },
        "surrounding_prompt": surrounding_prompt,
        "inference_params": dict(DEFAULT_INFERENCE_PARAMS),
        "checkpoint": DEFAULT_TOP_LEVEL_CHECKPOINT,
        "_meta": {
            "source_text": scene_text,
        },
    }


def strip_meta(scene_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(scene_dict)
    cleaned.pop("_meta", None)
    return cleaned


def to_builtin(value: Any) -> Any:
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()

    if isinstance(value, dict):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    return value


def load_raw_pkl(pkl_path: str) -> Dict[str, Any]:
    try:
        return core.load_scene_pkl(pkl_path)
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            raise RuntimeError("读取该 pkl 需要 numpy，请在项目环境中运行。") from exc
        raise


def export_pkl_contents(pkl_path: str) -> Dict[str, Any]:
    raw_scene = load_raw_pkl(pkl_path)
    return to_builtin(raw_scene)


def save_exported_json(exported: Dict[str, Any], output_json_path: str) -> None:
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(exported, handle, ensure_ascii=False, indent=2)


def json_output_path_from_pkl(output_pkl_path: str) -> str:
    base_path, extension = os.path.splitext(output_pkl_path)
    if extension.lower() == ".pkl":
        return f"{base_path}.json"
    return f"{output_pkl_path}.json"


def main() -> None:
    args = parse_args()

    scene_text = args.scene_text.strip()
    if not scene_text:
        raise RuntimeError("场景描述为空。")

    asset_dimensions = core.load_asset_dimensions(ASSET_DIMENSIONS_PATH)
    object_scales = core.load_object_scales(OBJECT_SCALES_PATH)
    reference_dims_map = core.build_reference_dims_map(asset_dimensions, object_scales)
    default_dims_map = core.build_default_dims_map(asset_dimensions)

    api_key = core.load_api_key(args.api_key)
    prompt = build_prompt(scene_text, list(asset_dimensions.keys()))
    plan = core.call_llm_for_json(prompt, api_key=api_key, model=args.model)

    if args.print_plan:
        print("===== LLM Scene Plan =====")
        print(json.dumps(plan, ensure_ascii=False, indent=2))

    initial_entries = build_subject_entries_from_plan(
        plan=plan,
        asset_dimensions=asset_dimensions,
        reference_dims_map=reference_dims_map,
    )
    constrained_entries, parsed_mentions, parse_warnings = enforce_expected_object_counts(
        initial_entries,
        scene_text=scene_text,
        asset_dimensions=asset_dimensions,
        reference_dims_map=reference_dims_map,
    )
    if not constrained_entries:
        raise RuntimeError("模型没有返回可用 subjects，且无法从 prompt 中稳定恢复对象。")

    initial_camera_elevation_deg = core.clamp(
        core.to_float(plan.get("camera_elevation_deg"), DEFAULT_CAMERA_ELEVATION_DEG),
        0.0,
        90.0,
    )
    fit_result = repair_subject_layout(
        subjects=constrained_entries,
        scene_text=scene_text,
        reference_dims_map=reference_dims_map,
        default_dims_map=default_dims_map,
        initial_camera_elevation_deg=initial_camera_elevation_deg,
        initial_lens_mm=DEFAULT_LENS_MM,
    )

    base_scene_dict = build_base_scene_dict(plan, scene_text)
    scene_dict = core.build_scene_dict_from_subjects(
        base_scene_dict=base_scene_dict,
        subjects=fit_result["subjects"],
        camera_elevation_deg=fit_result["camera_elevation_deg"],
        lens_mm=fit_result["lens_mm"],
        scene_text=scene_text,
        global_scale=DEFAULT_GLOBAL_SCALE,
    )

    if parsed_mentions:
        scene_dict["_meta"]["parsed_mentions"] = parsed_mentions
    if parse_warnings:
        scene_dict["_meta"]["parse_warnings"] = parse_warnings
        print("解析提示：")
        for warning in parse_warnings:
            print(f"- {warning}")

    if args.print_scene_json:
        print("===== Final Scene Dict =====")
        print(json.dumps(strip_meta(scene_dict), ensure_ascii=False, indent=2))

    core.save_scene_pkl(strip_meta(scene_dict), args.output)
    output_json_path = json_output_path_from_pkl(args.output)
    exported_scene = export_pkl_contents(args.output)
    save_exported_json(exported_scene, output_json_path)
    print(f"场景文件已保存到: {args.output}")
    print(f"同名 JSON 文件已保存到: {output_json_path}")


if __name__ == "__main__":
    main()
