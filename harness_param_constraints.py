#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic pkl-level harness constraints for scene layout agents.

This module is intentionally script-friendly: it uses plain dictionaries and
JSON-serializable records so agent_text2pkl_v5.py, agent_opinion.py and
agent_reverse.py can share the same constraints without relying on prompt
memory.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import agent_check_pkl_v5 as core


CUSTOM_ASSET_TYPE = "Custom"
ASSET_DEFAULT_AZIMUTHS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference", "asset_default_azimuths.json")
DEFAULT_CENTER_X = -6.0
DEFAULT_CENTER_Y = 0.0
MARGIN_X_MIN = 0.05
MARGIN_Y_MIN = 0.05
DEFAULT_COMPOUND_LONGITUDINAL_MARGIN = 1.5
DEFAULT_COMPOUND_LATERAL_MARGIN = 0.6
SCREEN_DEPTH_DELTA_PX_BY_LEVEL = {
    "slight": 80.0,
    "default": 80.0,
    "clear": 260.0,
    "far": 340.0,
    "extra_far": 420.0,
}
SCREEN_LATERAL_GAP_PX_BY_LEVEL = {
    "slight": 20.0,
    "default": 24.0,
    "clear": 35.0,
    "far": 42.0,
    "extra_far": 52.0,
}
SCREEN_DEPTH_MIN_VISIBLE_GAP_PX = 80.0
SCREEN_DEPTH_MAX_WORLD_SHIFT = 10.0
SCREEN_LATERAL_MAX_WORLD_SHIFT = 8.0
PAIRWISE_OCCLUSION_MAX_RATIO = 0.20
SCREEN_SIZE_RATIO_RANGE = (0.8, 1.2)
SCREEN_SIZE_FRONT_RATIO_RANGE_BY_LEVEL = {
    "default": (0.8, 1.6),
    "clear": (0.75, 1.8),
    "far": (0.7, 2.0),
    "extra_far": (0.7, 2.0),
}
SCREEN_SIZE_BACK_RATIO_RANGE_BY_LEVEL = {
    "default": (0.55, 1.2),
    "clear": (0.45, 1.15),
    "far": (0.35, 1.1),
    "extra_far": (0.35, 1.1),
}
SCREEN_SIZE_MAX_PASSES = 10
SCREEN_SIZE_MAX_WORLD_SHIFT = 6.0
SCREEN_SIZE_MAX_DIM_SCALE = 1.25
DIM_REFERENCE_MIN_RATIO = 0.7
DIM_REFERENCE_MAX_RATIO = 1.4
COMPONENT_COMPACT_SINGLE_WIDTH_RATIO = 0.35
COMPONENT_COMPACT_SINGLE_HEIGHT_RATIO = 0.45
COMPONENT_COMPACT_SMALL_SCENE_WIDTH_RATIO = 0.55
COMPONENT_COMPACT_SMALL_SCENE_HEIGHT_RATIO = 0.38
COMPONENT_COMPACT_MIN_WIDTH_RATIO = 0.65
COMPONENT_COMPACT_MIN_HEIGHT_RATIO = 0.32
COMPONENT_COMPACT_HORIZONTAL_WIDTH_RATIO = 0.70
COMPONENT_COMPACT_HORIZONTAL_HEIGHT_RATIO = 0.30
COMPONENT_COMPACT_DEPTH_WIDTH_RATIO = 0.65
COMPONENT_COMPACT_DEPTH_HEIGHT_RATIO = 0.34
COMPONENT_COMPACT_MANY_OBJECT_WIDTH_RATIO = 0.78
COMPONENT_COMPACT_MANY_OBJECT_HEIGHT_RATIO = 0.28
COMPONENT_COMPACT_VERTICAL_WIDTH_RATIO = 0.45
COMPONENT_COMPACT_VERTICAL_HEIGHT_RATIO = 0.50
COMPONENT_COMPACT_MARGIN_PX = 32.0
COMPONENT_COMPACT_MIN_MARGIN_PX = 20.0
COMPONENT_COMPACT_CAMERA_STEP = 1.05
COMPONENT_COMPACT_ELEVATION_STEP_DEG = 2.0
COMPONENT_COMPACT_POST_REPAIR_PASSES = 3
COMPONENT_COMPACT_MAX_POST_REPAIR_CANDIDATES = 96
MARGIN_Z = 0.02
SUPPORT_INSET = 0.02
EPS = 1e-5
AZIMUTH_TOLERANCE_RAD = 1e-4
DEFAULT_FRONT_AZIMUTH_RAD = math.pi / 2.0
CAMERA_FACE_AZIMUTH_OFFSETS = {
    "front": 0.0,
    "left": math.pi / 2.0,
    "right": -math.pi / 2.0,
    "back": math.pi,
}
DEFAULT_LENS_RANGE = (18.0, 80.0)
DEFAULT_CAMERA_ELEVATION_DEG = core.DEFAULT_CAMERA_ELEVATION_DEG
MIN_CAMERA_ELEVATION_DEG = core.MIN_CAMERA_ELEVATION_DEG
MAX_CAMERA_ELEVATION_DEG = core.MAX_CAMERA_ELEVATION_DEG
DEFAULT_ELEVATION_DEG_RANGE = (MIN_CAMERA_ELEVATION_DEG, MAX_CAMERA_ELEVATION_DEG)
DEFAULT_GLOBAL_SCALE_RANGE = (0.4, 2.5)
IMAGE_CENTER_PX = 512.0
IMAGE_SIZE_PX = 1024.0
SCREEN_DEPTH_TOLERANCE_PX = 3.0
SCREEN_SIZE_RATIO_TOLERANCE = 0.01
DIM_TARGET_TOLERANCE_RATIO = 0.08
GUIDANCE_DIM_DECREASE_SCALE = 1.0 - DIM_TARGET_TOLERANCE_RATIO
GUIDANCE_DIM_INCREASE_SCALE = 1.0 + DIM_TARGET_TOLERANCE_RATIO
GUIDANCE_LENS_DECREASE_SCALE = 0.80
GUIDANCE_GLOBAL_SCALE_DECREASE_SCALE = 0.90
VISUAL_TARGET_DIMS = {
    # Deprecated: inference/asset_dimensions.json is now the harness source of
    # layout-readable cube dimensions for every supported asset type.
}
MIN_VISUAL_DIMS = [0.12, 0.12, 0.08]

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
COUNT_DETERMINERS = (
    "a",
    "an",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)
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
    "mice": "mouse",
    "wolves": "wolf",
    "shelves": "shelf",
}
PART_WORDS = {
    "countertop",
    "surface",
    "top",
    "side",
    "front",
    "back",
    "left",
    "right",
}
SUPPORT_HINT_RE = re.compile(
    r"\b(on top of|sits on|sitting on|placed on|positioned on|resting on|rests on|atop|"
    r"on the countertop of|on countertop of|on the surface of|on the top of)\b|"
    r"放在.+?上|放到.+?上|摆在.+?上|置于.+?上|在.+?顶部|放在.+?顶部|在.+?台面上|放在.+?台面上|在.+?表面上|放在.+?表面上",
    re.IGNORECASE,
)
TIGHT_HINT_RE = re.compile(
    r"\b(tightly|flush|touching|in contact with)\b|紧贴|贴在|贴着|紧挨|紧靠",
    re.IGNORECASE,
)
LEFT_RE = re.compile(r"\b(to the left of|on the left of|left of|left side of)\b|左边|左侧|左方|左面", re.IGNORECASE)
RIGHT_RE = re.compile(r"\b(to the right of|on the right of|right of|right side of)\b|右边|右侧|右方|右面", re.IGNORECASE)
FRONT_RE = re.compile(r"\b(in front of|front of|ahead of)\b|前面|前方", re.IGNORECASE)
BEHIND_RE = re.compile(r"\b(behind|at the back of|back of)\b|后面|后方|后边", re.IGNORECASE)
ABOVE_RE = re.compile(r"\b(above|higher than|over|on top of)\b|上方|上面|顶部|顶上", re.IGNORECASE)
BELOW_RE = re.compile(r"\b(below|beneath|under|lower than)\b|下方|下面|底部", re.IGNORECASE)
CENTER_RE = re.compile(r"\b(center of the image|in the center|at the center|middle of the image|center)\b|居中|中央|中间|正中", re.IGNORECASE)
SUPPORT_TARGET_RE = re.compile(
    r"\b(?:on\s+(?:the\s+)?(?:countertop|surface|top)\s+of|"
    r"on\s+top\s+of|atop|placed\s+on|positioned\s+on|resting\s+on|rests\s+on|"
    r"sits\s+on|sitting\s+on)\b|"
    r"放在.+?上|放到.+?上|摆在.+?上|置于.+?上|在.+?顶部|放在.+?顶部|在.+?台面上|放在.+?台面上|在.+?表面上|放在.+?表面上",
    re.IGNORECASE,
)
RELATION_RULE_PATTERNS = {
    "centered": (
        r"\b(?:in the center(?: of the image)?|at the center(?: of the image)?|centered(?: in the image)?|centers? the image|anchors? the image)\b",
        r"(?:在|位于)?(?:画面|图像)?(?:中央|中间)|居中",
    ),
    "left_of": (
        r"\b(?:(?:stands?|sits?) (?:to )?(?:the )?left of|to the left of|on the left of|left of|on the left side of|left side of|positioned to the left of|located to the left of)\b",
        r"(?:在|位于|放在|放到)?[^，。,.!?;；]*?(?:左边|左侧)",
    ),
    "right_of": (
        r"\b(?:(?:stands?|sits?) (?:to )?(?:the )?right of|to the right of|on the right of|right of|on the right side of|right side of|positioned to the right of|located to the right of)\b",
        r"(?:在|位于|放在|放到)?[^，。,.!?;；]*?(?:右边|右侧)",
    ),
    "in_front_of": (
        r"\b(?:in front of|front of|positioned in front of|located in front of)\b",
        r"(?:在|位于|放在|放到)?[^，。,.!?;；]*?(?:前面|前方)",
    ),
    "behind": (
        r"\b(?:behind|at the back of|in back of|back of|positioned behind|located behind)\b",
        r"(?:在|位于|放在|放到)?[^，。,.!?;；]*?(?:后面|后方|后边)",
    ),
    "above": (
        r"\b(?:above|over|higher than)\b",
        r"(?:在|位于)?[^，。,.!?;；]*?上方|高于",
    ),
    "below": (
        r"\b(?:below|under|beneath|lower than)\b",
        r"(?:在|位于)?[^，。,.!?;；]*?(?:下方|下面)|低于",
    ),
    "support": (
        r"\b(?:on top of|atop|placed on|positioned on|resting on|rests on|sitting on|sits on|on the countertop of|on countertop of|on the surface of|on the top of)\b",
        r"(?:在|放在|放到|摆在|置于)[^，。,.!?;；]*?(?:上|顶部|台面上|表面上)",
    ),
}
COMPOUND_RELATION_RULES = (
    ("front_right", ("in_front_of", "right_of"), (r"\bfront-right of\b|\blocated in front-right of\b|\bpositioned front-right of\b", r"(?:在|位于|放在)?[^，。,.!?;；]*?右前方")),
    ("front_left", ("in_front_of", "left_of"), (r"\bfront-left of\b|\blocated in front-left of\b|\bpositioned front-left of\b", r"(?:在|位于|放在)?[^，。,.!?;；]*?左前方")),
    ("back_right", ("behind", "right_of"), (r"\bback-right of\b|\blocated in back-right of\b|\bpositioned back-right of\b", r"(?:在|位于|放在)?[^，。,.!?;；]*?右后方")),
    ("back_left", ("behind", "left_of"), (r"\bback-left of\b|\blocated in back-left of\b|\bpositioned back-left of\b", r"(?:在|位于|放在)?[^，。,.!?;；]*?左后方")),
    ("behind_left", ("behind", "left_of"), (r"\bbehind and to the left of\b|\bbehind to the left of\b", "")),
    ("behind_right", ("behind", "right_of"), (r"\bbehind and to the right of\b|\bbehind to the right of\b", "")),
    ("front_left_joined", ("in_front_of", "left_of"), (r"\bin front of and to the left of\b|\bin front and to the left of\b", "")),
    ("front_right_joined", ("in_front_of", "right_of"), (r"\bin front of and to the right of\b|\bin front and to the right of\b", "")),
)
SCREEN_DEPTH_LEVEL_PATTERNS = (
    ("extra_far", (r"\b(?:very\s+far|extremely\s+far|deeply|much\s+farther|extra-far|extra\s+far)\b", r"(?:非常远|特别远|更远|更深|很深|远得多)")),
    ("far", (r"\bfar\b", r"远")),
    ("clear", (r"\b(?:clearly|obviously|noticeably|distinctly)\b", r"(?:明显|清楚|显著)")),
    ("slight", (r"\b(?:slightly|a\s+little|a\s+bit)\b", r"(?:稍微|稍稍|一点|略微)")),
)
CAMERA_FACE_LABELS = ("back", "left", "right", "front")


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    return value


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_reference_dims_map() -> Dict[str, List[float]]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference", "asset_dimensions.json")
    payload = load_json(path)
    return {
        str(asset_type): [float(values[0]), float(values[1]), float(values[2])]
        for asset_type, values in payload.items()
        if isinstance(values, list) and len(values) >= 3
    }


def save_json(payload: Dict[str, Any], path: str) -> str:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, ensure_ascii=False, indent=2)
    return path


def save_jsonl(records: Sequence[Dict[str, Any]], path: str) -> str:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_builtin(record), ensure_ascii=False) + "\n")
    return path


def normalize_angle_rad(value: float) -> float:
    normalized = math.fmod(float(value), math.tau)
    if normalized < 0.0:
        normalized += math.tau
    if normalized <= 1e-6 or math.tau - normalized <= 1e-6:
        normalized = 0.0
    return round(normalized, 6)


def angular_distance_rad(first: float, second: float) -> float:
    delta = abs(normalize_angle_rad(first) - normalize_angle_rad(second))
    return min(delta, math.tau - delta)


def load_asset_default_azimuths(allowed_types: Sequence[str]) -> Dict[str, float]:
    defaults: Dict[str, float] = {str(asset_type): normalize_angle_rad(DEFAULT_FRONT_AZIMUTH_RAD) for asset_type in allowed_types}
    if not os.path.isfile(ASSET_DEFAULT_AZIMUTHS_PATH):
        return defaults

    payload = load_json(ASSET_DEFAULT_AZIMUTHS_PATH)
    canonical_payload = {core.canonicalize_type(str(key)): value for key, value in payload.items()}
    for asset_type in allowed_types:
        key = core.canonicalize_type(str(asset_type))
        if key not in canonical_payload:
            continue
        defaults[str(asset_type)] = normalize_angle_rad(float(canonical_payload[key]))
    return defaults


def camera_face_azimuth(asset_type: str, face: str, default_azimuths_map: Dict[str, float]) -> float:
    base = float(default_azimuths_map.get(asset_type, DEFAULT_FRONT_AZIMUTH_RAD))
    if angular_distance_rad(base, DEFAULT_FRONT_AZIMUTH_RAD) <= AZIMUTH_TOLERANCE_RAD:
        base = DEFAULT_FRONT_AZIMUTH_RAD
    offset = CAMERA_FACE_AZIMUTH_OFFSETS.get(face, 0.0)
    return normalize_angle_rad(base + offset)


def sidecar_path(pkl_path: str, suffix: str) -> str:
    base, ext = os.path.splitext(pkl_path)
    if ext.lower() == ".pkl":
        return f"{base}.{suffix}"
    return f"{pkl_path}.{suffix}"


def canonical_id(text: str) -> str:
    normalized = core.canonicalize_type(str(text or ""))
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
    return normalized or "object"


def singularize_word(word: str) -> str:
    lowered = word.lower().strip()
    if lowered in IRREGULAR_SINGULARS:
        return IRREGULAR_SINGULARS[lowered]
    if lowered.endswith("ies") and len(lowered) > 3:
        return lowered[:-3] + "y"
    if lowered.endswith("s") and not lowered.endswith("ss"):
        return lowered[:-1]
    return lowered


def singularize_phrase(phrase: str) -> str:
    tokens = core.canonicalize_type(phrase).split()
    if not tokens:
        return ""
    tokens[-1] = singularize_word(tokens[-1])
    return " ".join(tokens)


def phrase_candidates(phrase: str) -> List[str]:
    normalized = core.canonicalize_type(phrase)
    tokens = normalized.split()
    candidates: List[str] = []
    seen = set()

    def add(candidate: str) -> None:
        item = core.canonicalize_type(candidate)
        if item and item not in seen:
            seen.add(item)
            candidates.append(item)

    add(normalized)
    add(singularize_phrase(normalized))
    for window in range(min(3, len(tokens)), 0, -1):
        suffix = " ".join(tokens[-window:])
        add(suffix)
        add(singularize_phrase(suffix))
    return candidates


def map_asset_type(mention: str, allowed_types: Sequence[str]) -> str:
    for candidate in phrase_candidates(mention):
        mapped = core.match_asset_type(None, candidate, list(allowed_types))
        if mapped != CUSTOM_ASSET_TYPE:
            return mapped
    return CUSTOM_ASSET_TYPE


def display_name_for_type(asset_type: str, index: int, total_count: int) -> str:
    if total_count <= 1:
        return asset_type
    return f"{asset_type} {index}"


def normalize_manifest(manifest: Dict[str, Any], allowed_types: Sequence[str]) -> Dict[str, Any]:
    objects = []
    counts: Counter = Counter()
    for raw in manifest.get("objects", []):
        if not isinstance(raw, dict):
            continue
        mention = str(raw.get("mention") or raw.get("name") or raw.get("type") or "").strip()
        raw_type = str(raw.get("canonical_type") or raw.get("type") or mention).strip()
        asset_type = map_asset_type(raw_type, allowed_types)
        if asset_type == CUSTOM_ASSET_TYPE:
            asset_type = map_asset_type(mention, allowed_types)
        if asset_type == CUSTOM_ASSET_TYPE:
            objects.append(
                {
                    "id": canonical_id(raw.get("id") or mention or "unknown"),
                    "name": str(raw.get("name") or mention or "unknown").strip(),
                    "type": CUSTOM_ASSET_TYPE,
                    "mention": mention,
                    "confidence": float(raw.get("confidence", 0.0) or 0.0),
                }
            )
            continue
        counts[asset_type] += 1
        object_id = str(raw.get("id") or f"{canonical_id(asset_type)}_{counts[asset_type]}").strip()
        name = str(raw.get("name") or raw.get("display_name") or display_name_for_type(asset_type, counts[asset_type], 99)).strip()
        objects.append(
            {
                "id": object_id,
                "name": name,
                "type": asset_type,
                "canonical_type": asset_type,
                "mention": mention or asset_type,
                "source_span": raw.get("source_span", mention or asset_type),
                "confidence": float(raw.get("confidence", 1.0) or 1.0),
            }
        )

    type_totals = Counter(obj["type"] for obj in objects if obj["type"] != CUSTOM_ASSET_TYPE)
    type_seen: Counter = Counter()
    for obj in objects:
        if obj["type"] == CUSTOM_ASSET_TYPE:
            continue
        type_seen[obj["type"]] += 1
        obj["id"] = obj.get("id") or f"{canonical_id(obj['type'])}_{type_seen[obj['type']]}"
        obj["name"] = display_name_for_type(obj["type"], type_seen[obj["type"]], type_totals[obj["type"]])

    return {
        "objects": objects,
        "non_objects": manifest.get("non_objects", []),
        "parse_warnings": manifest.get("parse_warnings", []),
    }


def parse_objects_spec(objects_spec: str, allowed_types: Sequence[str]) -> Dict[str, Any]:
    objects = []
    if not objects_spec.strip():
        return {"objects": objects, "non_objects": []}
    for raw_part in re.split(r"[,;]", objects_spec):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            raw_name, raw_count = part.split(":", 1)
        elif "=" in part:
            raw_name, raw_count = part.split("=", 1)
        else:
            raw_name, raw_count = part, "1"
        asset_type = map_asset_type(raw_name, allowed_types)
        count = int(NUMBER_WORDS.get(raw_count.strip().lower(), raw_count.strip() or 1))
        for _ in range(max(count, 1)):
            objects.append({"type": asset_type, "mention": raw_name.strip(), "confidence": 1.0})
    return normalize_manifest({"objects": objects}, allowed_types)


def parse_quantified_mentions(scene_text: str) -> List[Tuple[str, int]]:
    normalized_text = core.normalize_free_text(scene_text)
    if not normalized_text:
        return []
    determiner_re = re.compile(
        r"\b(?P<det>the|a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
        re.IGNORECASE,
    )
    token_re = re.compile(r"[a-zA-Z]+(?:-[a-zA-Z]+)?|\d+")
    mentions: List[Tuple[str, int]] = []
    for match in determiner_re.finditer(normalized_text):
        det = match.group("det").lower()
        if det == "the":
            continue
        tail = normalized_text[match.end() :]
        phrase_tokens: List[str] = []
        for token in token_re.findall(tail):
            lower = token.lower()
            if lower in PART_WORDS and not phrase_tokens:
                break
            if lower in ALL_DETERMINERS and phrase_tokens:
                break
            if lower in OBJECT_EXTRACTION_STOP_TOKENS:
                break
            if lower.isdigit() and phrase_tokens:
                break
            phrase_tokens.append(lower)
            if len(phrase_tokens) >= 6:
                break
        if not phrase_tokens:
            continue
        phrase = " ".join(phrase_tokens).strip()
        count = 1 if det in {"a", "an"} else int(NUMBER_WORDS.get(det, det))
        mentions.append((phrase, max(count, 1)))
    return mentions


def build_object_manifest(
    scene_text: str,
    allowed_types: Sequence[str],
    objects_spec: str = "",
    manifest_path: str = "",
    on_ambiguous: str = "best_effort",
) -> Dict[str, Any]:
    if manifest_path.strip():
        return normalize_manifest(load_json(manifest_path), allowed_types)
    if objects_spec.strip():
        return parse_objects_spec(objects_spec, allowed_types)

    objects = []
    warnings = []
    for mention, count in parse_quantified_mentions(scene_text):
        asset_type = map_asset_type(mention, allowed_types)
        if asset_type == CUSTOM_ASSET_TYPE:
            warnings.append(f"对象短语 `{mention}` 不能映射到资产类型。")
            if on_ambiguous == "fail":
                continue
        for _ in range(count):
            objects.append({"type": asset_type, "mention": mention, "confidence": 0.85 if asset_type != CUSTOM_ASSET_TYPE else 0.0})

    if not objects:
        message = "未能从 prompt 稳定解析物体主体；请提供 --objects 或 --object-manifest。"
        if on_ambiguous in {"fail", "ask"}:
            raise RuntimeError(message)
        warnings.append(message)
    return normalize_manifest({"objects": objects, "parse_warnings": warnings}, allowed_types)


def subject_field(subject: Dict[str, Any], field: str, default: float = 0.0) -> float:
    value = subject.get(field)
    if isinstance(value, list) and value:
        return float(value[0])
    if isinstance(value, tuple) and value:
        return float(value[0])
    return float(value if value is not None else default)


def set_subject_field(subject: Dict[str, Any], field: str, value: float) -> None:
    subject[field] = [round(float(value), 6)]


def dims(subject: Dict[str, Any]) -> List[float]:
    raw = subject.get("dims", [1.0, 1.0, 1.0])
    return [float(raw[0]), float(raw[1]), float(raw[2])]


def set_dims(subject: Dict[str, Any], new_dims: Sequence[float]) -> None:
    subject["dims"] = [round(max(float(v), 1e-6), 6) for v in new_dims[:3]]


def set_dims_action(subject: Dict[str, Any], new_dims: Sequence[float], reason: str, actions: List[Dict[str, Any]]) -> None:
    old_dims = dims(subject)
    bounded_dims = [round(max(float(v), MIN_VISUAL_DIMS[idx]), 6) for idx, v in enumerate(new_dims[:3])]
    if any(abs(old_dims[idx] - bounded_dims[idx]) > EPS for idx in range(3)):
        set_dims(subject, bounded_dims)
        actions.append(action_record("set_param", str(subject.get("name", "")), "dims", old_dims, bounded_dims, reason))


def edges(subject: Dict[str, Any]) -> Dict[str, float]:
    width, depth, height = dims(subject)
    x = subject_field(subject, "x")
    y = subject_field(subject, "y")
    z = subject_field(subject, "z")
    return {
        "left": y - width / 2.0,
        "right": y + width / 2.0,
        "back": x - depth / 2.0,
        "front": x + depth / 2.0,
        "bottom": z,
        "top": z + height,
    }


def margin_x(first: Dict[str, Any], second: Dict[str, Any]) -> float:
    return max(MARGIN_X_MIN, 0.05 * max(dims(first)[1], dims(second)[1]))


def margin_y(first: Dict[str, Any], second: Dict[str, Any]) -> float:
    return max(MARGIN_Y_MIN, 0.05 * max(dims(first)[0], dims(second)[0]))


def lateral_world_margin(first: Dict[str, Any], second: Dict[str, Any], predicate: Dict[str, Any], tight: bool = False) -> float:
    base_margin = 0.0 if tight else margin_y(first, second)
    return max(base_margin, float(predicate.get("compound_lateral_margin", 0.0) or 0.0))


def clone_scene(scene_dict: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(scene_dict)


def subjects(scene_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_subjects = scene_dict.setdefault("subjects_data", [])
    if not isinstance(raw_subjects, list):
        scene_dict["subjects_data"] = []
    return scene_dict["subjects_data"]


def make_subject(
    obj: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    slot_index: int,
) -> Dict[str, Any]:
    asset_type = obj["type"]
    ref_dims = reference_dims_map.get(asset_type, [1.0, 1.0, 1.0])
    ring = slot_index // 8
    pos = slot_index % 8
    radius = 1.2 + ring * 0.8
    angle = pos * (math.pi / 4.0)
    return {
        "name": obj["name"],
        "type": asset_type,
        "dims": [round(float(v), 6) for v in ref_dims],
        "x": [round(DEFAULT_CENTER_X + math.cos(angle) * radius, 6)],
        "y": [round(DEFAULT_CENTER_Y + math.sin(angle) * radius, 6)],
        "z": [0.0],
        "azimuth": [0.0],
        "bbox": [(0, 0, 0, 0)],
    }


def action_record(
    tool: str,
    obj: str,
    field: str,
    old: Any,
    new: Any,
    reason: str,
    **extra: Any,
) -> Dict[str, Any]:
    record = {
        "tool": tool,
        "object": obj,
        "field": field,
        "old": to_builtin(old),
        "new": to_builtin(new),
        "reason": reason,
    }
    record.update({key: to_builtin(value) for key, value in extra.items()})
    return record


def reconcile_scene_to_manifest(
    scene_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    new_scene = clone_scene(scene_dict)
    current_subjects = subjects(new_scene)
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for subject in current_subjects:
        by_type[str(subject.get("type", ""))].append(subject)

    used_ids = set()
    new_subjects = []
    actions: List[Dict[str, Any]] = []
    for slot_index, obj in enumerate(manifest.get("objects", [])):
        asset_type = obj.get("type")
        if asset_type == CUSTOM_ASSET_TYPE:
            continue
        bucket = by_type.get(asset_type, [])
        chosen = None
        for candidate in bucket:
            if id(candidate) not in used_ids:
                chosen = candidate
                used_ids.add(id(candidate))
                break
        if chosen is None:
            chosen = make_subject(obj, reference_dims_map, slot_index)
            actions.append(action_record("create_object", obj["name"], "subjects_data", None, chosen, "missing_object"))
        else:
            old_name = chosen.get("name")
            old_type = chosen.get("type")
            chosen = copy.deepcopy(chosen)
            chosen["name"] = obj["name"]
            chosen["type"] = asset_type
            if old_name != chosen["name"]:
                actions.append(action_record("rename_object", chosen["name"], "name", old_name, chosen["name"], "object_manifest"))
            if old_type != chosen["type"]:
                actions.append(action_record("set_type", chosen["name"], "type", old_type, chosen["type"], "object_manifest"))
        new_subjects.append(chosen)

    expected_kept_ids = {id(subject) for typed in by_type.values() for subject in typed if id(subject) in used_ids}
    for subject in current_subjects:
        if id(subject) not in expected_kept_ids and subject.get("type") not in [obj.get("type") for obj in manifest.get("objects", [])]:
            actions.append(action_record("delete_object", str(subject.get("name", "")), "subjects_data", subject, None, "extra_object"))
    new_scene["subjects_data"] = new_subjects
    return new_scene, actions


def visual_target_dims(obj: Dict[str, Any], reference_dims_map: Dict[str, List[float]]) -> List[float]:
    asset_type = str(obj.get("type") or "")
    ref = list(VISUAL_TARGET_DIMS.get(asset_type) or reference_dims_map.get(asset_type, [1.0, 1.0, 1.0]))
    target = [max(float(ref[idx]), MIN_VISUAL_DIMS[idx]) for idx in range(3)]
    mention = core.normalize_free_text(str(obj.get("mention", "")))
    if any(token in mention for token in ("small", "tiny", "little", "mini")):
        target = [max(target[idx] * 0.75, MIN_VISUAL_DIMS[idx]) for idx in range(3)]
    if any(token in mention for token in ("large", "big", "huge", "giant")):
        target = [target[idx] * 1.25 for idx in range(3)]
    return [round(value, 6) for value in target]


def dim_bounds(obj: Dict[str, Any], reference_dims_map: Dict[str, List[float]]) -> Tuple[List[float], List[float]]:
    target = visual_target_dims(obj, reference_dims_map)
    lower = [max(target[idx] * (1.0 - DIM_TARGET_TOLERANCE_RATIO), MIN_VISUAL_DIMS[idx]) for idx in range(3)]
    upper = [target[idx] * (1.0 + DIM_TARGET_TOLERANCE_RATIO) for idx in range(3)]
    return lower, upper


def dim_reference_safety_bounds(
    obj: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
) -> Tuple[List[float], List[float]]:
    target = visual_target_dims(obj, reference_dims_map)
    lower = [max(target[idx] * DIM_REFERENCE_MIN_RATIO, MIN_VISUAL_DIMS[idx]) for idx in range(3)]
    upper = [target[idx] * DIM_REFERENCE_MAX_RATIO for idx in range(3)]
    return lower, upper


def clamp_dims_to_reference_safety(
    new_dims: Sequence[float],
    obj: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
) -> List[float]:
    lower, upper = dim_reference_safety_bounds(obj, reference_dims_map)
    return [round(clamp(float(new_dims[idx]), lower[idx], upper[idx]), 6) for idx in range(3)]


def dims_at_reference_safety_bound(
    subject: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    bound: str,
) -> bool:
    obj = {"type": subject.get("type"), "mention": subject.get("name")}
    lower, upper = dim_reference_safety_bounds(obj, reference_dims_map)
    current_dims = dims(subject)
    if bound == "lower":
        return all(current_dims[idx] <= lower[idx] + 1e-4 for idx in range(3))
    if bound == "upper":
        return all(current_dims[idx] >= upper[idx] - 1e-4 for idx in range(3))
    return False


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def normalize_dimensions(
    scene_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    obj_by_name = {obj["name"]: obj for obj in manifest.get("objects", [])}
    for subject in subjects(scene_dict):
        if subject.get("_screen_size_dim_compensated"):
            obj = obj_by_name.get(subject.get("name"), {"type": subject.get("type"), "mention": subject.get("name")})
            old_dims = dims(subject)
            new_dims = clamp_dims_to_reference_safety(old_dims, obj, reference_dims_map)
            if any(abs(old_dims[idx] - new_dims[idx]) > EPS for idx in range(3)):
                set_dims(subject, new_dims)
                actions.append(action_record("set_param", subject["name"], "dims", old_dims, new_dims, "dimension_reference_safety"))
            continue
        obj = obj_by_name.get(subject.get("name"), {"type": subject.get("type"), "mention": subject.get("name")})
        old_dims = dims(subject)
        new_dims = visual_target_dims(obj, reference_dims_map)
        if any(abs(old_dims[idx] - new_dims[idx]) > EPS for idx in range(3)):
            set_dims(subject, new_dims)
            actions.append(action_record("project_dims", subject["name"], "dims", old_dims, new_dims, "visual_target_dims"))
    actions.extend(enforce_volume_order(scene_dict, reference_dims_map))
    return actions


def enforce_camera_facing_azimuths(
    scene_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    camera_facing_payload: Dict[str, Any],
    default_azimuths_map: Dict[str, float],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    id_to_name = manifest_id_to_name(manifest)
    obj_by_id = {obj["id"]: obj for obj in manifest.get("objects", [])}
    by_name = subject_by_name(scene_dict)
    for constraint in camera_facing_payload.get("camera_facing", []):
        obj_id = constraint.get("object")
        obj = obj_by_id.get(obj_id)
        subject_name = id_to_name.get(obj_id, "")
        subject = by_name.get(subject_name)
        if obj is None or subject is None:
            continue
        face = str(constraint.get("face") or "front")
        expected = camera_face_azimuth(str(obj.get("type", "")), face, default_azimuths_map)
        old = normalize_angle_rad(subject_field(subject, "azimuth"))
        if angular_distance_rad(old, expected) <= AZIMUTH_TOLERANCE_RAD:
            if abs(subject_field(subject, "azimuth") - old) > EPS:
                set_subject_field(subject, "azimuth", old)
            continue
        set_subject_field(subject, "azimuth", expected)
        actions.append(
            action_record(
                "set_param",
                subject["name"],
                "azimuth[0]",
                old,
                expected,
                f"camera_facing:{face}",
                evidence=constraint.get("evidence", ""),
            )
        )
    return actions


def volume_for_dims(values: Sequence[float]) -> float:
    return float(values[0]) * float(values[1]) * float(values[2])


def enforce_volume_order(scene_dict: Dict[str, Any], reference_dims_map: Dict[str, List[float]]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    current = subjects(scene_dict)
    for idx, first in enumerate(current):
        if first.get("_screen_size_dim_compensated"):
            continue
        first_ref = visual_target_dims({"type": first.get("type"), "mention": first.get("name")}, reference_dims_map)
        first_ref_volume = volume_for_dims(first_ref)
        first_volume = volume_for_dims(dims(first))
        for second in current[idx + 1 :]:
            if second.get("_screen_size_dim_compensated"):
                continue
            second_ref = visual_target_dims({"type": second.get("type"), "mention": second.get("name")}, reference_dims_map)
            second_ref_volume = volume_for_dims(second_ref)
            second_volume = volume_for_dims(dims(second))
            if first_ref_volume > second_ref_volume * 1.5 and first_volume < second_volume * 1.15:
                old = dims(first)
                set_dims(first, first_ref)
                actions.append(action_record("normalize_volume", first["name"], "dims", old, first_ref, "volume_order"))
            elif second_ref_volume > first_ref_volume * 1.5 and second_volume < first_volume * 1.15:
                old = dims(second)
                set_dims(second, second_ref)
                actions.append(action_record("normalize_volume", second["name"], "dims", old, second_ref, "volume_order"))

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for subject in current:
        by_type[str(subject.get("type"))].append(subject)
    for typed_subjects in by_type.values():
        typed_subjects = [subject for subject in typed_subjects if not subject.get("_screen_size_dim_compensated")]
        if len(typed_subjects) < 2:
            continue
        volumes = [volume_for_dims(dims(subject)) for subject in typed_subjects]
        if min(volumes) <= 0 or max(volumes) / min(volumes) <= 1.25:
            continue
        for subject in typed_subjects:
            ref = visual_target_dims({"type": subject.get("type"), "mention": subject.get("name")}, reference_dims_map)
            old = dims(subject)
            set_dims(subject, ref)
            actions.append(action_record("normalize_same_type_volume", subject["name"], "dims", old, ref, "same_type_consistency"))
    return actions


def object_aliases(obj: Dict[str, Any]) -> List[str]:
    raw_aliases = {
        str(obj.get("id", "")),
        str(obj.get("name", "")),
        str(obj.get("type", "")),
        str(obj.get("mention", "")),
        str(obj.get("id", "")).replace("_", " "),
    }
    aliases = []
    for alias in raw_aliases:
        normalized = core.normalize_free_text(alias)
        if normalized:
            aliases.append(normalized)
            aliases.append(singularize_phrase(normalized))
    return sorted(set(item for item in aliases if item), key=len, reverse=True)


def text_mentions_alias(text: str, obj: Dict[str, Any]) -> bool:
    normalized = core.normalize_free_text(text)
    for alias in object_aliases(obj):
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized):
            return True
    return False


def alias_position(text: str, obj: Dict[str, Any]) -> Optional[int]:
    normalized = core.normalize_free_text(text)
    positions = []
    for alias in object_aliases(obj):
        match = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized)
        if match:
            positions.append(match.start())
    return min(positions) if positions else None


def alias_positions(text: str, obj: Dict[str, Any]) -> List[int]:
    normalized = core.normalize_free_text(text)
    positions: List[int] = []
    for alias in object_aliases(obj):
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized):
            positions.append(match.start())
    return sorted(set(positions))


def nearest_object_after(text: str, objects: Sequence[Dict[str, Any]], start_pos: int) -> Optional[Dict[str, Any]]:
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for obj in objects:
        after_positions = [pos for pos in alias_positions(text, obj) if pos >= start_pos]
        if after_positions:
            candidates.append((min(after_positions), obj))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def object_has_mention_before(text: str, obj: Dict[str, Any], end_pos: int) -> bool:
    return any(pos < end_pos for pos in alias_positions(text, obj))


def relation_between(clause: str, first: Dict[str, Any], second: Dict[str, Any], pattern: re.Pattern[str]) -> bool:
    normalized = core.normalize_free_text(clause)
    first_pos = alias_position(normalized, first)
    second_pos = alias_position(normalized, second)
    relation_match = pattern.search(normalized)
    if first_pos is None or second_pos is None or relation_match is None:
        return False
    return first_pos < relation_match.start() < second_pos or first_pos < second_pos < relation_match.end()


def relation_rule_matches(clause: str, first: Dict[str, Any], second: Dict[str, Any], pattern_texts: Sequence[str]) -> bool:
    normalized = core.normalize_free_text(clause)
    first_positions = alias_positions(normalized, first)
    second_positions = alias_positions(normalized, second)
    if not first_positions or not second_positions:
        return False

    for pattern_text in pattern_texts:
        if not pattern_text:
            continue
        for relation_match in re.finditer(pattern_text, normalized, flags=re.IGNORECASE):
            for first_pos in first_positions:
                for second_pos in second_positions:
                    if first_pos >= second_pos:
                        continue
                    if first_pos < relation_match.start() < second_pos:
                        return True
                    if first_pos < second_pos < relation_match.end():
                        return True
    return False


def possessive_side_rule_matches(clause: str, first: Dict[str, Any], second: Dict[str, Any], side: str) -> bool:
    normalized = core.normalize_free_text(clause).replace("’", "'")
    verb_pattern = (
        r"(?:is\s+(?:sitting|standing|located|positioned)|are\s+(?:sitting|standing|located|positioned)|"
        r"stands?|stand|sits?|sit|lies?|lie|is|are|located|positioned)"
    )
    for first_alias in object_aliases(first):
        first_pattern = rf"(?<![a-z0-9]){re.escape(first_alias)}(?![a-z0-9])"
        for second_alias in object_aliases(second):
            second_pattern = rf"(?<![a-z0-9]){re.escape(second_alias)}(?![a-z0-9])"
            pattern = rf"{first_pattern}\s+{verb_pattern}\s+(?:to|on)\s+(?:the\s+)?{second_pattern}(?:'s)?\s+{side}\b"
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                return True
    return False


def centered_rule_matches(clause: str, obj: Dict[str, Any]) -> bool:
    normalized = core.normalize_free_text(clause)
    if alias_position(normalized, obj) is None:
        return False
    return any(re.search(pattern_text, normalized, flags=re.IGNORECASE) for pattern_text in RELATION_RULE_PATTERNS["centered"])


def collect_rule_predicate_types(clause: str, first: Dict[str, Any], second: Dict[str, Any]) -> List[str]:
    predicate_types: List[str] = []
    seen = set()

    def add(predicate_type: str) -> None:
        if predicate_type not in seen:
            seen.add(predicate_type)
            predicate_types.append(predicate_type)

    for _, compound_predicates, pattern_texts in COMPOUND_RELATION_RULES:
        if relation_rule_matches(clause, first, second, pattern_texts):
            for predicate_type in compound_predicates:
                add(predicate_type)

    for predicate_type in ("left_of", "right_of", "in_front_of", "behind", "above", "below", "support"):
        if relation_rule_matches(clause, first, second, RELATION_RULE_PATTERNS[predicate_type]):
            add(predicate_type)

    if possessive_side_rule_matches(clause, first, second, "left"):
        add("left_of")
    if possessive_side_rule_matches(clause, first, second, "right"):
        add("right_of")

    return predicate_types


def detect_compound_relation(clause: str, first: Dict[str, Any], second: Dict[str, Any]) -> str:
    for compound_name, _, pattern_texts in COMPOUND_RELATION_RULES:
        if relation_rule_matches(clause, first, second, pattern_texts):
            if compound_name.startswith("front_"):
                return "front"
            if compound_name.startswith("back_") or compound_name.startswith("behind_"):
                return "back"
    return ""


def detect_compound_lateral_relation(clause: str, first: Dict[str, Any], second: Dict[str, Any]) -> str:
    for compound_name, _, pattern_texts in COMPOUND_RELATION_RULES:
        if relation_rule_matches(clause, first, second, pattern_texts):
            if compound_name.endswith("_left") or "_left_" in compound_name:
                return "left"
            if compound_name.endswith("_right") or "_right_" in compound_name:
                return "right"
    return ""


def detect_screen_depth_level(clause: str) -> str:
    normalized = core.normalize_free_text(clause)
    for level, pattern_texts in SCREEN_DEPTH_LEVEL_PATTERNS:
        if any(re.search(pattern_text, normalized, flags=re.IGNORECASE) for pattern_text in pattern_texts):
            return level
    return "default"


def screen_depth_delta_px(level: str) -> float:
    return float(SCREEN_DEPTH_DELTA_PX_BY_LEVEL.get(level, SCREEN_DEPTH_DELTA_PX_BY_LEVEL["default"]))


def screen_lateral_gap_px(level: str) -> float:
    return float(SCREEN_LATERAL_GAP_PX_BY_LEVEL.get(level, SCREEN_LATERAL_GAP_PX_BY_LEVEL["default"]))


def camera_face_pattern_texts(face: str) -> Sequence[str]:
    english_nouns = {
        "front": r"front",
        "back": r"(?:back|rear)",
        "left": r"left(?:\s+side)?",
        "right": r"right(?:\s+side)?",
    }
    chinese_nouns = {
        "front": r"(?:正面|前面|前侧|前方)",
        "back": r"(?:背面|后面|后侧|后方)",
        "left": r"(?:左侧|左面|左边)",
        "right": r"(?:右侧|右面|右边)",
    }
    english_face = english_nouns[face]
    chinese_face = chinese_nouns[face]
    return (
        rf"\b(?:facing|face(?:s)?|with (?:its|the))\s+(?:the\s+)?{english_face}\s+(?:toward|to|facing)\s+(?:the\s+)?(?:camera|viewer|lens)\b",
        rf"\b(?:{english_face})\s+(?:of|side of)\s+[^，。,.!?;；]*?\s+(?:faces|facing|toward|to)\s+(?:the\s+)?(?:camera|viewer|lens)\b",
        rf"\b(?:camera|viewer|lens)\s+(?:sees|views)\s+(?:the\s+)?{english_face}\s+(?:of|side of)\b",
        rf"{chinese_face}[^，。,.!?;；]*?(?:朝向|面向|对着)(?:镜头|摄像机|相机|观众)",
        rf"(?:镜头|摄像机|相机|观众)[^，。,.!?;；]*?(?:看到|看见|正对){chinese_face}",
    )


def camera_face_rule_matches(clause: str, obj: Dict[str, Any], face: str) -> bool:
    normalized = core.normalize_free_text(clause)
    english_nouns = {
        "front": r"front",
        "back": r"(?:back|rear)",
        "left": r"left(?:\s+side)?",
        "right": r"right(?:\s+side)?",
    }
    chinese_nouns = {
        "front": r"(?:正面|前面|前侧|前方)",
        "back": r"(?:背面|后面|后侧|后方)",
        "left": r"(?:左侧|左面|左边)",
        "right": r"(?:右侧|右面|右边)",
    }
    english_face = english_nouns[face]
    chinese_face = chinese_nouns[face]
    camera = r"(?:the\s+)?(?:camera|viewer|lens)"
    gap = r"[^，。,.!?;；]*?"

    for alias in object_aliases(obj):
        alias_re = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        patterns = []
        if face == "front":
            patterns.extend(
                [
                    rf"{alias_re}{gap}\b(?:faces|is facing|facing|toward|looks toward)\s+{camera}\b",
                    rf"{alias_re}{gap}(?:朝向|面向|对着)(?:镜头|摄像机|相机|观众)",
                ]
            )
        patterns.extend(
            [
                rf"{alias_re}{gap}\b(?:with\s+(?:its|the)\s+)?(?:the\s+)?{english_face}{gap}\b(?:faces|is facing|facing|toward|to)\s+{camera}\b",
                rf"\b(?:the\s+)?{english_face}\s+(?:of\s+)?(?:the\s+)?{alias_re}{gap}\b(?:faces|is facing|facing|toward|to)\s+{camera}\b",
                rf"{alias_re}'s\s+{english_face}{gap}\b(?:faces|is facing|facing|toward|to)\s+{camera}\b",
                rf"{alias_re}{gap}{chinese_face}{gap}(?:朝向|面向|对着)(?:镜头|摄像机|相机|观众)",
                rf"(?:镜头|摄像机|相机|观众){gap}(?:看到|看见|正对){gap}{alias_re}{gap}{chinese_face}",
            ]
        )
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns):
            return True
    return False


def detect_camera_face_for_object(scene_text: str, obj: Dict[str, Any]) -> Tuple[str, str]:
    for clause in split_clauses(scene_text):
        if not text_mentions_alias(clause, obj):
            continue
        for face in CAMERA_FACE_LABELS:
            if camera_face_rule_matches(clause, obj, face):
                return face, clause
    return "front", "default_front_facing_camera"


def build_camera_facing_constraints(
    scene_text: str,
    manifest: Dict[str, Any],
    default_azimuths_map: Dict[str, float],
) -> Dict[str, Any]:
    constraints = []
    for obj in manifest.get("objects", []):
        asset_type = str(obj.get("type", ""))
        if asset_type == CUSTOM_ASSET_TYPE:
            continue
        face, evidence = detect_camera_face_for_object(scene_text, obj)
        constraints.append(
            {
                "object": obj["id"],
                "face": face,
                "azimuth": camera_face_azimuth(asset_type, face, default_azimuths_map),
                "evidence": evidence,
                "source": "prompt" if evidence != "default_front_facing_camera" else "default",
            }
        )
    return {"camera_facing": constraints}


def split_clauses(text: str) -> List[str]:
    try:
        clauses = core.split_text_clauses(text)
    except Exception:  # noqa: BLE001
        clauses = re.split(r"[.;!?]\s*", text)
    return [clause.strip() for clause in clauses if clause.strip()]


def clamp_camera_elevation_deg(value: float) -> float:
    return clamp(float(value), MIN_CAMERA_ELEVATION_DEG, MAX_CAMERA_ELEVATION_DEG)


def prompt_camera_elevation_constraint(scene_text: str) -> Dict[str, Any]:
    """Compatibility shim: camera angle is no longer forced from prompt text."""
    return {
        "camera_elevation_deg": round(DEFAULT_CAMERA_ELEVATION_DEG, 6),
        "source": "default_camera",
        "evidence": "",
        "lock": False,
    }


def camera_elevation_constraint_from_predicates(predicates_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = predicates_payload if isinstance(predicates_payload, dict) else {}
    return prompt_camera_elevation_constraint(str(payload.get("source_text", "")))


def apply_camera_elevation_constraint(
    scene_dict: Dict[str, Any],
    constraint: Dict[str, Any],
    reason_prefix: str = "prompt_camera_elevation",
    initialize_preference: bool = False,
) -> List[Dict[str, Any]]:
    return []


def strip_internal_camera_metadata(scene_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = clone_scene(scene_dict)
    camera = cleaned.get("camera_data")
    if isinstance(camera, dict):
        for key in list(camera.keys()):
            if key.startswith("_camera_elevation_"):
                camera.pop(key, None)
    return cleaned


def extract_predicates(scene_text: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    objects = [obj for obj in manifest.get("objects", []) if obj.get("type") != CUSTOM_ASSET_TYPE]
    predicates = []
    seen = set()

    def add(predicate_type: str, subject: Dict[str, Any], obj: Optional[Dict[str, Any]], evidence: str, tight: bool = False) -> None:
        compound_axis = detect_compound_relation(evidence, subject, obj) if obj is not None else ""
        lateral_axis = ""
        screen_axis = ""
        if obj is not None and predicate_type == "in_front_of":
            screen_axis = "front"
        elif obj is not None and predicate_type == "behind":
            screen_axis = "back"
        if obj is not None and predicate_type in {"left_of", "right_of"}:
            lateral_axis = "left" if predicate_type == "left_of" else "right"
        key = (predicate_type, subject.get("id"), obj.get("id") if obj else "", tight, compound_axis)
        if key in seen:
            return
        seen.add(key)
        predicate = {
            "id": f"p{len(predicates) + 1}",
            "type": predicate_type,
            "subject": subject["id"],
            "object": obj["id"] if obj else None,
            "tight": bool(tight),
            "evidence": evidence.strip(),
            "confidence": 0.90,
        }
        if compound_axis:
            predicate["compound_axis"] = compound_axis
            if predicate_type in {"in_front_of", "behind"}:
                predicate["compound_longitudinal_margin"] = DEFAULT_COMPOUND_LONGITUDINAL_MARGIN
        if lateral_axis:
            screen_lateral_level = detect_screen_depth_level(evidence)
            compound_lateral_axis = detect_compound_lateral_relation(evidence, subject, obj) if obj is not None else ""
            predicate["screen_lateral_axis"] = lateral_axis
            predicate["screen_lateral_level"] = screen_lateral_level
            predicate["screen_lateral_gap_px"] = screen_lateral_gap_px(screen_lateral_level)
            if compound_lateral_axis or predicate_type in {"left_of", "right_of"}:
                predicate["compound_lateral_margin"] = DEFAULT_COMPOUND_LATERAL_MARGIN
        if screen_axis:
            screen_depth_level = detect_screen_depth_level(evidence)
            predicate["screen_depth_axis"] = screen_axis
            predicate["screen_depth_level"] = screen_depth_level
            predicate["screen_depth_delta_px"] = screen_depth_delta_px(screen_depth_level)
        predicates.append(predicate)

    for clause in split_clauses(scene_text):
        tight = bool(TIGHT_HINT_RE.search(clause))
        mentioned = [obj for obj in objects if text_mentions_alias(clause, obj)]
        for obj in mentioned:
            if centered_rule_matches(clause, obj):
                add("centered", obj, None, clause)

        for first in mentioned:
            for second in mentioned:
                if first is second:
                    continue
                for predicate_type in collect_rule_predicate_types(clause, first, second):
                    add(predicate_type, first, second, clause, tight=(tight or predicate_type == "support"))

    return {
        "predicates": predicates,
        "source_text": scene_text,
    }


def manifest_id_to_name(manifest: Dict[str, Any]) -> Dict[str, str]:
    return {obj["id"]: obj["name"] for obj in manifest.get("objects", [])}


def subject_by_name(scene_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(subject.get("name")): subject for subject in subjects(scene_dict)}


def set_field_action(subject: Dict[str, Any], field: str, value: float, reason: str, actions: List[Dict[str, Any]]) -> None:
    old = subject_field(subject, field)
    if abs(old - value) <= EPS:
        return
    set_subject_field(subject, field, value)
    actions.append(action_record("set_param", subject["name"], f"{field}[0]", old, value, reason))


def set_dim_action(subject: Dict[str, Any], index: int, value: float, reason: str, actions: List[Dict[str, Any]]) -> None:
    old_dims = dims(subject)
    if abs(old_dims[index] - value) <= EPS:
        return
    new_dims = list(old_dims)
    new_dims[index] = max(value, 1e-6)
    if subject.get("_screen_size_dim_compensated"):
        new_dims = clamp_dims_to_reference_safety(
            new_dims,
            {"type": subject.get("type"), "mention": subject.get("name")},
            load_reference_dims_map(),
        )
    set_dims(subject, new_dims)
    actions.append(action_record("set_param", subject["name"], f"dims[{index}]", old_dims[index], new_dims[index], reason))


def clamp_subject_within_support_raw(subject: Dict[str, Any], support: Dict[str, Any], actions: List[Dict[str, Any]], reason: str) -> None:
    width, depth, _ = dims(subject)
    support_width, support_depth, _ = dims(support)
    if width > support_width - 2 * SUPPORT_INSET:
        set_dim_action(subject, 0, max(support_width - 2 * SUPPORT_INSET, 1e-6), reason + ":support_width", actions)
    if depth > support_depth - 2 * SUPPORT_INSET:
        set_dim_action(subject, 1, max(support_depth - 2 * SUPPORT_INSET, 1e-6), reason + ":support_depth", actions)

    width, depth, _ = dims(subject)
    support_width, support_depth, support_height = dims(support)
    sx = subject_field(support, "x")
    sy = subject_field(support, "y")
    sz = subject_field(support, "z")
    max_x_offset = max((support_depth - depth) / 2.0 - SUPPORT_INSET, 0.0)
    max_y_offset = max((support_width - width) / 2.0 - SUPPORT_INSET, 0.0)
    new_x = clamp(subject_field(subject, "x"), sx - max_x_offset, sx + max_x_offset)
    new_y = clamp(subject_field(subject, "y"), sy - max_y_offset, sy + max_y_offset)
    set_field_action(subject, "x", new_x, reason + ":support_x_interval", actions)
    set_field_action(subject, "y", new_y, reason + ":support_y_interval", actions)
    set_field_action(subject, "z", sz + support_height, reason + ":support_z", actions)


def spread_supported_subjects(scene_dict: Dict[str, Any], manifest: Dict[str, Any], predicates_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    id_to_name = manifest_id_to_name(manifest)
    by_name = subject_by_name(scene_dict)
    by_support: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for predicate in predicates_payload.get("predicates", []):
        if predicate.get("type") != "support":
            continue
        subject_name = id_to_name.get(predicate.get("subject", ""))
        support_name = id_to_name.get(predicate.get("object", ""))
        subject = by_name.get(subject_name)
        support = by_name.get(support_name)
        if subject is None or support is None:
            continue
        by_support[support_name].append(subject)

    for support_name, supported_subjects in by_support.items():
        if len(supported_subjects) < 2:
            continue
        support = by_name.get(support_name)
        if support is None:
            continue
        supported_subjects = sorted(supported_subjects, key=lambda item: volume_for_dims(dims(item)), reverse=True)
        support_width, _, support_height = dims(support)
        support_x = subject_field(support, "x")
        support_y = subject_field(support, "y")
        support_z = subject_field(support, "z")
        total_width = sum(dims(subject)[0] for subject in supported_subjects)
        available_gap = max(support_width - total_width, 0.0)
        gap = min(max(available_gap / (len(supported_subjects) + 1), 0.03), 0.12)
        cursor_y = support_y - min(total_width + gap * (len(supported_subjects) - 1), support_width) / 2.0
        for subject in supported_subjects:
            subject_width = dims(subject)[0]
            target_y = cursor_y + subject_width / 2.0
            cursor_y = target_y + subject_width / 2.0 + gap
            set_field_action(subject, "x", support_x, f"spread_on_support({subject['name']},{support_name}):align_x", actions)
            set_field_action(subject, "y", target_y, f"spread_on_support({subject['name']},{support_name}):slot_y", actions)
            set_field_action(subject, "z", support_z + support_height, f"spread_on_support({subject['name']},{support_name}):support_z", actions)
            clamp_subject_within_support_raw(subject, support, actions, f"spread_on_support({subject['name']},{support_name})")
    return actions


def apply_predicates(scene_dict: Dict[str, Any], manifest: Dict[str, Any], predicates_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    id_to_name = manifest_id_to_name(manifest)
    by_name = subject_by_name(scene_dict)

    pair_types: Dict[Tuple[str, str], set] = defaultdict(set)
    for predicate in predicates_payload.get("predicates", []):
        subject_name = id_to_name.get(predicate.get("subject", ""))
        object_name = id_to_name.get(predicate.get("object", ""))
        if subject_name and object_name:
            pair_types[(subject_name, object_name)].add(str(predicate.get("type", "")))

    for _ in range(8):
        changed_count = len(actions)
        for predicate in predicates_payload.get("predicates", []):
            subject_name = id_to_name.get(predicate.get("subject", ""))
            object_name = id_to_name.get(predicate.get("object", ""))
            first = by_name.get(subject_name)
            second = by_name.get(object_name) if object_name else None
            if first is None:
                continue
            ptype = predicate.get("type")
            reason = f"{ptype}({subject_name},{object_name})"
            tight = bool(predicate.get("tight"))
            if ptype == "centered":
                set_field_action(first, "x", DEFAULT_CENTER_X, reason, actions)
                set_field_action(first, "y", DEFAULT_CENTER_Y, reason, actions)
            if second is None:
                continue
            first_width, first_depth, first_height = dims(first)
            second_width, second_depth, second_height = dims(second)
            second_x = subject_field(second, "x")
            second_y = subject_field(second, "y")
            second_z = subject_field(second, "z")
            current_pair_types = pair_types.get((subject_name, object_name), set())
            has_lateral_pair = bool(current_pair_types & {"left_of", "right_of"})
            has_longitudinal_pair = bool(current_pair_types & {"in_front_of", "behind"})

            if ptype == "support":
                clamp_subject_within_support_raw(first, second, actions, reason)
            elif ptype == "above":
                set_field_action(first, "x", second_x, reason + ":align_x", actions)
                set_field_action(first, "y", second_y, reason + ":align_y", actions)
                target_z = second_z + second_height + (0.0 if tight else MARGIN_Z)
                if subject_field(first, "z") < target_z or tight:
                    set_field_action(first, "z", target_z, reason + ":z_interval", actions)
            elif ptype == "below":
                target_z = second_z - first_height - (0.0 if tight else MARGIN_Z)
                if target_z >= 0:
                    set_field_action(first, "x", second_x, reason + ":align_x", actions)
                    set_field_action(first, "y", second_y, reason + ":align_y", actions)
                    set_field_action(first, "z", target_z, reason + ":z_interval", actions)
            elif ptype == "left_of":
                margin = lateral_world_margin(first, second, predicate, tight=tight)
                target_y = second_y - second_width / 2.0 - first_width / 2.0 - margin
                set_field_action(first, "y", min(subject_field(first, "y"), target_y), reason + ":y_upper", actions)
                if not has_longitudinal_pair:
                    set_field_action(first, "x", second_x, reason + ":align_x", actions)
            elif ptype == "right_of":
                margin = lateral_world_margin(first, second, predicate, tight=tight)
                target_y = second_y + second_width / 2.0 + first_width / 2.0 + margin
                set_field_action(first, "y", max(subject_field(first, "y"), target_y), reason + ":y_lower", actions)
                if not has_longitudinal_pair:
                    set_field_action(first, "x", second_x, reason + ":align_x", actions)
            elif ptype == "in_front_of":
                margin = 0.0 if tight else margin_x(first, second)
                if predicate.get("compound_axis") == "front":
                    margin = max(margin, float(predicate.get("compound_longitudinal_margin", DEFAULT_COMPOUND_LONGITUDINAL_MARGIN)))
                target_x = second_x + second_depth / 2.0 + first_depth / 2.0 + margin
                set_field_action(first, "x", max(subject_field(first, "x"), target_x), reason + ":x_lower", actions)
                if not has_lateral_pair:
                    set_field_action(first, "y", second_y, reason + ":align_y", actions)
            elif ptype == "behind":
                margin = 0.0 if tight else margin_x(first, second)
                if predicate.get("compound_axis") == "back":
                    margin = max(margin, float(predicate.get("compound_longitudinal_margin", DEFAULT_COMPOUND_LONGITUDINAL_MARGIN)))
                target_x = second_x - second_depth / 2.0 - first_depth / 2.0 - margin
                set_field_action(first, "x", min(subject_field(first, "x"), target_x), reason + ":x_upper", actions)
                if not has_lateral_pair:
                    set_field_action(first, "y", second_y, reason + ":align_y", actions)
        if len(actions) == changed_count:
            break
    return actions


def enforce_ground_and_collision(scene_dict: Dict[str, Any], predicates_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    supported_names = set()
    for predicate in predicates_payload.get("predicates", []):
        if predicate.get("type") == "support":
            supported_names.add(predicate.get("subject"))
    for subject in subjects(scene_dict):
        if subject_field(subject, "z") < 0:
            set_field_action(subject, "z", 0.0, "below_ground", actions)
    return actions


def dot3(first: Tuple[float, float, float], second: Tuple[float, float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def cross3(first: Tuple[float, float, float], second: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def normalize3(vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
    length = math.sqrt(dot3(vector, vector))
    if length <= 0.0:
        return (0.0, 0.0, 0.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def cube_world_vertices(subject: Dict[str, Any], global_scale: float) -> List[Tuple[float, float, float]]:
    width, depth, height = [value * global_scale for value in dims(subject)]
    center_x = subject_field(subject, "x")
    center_y = subject_field(subject, "y")
    center_z = subject_field(subject, "z") + height / 2.0
    azimuth = subject_field(subject, "azimuth")
    cos_azimuth = math.cos(azimuth)
    sin_azimuth = math.sin(azimuth)
    local_vertices = [
        (-width / 2.0, -depth / 2.0, -height / 2.0),
        (width / 2.0, -depth / 2.0, -height / 2.0),
        (width / 2.0, depth / 2.0, -height / 2.0),
        (-width / 2.0, depth / 2.0, -height / 2.0),
        (-width / 2.0, -depth / 2.0, height / 2.0),
        (width / 2.0, -depth / 2.0, height / 2.0),
        (width / 2.0, depth / 2.0, height / 2.0),
        (-width / 2.0, depth / 2.0, height / 2.0),
    ]
    vertices = []
    for local_x, local_y, local_z in local_vertices:
        rotated_x = local_x * cos_azimuth - local_y * sin_azimuth
        rotated_y = local_x * sin_azimuth + local_y * cos_azimuth
        vertices.append((center_x + rotated_x, center_y + rotated_y, center_z + local_z))
    return vertices


def project_point_to_image(
    point: Tuple[float, float, float],
    elevation: float,
    lens: float,
    image_size: float = IMAGE_SIZE_PX,
) -> Optional[Tuple[float, float, float]]:
    camera_location = (
        6.0 * math.cos(elevation) - 6.0,
        0.0,
        6.0 * math.sin(elevation),
    )
    forward = normalize3((-1.0, 0.0, -math.tan(elevation)))
    right = normalize3(cross3(forward, (0.0, 0.0, 1.0)))
    up = normalize3(cross3(right, forward))
    relative = (
        point[0] - camera_location[0],
        point[1] - camera_location[1],
        point[2] - camera_location[2],
    )
    depth = dot3(relative, forward)
    if depth <= 1e-4:
        return None
    focal_scale = image_size * lens / 36.0
    image_x = image_size * 0.5 + dot3(relative, right) * focal_scale / depth
    image_y = image_size * 0.5 - dot3(relative, up) * focal_scale / depth
    return image_x, image_y, depth


def projected_component_bbox(scene_dict: Dict[str, Any]) -> Optional[Dict[str, float]]:
    camera = scene_dict.get("camera_data", {})
    elevation = float(camera.get("camera_elevation", math.radians(DEFAULT_CAMERA_ELEVATION_DEG)))
    lens = float(camera.get("lens", 50.0))
    global_scale = float(camera.get("global_scale", 1.0))
    projected_points: List[Tuple[float, float, float]] = []
    for subject in subjects(scene_dict):
        for vertex in cube_world_vertices(subject, global_scale):
            projected = project_point_to_image(vertex, elevation, lens)
            if projected is not None:
                projected_points.append(projected)
    if not projected_points:
        return None
    xs = [point[0] for point in projected_points]
    ys = [point[1] for point in projected_points]
    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "center_x": (min(xs) + max(xs)) / 2.0,
        "center_y": (min(ys) + max(ys)) / 2.0,
    }


def projected_subject_bbox(scene_dict: Dict[str, Any], subject: Dict[str, Any]) -> Optional[Dict[str, float]]:
    camera = scene_dict.get("camera_data", {})
    elevation = float(camera.get("camera_elevation", math.radians(DEFAULT_CAMERA_ELEVATION_DEG)))
    lens = float(camera.get("lens", 50.0))
    global_scale = float(camera.get("global_scale", 1.0))
    projected_points: List[Tuple[float, float, float]] = []
    for vertex in cube_world_vertices(subject, global_scale):
        projected = project_point_to_image(vertex, elevation, lens)
        if projected is not None:
            projected_points.append(projected)
    if not projected_points:
        return None
    xs = [point[0] for point in projected_points]
    ys = [point[1] for point in projected_points]
    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "center_x": (min(xs) + max(xs)) / 2.0,
        "center_y": (min(ys) + max(ys)) / 2.0,
    }


def bbox_fully_visible(bbox: Optional[Dict[str, float]], image_size: float = IMAGE_SIZE_PX) -> bool:
    if bbox is None:
        return False
    return (
        bbox["min_x"] >= -EPS
        and bbox["max_x"] <= image_size + EPS
        and bbox["min_y"] >= -EPS
        and bbox["max_y"] <= image_size + EPS
    )


def screen_bottom_gap(bbox: Dict[str, float], image_size: float = IMAGE_SIZE_PX) -> float:
    return image_size - bbox["max_y"]


def screen_top_gap(bbox: Dict[str, float]) -> float:
    return bbox["min_y"]


def projected_bbox_size(bbox: Dict[str, float]) -> Tuple[float, float, float]:
    width = max(float(bbox["max_x"] - bbox["min_x"]), 0.0)
    height = max(float(bbox["max_y"] - bbox["min_y"]), 0.0)
    return width, height, width * height


def projected_bbox_intersection_area(first: Dict[str, float], second: Dict[str, float]) -> float:
    overlap_width = max(0.0, min(first["max_x"], second["max_x"]) - max(first["min_x"], second["min_x"]))
    overlap_height = max(0.0, min(first["max_y"], second["max_y"]) - max(first["min_y"], second["min_y"]))
    return overlap_width * overlap_height


def projected_bbox_overlap_ratio(first: Dict[str, float], second: Dict[str, float]) -> Tuple[float, float]:
    first_area = projected_bbox_size(first)[2]
    second_area = projected_bbox_size(second)[2]
    intersection = projected_bbox_intersection_area(first, second)
    return intersection / max(min(first_area, second_area), EPS), intersection


def component_bbox_metrics(bbox: Optional[Dict[str, float]], image_size: float = IMAGE_SIZE_PX) -> Optional[Dict[str, float]]:
    if bbox is None:
        return None
    width, height, area = projected_bbox_size(bbox)
    margins = [
        float(bbox["min_x"]),
        float(image_size - bbox["max_x"]),
        float(bbox["min_y"]),
        float(image_size - bbox["max_y"]),
    ]
    return {
        "width": width,
        "height": height,
        "area": area,
        "width_ratio": width / image_size,
        "height_ratio": height / image_size,
        "area_ratio": area / (image_size * image_size),
        "min_margin_px": min(margins),
        "left_margin_px": margins[0],
        "right_margin_px": margins[1],
        "top_margin_px": margins[2],
        "bottom_margin_px": margins[3],
        "center_offset_x_px": float(bbox["center_x"] - image_size * 0.5),
        "center_offset_y_px": float(bbox["center_y"] - image_size * 0.5),
    }


def component_compactness_target(
    manifest: Optional[Dict[str, Any]],
    predicates_payload: Optional[Dict[str, Any]],
    reference_dims_map: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    object_count = len([obj for obj in (manifest or {}).get("objects", []) if obj.get("type") != CUSTOM_ASSET_TYPE])
    predicates = (predicates_payload or {}).get("predicates", [])
    predicate_types = {str(predicate.get("type", "")) for predicate in predicates}
    has_depth = bool(predicate_types & {"in_front_of", "behind"})
    has_horizontal = bool(predicate_types & {"left_of", "right_of"})

    width_ratio = COMPONENT_COMPACT_MIN_WIDTH_RATIO
    height_ratio = COMPONENT_COMPACT_MIN_HEIGHT_RATIO
    target_kind = "default"
    if object_count <= 1:
        width_ratio = COMPONENT_COMPACT_SINGLE_WIDTH_RATIO
        height_ratio = COMPONENT_COMPACT_SINGLE_HEIGHT_RATIO
        target_kind = "single_object"
    elif object_count <= 3:
        width_ratio = COMPONENT_COMPACT_SMALL_SCENE_WIDTH_RATIO
        height_ratio = COMPONENT_COMPACT_SMALL_SCENE_HEIGHT_RATIO
        target_kind = "small_scene"
    elif object_count >= 7:
        width_ratio = COMPONENT_COMPACT_MANY_OBJECT_WIDTH_RATIO
        height_ratio = COMPONENT_COMPACT_MANY_OBJECT_HEIGHT_RATIO
        target_kind = "many_objects"

    if has_horizontal and 4 <= object_count <= 6:
        width_ratio = max(width_ratio, COMPONENT_COMPACT_HORIZONTAL_WIDTH_RATIO)
        height_ratio = max(height_ratio, COMPONENT_COMPACT_HORIZONTAL_HEIGHT_RATIO)
        target_kind = "horizontal"
    if has_depth and 4 <= object_count <= 6:
        width_ratio = max(width_ratio, COMPONENT_COMPACT_DEPTH_WIDTH_RATIO)
        height_ratio = max(height_ratio, COMPONENT_COMPACT_DEPTH_HEIGHT_RATIO)
        target_kind = "depth" if target_kind == "default" else f"{target_kind}+depth"

    if reference_dims_map:
        vertical_count = 0
        typed_count = 0
        for obj in (manifest or {}).get("objects", []):
            asset_type = str(obj.get("type", ""))
            ref_dims = reference_dims_map.get(asset_type)
            if not ref_dims:
                continue
            typed_count += 1
            width, depth, height = [float(value) for value in ref_dims[:3]]
            if height >= max(width, depth) * 1.6:
                vertical_count += 1
        if typed_count > 0 and vertical_count / typed_count >= 0.5:
            width_ratio = max(width_ratio, COMPONENT_COMPACT_VERTICAL_WIDTH_RATIO)
            height_ratio = max(height_ratio, COMPONENT_COMPACT_VERTICAL_HEIGHT_RATIO)
            target_kind = "vertical" if target_kind == "default" else f"{target_kind}+vertical"

    margin_px = COMPONENT_COMPACT_MARGIN_PX
    if object_count >= 7 or (has_horizontal and object_count >= 4):
        margin_px = COMPONENT_COMPACT_MIN_MARGIN_PX
    return {
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "margin_px": margin_px,
        "kind": target_kind,
        "object_count": object_count,
        "has_horizontal": has_horizontal,
        "has_depth": has_depth,
    }


def translate_scene_xy(scene_dict: Dict[str, Any], delta_x: float, delta_y: float) -> None:
    for subject in subjects(scene_dict):
        set_subject_field(subject, "x", subject_field(subject, "x") + delta_x)
        set_subject_field(subject, "y", subject_field(subject, "y") + delta_y)


def _predicate_screen_depth_fields(predicate: Dict[str, Any]) -> Tuple[str, str, float]:
    ptype = str(predicate.get("type", ""))
    if ptype == "in_front_of":
        axis = "front"
    elif ptype == "behind":
        axis = "back"
    else:
        return "", "", 0.0
    level = str(predicate.get("screen_depth_level") or detect_screen_depth_level(str(predicate.get("evidence", ""))))
    return axis, level, screen_depth_delta_px(level)


def _predicate_screen_lateral_fields(predicate: Dict[str, Any]) -> Tuple[str, str, float]:
    ptype = str(predicate.get("type", ""))
    if ptype == "left_of":
        axis = "left"
    elif ptype == "right_of":
        axis = "right"
    else:
        return "", "", 0.0
    level = str(predicate.get("screen_lateral_level") or detect_screen_depth_level(str(predicate.get("evidence", ""))))
    return axis, level, screen_lateral_gap_px(level)


def _set_subject_x(subject: Dict[str, Any], value: float) -> None:
    set_subject_field(subject, "x", value)


def _subject_gap_for_axis(scene_dict: Dict[str, Any], subject: Dict[str, Any], axis: str) -> Optional[float]:
    bbox = projected_subject_bbox(scene_dict, subject)
    if bbox is None:
        return None
    if axis == "front":
        return screen_bottom_gap(bbox)
    if axis == "back":
        return screen_top_gap(bbox)
    return None


def _screen_depth_predicate_satisfied(
    scene_dict: Dict[str, Any],
    subject: Dict[str, Any],
    target_gap: float,
    axis: str,
) -> bool:
    gap = _subject_gap_for_axis(scene_dict, subject, axis)
    return gap is not None and gap <= target_gap + SCREEN_DEPTH_TOLERANCE_PX


def _find_screen_depth_candidate_x(
    scene_dict: Dict[str, Any],
    subject: Dict[str, Any],
    start_x: float,
    target_gap: float,
    axis: str,
) -> Optional[float]:
    direction = 1.0 if axis == "front" else -1.0
    last_visible_x = start_x
    last_not_satisfied_x = start_x
    first_satisfied_x: Optional[float] = None
    first_invalid_x: Optional[float] = None

    step = 0.25
    total_shift = 0.0
    while total_shift < SCREEN_DEPTH_MAX_WORLD_SHIFT:
        candidate_x = start_x + direction * min(total_shift + step, SCREEN_DEPTH_MAX_WORLD_SHIFT)
        _set_subject_x(subject, candidate_x)
        bbox = projected_subject_bbox(scene_dict, subject)
        if not bbox_fully_visible(bbox):
            first_invalid_x = candidate_x
            break
        last_visible_x = candidate_x
        if _screen_depth_predicate_satisfied(scene_dict, subject, target_gap, axis):
            first_satisfied_x = candidate_x
            break
        last_not_satisfied_x = candidate_x
        total_shift = abs(candidate_x - start_x)
        step = min(step * 1.6, 1.0)

    if first_invalid_x is not None:
        visible_x = last_visible_x
        invalid_x = first_invalid_x
        for _ in range(24):
            mid_x = (visible_x + invalid_x) / 2.0
            _set_subject_x(subject, mid_x)
            bbox = projected_subject_bbox(scene_dict, subject)
            if bbox_fully_visible(bbox):
                visible_x = mid_x
            else:
                invalid_x = mid_x
        _set_subject_x(subject, visible_x)
        if _screen_depth_predicate_satisfied(scene_dict, subject, target_gap, axis):
            first_satisfied_x = visible_x
        else:
            _set_subject_x(subject, start_x)
            if abs(visible_x - start_x) > EPS:
                return visible_x
            return None

    if first_satisfied_x is not None:
        # Binary search the smallest visible x movement that satisfies the
        # target, so extreme prompt values do not overshoot unnecessarily.
        low_x = last_not_satisfied_x
        high_x = first_satisfied_x
        for _ in range(24):
            mid_x = (low_x + high_x) / 2.0
            _set_subject_x(subject, mid_x)
            bbox = projected_subject_bbox(scene_dict, subject)
            if bbox_fully_visible(bbox) and _screen_depth_predicate_satisfied(scene_dict, subject, target_gap, axis):
                high_x = mid_x
            else:
                low_x = mid_x
        _set_subject_x(subject, start_x)
        return high_x

    _set_subject_x(subject, start_x)
    if abs(last_visible_x - start_x) > EPS:
        return last_visible_x
    return None


def _screen_lateral_predicate_satisfied(
    scene_dict: Dict[str, Any],
    subject: Dict[str, Any],
    reference: Dict[str, Any],
    axis: str,
    gap_px: float,
) -> bool:
    subject_bbox = projected_subject_bbox(scene_dict, subject)
    reference_bbox = projected_subject_bbox(scene_dict, reference)
    if subject_bbox is None or reference_bbox is None:
        return False
    if axis == "left":
        return subject_bbox["max_x"] <= reference_bbox["min_x"] - gap_px + SCREEN_DEPTH_TOLERANCE_PX
    if axis == "right":
        return subject_bbox["min_x"] >= reference_bbox["max_x"] + gap_px - SCREEN_DEPTH_TOLERANCE_PX
    return True


def _find_screen_lateral_candidate_y(
    scene_dict: Dict[str, Any],
    subject: Dict[str, Any],
    reference: Dict[str, Any],
    start_y: float,
    axis: str,
    gap_px: float,
    y_lower: Optional[float] = None,
    y_upper: Optional[float] = None,
) -> Optional[float]:
    direction = -1.0 if axis == "left" else 1.0
    last_visible_y = start_y
    last_not_satisfied_y = start_y
    first_satisfied_y: Optional[float] = None
    first_invalid_y: Optional[float] = None
    step = 0.15
    total_shift = 0.0
    while total_shift < SCREEN_LATERAL_MAX_WORLD_SHIFT:
        raw_y = start_y + direction * min(total_shift + step, SCREEN_LATERAL_MAX_WORLD_SHIFT)
        candidate_y = _clamp_optional(raw_y, y_lower, y_upper)
        if abs(candidate_y - start_y) <= EPS:
            break
        set_subject_field(subject, "y", candidate_y)
        bbox = projected_subject_bbox(scene_dict, subject)
        if not bbox_fully_visible(bbox):
            first_invalid_y = candidate_y
            break
        last_visible_y = candidate_y
        if _screen_lateral_predicate_satisfied(scene_dict, subject, reference, axis, gap_px):
            first_satisfied_y = candidate_y
            break
        last_not_satisfied_y = candidate_y
        total_shift = abs(candidate_y - start_y)
        step = min(step * 1.5, 0.75)

    if first_invalid_y is not None:
        visible_y = last_visible_y
        invalid_y = first_invalid_y
        for _ in range(24):
            mid_y = (visible_y + invalid_y) / 2.0
            set_subject_field(subject, "y", mid_y)
            bbox = projected_subject_bbox(scene_dict, subject)
            if bbox_fully_visible(bbox):
                visible_y = mid_y
            else:
                invalid_y = mid_y
        set_subject_field(subject, "y", visible_y)
        if _screen_lateral_predicate_satisfied(scene_dict, subject, reference, axis, gap_px):
            first_satisfied_y = visible_y
        else:
            set_subject_field(subject, "y", start_y)
            if abs(visible_y - start_y) > EPS:
                return visible_y
            return None

    if first_satisfied_y is not None:
        low_y = last_not_satisfied_y
        high_y = first_satisfied_y
        for _ in range(24):
            mid_y = (low_y + high_y) / 2.0
            set_subject_field(subject, "y", mid_y)
            if bbox_fully_visible(projected_subject_bbox(scene_dict, subject)) and _screen_lateral_predicate_satisfied(
                scene_dict,
                subject,
                reference,
                axis,
                gap_px,
            ):
                high_y = mid_y
            else:
                low_y = mid_y
        set_subject_field(subject, "y", start_y)
        return high_y

    set_subject_field(subject, "y", start_y)
    if abs(last_visible_y - start_y) > EPS:
        return last_visible_y
    return None


def enforce_screen_lateral_gaps(
    scene_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    predicates_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    id_to_name = manifest_id_to_name(manifest)
    by_name = subject_by_name(scene_dict)
    for predicate in predicates_payload.get("predicates", []):
        axis, level, gap_px = _predicate_screen_lateral_fields(predicate)
        if not axis:
            continue
        subject_name = id_to_name.get(predicate.get("subject", ""))
        object_name = id_to_name.get(predicate.get("object", ""))
        first = by_name.get(subject_name)
        second = by_name.get(object_name)
        if first is None or second is None:
            continue
        if _screen_lateral_predicate_satisfied(scene_dict, first, second, axis, gap_px):
            continue
        bounds = _predicate_bounds_for_subject(subject_name, by_name, manifest, predicates_payload)
        old_y = subject_field(first, "y")
        candidate_y = _find_screen_lateral_candidate_y(
            scene_dict,
            first,
            second,
            old_y,
            axis,
            gap_px,
            y_lower=bounds["y_lower"],
            y_upper=bounds["y_upper"],
        )
        if candidate_y is not None and abs(candidate_y - old_y) > EPS:
            set_field_action(first, "y", candidate_y, f"screen_lateral:{axis}:{level}", actions)
            actions[-1]["screen_lateral_gap_px"] = round(gap_px, 6)
    return actions


def _predicate_pair_axis_map(
    manifest: Dict[str, Any],
    predicates_payload: Optional[Dict[str, Any]],
) -> Dict[Tuple[str, str], str]:
    id_to_name = manifest_id_to_name(manifest)
    pair_axis: Dict[Tuple[str, str], str] = {}
    for predicate in (predicates_payload or {}).get("predicates", []):
        subject_name = id_to_name.get(predicate.get("subject", ""))
        object_name = id_to_name.get(predicate.get("object", ""))
        if not subject_name or not object_name:
            continue
        ptype = str(predicate.get("type", ""))
        if ptype == "left_of":
            pair_axis[(subject_name, object_name)] = "left"
        elif ptype == "right_of":
            pair_axis[(subject_name, object_name)] = "right"
    return pair_axis


def enforce_pairwise_screen_occlusion(
    scene_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    predicates_payload: Optional[Dict[str, Any]],
    max_ratio: float = PAIRWISE_OCCLUSION_MAX_RATIO,
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    pair_axis = _predicate_pair_axis_map(manifest, predicates_payload)
    by_name = subject_by_name(scene_dict)
    for _ in range(6):
        changed = False
        current_subjects = subjects(scene_dict)
        bboxes = {str(subject.get("name", "")): projected_subject_bbox(scene_dict, subject) for subject in current_subjects}
        worst_pair: Optional[Tuple[float, str, str]] = None
        for idx, first in enumerate(current_subjects):
            first_name = str(first.get("name", ""))
            first_bbox = bboxes.get(first_name)
            if first_bbox is None:
                continue
            for second in current_subjects[idx + 1 :]:
                second_name = str(second.get("name", ""))
                second_bbox = bboxes.get(second_name)
                if second_bbox is None:
                    continue
                ratio, _ = projected_bbox_overlap_ratio(first_bbox, second_bbox)
                if ratio <= max_ratio + 1e-4:
                    continue
                if worst_pair is None or ratio > worst_pair[0]:
                    worst_pair = (ratio, first_name, second_name)
        if worst_pair is None:
            break
        ratio, first_name, second_name = worst_pair
        first = by_name.get(first_name)
        second = by_name.get(second_name)
        if first is None or second is None:
            break

        direct_axis = pair_axis.get((first_name, second_name))
        reverse_axis = pair_axis.get((second_name, first_name))
        preferred_axis = direct_axis
        if preferred_axis is None and reverse_axis is None:
            first_y = subject_field(first, "y")
            second_y = subject_field(second, "y")
            preferred_axis = "left" if first_y <= second_y else "right"
        mover = first
        reference = second
        mover_name = first_name
        axis = preferred_axis
        if direct_axis is None and reverse_axis is not None:
            mover = second
            reference = first
            mover_name = second_name
            axis = reverse_axis

        bounds = _predicate_bounds_for_subject(mover_name, by_name, manifest, predicates_payload)
        old_y = subject_field(mover, "y")
        candidate_y = _find_screen_lateral_candidate_y(
            scene_dict,
            mover,
            reference,
            old_y,
            axis,
            screen_lateral_gap_px("default"),
            y_lower=bounds["y_lower"],
            y_upper=bounds["y_upper"],
        )
        if candidate_y is None or abs(candidate_y - old_y) <= EPS:
            break
        set_field_action(mover, "y", candidate_y, "pairwise_occlusion:separate_y", actions)
        actions[-1]["pairwise_occlusion_ratio_before"] = round(ratio, 6)
        actions[-1]["pairwise_occlusion_max_ratio"] = round(max_ratio, 6)
        actions[-1]["pairwise_occlusion_pair"] = [first_name, second_name]
        changed = True
        if not changed:
            break
    return actions


def _relax_camera_for_screen_depth(scene_dict: Dict[str, Any], actions: List[Dict[str, Any]], reason: str) -> bool:
    action_count_before = len(actions)
    _decrease_camera_for_visibility(scene_dict, reason, actions)
    return len(actions) > action_count_before


def _target_gap_for_screen_depth(scene_dict: Dict[str, Any], reference: Dict[str, Any], axis: str, delta_px: float) -> Optional[float]:
    reference_bbox = projected_subject_bbox(scene_dict, reference)
    if reference_bbox is None:
        return None
    reference_gap = screen_bottom_gap(reference_bbox) if axis == "front" else screen_top_gap(reference_bbox)
    return max(SCREEN_DEPTH_MIN_VISIBLE_GAP_PX, reference_gap - delta_px)


def _screen_depth_constraints_for_subject(
    subject_name: str,
    by_name: Dict[str, Dict[str, Any]],
    manifest: Dict[str, Any],
    predicates_payload: Optional[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any], float]]:
    if not predicates_payload:
        return []
    id_to_name = manifest_id_to_name(manifest)
    constraints: List[Tuple[str, Dict[str, Any], float]] = []
    for predicate in predicates_payload.get("predicates", []):
        if id_to_name.get(predicate.get("subject", "")) != subject_name:
            continue
        axis, _, delta_px = _predicate_screen_depth_fields(predicate)
        if not axis:
            continue
        reference_name = id_to_name.get(predicate.get("object", ""))
        reference = by_name.get(reference_name)
        if reference is None:
            continue
        constraints.append((axis, reference, delta_px))
    return constraints


def _screen_depth_allows_x(
    scene_dict: Dict[str, Any],
    subject: Dict[str, Any],
    constraints: Sequence[Tuple[str, Dict[str, Any], float]],
    candidate_x: float,
) -> bool:
    if not constraints:
        return True
    old_x = subject_field(subject, "x")
    set_subject_field(subject, "x", candidate_x)
    try:
        subject_bbox = projected_subject_bbox(scene_dict, subject)
        if subject_bbox is None:
            return False
        for axis, reference, delta_px in constraints:
            target_gap = _target_gap_for_screen_depth(scene_dict, reference, axis, delta_px)
            if target_gap is None:
                return False
            current_gap = screen_bottom_gap(subject_bbox) if axis == "front" else screen_top_gap(subject_bbox)
            if current_gap > target_gap + SCREEN_DEPTH_TOLERANCE_PX:
                return False
        return True
    finally:
        set_subject_field(subject, "x", old_x)


def _screen_size_records(
    scene_dict: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for subject in subjects(scene_dict):
        asset_type = str(subject.get("type", ""))
        reference_dims = reference_dims_map.get(asset_type)
        if not reference_dims:
            continue
        bbox = projected_subject_bbox(scene_dict, subject)
        if bbox is None:
            continue
        bbox_width, bbox_height, bbox_area = projected_bbox_size(bbox)
        ref_width = max(float(reference_dims[0]), EPS)
        ref_height = max(float(reference_dims[2]), EPS)
        ref_area = max(ref_width * ref_height, EPS)
        height_density = bbox_height / ref_height
        area_density = math.sqrt(max(bbox_area, EPS) / ref_area)
        # Height is the primary signal; sqrt(area) is a secondary stabilizer for
        # very wide or very narrow cuboids.
        density = 0.75 * height_density + 0.25 * area_density
        records.append(
            {
                "subject": subject,
                "name": str(subject.get("name", "")),
                "bbox": bbox,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "bbox_area": bbox_area,
                "height_density": height_density,
                "area_density": area_density,
                "density": density,
            }
        )
    return records


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _screen_size_target_density(records: Sequence[Dict[str, Any]]) -> Optional[float]:
    return _median([record["density"] for record in records if record.get("density", 0.0) > EPS])


def _screen_size_ratio(record: Dict[str, Any], target_density: float) -> float:
    return float(record["density"]) / max(target_density, EPS)


def _screen_size_range_for_axis_level(axis: str, level: str) -> Tuple[float, float]:
    normalized_level = level if level in SCREEN_DEPTH_DELTA_PX_BY_LEVEL else "default"
    if axis == "front":
        return SCREEN_SIZE_FRONT_RATIO_RANGE_BY_LEVEL.get(
            normalized_level,
            SCREEN_SIZE_FRONT_RATIO_RANGE_BY_LEVEL["default"],
        )
    if axis == "back":
        return SCREEN_SIZE_BACK_RATIO_RANGE_BY_LEVEL.get(
            normalized_level,
            SCREEN_SIZE_BACK_RATIO_RANGE_BY_LEVEL["default"],
        )
    return SCREEN_SIZE_RATIO_RANGE


def _screen_size_ranges_by_subject(
    manifest: Optional[Dict[str, Any]],
    predicates_payload: Optional[Dict[str, Any]],
) -> Dict[str, Tuple[float, float]]:
    ranges: Dict[str, Tuple[float, float]] = {}
    if manifest is None or not predicates_payload:
        return ranges
    id_to_name = manifest_id_to_name(manifest)
    for predicate in predicates_payload.get("predicates", []):
        axis, level, _ = _predicate_screen_depth_fields(predicate)
        if not axis:
            continue
        subject_name = id_to_name.get(predicate.get("subject", ""))
        if not subject_name:
            continue
        candidate = _screen_size_range_for_axis_level(axis, level)
        previous = ranges.get(subject_name)
        if previous is None:
            ranges[subject_name] = candidate
        else:
            ranges[subject_name] = (min(previous[0], candidate[0]), max(previous[1], candidate[1]))
    return ranges


def _screen_size_ratio_range_for_record(
    record: Dict[str, Any],
    ranges_by_subject: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    if not ranges_by_subject:
        return SCREEN_SIZE_RATIO_RANGE
    return ranges_by_subject.get(str(record.get("name", "")), SCREEN_SIZE_RATIO_RANGE)


def _screen_size_ratio_in_range(ratio: float, ratio_range: Tuple[float, float]) -> bool:
    return ratio_range[0] - SCREEN_SIZE_RATIO_TOLERANCE <= ratio <= ratio_range[1] + SCREEN_SIZE_RATIO_TOLERANCE


def _screen_size_expand_reference_subjects(
    scene_dict: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    ranges_by_subject: Dict[str, Tuple[float, float]],
    manifest: Optional[Dict[str, Any]],
    predicates_payload: Optional[Dict[str, Any]],
    actions: List[Dict[str, Any]],
) -> bool:
    records = _screen_size_records(scene_dict, reference_dims_map)
    target_density = _screen_size_target_density(records)
    if target_density is None:
        return False
    if not any(
        _screen_size_ratio(record, target_density) > _screen_size_ratio_range_for_record(record, ranges_by_subject)[1] + EPS
        for record in records
    ):
        return False

    changed = False
    for record in records:
        subject = record["subject"]
        name = str(record.get("name", ""))
        if name in ranges_by_subject:
            continue
        if manifest is not None and predicates_payload is not None:
            relation_bounds = _predicate_bounds_for_subject(name, subject_by_name(scene_dict), manifest, predicates_payload)
            if any(value is not None for value in relation_bounds.values()):
                continue
        ratio = _screen_size_ratio(record, target_density)
        ratio_range = _screen_size_ratio_range_for_record(record, ranges_by_subject)
        if ratio > ratio_range[1] + EPS:
            continue
        old_dims = dims(subject)
        target_dims = clamp_dims_to_reference_safety(
            [value * SCREEN_SIZE_MAX_DIM_SCALE for value in old_dims],
            {"type": subject.get("type"), "mention": subject.get("name")},
            reference_dims_map,
        )
        if all(abs(old_dims[idx] - target_dims[idx]) <= EPS for idx in range(3)):
            continue
        set_dims_action(subject, target_dims, "screen_size:expand_reference_subject", actions)
        old_marker = bool(subject.get("_screen_size_dim_compensated", False))
        subject["_screen_size_dim_compensated"] = True
        if not old_marker:
            actions.append(
                action_record(
                    "set_param",
                    str(subject.get("name", "")),
                    "_screen_size_dim_compensated",
                    old_marker,
                    True,
                    "screen_size:reference_expansion_marker",
                )
            )
        changed = True
    return changed


def _screen_size_find_x(
    scene_dict: Dict[str, Any],
    subject: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    target_density: float,
    target_ratio: float,
    direction: float,
    x_lower: Optional[float] = None,
    x_upper: Optional[float] = None,
    screen_depth_constraints: Optional[Sequence[Tuple[str, Dict[str, Any], float]]] = None,
) -> Optional[float]:
    old_x = subject_field(subject, "x")
    best_x: Optional[float] = None
    best_error = float("inf")
    step_count = 48
    for idx in range(1, step_count + 1):
        raw_candidate_x = old_x + direction * SCREEN_SIZE_MAX_WORLD_SHIFT * idx / step_count
        candidate_x = _clamp_optional(raw_candidate_x, x_lower, x_upper)
        if abs(candidate_x - old_x) <= EPS:
            continue
        set_subject_field(subject, "x", candidate_x)
        bbox = projected_subject_bbox(scene_dict, subject)
        if not bbox_fully_visible(bbox):
            continue
        if screen_depth_constraints and not _screen_depth_allows_x(scene_dict, subject, screen_depth_constraints, candidate_x):
            continue
        records = _screen_size_records(scene_dict, reference_dims_map)
        candidate_record = next((record for record in records if record["subject"] is subject), None)
        if candidate_record is None:
            continue
        ratio = _screen_size_ratio(candidate_record, target_density)
        error = abs(ratio - target_ratio)
        if error < best_error:
            best_error = error
            best_x = candidate_x
        if direction > 0.0 and ratio >= target_ratio - 0.01:
            break
        if direction < 0.0 and ratio <= target_ratio + 0.01:
            break
    set_subject_field(subject, "x", old_x)
    return best_x


def _scale_subject_dims_uniform(
    subject: Dict[str, Any],
    scale: float,
    reason: str,
    actions: List[Dict[str, Any]],
    reference_dims_map: Optional[Dict[str, List[float]]] = None,
) -> None:
    bounded_scale = clamp(float(scale), 1.0 / SCREEN_SIZE_MAX_DIM_SCALE, SCREEN_SIZE_MAX_DIM_SCALE)
    old_dims = dims(subject)
    new_dims = [max(value * bounded_scale, MIN_VISUAL_DIMS[idx]) for idx, value in enumerate(old_dims)]
    if reference_dims_map is not None:
        new_dims = clamp_dims_to_reference_safety(
            new_dims,
            {"type": subject.get("type"), "mention": subject.get("name")},
            reference_dims_map,
        )
    set_dims_action(subject, new_dims, reason, actions)
    if any(abs(old_dims[idx] - new_dims[idx]) > EPS for idx in range(3)):
        old_marker = bool(subject.get("_screen_size_dim_compensated", False))
        subject["_screen_size_dim_compensated"] = True
        if not old_marker:
            actions.append(
                action_record(
                    "set_param",
                    str(subject.get("name", "")),
                    "_screen_size_dim_compensated",
                    old_marker,
                    True,
                    "screen_size:dim_compensation_marker",
                )
            )


def _screen_size_violation_score(
    records: Sequence[Dict[str, Any]],
    target_density: Optional[float],
    ranges_by_subject: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    if target_density is None:
        return float("inf")
    score = 0.0
    for record in records:
        ratio = _screen_size_ratio(record, target_density)
        ratio_range = _screen_size_ratio_range_for_record(record, ranges_by_subject)
        if ratio > ratio_range[1]:
            score += ratio - ratio_range[1]
        elif ratio < ratio_range[0]:
            score += ratio_range[0] - ratio
    return score


def _screen_depth_violation_score(
    scene_dict: Dict[str, Any],
    manifest: Optional[Dict[str, Any]],
    predicates_payload: Optional[Dict[str, Any]],
) -> float:
    if manifest is None or not predicates_payload:
        return 0.0
    id_to_name = manifest_id_to_name(manifest)
    by_name = subject_by_name(scene_dict)
    score = 0.0
    for predicate in predicates_payload.get("predicates", []):
        axis, _, delta_px = _predicate_screen_depth_fields(predicate)
        if not axis:
            continue
        first = by_name.get(id_to_name.get(predicate.get("subject", ""), ""))
        second = by_name.get(id_to_name.get(predicate.get("object", ""), ""))
        if first is None or second is None:
            continue
        first_bbox = projected_subject_bbox(scene_dict, first)
        target_gap = _target_gap_for_screen_depth(scene_dict, second, axis, delta_px)
        if first_bbox is None or target_gap is None:
            score += IMAGE_SIZE_PX
            continue
        actual_gap = screen_bottom_gap(first_bbox) if axis == "front" else screen_top_gap(first_bbox)
        score += max(actual_gap - target_gap, 0.0)
    return score


def _component_compactness_score(metrics: Optional[Dict[str, float]], target: Dict[str, Any]) -> float:
    if metrics is None:
        return float("inf")
    width_error = max(float(target["width_ratio"]) - float(metrics["width_ratio"]), 0.0)
    height_error = max(float(target["height_ratio"]) - float(metrics["height_ratio"]), 0.0)
    return width_error + height_error


def _component_compactness_hard_constraints_valid(
    scene_dict: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    manifest: Dict[str, Any],
    predicates_payload: Dict[str, Any],
    target: Dict[str, Any],
) -> bool:
    metrics = component_bbox_metrics(projected_component_bbox(scene_dict))
    if metrics is None:
        return False
    if not all(bbox_fully_visible(projected_subject_bbox(scene_dict, subject)) for subject in subjects(scene_dict)):
        return False
    if validate_predicates(scene_dict, manifest, predicates_payload):
        return False
    if validate_screen_lateral_gaps(scene_dict, manifest, predicates_payload):
        return False
    if validate_pairwise_screen_occlusion(scene_dict):
        return False
    if validate_camera(scene_dict):
        return False
    return True


def _component_compactness_candidate_valid(
    scene_dict: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    manifest: Dict[str, Any],
    predicates_payload: Dict[str, Any],
    target: Dict[str, Any],
) -> bool:
    if not _component_compactness_hard_constraints_valid(scene_dict, reference_dims_map, manifest, predicates_payload, target):
        return False
    metrics = component_bbox_metrics(projected_component_bbox(scene_dict))
    if metrics is None or metrics["min_margin_px"] < float(target["margin_px"]) - EPS:
        return False
    return validate_screen_size_reasonableness(scene_dict, reference_dims_map, manifest, predicates_payload) == []


def _repair_after_component_camera_candidate(
    scene_dict: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    manifest: Dict[str, Any],
    predicates_payload: Dict[str, Any],
    target: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    actions: List[Dict[str, Any]] = []
    for _ in range(COMPONENT_COMPACT_POST_REPAIR_PASSES):
        before_actions = len(actions)
        actions.extend(enforce_screen_size_reasonableness(scene_dict, reference_dims_map, manifest, predicates_payload))
        actions.extend(enforce_screen_depth_levels(scene_dict, manifest, predicates_payload))
        actions.extend(enforce_screen_lateral_gaps(scene_dict, manifest, predicates_payload))
        actions.extend(enforce_pairwise_screen_occlusion(scene_dict, manifest, predicates_payload))
        actions.extend(apply_camera_ranges(scene_dict))
        if _component_compactness_candidate_valid(scene_dict, reference_dims_map, manifest, predicates_payload, target):
            return actions
        if len(actions) == before_actions:
            break
    return actions if _component_compactness_candidate_valid(scene_dict, reference_dims_map, manifest, predicates_payload, target) else None


def enforce_component_compactness(
    scene_dict: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    manifest: Dict[str, Any],
    predicates_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    camera = scene_dict.setdefault("camera_data", {})
    target = component_compactness_target(manifest, predicates_payload, reference_dims_map)
    old_elevation = float(camera.get("camera_elevation", math.radians(DEFAULT_CAMERA_ELEVATION_DEG)))
    old_lens = float(camera.get("lens", 50.0))
    old_scale = float(camera.get("global_scale", 1.0))
    elevation_locked = False
    old_metrics = component_bbox_metrics(projected_component_bbox(scene_dict))
    old_score = _component_compactness_score(old_metrics, target)
    if old_metrics is None:
        return actions

    original_scene = clone_scene(scene_dict)
    lower_elevation = math.radians(DEFAULT_ELEVATION_DEG_RANGE[0])
    upper_elevation = math.radians(DEFAULT_ELEVATION_DEG_RANGE[1])
    elevation_step = math.radians(COMPONENT_COMPACT_ELEVATION_STEP_DEG)
    best_lens = old_lens
    best_scale = old_scale
    best_elevation = old_elevation
    best_score = old_score
    best_metrics = old_metrics
    best_scene: Optional[Dict[str, Any]] = None
    best_post_actions: List[Dict[str, Any]] = []

    lens_values = [old_lens]
    current_lens = old_lens
    for _ in range(40):
        current_lens = clamp(current_lens * COMPONENT_COMPACT_CAMERA_STEP, *DEFAULT_LENS_RANGE)
        if current_lens <= lens_values[-1] + EPS:
            break
        lens_values.append(current_lens)
        if abs(current_lens - DEFAULT_LENS_RANGE[1]) <= EPS:
            break

    scale_values = [old_scale]
    current_scale = old_scale
    for _ in range(24):
        current_scale = clamp(current_scale * COMPONENT_COMPACT_CAMERA_STEP, *DEFAULT_GLOBAL_SCALE_RANGE)
        if current_scale <= scale_values[-1] + EPS:
            break
        scale_values.append(current_scale)
        if abs(current_scale - DEFAULT_GLOBAL_SCALE_RANGE[1]) <= EPS:
            break

    elevation_values = {clamp(old_elevation, lower_elevation, upper_elevation)}
    if not elevation_locked:
        for direction in (-1.0, 1.0):
            current_elevation = old_elevation
            for _ in range(24):
                current_elevation = clamp(current_elevation + direction * elevation_step, lower_elevation, upper_elevation)
                if current_elevation in elevation_values:
                    break
                elevation_values.add(current_elevation)
                if abs(current_elevation - lower_elevation) <= EPS or abs(current_elevation - upper_elevation) <= EPS:
                    break

    def candidate_better(
        candidate_score: float,
        candidate_metrics: Dict[str, float],
        candidate_elevation: float,
        candidate_lens: float,
        candidate_scale: float,
    ) -> bool:
        nonlocal best_score, best_metrics, best_elevation, best_lens, best_scale
        if candidate_score < best_score - 1e-4:
            return True
        if abs(candidate_score - best_score) > 1e-4:
            return False
        if candidate_score <= EPS:
            candidate_area = float(candidate_metrics["area_ratio"])
            best_area = float(best_metrics["area_ratio"])
            if candidate_area > best_area + 1e-4:
                return True
        candidate_change = (
            abs(candidate_lens - old_lens) / max(old_lens, EPS)
            + 0.5 * abs(candidate_scale - old_scale) / max(old_scale, EPS)
            + 0.2 * abs(candidate_elevation - old_elevation) / max(math.radians(1.0), EPS)
        )
        best_change = (
            abs(best_lens - old_lens) / max(old_lens, EPS)
            + 0.5 * abs(best_scale - old_scale) / max(old_scale, EPS)
            + 0.2 * abs(best_elevation - old_elevation) / max(math.radians(1.0), EPS)
        )
        return candidate_change < best_change - 1e-4

    candidates: List[Tuple[float, float, float, float]] = []
    for candidate_elevation in sorted(elevation_values, key=lambda value: abs(value - old_elevation)):
        for candidate_scale in scale_values:
            for candidate_lens in lens_values:
                if abs(candidate_lens - old_lens) <= EPS and abs(candidate_scale - old_scale) <= EPS and abs(candidate_elevation - old_elevation) <= EPS:
                    continue
                estimated_scene = clone_scene(original_scene)
                estimated_camera = estimated_scene.setdefault("camera_data", {})
                estimated_camera["camera_elevation"] = round(candidate_elevation, 6)
                estimated_camera["lens"] = round(candidate_lens, 6)
                estimated_camera["global_scale"] = round(candidate_scale, 6)
                estimated_metrics = component_bbox_metrics(projected_component_bbox(estimated_scene))
                estimated_score = _component_compactness_score(estimated_metrics, target)
                candidates.append((estimated_score, candidate_elevation, candidate_lens, candidate_scale))

    candidates = sorted(
        candidates,
        key=lambda item: (item[0], abs(item[1] - old_elevation), abs(item[2] - old_lens), abs(item[3] - old_scale)),
    )
    checked_hard_candidates = 0
    for _, candidate_elevation, candidate_lens, candidate_scale in candidates:
        trial_scene = clone_scene(original_scene)
        trial_camera = trial_scene.setdefault("camera_data", {})
        trial_camera["camera_elevation"] = round(candidate_elevation, 6)
        trial_camera["lens"] = round(candidate_lens, 6)
        trial_camera["global_scale"] = round(candidate_scale, 6)
        if not _component_compactness_hard_constraints_valid(trial_scene, reference_dims_map, manifest, predicates_payload, target):
            continue
        checked_hard_candidates += 1
        if checked_hard_candidates > COMPONENT_COMPACT_MAX_POST_REPAIR_CANDIDATES:
            break
        post_actions = _repair_after_component_camera_candidate(
            trial_scene,
            reference_dims_map,
            manifest,
            predicates_payload,
            target,
        )
        if post_actions is None:
            continue
        candidate_metrics = component_bbox_metrics(projected_component_bbox(trial_scene))
        if candidate_metrics is None:
            continue
        candidate_score = _component_compactness_score(candidate_metrics, target)
        if candidate_better(candidate_score, candidate_metrics, candidate_elevation, candidate_lens, candidate_scale):
            best_elevation = candidate_elevation
            best_lens = candidate_lens
            best_scale = candidate_scale
            best_score = candidate_score
            best_metrics = candidate_metrics
            best_scene = trial_scene
            best_post_actions = post_actions

    scene_dict.clear()
    scene_dict.update(original_scene)
    if best_scene is None:
        return actions

    scene_dict.clear()
    scene_dict.update(best_scene)
    if abs(best_elevation - old_elevation) > EPS:
        actions.append(
            action_record(
                "set_param",
                "camera",
                "camera_data.camera_elevation",
                old_elevation,
                best_elevation,
                "component_compactness:adjust_camera_elevation",
                component_compactness_target=target,
                component_compactness_score_before=round(old_score, 6),
                component_compactness_score_after=round(best_score, 6),
            )
        )
    if abs(best_lens - old_lens) > EPS:
        actions.append(
            action_record(
                "set_param",
                "camera",
                "camera_data.lens",
                old_lens,
                best_lens,
                "component_compactness:increase_lens",
                component_compactness_target=target,
                component_compactness_score_before=round(old_score, 6),
                component_compactness_score_after=round(best_score, 6),
            )
        )
    if abs(best_scale - old_scale) > EPS:
        actions.append(
            action_record(
                "set_param",
                "camera",
                "camera_data.global_scale",
                old_scale,
                best_scale,
                "component_compactness:increase_global_scale",
                component_compactness_target=target,
                component_compactness_score_before=round(old_score, 6),
                component_compactness_score_after=round(best_score, 6),
            )
        )
    if best_post_actions:
        actions.extend(best_post_actions)
    return actions


def _try_camera_for_screen_size_reasonableness(
    scene_dict: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    manifest: Optional[Dict[str, Any]],
    predicates_payload: Optional[Dict[str, Any]],
    actions: List[Dict[str, Any]],
) -> bool:
    camera = scene_dict.setdefault("camera_data", {})
    old_lens = float(camera.get("lens", 50.0))
    old_scale = float(camera.get("global_scale", 1.0))
    ranges_by_subject = _screen_size_ranges_by_subject(manifest, predicates_payload)
    old_records = _screen_size_records(scene_dict, reference_dims_map)
    old_target = _screen_size_target_density(old_records)
    old_score = _screen_size_violation_score(old_records, old_target, ranges_by_subject)
    old_depth_score = _screen_depth_violation_score(scene_dict, manifest, predicates_payload)
    if old_score <= EPS:
        return False

    candidates: List[Tuple[float, float]] = []
    for lens_factor in (0.8, 0.9, 1.1, 1.2):
        candidates.append((clamp(old_lens * lens_factor, *DEFAULT_LENS_RANGE), old_scale))
    for scale_factor in (0.85, 0.92, 1.08, 1.15):
        candidates.append((old_lens, clamp(old_scale * scale_factor, *DEFAULT_GLOBAL_SCALE_RANGE)))
    for lens_factor, scale_factor in ((0.9, 0.92), (1.1, 1.08), (0.8, 0.85), (1.2, 1.15)):
        candidates.append(
            (
                clamp(old_lens * lens_factor, *DEFAULT_LENS_RANGE),
                clamp(old_scale * scale_factor, *DEFAULT_GLOBAL_SCALE_RANGE),
            )
        )

    best_lens = old_lens
    best_scale = old_scale
    best_score = old_score
    for candidate_lens, candidate_scale in candidates:
        if abs(candidate_lens - old_lens) <= EPS and abs(candidate_scale - old_scale) <= EPS:
            continue
        camera["lens"] = round(candidate_lens, 6)
        camera["global_scale"] = round(candidate_scale, 6)
        if not all(bbox_fully_visible(projected_subject_bbox(scene_dict, subject)) for subject in subjects(scene_dict)):
            continue
        records = _screen_size_records(scene_dict, reference_dims_map)
        target = _screen_size_target_density(records)
        score = _screen_size_violation_score(records, target, ranges_by_subject)
        depth_score = _screen_depth_violation_score(scene_dict, manifest, predicates_payload)
        if depth_score > old_depth_score + 1.0:
            continue
        if score < best_score - 0.01:
            best_score = score
            best_lens = candidate_lens
            best_scale = candidate_scale

    camera["lens"] = round(old_lens, 6)
    camera["global_scale"] = round(old_scale, 6)
    if best_score >= old_score - 0.01:
        return False
    if abs(best_lens - old_lens) > EPS:
        camera["lens"] = round(best_lens, 6)
        actions.append(action_record("set_param", "camera", "camera_data.lens", old_lens, best_lens, "screen_size:camera_lens"))
    if abs(best_scale - old_scale) > EPS:
        camera["global_scale"] = round(best_scale, 6)
        actions.append(action_record("set_param", "camera", "camera_data.global_scale", old_scale, best_scale, "screen_size:camera_global_scale"))
    return True


def _restore_screen_depth_after_screen_size(
    scene_dict: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    manifest: Dict[str, Any],
    predicates_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    id_to_name = manifest_id_to_name(manifest)
    by_name = subject_by_name(scene_dict)
    ranges_by_subject = _screen_size_ranges_by_subject(manifest, predicates_payload)
    for _ in range(4):
        changed = False
        for predicate in predicates_payload.get("predicates", []):
            axis, level, delta_px = _predicate_screen_depth_fields(predicate)
            if not axis:
                continue
            subject_name = id_to_name.get(predicate.get("subject", ""))
            object_name = id_to_name.get(predicate.get("object", ""))
            first = by_name.get(subject_name)
            second = by_name.get(object_name)
            if first is None or second is None:
                continue
            first_bbox = projected_subject_bbox(scene_dict, first)
            target_gap = _target_gap_for_screen_depth(scene_dict, second, axis, delta_px)
            if first_bbox is None or target_gap is None:
                continue
            current_gap = screen_bottom_gap(first_bbox) if axis == "front" else screen_top_gap(first_bbox)
            if current_gap <= target_gap + SCREEN_DEPTH_TOLERANCE_PX:
                continue
            old_x = subject_field(first, "x")
            candidate_x = _find_screen_depth_candidate_x(scene_dict, first, old_x, target_gap, axis)
            if candidate_x is None or abs(candidate_x - old_x) <= EPS:
                continue
            set_field_action(first, "x", candidate_x, f"screen_size:restore_screen_depth:{axis}:{level}", actions)
            records = _screen_size_records(scene_dict, reference_dims_map)
            target_density = _screen_size_target_density(records)
            record = next((item for item in records if item["subject"] is first), None)
            if record is not None and target_density is not None:
                ratio = _screen_size_ratio(record, target_density)
                ratio_range = _screen_size_ratio_range_for_record(record, ranges_by_subject)
                if ratio > ratio_range[1]:
                    _scale_subject_dims_uniform(
                        first,
                        ratio_range[1] / max(ratio, EPS),
                        "screen_size:restore_depth_dim_compensation",
                        actions,
                        reference_dims_map,
                    )
                elif ratio < ratio_range[0]:
                    _scale_subject_dims_uniform(
                        first,
                        ratio_range[0] / max(ratio, EPS),
                        "screen_size:restore_depth_dim_compensation",
                        actions,
                        reference_dims_map,
                    )
            changed = True
        if not changed:
            break
    return actions


def enforce_screen_size_reasonableness(
    scene_dict: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    manifest: Optional[Dict[str, Any]] = None,
    predicates_payload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    by_name = subject_by_name(scene_dict)
    ranges_by_subject = _screen_size_ranges_by_subject(manifest, predicates_payload)
    for _ in range(SCREEN_SIZE_MAX_PASSES):
        records = _screen_size_records(scene_dict, reference_dims_map)
        target_density = _screen_size_target_density(records)
        if target_density is None:
            break
        changed = False
        dim_candidates: List[Tuple[Dict[str, Any], float, float, Tuple[float, float]]] = []
        for record in sorted(records, key=lambda item: abs(_screen_size_ratio(item, target_density) - 1.0), reverse=True):
            subject = record["subject"]
            ratio = _screen_size_ratio(record, target_density)
            ratio_range = _screen_size_ratio_range_for_record(record, ranges_by_subject)
            if _screen_size_ratio_in_range(ratio, ratio_range):
                continue
            old_x = subject_field(subject, "x")
            bounds = {"x_lower": None, "x_upper": None}
            if manifest is not None:
                bounds = _predicate_bounds_for_subject(str(subject.get("name", "")), by_name, manifest, predicates_payload)
            screen_depth_constraints = []
            if manifest is not None:
                screen_depth_constraints = _screen_depth_constraints_for_subject(
                    str(subject.get("name", "")),
                    by_name,
                    manifest,
                    predicates_payload,
                )
            if not screen_depth_constraints and any(value is not None for value in bounds.values()):
                dim_candidates.append((subject, ratio, target_density, ratio_range))
                continue
            if ratio > ratio_range[1]:
                # Too large on screen: move away from the camera first.
                candidate_x = _screen_size_find_x(
                    scene_dict,
                    subject,
                    reference_dims_map,
                    target_density,
                    ratio_range[1],
                    direction=-1.0,
                    x_lower=bounds["x_lower"],
                    x_upper=bounds["x_upper"],
                    screen_depth_constraints=screen_depth_constraints,
                )
                reason = "screen_size:move_away"
            else:
                # Too small on screen: move toward the camera first.
                candidate_x = _screen_size_find_x(
                    scene_dict,
                    subject,
                    reference_dims_map,
                    target_density,
                    ratio_range[0],
                    direction=1.0,
                    x_lower=bounds["x_lower"],
                    x_upper=bounds["x_upper"],
                    screen_depth_constraints=screen_depth_constraints,
                )
                reason = "screen_size:move_closer"
            if candidate_x is not None and abs(candidate_x - old_x) > EPS:
                set_field_action(subject, "x", candidate_x, reason, actions)
                actions[-1]["screen_size_ratio_before"] = round(ratio, 6)
                actions[-1]["screen_size_target_density"] = round(target_density, 6)
                actions[-1]["screen_size_ratio_range"] = [round(ratio_range[0], 6), round(ratio_range[1], 6)]
                changed = True
                updated_records = _screen_size_records(scene_dict, reference_dims_map)
                updated_target_density = _screen_size_target_density(updated_records)
                updated_record = next((item for item in updated_records if item["subject"] is subject), None)
                if updated_record is None or updated_target_density is None:
                    continue
                target_density = updated_target_density
                ratio = _screen_size_ratio(updated_record, target_density)
                ratio_range = _screen_size_ratio_range_for_record(updated_record, ranges_by_subject)
                if _screen_size_ratio_in_range(ratio, ratio_range):
                    continue
            dim_candidates.append((subject, ratio, target_density, ratio_range))
        if dim_candidates and _try_camera_for_screen_size_reasonableness(scene_dict, reference_dims_map, manifest, predicates_payload, actions):
            changed = True
            continue
        if dim_candidates and _screen_size_expand_reference_subjects(scene_dict, reference_dims_map, ranges_by_subject, manifest, predicates_payload, actions):
            changed = True
            continue
        for subject, ratio, target_density, ratio_range in dim_candidates:
            updated_records = _screen_size_records(scene_dict, reference_dims_map)
            updated_target_density = _screen_size_target_density(updated_records)
            updated_record = next((item for item in updated_records if item["subject"] is subject), None)
            if updated_record is not None and updated_target_density is not None:
                target_density = updated_target_density
                ratio = _screen_size_ratio(updated_record, target_density)
            ratio_range = _screen_size_ratio_range_for_record(
                updated_record if updated_record is not None else {"name": str(subject.get("name", ""))},
                ranges_by_subject,
            )
            if _screen_size_ratio_in_range(ratio, ratio_range):
                continue
            dim_scale = ratio_range[1] / max(ratio, EPS) if ratio > ratio_range[1] else ratio_range[0] / max(ratio, EPS)
            before_count = len(actions)
            _scale_subject_dims_uniform(subject, dim_scale, "screen_size:dim_compensation", actions, reference_dims_map)
            if len(actions) > before_count:
                for action in actions[before_count:]:
                    action["screen_size_ratio_before"] = round(ratio, 6)
                    action["screen_size_target_density"] = round(target_density, 6)
                    action["screen_size_ratio_range"] = [round(ratio_range[0], 6), round(ratio_range[1], 6)]
                changed = True
        if not changed:
            break
    if manifest is not None and predicates_payload is not None:
        actions.extend(_restore_screen_depth_after_screen_size(scene_dict, reference_dims_map, manifest, predicates_payload))
    return actions


def enforce_screen_depth_levels(
    scene_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    predicates_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    id_to_name = manifest_id_to_name(manifest)
    by_name = subject_by_name(scene_dict)
    for predicate in predicates_payload.get("predicates", []):
        axis, level, delta_px = _predicate_screen_depth_fields(predicate)
        if not axis:
            continue
        subject_name = id_to_name.get(predicate.get("subject", ""))
        object_name = id_to_name.get(predicate.get("object", ""))
        first = by_name.get(subject_name)
        second = by_name.get(object_name)
        if first is None or second is None:
            continue
        first_bbox = projected_subject_bbox(scene_dict, first)
        target_gap = _target_gap_for_screen_depth(scene_dict, second, axis, delta_px)
        if first_bbox is None or target_gap is None:
            continue
        current_gap = screen_bottom_gap(first_bbox) if axis == "front" else screen_top_gap(first_bbox)
        if current_gap <= target_gap + SCREEN_DEPTH_TOLERANCE_PX:
            continue
        old_x = subject_field(first, "x")
        candidate_x = _find_screen_depth_candidate_x(scene_dict, first, old_x, target_gap, axis)
        if candidate_x is not None:
            set_field_action(
                first,
                "x",
                candidate_x,
                f"screen_depth:{axis}:{level}",
                actions,
            )
            actions[-1]["screen_depth_delta_px"] = round(delta_px, 6)
            actions[-1]["screen_depth_target_gap_px"] = round(target_gap, 6)
            actions[-1]["screen_depth_initial_gap_px"] = round(current_gap, 6)
        for relax_idx in range(4):
            updated_bbox = projected_subject_bbox(scene_dict, first)
            if updated_bbox is None:
                break
            updated_gap = screen_bottom_gap(updated_bbox) if axis == "front" else screen_top_gap(updated_bbox)
            if updated_gap <= target_gap + SCREEN_DEPTH_TOLERANCE_PX:
                break
            if not _relax_camera_for_screen_depth(scene_dict, actions, f"screen_depth:{axis}:{level}:relax_camera"):
                break
            recalculated_target_gap = _target_gap_for_screen_depth(scene_dict, second, axis, delta_px)
            if recalculated_target_gap is not None:
                target_gap = recalculated_target_gap
            old_x = subject_field(first, "x")
            candidate_x = _find_screen_depth_candidate_x(scene_dict, first, old_x, target_gap, axis)
            if candidate_x is None or abs(candidate_x - old_x) <= EPS:
                continue
            set_field_action(
                first,
                "x",
                candidate_x,
                f"screen_depth:{axis}:{level}:after_camera_relax_{relax_idx + 1}",
                actions,
            )
            actions[-1]["screen_depth_delta_px"] = round(delta_px, 6)
            actions[-1]["screen_depth_target_gap_px"] = round(target_gap, 6)
            actions[-1]["screen_depth_initial_gap_px"] = round(updated_gap, 6)
    return actions


def center_scene_component(scene_dict: Dict[str, Any], pinned_names: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    if pinned_names:
        return actions
    current_subjects = subjects(scene_dict)
    if not current_subjects:
        return actions

    all_edges = [edges(subject) for subject in current_subjects]
    min_x = min(edge["back"] for edge in all_edges)
    max_x = max(edge["front"] for edge in all_edges)
    min_y = min(edge["left"] for edge in all_edges)
    max_y = max(edge["right"] for edge in all_edges)
    component_center_x = (min_x + max_x) / 2.0
    component_center_y = (min_y + max_y) / 2.0
    delta_x = DEFAULT_CENTER_X - component_center_x
    delta_y = DEFAULT_CENTER_Y - component_center_y
    if abs(delta_x) <= EPS and abs(delta_y) <= EPS:
        return actions

    for subject in current_subjects:
        old_x = subject_field(subject, "x")
        old_y = subject_field(subject, "y")
        new_x = round(old_x + delta_x, 6)
        new_y = round(old_y + delta_y, 6)
        set_subject_field(subject, "x", new_x)
        set_subject_field(subject, "y", new_y)
        actions.append(
            action_record(
                "set_param",
                subject["name"],
                "x[0]",
                old_x,
                new_x,
                "center_scene_component:x",
                component_delta=round(delta_x, 6),
            )
        )
        actions.append(
            action_record(
                "set_param",
                subject["name"],
                "y[0]",
                old_y,
                new_y,
                "center_scene_component:y",
                component_delta=round(delta_y, 6),
            )
        )
    return actions


def apply_camera_ranges(scene_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    camera = scene_dict.setdefault("camera_data", {})
    old_scale = float(camera.get("global_scale", 1.0))
    new_scale = clamp(old_scale, *DEFAULT_GLOBAL_SCALE_RANGE)
    if abs(old_scale - new_scale) > EPS:
        camera["global_scale"] = round(new_scale, 6)
        actions.append(action_record("clamp_camera", "camera", "camera_data.global_scale", old_scale, new_scale, "camera_range"))
    old_lens = float(camera.get("lens", 50.0))
    new_lens = clamp(old_lens, *DEFAULT_LENS_RANGE)
    if abs(old_lens - new_lens) > EPS:
        camera["lens"] = round(new_lens, 6)
        actions.append(action_record("clamp_camera", "camera", "camera_data.lens", old_lens, new_lens, "camera_range"))
    old_elevation = float(camera.get("camera_elevation", math.radians(DEFAULT_CAMERA_ELEVATION_DEG)))
    lower = math.radians(DEFAULT_ELEVATION_DEG_RANGE[0])
    upper = math.radians(DEFAULT_ELEVATION_DEG_RANGE[1])
    new_elevation = clamp(old_elevation, lower, upper)
    if abs(old_elevation - new_elevation) > EPS:
        camera["camera_elevation"] = round(new_elevation, 6)
        actions.append(action_record("clamp_camera", "camera", "camera_data.camera_elevation", old_elevation, new_elevation, "camera_range"))
    return actions


def _issue_object_pair(issue: Dict[str, Any]) -> Tuple[str, str]:
    raw = str(issue.get("object", ""))
    first, second = (raw.split("|", 1) + [""])[:2]
    return first.strip(), second.strip()


def _resize_subject_for_feedback(
    subject: Dict[str, Any],
    scale: float,
    reason: str,
    actions: List[Dict[str, Any]],
    reference_dims_map: Optional[Dict[str, List[float]]] = None,
    manifest_obj: Optional[Dict[str, Any]] = None,
) -> None:
    old_dims = dims(subject)
    new_dims = [value * scale for value in old_dims]
    if reference_dims_map is not None:
        obj = manifest_obj or {"type": subject.get("type"), "mention": subject.get("name")}
        lower, upper = dim_bounds(obj, reference_dims_map)
        if scale < 1.0:
            new_dims = [max(new_dims[idx], lower[idx]) for idx in range(3)]
        elif scale > 1.0:
            new_dims = [min(new_dims[idx], upper[idx]) for idx in range(3)]
    set_dims_action(subject, new_dims, reason, actions)


def _predicate_bounds_for_subject(
    subject_name: str,
    by_name: Dict[str, Dict[str, Any]],
    manifest: Dict[str, Any],
    predicates_payload: Optional[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    bounds: Dict[str, Optional[float]] = {"x_lower": None, "x_upper": None, "y_lower": None, "y_upper": None}
    if not predicates_payload:
        return bounds
    id_to_name = manifest_id_to_name(manifest)
    first = by_name.get(subject_name)
    if first is None:
        return bounds
    for predicate in predicates_payload.get("predicates", []):
        if id_to_name.get(predicate.get("subject", "")) != subject_name:
            continue
        second_name = id_to_name.get(predicate.get("object", ""))
        second = by_name.get(second_name)
        if second is None:
            continue
        tight = bool(predicate.get("tight"))
        ptype = predicate.get("type")
        if ptype == "left_of":
            value = subject_field(second, "y") - dims(second)[0] / 2.0 - dims(first)[0] / 2.0 - lateral_world_margin(first, second, predicate, tight=tight)
            bounds["y_upper"] = value if bounds["y_upper"] is None else min(bounds["y_upper"], value)
        elif ptype == "right_of":
            value = subject_field(second, "y") + dims(second)[0] / 2.0 + dims(first)[0] / 2.0 + lateral_world_margin(first, second, predicate, tight=tight)
            bounds["y_lower"] = value if bounds["y_lower"] is None else max(bounds["y_lower"], value)
        elif ptype == "in_front_of":
            margin = 0.0 if tight else margin_x(first, second)
            if predicate.get("compound_axis") == "front":
                margin = max(margin, float(predicate.get("compound_longitudinal_margin", DEFAULT_COMPOUND_LONGITUDINAL_MARGIN)))
            value = subject_field(second, "x") + dims(second)[1] / 2.0 + dims(first)[1] / 2.0 + margin
            bounds["x_lower"] = value if bounds["x_lower"] is None else max(bounds["x_lower"], value)
        elif ptype == "behind":
            margin = 0.0 if tight else margin_x(first, second)
            if predicate.get("compound_axis") == "back":
                margin = max(margin, float(predicate.get("compound_longitudinal_margin", DEFAULT_COMPOUND_LONGITUDINAL_MARGIN)))
            value = subject_field(second, "x") - dims(second)[1] / 2.0 - dims(first)[1] / 2.0 - margin
            bounds["x_upper"] = value if bounds["x_upper"] is None else min(bounds["x_upper"], value)
    return bounds


def _clamp_optional(value: float, lower: Optional[float], upper: Optional[float]) -> float:
    if lower is not None:
        value = max(value, lower)
    if upper is not None:
        value = min(value, upper)
    return value


def _predicate_name_graph(
    manifest: Dict[str, Any],
    predicates_payload: Optional[Dict[str, Any]],
    predicate_types: Set[str],
) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    if not predicates_payload:
        return graph
    id_to_name = manifest_id_to_name(manifest)
    for predicate in predicates_payload.get("predicates", []):
        if str(predicate.get("type", "")) not in predicate_types:
            continue
        first_name = id_to_name.get(predicate.get("subject", ""))
        second_name = id_to_name.get(predicate.get("object", ""))
        if not first_name or not second_name:
            continue
        graph[first_name].add(second_name)
        graph[second_name].add(first_name)
    return graph


def _predicate_component_names(
    subject_name: str,
    manifest: Dict[str, Any],
    predicates_payload: Optional[Dict[str, Any]],
    predicate_types: Set[str],
) -> Set[str]:
    graph = _predicate_name_graph(manifest, predicates_payload, predicate_types)
    if subject_name not in graph:
        return {subject_name}
    visited = {subject_name}
    stack = [subject_name]
    while stack:
        current = stack.pop()
        for neighbor in graph.get(current, set()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            stack.append(neighbor)
    return visited


def _centered_subject_names(manifest: Dict[str, Any], predicates_payload: Optional[Dict[str, Any]]) -> Set[str]:
    if not predicates_payload:
        return set()
    id_to_name = manifest_id_to_name(manifest)
    return {
        id_to_name.get(predicate.get("subject", ""))
        for predicate in predicates_payload.get("predicates", [])
        if predicate.get("type") == "centered"
    } - {""}


def _move_y_relation_component_toward_image_center(
    subject: Dict[str, Any],
    by_name: Dict[str, Dict[str, Any]],
    manifest: Dict[str, Any],
    predicates_payload: Optional[Dict[str, Any]],
    actions: List[Dict[str, Any]],
) -> bool:
    subject_name = str(subject.get("name", ""))
    component_names = _predicate_component_names(
        subject_name,
        manifest,
        predicates_payload,
        {"left_of", "right_of", "support", "above", "below"},
    )
    if len(component_names) <= 1:
        return False
    if component_names & _centered_subject_names(manifest, predicates_payload):
        return False

    old_subject_y = subject_field(subject, "y")
    target_subject_y = old_subject_y + (DEFAULT_CENTER_Y - old_subject_y) * 0.35
    delta_y = target_subject_y - old_subject_y
    if abs(delta_y) <= EPS:
        return False

    action_count_before = len(actions)
    for name in sorted(component_names):
        related_subject = by_name.get(name)
        if related_subject is None:
            continue
        old_y = subject_field(related_subject, "y")
        new_y = old_y + delta_y
        set_field_action(
            related_subject,
            "y",
            new_y,
            f"frame_clipping:move_y_relation_component({subject_name})",
            actions,
        )
    return len(actions) > action_count_before


def _move_clipped_subject_toward_image_center(
    scene_dict: Dict[str, Any],
    subject: Dict[str, Any],
    by_name: Dict[str, Dict[str, Any]],
    manifest: Dict[str, Any],
    predicates_payload: Optional[Dict[str, Any]],
    actions: List[Dict[str, Any]],
    axes: str = "xy",
) -> bool:
    action_count_before = len(actions)
    bounds = _predicate_bounds_for_subject(str(subject.get("name", "")), by_name, manifest, predicates_payload)
    old_x = subject_field(subject, "x")
    old_y = subject_field(subject, "y")
    if "x" in axes:
        target_x = _clamp_optional(old_x + (DEFAULT_CENTER_X - old_x) * 0.35, bounds["x_lower"], bounds["x_upper"])
        screen_depth_constraints = _screen_depth_constraints_for_subject(
            str(subject.get("name", "")),
            by_name,
            manifest,
            predicates_payload,
        )
        if screen_depth_constraints and not _screen_depth_allows_x(scene_dict, subject, screen_depth_constraints, target_x):
            target_x = old_x
        set_field_action(subject, "x", target_x, "frame_clipping:move_toward_image_center_x", actions)
    if "y" in axes:
        target_y = _clamp_optional(old_y + (DEFAULT_CENTER_Y - old_y) * 0.35, bounds["y_lower"], bounds["y_upper"])
        set_field_action(subject, "y", target_y, "frame_clipping:move_toward_image_center_y", actions)
    return len(actions) > action_count_before


def _decrease_camera_for_visibility(scene_dict: Dict[str, Any], reason_prefix: str, actions: List[Dict[str, Any]]) -> None:
    camera = scene_dict.setdefault("camera_data", {})
    old_lens = float(camera.get("lens", 50.0))
    new_lens = clamp(old_lens * GUIDANCE_LENS_DECREASE_SCALE, DEFAULT_LENS_RANGE[0], DEFAULT_LENS_RANGE[1])
    if abs(old_lens - new_lens) > EPS:
        camera["lens"] = round(new_lens, 6)
        actions.append(action_record("set_param", "camera", "camera_data.lens", old_lens, new_lens, f"{reason_prefix}:decrease_lens"))
    old_scale = float(camera.get("global_scale", 1.0))
    new_scale = clamp(old_scale * GUIDANCE_GLOBAL_SCALE_DECREASE_SCALE, DEFAULT_GLOBAL_SCALE_RANGE[0], DEFAULT_GLOBAL_SCALE_RANGE[1])
    if abs(old_scale - new_scale) > EPS:
        camera["global_scale"] = round(new_scale, 6)
        actions.append(action_record("set_param", "camera", "camera_data.global_scale", old_scale, new_scale, f"{reason_prefix}:decrease_global_scale"))


def apply_feedback_issues(
    scene_dict: Dict[str, Any],
    issues: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    predicates_payload: Optional[Dict[str, Any]] = None,
    camera_facing_payload: Optional[Dict[str, Any]] = None,
    default_azimuths_map: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Compile A-F validation issues into deterministic pkl edits.

    This bridges natural-language suggestions such as "Decrease all dimensions
    of lamp" into field-level actions, so the harness loop does not depend on
    an LLM or free-form guidance execution.
    """
    actions: List[Dict[str, Any]] = []
    by_name = subject_by_name(scene_dict)
    obj_by_name = {obj["name"]: obj for obj in manifest.get("objects", [])}
    if default_azimuths_map is None:
        default_azimuths_map = load_asset_default_azimuths(reference_dims_map.keys())
    if camera_facing_payload is None:
        camera_facing_payload = build_camera_facing_constraints(
            (predicates_payload or {}).get("source_text", ""),
            manifest,
            default_azimuths_map,
        )
    camera_facing_by_name = {
        manifest_id_to_name(manifest).get(constraint.get("object", "")): constraint
        for constraint in camera_facing_payload.get("camera_facing", [])
    }
    clipping_objects = [
        str(issue.get("object", "")).strip()
        for issue in issues
        if issue.get("category") == "frame_clipping" and str(issue.get("object", "")).strip()
    ]
    if any(issue.get("category") == "screen_depth_relation" for issue in issues):
        actions.extend(enforce_screen_depth_levels(scene_dict, manifest, predicates_payload or {"predicates": []}))
    if any(issue.get("category") == "screen_lateral_gap" for issue in issues):
        actions.extend(enforce_screen_lateral_gaps(scene_dict, manifest, predicates_payload or {"predicates": []}))
    if any(issue.get("category") == "pairwise_screen_occlusion" for issue in issues):
        actions.extend(enforce_pairwise_screen_occlusion(scene_dict, manifest, predicates_payload or {"predicates": []}))
    if any(issue.get("category") == "screen_size_reasonableness" for issue in issues):
        actions.extend(enforce_screen_size_reasonableness(scene_dict, reference_dims_map, manifest, predicates_payload))
    if any(issue.get("category") == "component_compactness" for issue in issues):
        actions.extend(enforce_component_compactness(scene_dict, reference_dims_map, manifest, predicates_payload or {"predicates": []}))
    unique_clipping_objects = sorted(set(clipping_objects))

    clipping_subjects = [by_name[name] for name in unique_clipping_objects if name in by_name]
    moved_clipping_names: Set[str] = set()
    for subject in clipping_subjects:
        moved = _move_clipped_subject_toward_image_center(
            scene_dict,
            subject,
            by_name,
            manifest,
            predicates_payload,
            actions,
            axes="y",
        )
        if not moved:
            moved = _move_y_relation_component_toward_image_center(
                subject,
                by_name,
                manifest,
                predicates_payload,
                actions,
            )
        if moved:
            moved_clipping_names.add(str(subject.get("name", "")))

    if len(unique_clipping_objects) >= 3:
        _decrease_camera_for_visibility(scene_dict, "frame_clipping", actions)

    for issue in issues:
        category = str(issue.get("category", ""))
        obj_name = str(issue.get("object", "")).strip()

        if category == "frame_clipping":
            if len(unique_clipping_objects) >= 3:
                continue
            subject = by_name.get(obj_name)
            if subject is not None:
                moved = obj_name in moved_clipping_names
                moved = _move_clipped_subject_toward_image_center(scene_dict, subject, by_name, manifest, predicates_payload, actions, axes="x") or moved
                if not moved:
                    _decrease_camera_for_visibility(scene_dict, "frame_clipping:predicate_bound_fallback", actions)
            continue

        if category == "size_implausible":
            subject = by_name.get(obj_name)
            if subject is not None:
                obj = obj_by_name.get(obj_name, {"type": subject.get("type"), "mention": subject.get("name")})
                if issue.get("reason") == "dimension_reference_safety":
                    target = clamp_dims_to_reference_safety(dims(subject), obj, reference_dims_map)
                else:
                    target = visual_target_dims(obj, reference_dims_map)
                set_dims_action(subject, target, "size_implausible:reference_dims", actions)
            continue

        if category == "camera_facing_azimuth":
            subject = by_name.get(obj_name)
            obj = obj_by_name.get(obj_name)
            constraint = camera_facing_by_name.get(obj_name)
            if subject is not None and obj is not None and constraint is not None:
                face = str(constraint.get("face") or "front")
                expected = camera_face_azimuth(str(obj.get("type", "")), face, default_azimuths_map)
                set_field_action(subject, "azimuth", expected, f"camera_facing_azimuth:{face}", actions)
            continue

        if category == "size_suspicious":
            subject = by_name.get(obj_name)
            if subject is not None:
                obj = obj_by_name.get(obj_name, {"type": subject.get("type"), "mention": subject.get("name")})
                if issue.get("reason") == "dimension_reference_safety":
                    target = clamp_dims_to_reference_safety(dims(subject), obj, reference_dims_map)
                else:
                    target = visual_target_dims(obj, reference_dims_map)
                current = dims(subject)
                blended = [(current[idx] + target[idx]) / 2.0 for idx in range(3)]
                set_dims_action(subject, blended, "size_suspicious:toward_reference_dims", actions)
            continue

        if category == "below_ground":
            subject = by_name.get(obj_name)
            if subject is not None:
                set_field_action(subject, "z", 0.0, "below_ground:move_to_ground", actions)
            continue

        if category == "floating_object":
            subject = by_name.get(obj_name)
            if subject is not None:
                set_field_action(subject, "z", 0.0, "floating_object:move_to_ground", actions)
            continue

        if category == "relation_center_mismatch":
            subject = by_name.get(obj_name)
            if subject is not None:
                set_field_action(subject, "x", DEFAULT_CENTER_X, "relation_center_mismatch:center_x", actions)
                set_field_action(subject, "y", DEFAULT_CENTER_Y, "relation_center_mismatch:center_y", actions)
            continue

        if category in {"relation_left_mismatch", "relation_right_mismatch", "relation_front_mismatch", "relation_behind_mismatch"}:
            first_name, second_name = _issue_object_pair(issue)
            first = by_name.get(first_name)
            second = by_name.get(second_name)
            if first is None or second is None:
                continue
            if category == "relation_left_mismatch":
                target_y = subject_field(second, "y") - dims(second)[0] / 2.0 - dims(first)[0] / 2.0 - margin_y(first, second)
                set_field_action(first, "y", min(subject_field(first, "y"), target_y), "relation_left_mismatch:y_upper", actions)
            elif category == "relation_right_mismatch":
                target_y = subject_field(second, "y") + dims(second)[0] / 2.0 + dims(first)[0] / 2.0 + margin_y(first, second)
                set_field_action(first, "y", max(subject_field(first, "y"), target_y), "relation_right_mismatch:y_lower", actions)
            elif category == "relation_front_mismatch":
                target_x = subject_field(second, "x") + dims(second)[1] / 2.0 + dims(first)[1] / 2.0 + margin_x(first, second)
                set_field_action(first, "x", max(subject_field(first, "x"), target_x), "relation_front_mismatch:x_lower", actions)
            elif category == "relation_behind_mismatch":
                target_x = subject_field(second, "x") - dims(second)[1] / 2.0 - dims(first)[1] / 2.0 - margin_x(first, second)
                set_field_action(first, "x", min(subject_field(first, "x"), target_x), "relation_behind_mismatch:x_upper", actions)
            continue

        if category == "support_width_oversize":
            first_name, _ = _issue_object_pair(issue)
            subject = by_name.get(first_name)
            if subject is not None:
                old = dims(subject)
                new = list(old)
                new[0] = max(old[0] * GUIDANCE_DIM_DECREASE_SCALE, MIN_VISUAL_DIMS[0])
                set_dims_action(subject, new, "support_width_oversize:decrease_width", actions)
            continue

        if category == "support_depth_oversize":
            first_name, _ = _issue_object_pair(issue)
            subject = by_name.get(first_name)
            if subject is not None:
                old = dims(subject)
                new = list(old)
                new[1] = max(old[1] * GUIDANCE_DIM_DECREASE_SCALE, MIN_VISUAL_DIMS[1])
                set_dims_action(subject, new, "support_depth_oversize:decrease_depth", actions)
            continue

    return actions


def build_feedback_repair_plan(
    scene_dict: Dict[str, Any],
    issues: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    predicates_payload: Optional[Dict[str, Any]] = None,
    camera_facing_payload: Optional[Dict[str, Any]] = None,
    default_azimuths_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    repaired_scene = clone_scene(scene_dict)
    actions: List[Dict[str, Any]] = []
    actions.extend(apply_feedback_issues(
        scene_dict=repaired_scene,
        issues=issues,
        manifest=manifest,
        reference_dims_map=reference_dims_map,
        predicates_payload=predicates_payload,
        camera_facing_payload=camera_facing_payload,
        default_azimuths_map=default_azimuths_map,
    ))
    return build_repair_plan(scene_dict, repaired_scene, actions)


def merge_repair_plans(*plans: Dict[str, Any]) -> Dict[str, Any]:
    merged_actions: List[Dict[str, Any]] = []
    for plan in plans:
        merged_actions.extend(copy.deepcopy(plan.get("actions", [])))
    return {
        "version": 1,
        "mode": "deterministic_harness",
        "manifest_path": next((str(plan.get("manifest_path", "")) for plan in plans if plan.get("manifest_path")), ""),
        "predicates_path": next((str(plan.get("predicates_path", "")) for plan in plans if plan.get("predicates_path")), ""),
        "constraints_path": next((str(plan.get("constraints_path", "")) for plan in plans if plan.get("constraints_path")), ""),
        "actions": merged_actions,
        "action_count": len(merged_actions),
    }


def apply_full_harness(
    scene_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    predicates_payload: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    camera_facing_payload: Optional[Dict[str, Any]] = None,
    default_azimuths_map: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if default_azimuths_map is None:
        default_azimuths_map = load_asset_default_azimuths(reference_dims_map.keys())
    if camera_facing_payload is None:
        camera_facing_payload = build_camera_facing_constraints(
            predicates_payload.get("source_text", ""),
            manifest,
            default_azimuths_map,
        )
    working_scene, actions = reconcile_scene_to_manifest(scene_dict, manifest, reference_dims_map)
    id_to_name = manifest_id_to_name(manifest)
    pinned_center_names = {
        id_to_name.get(predicate.get("subject", ""))
        for predicate in predicates_payload.get("predicates", [])
        if predicate.get("type") == "centered"
    }
    pinned_center_names.discard("")
    actions.extend(normalize_dimensions(working_scene, manifest, reference_dims_map))
    actions.extend(enforce_camera_facing_azimuths(working_scene, manifest, camera_facing_payload, default_azimuths_map))
    actions.extend(apply_predicates(working_scene, manifest, predicates_payload))
    actions.extend(spread_supported_subjects(working_scene, manifest, predicates_payload))
    actions.extend(normalize_dimensions(working_scene, manifest, reference_dims_map))
    actions.extend(enforce_camera_facing_azimuths(working_scene, manifest, camera_facing_payload, default_azimuths_map))
    actions.extend(spread_supported_subjects(working_scene, manifest, predicates_payload))
    actions.extend(enforce_ground_and_collision(working_scene, predicates_payload))
    actions.extend(center_scene_component(working_scene, pinned_names=pinned_center_names))
    actions.extend(apply_camera_ranges(working_scene))
    actions.extend(enforce_screen_depth_levels(working_scene, manifest, predicates_payload))
    actions.extend(enforce_screen_lateral_gaps(working_scene, manifest, predicates_payload))
    actions.extend(enforce_pairwise_screen_occlusion(working_scene, manifest, predicates_payload))
    actions.extend(apply_camera_ranges(working_scene))
    actions.extend(enforce_screen_size_reasonableness(working_scene, reference_dims_map, manifest, predicates_payload))
    actions.extend(enforce_screen_depth_levels(working_scene, manifest, predicates_payload))
    actions.extend(enforce_screen_lateral_gaps(working_scene, manifest, predicates_payload))
    actions.extend(enforce_pairwise_screen_occlusion(working_scene, manifest, predicates_payload))
    actions.extend(apply_predicates(working_scene, manifest, predicates_payload))
    actions.extend(spread_supported_subjects(working_scene, manifest, predicates_payload))
    actions.extend(center_scene_component(working_scene, pinned_names=pinned_center_names))
    actions.extend(enforce_screen_size_reasonableness(working_scene, reference_dims_map, manifest, predicates_payload))
    actions.extend(apply_camera_ranges(working_scene))
    actions.extend(enforce_component_compactness(working_scene, reference_dims_map, manifest, predicates_payload))
    actions.extend(enforce_screen_depth_levels(working_scene, manifest, predicates_payload))
    actions.extend(enforce_screen_lateral_gaps(working_scene, manifest, predicates_payload))
    actions.extend(enforce_pairwise_screen_occlusion(working_scene, manifest, predicates_payload))
    actions.extend(enforce_screen_size_reasonableness(working_scene, reference_dims_map, manifest, predicates_payload))
    actions.extend(apply_camera_ranges(working_scene))
    return working_scene, actions


def validate_objects(scene_dict: Dict[str, Any], manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    expected_names = [obj["name"] for obj in manifest.get("objects", []) if obj.get("type") != CUSTOM_ASSET_TYPE]
    actual_names = [str(subject.get("name")) for subject in subjects(scene_dict)]
    for name in sorted(set(expected_names) - set(actual_names)):
        issues.append({"category": "object_exact_match", "object": name, "details": "prompt 主体缺失。"})
    for name in sorted(set(actual_names) - set(expected_names)):
        issues.append({"category": "object_exact_match", "object": name, "details": "pkl 中存在 prompt 未声明的物体。"})
    expected_counts = Counter(obj["type"] for obj in manifest.get("objects", []) if obj.get("type") != CUSTOM_ASSET_TYPE)
    actual_counts = Counter(str(subject.get("type")) for subject in subjects(scene_dict))
    if expected_counts != actual_counts:
        issues.append(
            {
                "category": "object_type_count",
                "object": "",
                "details": f"类型数量不一致，expected={dict(expected_counts)}, actual={dict(actual_counts)}。",
            }
        )
    for obj in manifest.get("objects", []):
        if obj.get("type") == CUSTOM_ASSET_TYPE:
            issues.append({"category": "unmapped_object", "object": obj.get("mention", ""), "details": "主体无法映射到资产类型。"})
    return issues


def validate_dimensions(
    scene_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    obj_by_name = {obj["name"]: obj for obj in manifest.get("objects", [])}
    for subject in subjects(scene_dict):
        obj = obj_by_name.get(subject.get("name"), {"type": subject.get("type"), "mention": subject.get("name")})
        if subject.get("_screen_size_dim_compensated"):
            lower, upper = dim_reference_safety_bounds(obj, reference_dims_map)
            reason = "dimension_reference_safety"
        else:
            lower, upper = dim_bounds(obj, reference_dims_map)
            reason = "dimension_range"
        current_dims = dims(subject)
        for idx, label in enumerate(("width", "depth", "height")):
            if current_dims[idx] < lower[idx] - EPS or current_dims[idx] > upper[idx] + EPS:
                issues.append(
                    {
                        "category": "dimension_range",
                        "object": subject.get("name"),
                        "field": f"dims[{idx}]",
                        "expected": f"{lower[idx]:.6f} <= {label} <= {upper[idx]:.6f}",
                        "actual": current_dims[idx],
                        "reason": reason,
                    }
                )
    return issues


def validate_camera_facing_azimuths(
    scene_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    camera_facing_payload: Dict[str, Any],
    default_azimuths_map: Dict[str, float],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    id_to_name = manifest_id_to_name(manifest)
    obj_by_id = {obj["id"]: obj for obj in manifest.get("objects", [])}
    by_name = subject_by_name(scene_dict)
    for constraint in camera_facing_payload.get("camera_facing", []):
        obj_id = constraint.get("object")
        obj = obj_by_id.get(obj_id)
        subject_name = id_to_name.get(obj_id, "")
        subject = by_name.get(subject_name)
        if obj is None or subject is None:
            continue
        face = str(constraint.get("face") or "front")
        expected = camera_face_azimuth(str(obj.get("type", "")), face, default_azimuths_map)
        actual = normalize_angle_rad(subject_field(subject, "azimuth"))
        if angular_distance_rad(actual, expected) > AZIMUTH_TOLERANCE_RAD:
            issues.append(
                {
                    "category": "camera_facing_azimuth",
                    "object": subject_name,
                    "field": "azimuth[0]",
                    "expected": f"{face} face toward camera -> azimuth={expected:.6f}",
                    "actual": actual,
                    "evidence": constraint.get("evidence", ""),
                }
            )
    return issues


def validate_volume(scene_dict: Dict[str, Any], reference_dims_map: Dict[str, List[float]]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    current = subjects(scene_dict)
    for idx, first in enumerate(current):
        if first.get("_screen_size_dim_compensated"):
            continue
        first_ref = reference_dims_map.get(first.get("type"), dims(first))
        first_ref_volume = volume_for_dims(first_ref)
        first_volume = volume_for_dims(dims(first))
        for second in current[idx + 1 :]:
            if second.get("_screen_size_dim_compensated"):
                continue
            second_ref = reference_dims_map.get(second.get("type"), dims(second))
            second_ref_volume = volume_for_dims(second_ref)
            second_volume = volume_for_dims(dims(second))
            if first_ref_volume > second_ref_volume * 1.5 and first_volume < second_volume * 1.15:
                issues.append(
                    {
                        "category": "volume_order",
                        "object": f"{first.get('name')}|{second.get('name')}",
                        "details": "参考体积更大的物体在当前 pkl 中没有保持更大体积。",
                    }
                )
            if second_ref_volume > first_ref_volume * 1.5 and second_volume < first_volume * 1.15:
                issues.append(
                    {
                        "category": "volume_order",
                        "object": f"{second.get('name')}|{first.get('name')}",
                        "details": "参考体积更大的物体在当前 pkl 中没有保持更大体积。",
                    }
                )
    return issues


def validate_predicates(scene_dict: Dict[str, Any], manifest: Dict[str, Any], predicates_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    id_to_name = manifest_id_to_name(manifest)
    by_name = subject_by_name(scene_dict)
    for predicate in predicates_payload.get("predicates", []):
        subject_name = id_to_name.get(predicate.get("subject", ""))
        object_name = id_to_name.get(predicate.get("object", ""))
        first = by_name.get(subject_name)
        second = by_name.get(object_name) if object_name else None
        if first is None:
            continue
        ptype = predicate.get("type")
        tight = bool(predicate.get("tight"))
        if ptype == "centered":
            if abs(subject_field(first, "x") - DEFAULT_CENTER_X) > 0.15 or abs(subject_field(first, "y") - DEFAULT_CENTER_Y) > 0.15:
                issues.append({"category": "spatial_relation", "predicate_id": predicate.get("id"), "object": subject_name, "details": "centered 未满足。"})
            continue
        if second is None:
            continue
        first_edges = edges(first)
        second_edges = edges(second)
        passed = True
        expected = ""
        if ptype == "support":
            passed = (
                abs(first_edges["bottom"] - second_edges["top"]) <= 0.03
                and first_edges["left"] >= second_edges["left"] + SUPPORT_INSET - EPS
                and first_edges["right"] <= second_edges["right"] - SUPPORT_INSET + EPS
                and first_edges["back"] >= second_edges["back"] + SUPPORT_INSET - EPS
                and first_edges["front"] <= second_edges["front"] - SUPPORT_INSET + EPS
            )
            expected = "bottom(A)=top(B) and footprint(A) inside footprint(B)"
        elif ptype == "above":
            lower = second_edges["top"] + (0.0 if tight else MARGIN_Z)
            passed = first_edges["bottom"] >= lower - EPS
            expected = f"bottom(A) >= {lower:.6f}"
        elif ptype == "below":
            upper = second_edges["bottom"] - (0.0 if tight else MARGIN_Z)
            passed = first_edges["top"] <= upper + EPS
            expected = f"top(A) <= {upper:.6f}"
        elif ptype == "left_of":
            margin = lateral_world_margin(first, second, predicate, tight=tight)
            upper = second_edges["left"] - margin
            passed = first_edges["right"] <= upper + EPS
            expected = f"right(A) <= {upper:.6f}"
        elif ptype == "right_of":
            margin = lateral_world_margin(first, second, predicate, tight=tight)
            lower = second_edges["right"] + margin
            passed = first_edges["left"] >= lower - EPS
            expected = f"left(A) >= {lower:.6f}"
        elif ptype == "in_front_of":
            margin = 0.0 if tight else margin_x(first, second)
            if predicate.get("compound_axis") == "front":
                margin = max(margin, float(predicate.get("compound_longitudinal_margin", DEFAULT_COMPOUND_LONGITUDINAL_MARGIN)))
            lower = second_edges["front"] + margin
            passed = first_edges["back"] >= lower - EPS
            expected = f"back(A) >= {lower:.6f}"
        elif ptype == "behind":
            margin = 0.0 if tight else margin_x(first, second)
            if predicate.get("compound_axis") == "back":
                margin = max(margin, float(predicate.get("compound_longitudinal_margin", DEFAULT_COMPOUND_LONGITUDINAL_MARGIN)))
            upper = second_edges["back"] - margin
            passed = first_edges["front"] <= upper + EPS
            expected = f"front(A) <= {upper:.6f}"
        if not passed:
            issues.append(
                {
                    "category": "spatial_relation",
                    "predicate_id": predicate.get("id"),
                    "predicate": f"{ptype}({subject_name},{object_name})",
                    "object": f"{subject_name}|{object_name}",
                    "expected": expected,
                    "actual": {"first": first_edges, "second": second_edges},
                    "repair_tool": "project_scene_constraints",
                }
            )
    return issues


def validate_screen_depth_levels(scene_dict: Dict[str, Any], manifest: Dict[str, Any], predicates_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    id_to_name = manifest_id_to_name(manifest)
    by_name = subject_by_name(scene_dict)
    for predicate in predicates_payload.get("predicates", []):
        axis, level, delta_px = _predicate_screen_depth_fields(predicate)
        if not axis:
            continue
        subject_name = id_to_name.get(predicate.get("subject", ""))
        object_name = id_to_name.get(predicate.get("object", ""))
        first = by_name.get(subject_name)
        second = by_name.get(object_name)
        if first is None or second is None:
            continue
        first_bbox = projected_subject_bbox(scene_dict, first)
        target_gap = _target_gap_for_screen_depth(scene_dict, second, axis, delta_px)
        if first_bbox is None or target_gap is None:
            continue
        actual_gap = screen_bottom_gap(first_bbox) if axis == "front" else screen_top_gap(first_bbox)
        if actual_gap <= target_gap + SCREEN_DEPTH_TOLERANCE_PX:
            continue
        gap_name = "bottom_gap" if axis == "front" else "top_gap"
        issues.append(
            {
                "category": "screen_depth_relation",
                "predicate_id": predicate.get("id"),
                "predicate": f"{predicate.get('type')}({subject_name},{object_name})",
                "object": f"{subject_name}|{object_name}",
                "details": f"{gap_name}(A)={actual_gap:.3f}px 未达到 {level} 分级目标。",
                "expected": f"{gap_name}(A) <= {target_gap:.3f}px",
                "actual": {gap_name: round(actual_gap, 6), "target_gap": round(target_gap, 6), "level": level, "delta_px": delta_px},
                "repair_tool": "project_scene_constraints",
            }
        )
    return issues


def validate_screen_lateral_gaps(scene_dict: Dict[str, Any], manifest: Dict[str, Any], predicates_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    id_to_name = manifest_id_to_name(manifest)
    by_name = subject_by_name(scene_dict)
    for predicate in predicates_payload.get("predicates", []):
        axis, level, gap_px = _predicate_screen_lateral_fields(predicate)
        if not axis:
            continue
        subject_name = id_to_name.get(predicate.get("subject", ""))
        object_name = id_to_name.get(predicate.get("object", ""))
        first = by_name.get(subject_name)
        second = by_name.get(object_name)
        if first is None or second is None:
            continue
        first_bbox = projected_subject_bbox(scene_dict, first)
        second_bbox = projected_subject_bbox(scene_dict, second)
        if first_bbox is None or second_bbox is None:
            continue
        if axis == "left":
            actual_gap = second_bbox["min_x"] - first_bbox["max_x"]
            expected = f"screen_gap_x(A.right, B.left) >= {gap_px:.3f}px"
        else:
            actual_gap = first_bbox["min_x"] - second_bbox["max_x"]
            expected = f"screen_gap_x(A.left, B.right) >= {gap_px:.3f}px"
        if actual_gap + SCREEN_DEPTH_TOLERANCE_PX >= gap_px:
            continue
        issues.append(
            {
                "category": "screen_lateral_gap",
                "predicate_id": predicate.get("id"),
                "predicate": f"{predicate.get('type')}({subject_name},{object_name})",
                "object": f"{subject_name}|{object_name}",
                "details": f"screen lateral gap={actual_gap:.3f}px 未达到 {level} 分级目标。",
                "expected": expected,
                "actual": {
                    "screen_gap_px": round(actual_gap, 6),
                    "target_gap_px": round(gap_px, 6),
                    "level": level,
                    "axis": axis,
                },
                "repair_tool": "project_scene_constraints",
            }
        )
    return issues


def validate_pairwise_screen_occlusion(
    scene_dict: Dict[str, Any],
    max_ratio: float = PAIRWISE_OCCLUSION_MAX_RATIO,
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    current_subjects = subjects(scene_dict)
    bboxes = {str(subject.get("name", "")): projected_subject_bbox(scene_dict, subject) for subject in current_subjects}
    for idx, first in enumerate(current_subjects):
        first_name = str(first.get("name", ""))
        first_bbox = bboxes.get(first_name)
        if first_bbox is None:
            continue
        first_area = projected_bbox_size(first_bbox)[2]
        for second in current_subjects[idx + 1 :]:
            second_name = str(second.get("name", ""))
            second_bbox = bboxes.get(second_name)
            if second_bbox is None:
                continue
            second_area = projected_bbox_size(second_bbox)[2]
            ratio, intersection = projected_bbox_overlap_ratio(first_bbox, second_bbox)
            if ratio <= max_ratio + 1e-4:
                continue
            issues.append(
                {
                    "category": "pairwise_screen_occlusion",
                    "object": f"{first_name}|{second_name}",
                    "details": f"两个 cube 的屏幕 bbox 重叠比例={ratio:.3f}，超过 {max_ratio:.2f}。",
                    "expected": f"intersection_area / min(area(A), area(B)) <= {max_ratio:.2f}",
                    "actual": {
                        "overlap_ratio": round(ratio, 6),
                        "intersection_area": round(intersection, 6),
                        "first_area": round(first_area, 6),
                        "second_area": round(second_area, 6),
                        "max_ratio": round(max_ratio, 6),
                    },
                    "repair_tool": "project_scene_constraints",
                }
            )
    return issues


def validate_screen_size_reasonableness(
    scene_dict: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    manifest: Optional[Dict[str, Any]] = None,
    predicates_payload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    ranges_by_subject = _screen_size_ranges_by_subject(manifest, predicates_payload)
    records = _screen_size_records(scene_dict, reference_dims_map)
    target_density = _screen_size_target_density(records)
    if target_density is None:
        return issues
    for record in records:
        ratio = _screen_size_ratio(record, target_density)
        ratio_range = _screen_size_ratio_range_for_record(record, ranges_by_subject)
        subject = record["subject"]
        if ratio > ratio_range[1] and dims_at_reference_safety_bound(subject, reference_dims_map, "lower"):
            continue
        if ratio < ratio_range[0] and dims_at_reference_safety_bound(subject, reference_dims_map, "upper"):
            continue
        if _screen_size_ratio_in_range(ratio, ratio_range):
            continue
        issues.append(
            {
                "category": "screen_size_reasonableness",
                "object": record["name"],
                "details": f"screen size ratio={ratio:.3f} 超出 [{ratio_range[0]:.2f}, {ratio_range[1]:.2f}]。",
                "expected": f"{ratio_range[0]:.2f} <= screen_size_ratio <= {ratio_range[1]:.2f}",
                "actual": {
                    "ratio": round(ratio, 6),
                    "ratio_range": [round(ratio_range[0], 6), round(ratio_range[1], 6)],
                    "density": round(float(record["density"]), 6),
                    "target_density": round(float(target_density), 6),
                    "bbox_height": round(float(record["bbox_height"]), 6),
                    "bbox_area": round(float(record["bbox_area"]), 6),
                },
                "repair_tool": "project_scene_constraints",
            }
        )
    return issues


def validate_component_compactness(
    scene_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    predicates_payload: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    bbox = projected_component_bbox(scene_dict)
    metrics = component_bbox_metrics(bbox)
    target = component_compactness_target(manifest, predicates_payload, reference_dims_map)
    if metrics is None:
        return [
            {
                "category": "component_compactness",
                "object": "scene",
                "details": "所有 cube 的 union bbox 无法投影到屏幕空间。",
                "expected": "projected component bbox exists",
                "actual": {},
                "repair_tool": "project_scene_constraints",
            }
        ]
    width_ok = metrics["width_ratio"] + EPS >= float(target["width_ratio"])
    height_ok = metrics["height_ratio"] + EPS >= float(target["height_ratio"])
    if width_ok and height_ok:
        return []
    return [
        {
            "category": "component_compactness",
            "object": "scene",
            "details": (
                "所有 cube 的 union bbox 占屏不足："
                f"width_ratio={metrics['width_ratio']:.3f}, height_ratio={metrics['height_ratio']:.3f}。"
            ),
            "expected": (
                f"component_width_ratio >= {float(target['width_ratio']):.2f}, "
                f"component_height_ratio >= {float(target['height_ratio']):.2f}"
            ),
            "actual": {
                "width_ratio": round(float(metrics["width_ratio"]), 6),
                "height_ratio": round(float(metrics["height_ratio"]), 6),
                "area_ratio": round(float(metrics["area_ratio"]), 6),
                "min_margin_px": round(float(metrics["min_margin_px"]), 6),
                "target": target,
            },
            "repair_tool": "project_scene_constraints",
        }
    ]


def validate_camera(scene_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    camera = scene_dict.get("camera_data", {})
    lens = float(camera.get("lens", 50.0))
    scale = float(camera.get("global_scale", 1.0))
    elevation = float(camera.get("camera_elevation", math.radians(DEFAULT_CAMERA_ELEVATION_DEG)))
    if not (DEFAULT_LENS_RANGE[0] <= lens <= DEFAULT_LENS_RANGE[1]):
        issues.append({"category": "camera_range", "object": "lens", "details": f"lens={lens} 超出范围。"})
    if not (DEFAULT_GLOBAL_SCALE_RANGE[0] <= scale <= DEFAULT_GLOBAL_SCALE_RANGE[1]):
        issues.append({"category": "camera_range", "object": "global_scale", "details": f"global_scale={scale} 超出范围。"})
    lower = math.radians(DEFAULT_ELEVATION_DEG_RANGE[0])
    upper = math.radians(DEFAULT_ELEVATION_DEG_RANGE[1])
    if not (lower <= elevation <= upper):
        issues.append({"category": "camera_range", "object": "camera_elevation", "details": f"camera_elevation={elevation} 超出范围。"})
    return issues


def validate_scene(
    scene_dict: Dict[str, Any],
    manifest: Dict[str, Any],
    predicates_payload: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    camera_facing_payload: Optional[Dict[str, Any]] = None,
    default_azimuths_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    if default_azimuths_map is None:
        default_azimuths_map = load_asset_default_azimuths(reference_dims_map.keys())
    if camera_facing_payload is None:
        camera_facing_payload = build_camera_facing_constraints(
            predicates_payload.get("source_text", ""),
            manifest,
            default_azimuths_map,
        )
    object_issues = validate_objects(scene_dict, manifest)
    dim_issues = validate_dimensions(scene_dict, manifest, reference_dims_map)
    azimuth_issues = validate_camera_facing_azimuths(scene_dict, manifest, camera_facing_payload, default_azimuths_map)
    volume_issues = validate_volume(scene_dict, reference_dims_map)
    predicate_issues = validate_predicates(scene_dict, manifest, predicates_payload)
    screen_depth_issues = validate_screen_depth_levels(scene_dict, manifest, predicates_payload)
    screen_lateral_issues = validate_screen_lateral_gaps(scene_dict, manifest, predicates_payload)
    screen_size_issues = validate_screen_size_reasonableness(scene_dict, reference_dims_map, manifest, predicates_payload)
    occlusion_issues = validate_pairwise_screen_occlusion(scene_dict)
    compactness_issues = validate_component_compactness(scene_dict, manifest, predicates_payload, reference_dims_map)
    camera_issues = validate_camera(scene_dict)
    issues = (
        object_issues
        + dim_issues
        + azimuth_issues
        + volume_issues
        + predicate_issues
        + screen_depth_issues
        + screen_lateral_issues
        + screen_size_issues
        + occlusion_issues
        + compactness_issues
        + camera_issues
    )
    issue_counts = Counter(issue.get("category", "other") for issue in issues)
    return {
        "overall_pass": not issues,
        "issue_counts": dict(issue_counts),
        "criteria": {
            "object_exact_match": {"pass": not object_issues, "issues": object_issues},
            "dimension_ranges": {"pass": not dim_issues, "issues": dim_issues},
            "camera_facing_azimuths": {"pass": not azimuth_issues, "issues": azimuth_issues},
            "volume_order": {"pass": not volume_issues, "issues": volume_issues},
            "spatial_predicates": {"pass": not predicate_issues, "issues": predicate_issues},
            "screen_depth_levels": {"pass": not screen_depth_issues, "issues": screen_depth_issues},
            "screen_lateral_gaps": {"pass": not screen_lateral_issues, "issues": screen_lateral_issues},
            "screen_size_reasonableness": {"pass": not screen_size_issues, "issues": screen_size_issues},
            "pairwise_screen_occlusion": {"pass": not occlusion_issues, "issues": occlusion_issues},
            "component_compactness": {"pass": not compactness_issues, "issues": compactness_issues},
            "camera_ranges": {"pass": not camera_issues, "issues": camera_issues},
        },
        "issues": issues,
    }


def build_repair_plan(
    original_scene: Dict[str, Any],
    repaired_scene: Dict[str, Any],
    actions: Sequence[Dict[str, Any]],
    manifest_path: str = "",
    predicates_path: str = "",
    constraints_path: str = "",
) -> Dict[str, Any]:
    return {
        "version": 1,
        "mode": "deterministic_harness",
        "manifest_path": manifest_path,
        "predicates_path": predicates_path,
        "constraints_path": constraints_path,
        "actions": [to_repair_action(action) for action in actions],
        "action_count": len(actions),
    }


def to_repair_action(action: Dict[str, Any]) -> Dict[str, Any]:
    tool = action.get("tool", "set_param")
    if tool in {"delete_object", "create_object"}:
        return action
    field = str(action.get("field", ""))
    obj = str(action.get("object", ""))
    return {
        "tool": "set_param",
        "object": obj,
        "param": field,
        "value": action.get("new"),
        "reason": action.get("reason", ""),
    }


def set_repair_param(scene_dict: Dict[str, Any], obj_name: str, param: str, value: Any, reason: str = "repair_plan") -> Optional[Dict[str, Any]]:
    action_reason = reason or "repair_plan"
    if obj_name == "camera":
        camera = scene_dict.setdefault("camera_data", {})
        key = param.split(".")[-1]
        old = camera.get(key)
        if key == "camera_elevation":
            target_deg = math.degrees(float(value))
            value = math.radians(clamp_camera_elevation_deg(target_deg))
        camera[key] = value
        return action_record("set_param", obj_name, param, old, value, action_reason)
    by_name = subject_by_name(scene_dict)
    subject = by_name.get(obj_name)
    if subject is None:
        return None
    reference_dims_map = load_reference_dims_map()
    manifest_obj = {"type": subject.get("type"), "mention": subject.get("name")}
    if param == "dims":
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"repair_plan 中 {obj_name}.dims 必须是长度为 3 的列表。")
        old_dims = dims(subject)
        new_dims = [float(item) for item in value]
        ratios = [new_dims[idx] / old_dims[idx] for idx in range(3) if abs(old_dims[idx]) > EPS]
        if len(ratios) == 3:
            same_ratio = max(ratios) - min(ratios) <= 1e-3
            all_shrink = all(new_dims[idx] <= old_dims[idx] + EPS for idx in range(3))
            screen_size_compensation = bool(subject.get("_screen_size_dim_compensated")) or action_reason.startswith("screen_size:")
            if all_shrink and not same_ratio and not screen_size_compensation:
                raise ValueError(
                    f"repair_plan 要求缩小 {obj_name}.dims 时必须等比例缩小，old={old_dims}, new={new_dims}。"
                )
        new_dims = clamp_dims_to_reference_safety(new_dims, manifest_obj, reference_dims_map)
        set_dims(subject, new_dims)
        return action_record("set_param", obj_name, param, old_dims, dims(subject), action_reason)
    if param == "_screen_size_dim_compensated":
        old = bool(subject.get("_screen_size_dim_compensated", False))
        subject["_screen_size_dim_compensated"] = bool(value)
        return action_record("set_param", obj_name, param, old, bool(value), action_reason)
    if param.startswith("dims["):
        idx = int(re.search(r"\[(\d+)\]", param).group(1))  # type: ignore[union-attr]
        old_dims = dims(subject)
        new_dims = list(old_dims)
        new_dims[idx] = float(value)
        new_dims = clamp_dims_to_reference_safety(new_dims, manifest_obj, reference_dims_map)
        set_dims(subject, new_dims)
        return action_record("set_param", obj_name, param, old_dims[idx], new_dims[idx], action_reason)
    field = param.split("[", 1)[0]
    if field in {"x", "y", "z", "azimuth"}:
        old = subject_field(subject, field)
        set_subject_field(subject, field, float(value))
        return action_record("set_param", obj_name, param, old, value, action_reason)
    if field in {"name", "type"}:
        old = subject.get(field)
        subject[field] = value
        return action_record("set_param", obj_name, param, old, value, action_reason)
    return None


def apply_repair_plan(scene_dict: Dict[str, Any], repair_plan: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    working_scene = clone_scene(scene_dict)
    actions: List[Dict[str, Any]] = []
    for action in repair_plan.get("actions", []):
        tool = action.get("tool")
        if tool == "set_camera_elevation":
            raw_deg = action.get("camera_elevation_deg")
            if raw_deg is None:
                raw_deg = math.degrees(float(action.get("value", math.radians(DEFAULT_CAMERA_ELEVATION_DEG))))
            record = set_repair_param(
                working_scene,
                "camera",
                "camera_data.camera_elevation",
                math.radians(float(raw_deg)),
                str(action.get("reason", "repair_plan")),
            )
            if record is not None:
                actions.append(record)
        elif tool == "set_param":
            record = set_repair_param(
                working_scene,
                str(action.get("object", "")),
                str(action.get("param", "")),
                action.get("value"),
                str(action.get("reason") or "repair_plan"),
            )
            if record is None:
                raise ValueError(f"repair_plan action 未执行: {action}")
            actions.append(record)
        elif tool == "delete_object":
            target = str(action.get("object", ""))
            old_subjects = subjects(working_scene)
            working_scene["subjects_data"] = [subject for subject in old_subjects if subject.get("name") != target]
            actions.append(action_record("delete_object", target, "subjects_data", target, None, "repair_plan"))
        elif tool == "create_object":
            subject = copy.deepcopy(action.get("subject") or action.get("new"))
            if isinstance(subject, dict):
                subjects(working_scene).append(subject)
                actions.append(action_record("create_object", str(subject.get("name", "")), "subjects_data", None, subject, "repair_plan"))
    return working_scene, actions


def run_harness_on_scene(
    scene_dict: Dict[str, Any],
    scene_text: str,
    allowed_types: Sequence[str],
    reference_dims_map: Dict[str, List[float]],
    objects_spec: str = "",
    object_manifest_path: str = "",
    on_ambiguous: str = "best_effort",
) -> Dict[str, Any]:
    manifest = build_object_manifest(
        scene_text=scene_text,
        allowed_types=allowed_types,
        objects_spec=objects_spec,
        manifest_path=object_manifest_path,
        on_ambiguous=on_ambiguous,
    )
    predicates_payload = extract_predicates(scene_text, manifest)
    default_azimuths_map = load_asset_default_azimuths(allowed_types)
    camera_facing_payload = build_camera_facing_constraints(scene_text, manifest, default_azimuths_map)
    repaired_scene, repair_actions = apply_full_harness(
        scene_dict,
        manifest,
        predicates_payload,
        reference_dims_map,
        camera_facing_payload=camera_facing_payload,
        default_azimuths_map=default_azimuths_map,
    )
    validation = validate_scene(
        repaired_scene,
        manifest,
        predicates_payload,
        reference_dims_map,
        camera_facing_payload=camera_facing_payload,
        default_azimuths_map=default_azimuths_map,
    )
    return {
        "scene": repaired_scene,
        "object_manifest": manifest,
        "predicates": predicates_payload,
        "camera_facing": camera_facing_payload,
        "constraints": {
            "actions": repair_actions,
            "predicate_count": len(predicates_payload.get("predicates", [])),
            "camera_facing": camera_facing_payload.get("camera_facing", []),
        },
        "repair_actions": repair_actions,
        "validation": validation,
        "repair_plan": build_repair_plan(scene_dict, repaired_scene, repair_actions),
    }
