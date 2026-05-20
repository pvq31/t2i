#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审核并修正已有的场景 PKL（单文件入口）。

输入：
1. 空间布局文字（中文或英文）
2. 已有的场景 pkl

流程：
1. 本地读取 pkl，恢复成可读的布局摘要
2. 做一轮确定性的几何检查
3. 调用与 agent_text2pkl.py 相同的 ChatAnywhere Responses API，请大模型严格审核
4. 如果合适，直接输出结论并结束
5. 如果不合适，输出问题、生成修正版，并保存为新的 pkl，不修改原文件
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import http.client
import json
import math
import os
import pickle
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSET_DIMENSIONS_PATH = os.path.join(REPO_ROOT, "inference", "asset_dimensions.json")
OBJECT_SCALES_PATH = os.path.join(REPO_ROOT, "inference", "object_scales.py")

API_HOST = "api.chatanywhere.tech"
API_PATH = "/v1/responses"
API_MODEL = "gpt-5.2"

SCENE_X_SAVE_OFFSET = 6.0
DEFAULT_CAMERA_ELEVATION_DEG = 12.0
MIN_CAMERA_ELEVATION_DEG = 0.0
MAX_CAMERA_ELEVATION_DEG = 90.0
DEFAULT_LENS_MM = 50.0
DEFAULT_GLOBAL_SCALE = 1.0
DEFAULT_REFERENCE_DIM_SCALE = 2.25

CAMERA_CENTER_X = -6.0
CAMERA_RADIUS = 6.0
CAMERA_SENSOR_MM = 36.0
VISIBILITY_MARGIN = 0.05
MIN_CAMERA_LENS_MM = 18.0
MAX_CAMERA_LENS_MM = 60.0
MAX_REPAIR_ATTEMPTS = 4
SIZE_REFERENCE_IGNORE_RATIO = 3.0
GUIDANCE_SIZE_MIN_RATIO = 0.8
GUIDANCE_SIZE_MAX_RATIO = 1.2
EXTREME_SIZE_NORMALIZE_RATIO = 1.7

SUPPORT_SURFACE_TYPES = {
    "desk",
    "table",
    "drawer",
    "bookshelf",
    "bed",
    "refrigerator",
    "oven",
    "microwave",
}
SUPPORT_DIRECT_HINT_TOKENS = (
    "on top of",
    "sits on",
    "sitting on",
    "placed on",
    "positioned on",
    "resting on",
    "rests on",
    "lying on",
    "lie on",
    "atop",
    "桌上",
    "台上",
    "顶上",
    "顶部",
)
SUPPORT_SURFACE_SLOT_TOKENS = (
    "corner of",
    "center of",
    "back-center",
    "back center",
    "front-center",
    "front center",
    "角落",
    "中央",
)
SUPPORT_VERB_HINT_TOKENS = (
    "sit",
    "sits",
    "sitting",
    "stand",
    "stands",
    "standing",
    "rest",
    "rests",
    "resting",
    "place",
    "placed",
    "placing",
    "position",
    "positioned",
    "positioning",
    "lie",
    "lies",
    "lying",
    "放",
    "放置",
    "摆",
    "摆放",
    "坐",
)
SUPPORT_INHERIT_HINT_TOKENS = (
    "front-right",
    "front right",
    "front-left",
    "front left",
    "back-right",
    "back right",
    "back-left",
    "back left",
    "right of",
    "left of",
    "to the right of",
    "to the left of",
    "behind",
    "back of",
    "in front of",
    "front of",
    "前面",
    "后面",
    "左边",
    "右边",
)

RELATION_LEFT_PATTERN = r"(?:to the left of|on the left of|left of|left side of|左边|左侧|左面|左方)"
RELATION_RIGHT_PATTERN = r"(?:to the right of|on the right of|right of|right side of|右边|右侧|右面|右方)"
RELATION_FRONT_PATTERN = r"(?:in front of|front of|ahead of|前面|前方)"
RELATION_BEHIND_PATTERN = r"(?:behind|at the back of|back of|后面|后方|后边)"

CENTER_HINT_TOKENS = (
    "center of the image",
    "in the center",
    "at the center",
    "middle of the image",
    "居中",
    "中央",
    "中间",
    "正中",
)
ENLARGE_HINT_TOKENS = (
    "larger",
    "bigger",
    "increase",
    "enlarge",
    "grow",
    "更大",
    "放大",
    "增大",
)
REDUCE_HINT_TOKENS = (
    "smaller",
    "reduce",
    "decrease",
    "shrink",
    "tiny",
    "更小",
    "缩小",
    "减小",
)
GUIDANCE_MOVE_ACTION_TOKENS = (
    "move",
    "shift",
    "bring",
    "drag",
    "pull",
    "push",
    "keep",
    "stay",
    "移",
    "挪",
    "往",
)
GUIDANCE_MOVE_LEFT_TOKENS = (
    "to the left",
    "move left",
    "shift left",
    "leftward",
    "toward the left",
    "left",
    "向左",
    "左移",
    "往左",
)
GUIDANCE_MOVE_RIGHT_TOKENS = (
    "to the right",
    "move right",
    "shift right",
    "rightward",
    "toward the right",
    "right",
    "向右",
    "右移",
    "往右",
)
GUIDANCE_MOVE_FRONT_TOKENS = (
    "move forward",
    "shift forward",
    "toward the camera",
    "closer to the camera",
    "closer to viewer",
    "forward",
    "front",
    "前移",
    "往前",
    "向前",
)
GUIDANCE_MOVE_BACK_TOKENS = (
    "move backward",
    "move back",
    "shift back",
    "away from the camera",
    "farther from the camera",
    "backward",
    "back",
    "后移",
    "往后",
    "向后",
)
GUIDANCE_MOVE_UP_TOKENS = ("move up", "shift up", "upward", "up", "raise", "lift", "向上", "上移", "往上", "抬高")
GUIDANCE_MOVE_DOWN_TOKENS = ("move down", "shift down", "downward", "down", "lower", "向下", "下移", "往下", "降低")
GUIDANCE_WIDTH_TOKENS = ("width", "wide", "x dimension", "左右宽度", "宽度", "宽")
GUIDANCE_DEPTH_TOKENS = ("depth", "deep", "y dimension", "前后深度", "深度", "深")
GUIDANCE_HEIGHT_TOKENS = ("height", "tall", "z dimension", "vertical dimension", "高度", "高")
GUIDANCE_ALL_DIMENSION_TOKENS = ("all dimensions", "all dims", "all size", "overall size", "所有维度", "整体尺寸")
GUIDANCE_WIDER_TOKENS = ("wider", "widen", "broader", "increase width", "more width", "更宽", "加宽", "变宽")
GUIDANCE_NARROWER_TOKENS = ("narrower", "less wide", "decrease width", "reduce width", "更窄", "变窄", "缩窄")
GUIDANCE_DEEPER_TOKENS = ("deeper", "increase depth", "more depth", "更深", "加深")
GUIDANCE_SHALLOWER_TOKENS = ("shallower", "reduce depth", "less deep", "decrease depth", "更浅", "变浅")
GUIDANCE_TALLER_TOKENS = ("taller", "higher", "more tall", "increase height", "more height", "更高", "加高", "变高")
GUIDANCE_SHORTER_TOKENS = ("shorter", "lower", "less tall", "decrease height", "reduce height", "更矮", "变矮", "降低高度")
GUIDANCE_SLIGHT_TOKENS = ("slightly", "a bit", "a little", "slight", "稍微", "一点")
GUIDANCE_STRONG_TOKENS = ("significantly", "much", "far", "way", "明显", "大幅")
GUIDANCE_ROTATE_ACTION_TOKENS = ("rotate", "turn", "spin", "旋转", "转动")
GUIDANCE_ROTATE_CCW_TOKENS = (
    "counterclockwise",
    "counter-clockwise",
    "anticlockwise",
    "anti-clockwise",
    "逆时针",
)
GUIDANCE_ROTATE_CW_TOKENS = ("clockwise", "顺时针")
ELEVATION_HINT_TOKENS = (
    "in the air",
    "airborne",
    "flying",
    "hover",
    "above",
    "over",
    "on top of",
    "atop",
    "空中",
    "飞",
    "悬浮",
    "上方",
    "上面",
    "顶部",
    "顶上",
    "桌上",
    "台上",
    "move up",
    "move down",
    "shift up",
    "shift down",
    "向上",
    "向下",
    "上移",
    "下移",
)
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

DEFAULT_INFERENCE_PARAMS = {
    "height": 1024,
    "width": 1024,
    "seed": 42,
    "guidance_scale": 3.5,
    "num_inference_steps": 25,
    "checkpoint": "rgb__finetune_1024/epoch-1__checkpoint-5000",
}
DEFAULT_TOP_LEVEL_CHECKPOINT = "seethrough3d_release/seethrough3d_release"

ASSET_TYPE_ALIASES = {
    "adult": "man",
    "automobile": "sedan",
    "backyard chair": "chair",
    "beetle": "vw beetle",
    "bike": "bicycle",
    "biker": "man",
    "bird": "sparrow",
    "car": "sedan",
    "cat": "cat",
    "chair": "chair",
    "coupe car": "coupe",
    "dog": "dog",
    "human": "man",
    "人": "man",
    "男人": "man",
    "女人": "man",
    "kitten": "cat",
    "猫": "cat",
    "motorcycle": "motorbike",
    "摩托车": "motorbike",
    "person": "man",
    "pickup": "pickup truck",
    "pickuptruck": "pickup truck",
    "皮卡": "pickup truck",
    "puppy": "dog",
    "狗": "dog",
    "road bike": "bicycle",
    "自行车": "bicycle",
    "scooter": "scooter",
    "踏板车": "scooter",
    "sports car": "coupe",
    "table": "table",
    "桌子": "table",
    "truck": "pickup truck",
    "卡车": "pickup truck",
    "van": "van",
    "货车": "van",
    "厢式车": "van",
    "汽车": "sedan",
    "轿车": "sedan",
    "椅子": "chair",
    "凳子": "chair",
    "公交车": "bus",
    "巴士": "bus",
    "公交": "bus",
    "拖拉机": "tractor",
    "推土机": "bulldozer",
    "摩托": "motorbike",
    "直升机": "helicopter",
    "鞋": "shoe",
    "熊": "bear",
    "牛": "cow",
    "鹿": "deer",
    "大象": "elephant",
    "长颈鹿": "giraffe",
    "马": "horse",
    "狮子": "lion",
    "猪": "pig",
    "兔子": "rabbit",
    "羊": "sheep",
    "老虎": "tiger",
    "狼": "wolf",
    "斑马": "zebra",
}


def load_asset_dimensions(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(k): [float(x) for x in v] for k, v in data.items()}


def load_object_scales(path: str) -> Dict[str, float]:
    if not os.path.exists(path):
        return {}

    namespace: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as handle:
        code = compile(handle.read(), path, "exec")
        exec(code, namespace)  # noqa: S102

    raw_scales = namespace.get("scales", {})
    if not isinstance(raw_scales, dict):
        return {}
    return {canonicalize_type(str(k)): float(v) for k, v in raw_scales.items()}


def read_api_key_from_example(example_path: str) -> Optional[str]:
    if not os.path.exists(example_path):
        return None
    with open(example_path, "r", encoding="utf-8") as handle:
        content = handle.read()
    match = re.search(r"Bearer\s+([A-Za-z0-9._\-]+)", content)
    if match:
        return match.group(1)
    return None


def load_api_key(explicit_api_key: Optional[str]) -> str:
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()

    for env_name in ("CHATANYWHERE_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value

    fallback = read_api_key_from_example(os.path.join(REPO_ROOT, "chatgpt_api.py"))
    if fallback:
        return fallback

    raise RuntimeError(
        "没有找到可用的 API Key。请通过 --api-key、CHATANYWHERE_API_KEY、OPENAI_API_KEY "
        "或在 chatgpt_api.py 中提供 Bearer Token。"
    )


def call_llm(prompt: str, api_key: str, model: str) -> Dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "input": prompt,
        }
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    conn = http.client.HTTPSConnection(API_HOST)
    try:
        conn.request("POST", API_PATH, payload, headers)
        response = conn.getresponse()
        raw_text = response.read().decode("utf-8")
    finally:
        conn.close()

    if response.status >= 400:
        raise RuntimeError(
            f"API 请求失败，status={response.status}, body={raw_text[:500]}"
        )

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API 返回的不是合法 JSON：{raw_text[:500]}") from exc


def extract_output_text(api_response: Dict[str, Any]) -> str:
    if not isinstance(api_response, dict):
        return str(api_response)

    if isinstance(api_response.get("output_text"), str) and api_response["output_text"].strip():
        return api_response["output_text"]

    texts: List[str] = []
    output = api_response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    texts.append(block["text"])

    if texts:
        return "\n".join(texts)

    choices = api_response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]

    return json.dumps(api_response, ensure_ascii=False)


def extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"模型输出中没有找到 JSON 对象：{text[:500]}")

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"模型输出无法解析为 JSON：{candidate[:500]}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"模型输出不是对象：{candidate[:500]}")
        return parsed


def canonicalize_type(raw_type: str) -> str:
    text = re.sub(r"\s+", " ", raw_type.strip().lower())
    text = text.replace("_", " ").replace("-", " ")
    return text.strip()


def sanitize_name(name: Any, default_name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return default_name
    return re.sub(r"\s+", " ", text)


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def first_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (list, tuple)) and value:
        return to_float(value[0], default)
    return to_float(value, default)


def normalize_dims(value: Any, fallback: List[float]) -> List[float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        dims = [to_float(value[i], fallback[i]) for i in range(3)]
    else:
        dims = [float(fallback[0]), float(fallback[1]), float(fallback[2])]
    return [max(0.01, float(v)) for v in dims]


def dims_ratio_distance(first_dims: List[float], second_dims: List[float]) -> float:
    ratios = []
    for idx in range(3):
        first_value = max(to_float(first_dims[idx], 0.01), 0.01)
        second_value = max(to_float(second_dims[idx], 0.01), 0.01)
        ratios.append(max(first_value / second_value, second_value / first_value))
    return max(ratios) if ratios else 1.0


def reference_dims_is_reliable(current_dims: List[float], reference_dims: List[float]) -> bool:
    return dims_ratio_distance(current_dims, reference_dims) <= SIZE_REFERENCE_IGNORE_RATIO


def pick_reference_dims_for_size_check(
    subject: Dict[str, Any],
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
) -> Optional[List[float]]:
    scaled_reference_dims = reference_dims_map.get(subject["type"])
    if scaled_reference_dims:
        return [float(v) for v in scaled_reference_dims]

    default_dims = default_dims_map.get(subject["type"])
    if default_dims:
        return [float(v) for v in default_dims]
    return None


def find_matching_subject(
    subject: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    target_name = canonicalize_type(str(subject.get("name") or ""))
    target_base_name = re.sub(r"\s+\d+$", "", target_name).strip()
    target_type = canonicalize_type(str(subject.get("type") or ""))

    for candidate in candidates:
        candidate_name = canonicalize_type(str(candidate.get("name") or ""))
        candidate_base_name = re.sub(r"\s+\d+$", "", candidate_name).strip()
        candidate_type = canonicalize_type(str(candidate.get("type") or ""))
        if target_name and candidate_name == target_name:
            return candidate
        if target_base_name and candidate_base_name == target_base_name and candidate_type == target_type:
            return candidate

    for candidate in candidates:
        candidate_type = canonicalize_type(str(candidate.get("type") or ""))
        if candidate_type == target_type:
            return candidate
    return None


def match_asset_type(
    raw_type: Any,
    raw_name: str,
    allowed_types: List[str],
) -> str:
    allowed_lookup = {canonicalize_type(t): t for t in allowed_types}
    candidates: List[str] = []

    if raw_type is not None:
        candidates.append(str(raw_type))
    if raw_name:
        candidates.append(raw_name)
        candidates.extend(re.split(r"[^a-zA-Z ]+", raw_name))

    normalized_candidates = []
    for candidate in candidates:
        normalized = canonicalize_type(candidate)
        if normalized:
            normalized_candidates.append(normalized)

    for normalized in normalized_candidates:
        if normalized in allowed_lookup:
            return allowed_lookup[normalized]
        alias = ASSET_TYPE_ALIASES.get(normalized)
        if alias:
            alias_norm = canonicalize_type(alias)
            if alias_norm in allowed_lookup:
                return allowed_lookup[alias_norm]

    search_space = list(allowed_lookup.keys()) + list(ASSET_TYPE_ALIASES.keys())
    for normalized in normalized_candidates:
        matches = difflib.get_close_matches(normalized, search_space, n=1, cutoff=0.6)
        if not matches:
            continue
        best = matches[0]
        if best in allowed_lookup:
            return allowed_lookup[best]
        alias = ASSET_TYPE_ALIASES.get(best)
        if alias:
            alias_norm = canonicalize_type(alias)
            if alias_norm in allowed_lookup:
                return allowed_lookup[alias_norm]

    return "Custom"


def ensure_unique_names(subjects: List[Dict[str, Any]]) -> None:
    counts: Dict[str, int] = {}
    for subject in subjects:
        name = subject["name"]
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > 1:
            subject["name"] = f"{name} {counts[name]}"


def build_reference_dims_map(
    asset_dimensions: Dict[str, List[float]],
    object_scales: Dict[str, float],
) -> Dict[str, List[float]]:
    reference_dims: Dict[str, List[float]] = {}
    for asset_type, dims in asset_dimensions.items():
        type_key = canonicalize_type(asset_type)
        type_scale = object_scales.get(type_key, 1.0)
        realistic_dims = [
            round(float(dim) * type_scale * DEFAULT_REFERENCE_DIM_SCALE, 6)
            for dim in dims
        ]
        reference_dims[asset_type] = realistic_dims
    return reference_dims


def build_default_dims_map(asset_dimensions: Dict[str, List[float]]) -> Dict[str, List[float]]:
    return {str(k): [float(x) for x in v] for k, v in asset_dimensions.items()}


def load_scene_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        scene_dict = pickle.load(handle)
    if not isinstance(scene_dict, dict):
        raise RuntimeError(f"读取到的 pkl 顶层不是 dict：{type(scene_dict).__name__}")
    return scene_dict


def summarize_scene_subjects(
    scene_dict: Dict[str, Any],
    allowed_types: List[str],
    reference_dims_map: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    subjects_data = scene_dict.get("subjects_data")
    if not isinstance(subjects_data, list):
        raise RuntimeError("场景 pkl 中的 subjects_data 不是列表。")

    subjects: List[Dict[str, Any]] = []
    for idx, raw_subject in enumerate(subjects_data):
        if not isinstance(raw_subject, dict):
            continue

        name = sanitize_name(raw_subject.get("name"), default_name=f"object_{idx + 1}")
        asset_type = match_asset_type(raw_subject.get("type"), name, allowed_types)
        fallback_dims = reference_dims_map.get(asset_type, [1.0, 1.0, 1.0])
        dims = normalize_dims(raw_subject.get("dims"), fallback_dims)
        pkl_x = first_number(raw_subject.get("x"), -SCENE_X_SAVE_OFFSET)
        scene_x = pkl_x + SCENE_X_SAVE_OFFSET
        scene_y = first_number(raw_subject.get("y"), 0.0)
        scene_z = first_number(raw_subject.get("z"), 0.0)
        azimuth_deg = math.degrees(first_number(raw_subject.get("azimuth"), 0.0))

        subjects.append(
            {
                "name": name,
                "type": asset_type,
                "dims": [round(v, 6) for v in dims],
                "pkl_x": round(pkl_x, 6),
                "scene_x": round(scene_x, 6),
                "scene_y": round(scene_y, 6),
                "scene_z": round(scene_z, 6),
                "azimuth_deg": round(azimuth_deg, 6),
            }
        )

    ensure_unique_names(subjects)
    return subjects


def summarize_camera(scene_dict: Dict[str, Any]) -> Dict[str, float]:
    camera_data = scene_dict.get("camera_data")
    if not isinstance(camera_data, dict):
        camera_data = {}

    camera_elevation_deg = math.degrees(
        to_float(camera_data.get("camera_elevation"), math.radians(DEFAULT_CAMERA_ELEVATION_DEG))
    )
    lens_mm = to_float(camera_data.get("lens"), DEFAULT_LENS_MM)
    global_scale = to_float(camera_data.get("global_scale"), DEFAULT_GLOBAL_SCALE)

    return {
        "camera_elevation_deg": round(
            clamp(camera_elevation_deg, MIN_CAMERA_ELEVATION_DEG, MAX_CAMERA_ELEVATION_DEG),
            6,
        ),
        "lens_mm": round(clamp(lens_mm, MIN_CAMERA_LENS_MM, 200.0), 6),
        "global_scale": round(global_scale, 6),
    }


def get_camera_pose(camera_elevation_deg: float) -> Dict[str, Tuple[float, float, float]]:
    elevation_rad = math.radians(camera_elevation_deg)
    cos_elevation = math.cos(elevation_rad)
    sin_elevation = math.sin(elevation_rad)

    camera_position = (
        CAMERA_RADIUS * cos_elevation + CAMERA_CENTER_X,
        0.0,
        CAMERA_RADIUS * sin_elevation,
    )
    forward = (-cos_elevation, 0.0, -sin_elevation)
    right = (0.0, 1.0, 0.0)
    up = (-sin_elevation, 0.0, cos_elevation)

    return {
        "position": camera_position,
        "forward": forward,
        "right": right,
        "up": up,
    }


def dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def build_subject_world_corners(subject: Dict[str, Any]) -> List[Tuple[float, float, float]]:
    width, depth, height = subject["dims"]
    half_width = width / 2.0
    half_depth = depth / 2.0
    half_height = height / 2.0
    azimuth_rad = math.radians(subject["azimuth_deg"])
    cos_a = math.cos(azimuth_rad)
    sin_a = math.sin(azimuth_rad)

    center_x = subject["scene_x"] - SCENE_X_SAVE_OFFSET
    center_y = subject["scene_y"]
    center_z = subject["scene_z"] + half_height

    corners: List[Tuple[float, float, float]] = []
    for local_x in (-half_width, half_width):
        for local_y in (-half_depth, half_depth):
            rotated_x = local_x * cos_a - local_y * sin_a
            rotated_y = local_x * sin_a + local_y * cos_a
            for local_z in (-half_height, half_height):
                corners.append(
                    (
                        center_x + rotated_x,
                        center_y + rotated_y,
                        center_z + local_z,
                    )
                )
    return corners


def project_world_point(
    point: Tuple[float, float, float],
    camera_elevation_deg: float,
    lens_mm: float,
) -> Optional[Dict[str, float]]:
    camera_pose = get_camera_pose(camera_elevation_deg)
    camera_position = camera_pose["position"]
    forward = camera_pose["forward"]
    right = camera_pose["right"]
    up = camera_pose["up"]

    rel = (
        point[0] - camera_position[0],
        point[1] - camera_position[1],
        point[2] - camera_position[2],
    )
    depth = dot(rel, forward)
    if depth <= 1e-6:
        return None

    tan_half_fov = CAMERA_SENSOR_MM / (2.0 * lens_mm)
    x_cam = dot(rel, right)
    y_cam = dot(rel, up)

    ndc_x = x_cam / (depth * tan_half_fov)
    ndc_y = y_cam / (depth * tan_half_fov)
    u = 0.5 * (1.0 + ndc_x)
    v = 0.5 * (1.0 - ndc_y)

    return {
        "u": u,
        "v": v,
        "depth": depth,
        "ndc_x": ndc_x,
        "ndc_y": ndc_y,
    }


def evaluate_subject_visibility(
    subject: Dict[str, Any],
    camera_elevation_deg: float,
    lens_mm: float,
    margin: float = 0.0,
) -> Dict[str, Any]:
    projections: List[Dict[str, float]] = []
    for corner in build_subject_world_corners(subject):
        projected = project_world_point(corner, camera_elevation_deg, lens_mm)
        if projected is None:
            return {
                "name": subject["name"],
                "type": subject["type"],
                "fully_inside": False,
                "fully_inside_margin": False,
                "bbox": None,
                "reason": "有角点落在相机后方。",
            }
        projections.append(projected)

    min_u = min(point["u"] for point in projections)
    max_u = max(point["u"] for point in projections)
    min_v = min(point["v"] for point in projections)
    max_v = max(point["v"] for point in projections)

    fully_inside = min_u >= 0.0 and max_u <= 1.0 and min_v >= 0.0 and max_v <= 1.0
    fully_inside_margin = (
        min_u >= margin
        and max_u <= 1.0 - margin
        and min_v >= margin
        and max_v <= 1.0 - margin
    )

    return {
        "name": subject["name"],
        "type": subject["type"],
        "fully_inside": fully_inside,
        "fully_inside_margin": fully_inside_margin,
        "bbox": {
            "min_u": round(min_u, 6),
            "max_u": round(max_u, 6),
            "min_v": round(min_v, 6),
            "max_v": round(max_v, 6),
        },
        "reason": "" if fully_inside else "cube 没有完整落在画面内部。",
    }


def evaluate_scene_visibility(
    subjects: List[Dict[str, Any]],
    camera_elevation_deg: float,
    lens_mm: float,
    margin: float = 0.0,
) -> Dict[str, Any]:
    subject_results = [
        evaluate_subject_visibility(subject, camera_elevation_deg, lens_mm, margin=margin)
        for subject in subjects
    ]
    visible_results = [result for result in subject_results if isinstance(result.get("bbox"), dict)]

    if visible_results:
        union_bbox = {
            "min_u": min(result["bbox"]["min_u"] for result in visible_results),
            "max_u": max(result["bbox"]["max_u"] for result in visible_results),
            "min_v": min(result["bbox"]["min_v"] for result in visible_results),
            "max_v": max(result["bbox"]["max_v"] for result in visible_results),
        }
    else:
        union_bbox = None

    all_inside = all(result["fully_inside"] for result in subject_results)
    all_inside_margin = all(result["fully_inside_margin"] for result in subject_results)

    return {
        "all_inside": all_inside,
        "all_inside_margin": all_inside_margin,
        "subjects": subject_results,
        "union_bbox": union_bbox,
    }


def run_local_checks(
    subjects: List[Dict[str, Any]],
    camera_elevation_deg: float,
    lens_mm: float,
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
    scene_text: str = "",
) -> Dict[str, Any]:
    hard_issues: List[Dict[str, str]] = []
    soft_issues: List[Dict[str, str]] = []
    support_map = build_support_map(subjects, scene_text) if scene_text.strip() else {}
    allowed_types = sorted(set(reference_dims_map.keys()) | set(default_dims_map.keys()))

    if scene_text.strip() and allowed_types:
        expected_counts, parse_warnings = infer_expected_type_counts(scene_text, allowed_types)
        actual_counts = Counter(subject["type"] for subject in subjects)
        all_types = sorted(set(expected_counts.keys()) | set(actual_counts.keys()))
        for asset_type in all_types:
            expected = int(expected_counts.get(asset_type, 0))
            actual = int(actual_counts.get(asset_type, 0))
            if actual < expected:
                hard_issues.append(
                    {
                        "category": "missing_object",
                        "object": asset_type,
                        "details": f"类型 `{asset_type}` 缺少 {expected - actual} 个，期望数量={expected}，实际数量={actual}。",
                    }
                )
            elif actual > expected:
                hard_issues.append(
                    {
                        "category": "extra_object",
                        "object": asset_type,
                        "details": f"类型 `{asset_type}` 多出 {actual - expected} 个，期望数量={expected}，实际数量={actual}。",
                    }
                )
        for warning in parse_warnings:
            soft_issues.append(
                {
                    "category": "type_count_unverifiable",
                    "object": "",
                    "details": warning,
                }
            )

    if not subjects:
        hard_issues.append(
            {
                "category": "empty_scene",
                "object": "",
                "details": "subjects_data 为空，无法构成有效场景。",
            }
        )

    visibility = evaluate_scene_visibility(subjects, camera_elevation_deg, lens_mm, margin=0.0)
    visibility_margin = evaluate_scene_visibility(
        subjects,
        camera_elevation_deg,
        lens_mm,
        margin=VISIBILITY_MARGIN,
    )

    for subject in subjects:
        dims = subject["dims"]
        if subject["type"] == "Custom":
            hard_issues.append(
                {
                    "category": "unknown_asset_type",
                    "object": subject["name"],
                    "details": "物体类型未能匹配到已知资产，无法保证尺寸与外形合理性。",
                }
            )

        if any(not math.isfinite(v) or v <= 0.0 for v in dims):
            hard_issues.append(
                {
                    "category": "invalid_dims",
                    "object": subject["name"],
                    "details": f"dims 非法: {dims}",
                }
            )

        support_subject = support_map.get(subject["name"])
        if support_subject is not None:
            expected_scene_z = round(support_subject["scene_z"] + support_subject["dims"][2], 6)
            tolerance = max(0.05, dims[2] * 0.2)
            if abs(subject["scene_z"] - expected_scene_z) > tolerance:
                hard_issues.append(
                    {
                        "category": "physical_implausibility",
                        "object": subject["name"],
                        "details": (
                            f"scene_z={subject['scene_z']}，未落在支撑物 {support_subject['name']} 顶部，"
                            f"合理值应接近 {expected_scene_z}。"
                        ),
                    }
                )
        else:
            if subject["scene_z"] < -0.05:
                hard_issues.append(
                    {
                        "category": "below_ground",
                        "object": subject["name"],
                        "details": f"scene_z={subject['scene_z']}，cube 底面落在地面以下。",
                    }
                )
            elif subject["scene_z"] > max(0.2, dims[2] * 0.25):
                soft_issues.append(
                    {
                        "category": "floating_object",
                        "object": subject["name"],
                        "details": f"scene_z={subject['scene_z']}，看起来像漂浮在空中。",
                    }
                )

        reference_dims = pick_reference_dims_for_size_check(
            subject,
            reference_dims_map,
            default_dims_map,
        )
        if reference_dims:
            ratios = [
                dims[idx] / reference_dims[idx] if reference_dims[idx] > 1e-6 else 1.0
                for idx in range(3)
            ]
            if any(ratio < 0.7 or ratio > 1.35 for ratio in ratios):
                hard_issues.append(
                    {
                        "category": "size_implausible",
                        "object": subject["name"],
                        "details": (
                            f"当前 dims={dims} 与 {subject['type']} 的参考尺寸 "
                            f"{reference_dims} 偏差过大。"
                        ),
                    }
                )
            elif any(ratio < 0.85 or ratio > 1.15 for ratio in ratios):
                soft_issues.append(
                    {
                        "category": "size_suspicious",
                        "object": subject["name"],
                        "details": (
                            f"当前 dims={dims} 与 {subject['type']} 的参考尺寸 "
                            f"{reference_dims} 偏差较大。"
                        ),
                    }
                )

    for result in visibility["subjects"]:
        if not result["fully_inside"]:
            hard_issues.append(
                {
                    "category": "frame_clipping",
                    "object": result["name"],
                    "details": result["reason"],
                }
            )
    for result in visibility_margin["subjects"]:
        if not result["fully_inside"]:
            continue
        if not result["fully_inside_margin"]:
            soft_issues.append(
                {
                    "category": "frame_too_tight",
                    "object": result["name"],
                    "details": "cube 虽然还在画面里，但离边缘过近。",
                }
            )

    for idx, first_subject in enumerate(subjects):
        first_reference = pick_reference_dims_for_size_check(
            first_subject,
            reference_dims_map,
            default_dims_map,
        )
        if not first_reference:
            continue
        first_scalar = max(first_reference)
        first_actual = max(first_subject["dims"])

        for second_subject in subjects[idx + 1 :]:
            second_reference = pick_reference_dims_for_size_check(
                second_subject,
                reference_dims_map,
                default_dims_map,
            )
            if not second_reference:
                continue
            second_scalar = max(second_reference)
            second_actual = max(second_subject["dims"])

            if first_scalar < second_scalar * 0.75 and first_actual > second_actual * 1.2:
                soft_issues.append(
                    {
                        "category": "size_order_conflict",
                        "object": f"{first_subject['name']} vs {second_subject['name']}",
                        "details": (
                            f"参考上 {first_subject['type']} 应明显小于 {second_subject['type']}，"
                            "但当前尺寸排序相反。"
                        ),
                    }
                )
            elif second_scalar < first_scalar * 0.75 and second_actual > first_actual * 1.2:
                soft_issues.append(
                    {
                        "category": "size_order_conflict",
                        "object": f"{second_subject['name']} vs {first_subject['name']}",
                        "details": (
                            f"参考上 {second_subject['type']} 应明显小于 {first_subject['type']}，"
                            "但当前尺寸排序相反。"
                        ),
                    }
                )

    return {
        "hard_issues": hard_issues,
        "soft_issues": soft_issues,
        "visibility": visibility,
        "visibility_with_margin": visibility_margin,
    }


def build_relevant_reference_dims(
    current_subjects: List[Dict[str, Any]],
    reference_dims_map: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    relevant: Dict[str, List[float]] = {}
    for subject in current_subjects:
        asset_type = subject["type"]
        if asset_type in reference_dims_map:
            relevant[asset_type] = reference_dims_map[asset_type]
    return relevant


def normalize_free_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


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
    tokens = canonicalize_type(phrase).split()
    if not tokens:
        return ""
    tokens[-1] = singularize_word(tokens[-1])
    return " ".join(tokens).strip()


def build_phrase_mapping_candidates(phrase: str) -> List[str]:
    normalized = canonicalize_type(phrase)
    if not normalized:
        return []

    tokens = normalized.split()
    candidates: List[str] = []
    seen: Set[str] = set()

    def add(candidate: str) -> None:
        normalized_candidate = canonicalize_type(candidate)
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
        mapped_type = match_asset_type(None, candidate, allowed_types)
        if mapped_type != "Custom":
            return mapped_type
    return "Custom"


def parse_quantified_object_mentions(scene_text: str) -> List[Tuple[str, int]]:
    normalized_text = normalize_free_text(scene_text)
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


def infer_expected_type_counts(
    scene_text: str,
    allowed_types: List[str],
) -> Tuple[Counter, List[str]]:
    expected_counts: Counter = Counter()
    parse_warnings: List[str] = []

    quantified_mentions = parse_quantified_object_mentions(scene_text)
    if not quantified_mentions:
        if normalize_free_text(scene_text):
            parse_warnings.append(
                "未按 a/an/明确数字 规则解析到任何对象；只有 a/an 或明确数字标注的名词会被计入，the 不会被当作新物体。"
            )
        return expected_counts, parse_warnings

    for mention, mention_count in quantified_mentions:
        mapped_type = map_mention_to_asset_type(mention, allowed_types)
        if mapped_type == "Custom":
            parse_warnings.append(f"对象短语 `{mention}` 不能稳定映射到已知资产类型。")
            continue
        expected_counts[mapped_type] += mention_count
    return expected_counts, parse_warnings


def text_contains_alias(text: str, alias: str) -> bool:
    normalized_text = normalize_free_text(text)
    normalized_alias = canonicalize_type(alias)
    if not normalized_text or not normalized_alias:
        return False

    if re.search(r"[a-z]", normalized_alias):
        return re.search(
            rf"(?<![a-z]){re.escape(normalized_alias)}(?![a-z])",
            normalized_text,
        ) is not None
    return normalized_alias in normalized_text


def split_text_clauses(text: str) -> List[str]:
    protected_text = re.sub(r"(?<=\d)\.(?=\d)", "<DECIMAL_POINT>", str(text or ""))
    return [
        clause.strip().replace("<DECIMAL_POINT>", ".")
        for clause in re.split(r"[\n\r,，。.!?;；:：]+", protected_text)
        if clause.strip()
    ]


def normalize_guidance_clause_text(clause: str) -> str:
    return re.sub(r"^\s*\d+\s*[\.\):：-]?\s*", "", str(clause or "").strip())


def build_subject_aliases(subject: Dict[str, Any]) -> List[str]:
    aliases = set()

    for raw_value in (subject.get("name"), subject.get("type")):
        normalized = canonicalize_type(str(raw_value or ""))
        if not normalized:
            continue
        aliases.add(normalized)
        aliases.add(re.sub(r"\s+\d+$", "", normalized).strip())
        if " " in normalized:
            for token in normalized.split():
                if len(token) >= 3:
                    aliases.add(token)

    target_type = canonicalize_type(str(subject.get("type") or ""))
    for alias, mapped_type in ASSET_TYPE_ALIASES.items():
        if canonicalize_type(mapped_type) == target_type:
            aliases.add(canonicalize_type(alias))

    aliases.discard("")
    return sorted(aliases, key=len, reverse=True)


def clause_mentions_subject(clause: str, subject: Dict[str, Any]) -> bool:
    return any(text_contains_alias(clause, alias) for alias in build_subject_aliases(subject))


def subject_relevant_clauses(text: str, subject: Dict[str, Any]) -> List[str]:
    return [clause for clause in split_text_clauses(text) if clause_mentions_subject(clause, subject)]


def collect_subject_names_mentioned_in_text(
    subjects: List[Dict[str, Any]],
    text: str,
) -> Set[str]:
    mentioned_names: Set[str] = set()
    if not str(text or "").strip():
        return mentioned_names

    for subject in subjects:
        if subject_relevant_clauses(text, subject):
            mentioned_names.add(subject["name"])
    return mentioned_names


def clause_contains_any(clause: str, tokens: Tuple[str, ...]) -> bool:
    normalized_clause = normalize_free_text(clause)
    return any(token in normalized_clause for token in tokens)


def ordered_relation_in_clause(
    clause: str,
    first_subject: Dict[str, Any],
    second_subject: Dict[str, Any],
    relation_pattern: str,
) -> bool:
    normalized_clause = normalize_free_text(clause)

    def alias_pattern(alias: str) -> str:
        escaped = re.escape(alias)
        if re.search(r"[a-z]", alias):
            return rf"(?<![a-z]){escaped}(?![a-z])"
        return escaped

    for first_alias in build_subject_aliases(first_subject):
        if not text_contains_alias(normalized_clause, first_alias):
            continue
        for second_alias in build_subject_aliases(second_subject):
            if first_alias == second_alias:
                continue
            if not text_contains_alias(normalized_clause, second_alias):
                continue
            first_pattern = alias_pattern(first_alias)
            second_pattern = alias_pattern(second_alias)
            patterns = (
                rf"{first_pattern}.*?{relation_pattern}.*?{second_pattern}",
                rf"{first_pattern}.*?{second_pattern}.*?{relation_pattern}",
            )
            if any(re.search(pattern, normalized_clause) for pattern in patterns):
                return True
    return False


def clause_requests_tight_contact(clause: str) -> bool:
    return clause_contains_any(clause, TIGHT_CONTACT_TOKENS)


def subject_size_multiplier_from_guidance(
    subject: Dict[str, Any],
    guidance_text: str,
) -> float:
    multiplier = 1.0
    for clause in subject_relevant_clauses(guidance_text, subject):
        if clause_requests_specific_dimension_resize(clause):
            continue
        if clause_contains_any(clause, ENLARGE_HINT_TOKENS):
            multiplier *= 1.12
        if clause_contains_any(clause, REDUCE_HINT_TOKENS):
            multiplier *= 0.88
    return clamp(multiplier, GUIDANCE_SIZE_MIN_RATIO, GUIDANCE_SIZE_MAX_RATIO)


def merge_subjects_by_guidance_scope(
    candidate_subjects: List[Dict[str, Any]],
    baseline_subjects: List[Dict[str, Any]],
    guidance_target_names: Set[str],
) -> List[Dict[str, Any]]:
    if not guidance_target_names:
        return clone_subjects(candidate_subjects)

    merged_subjects: List[Dict[str, Any]] = []
    for candidate_subject in candidate_subjects:
        baseline_subject = find_matching_subject(candidate_subject, baseline_subjects)
        if candidate_subject["name"] in guidance_target_names:
            merged_subject = copy.deepcopy(candidate_subject)
            if baseline_subject is not None:
                merged_subject["name"] = baseline_subject["name"]
            merged_subjects.append(merged_subject)
        else:
            merged_subjects.append(copy.deepcopy(baseline_subject or candidate_subject))
    return merged_subjects


def preserve_non_guidance_subjects(
    candidate_subjects: List[Dict[str, Any]],
    baseline_subjects: List[Dict[str, Any]],
    guidance_target_names: Set[str],
) -> List[Dict[str, Any]]:
    if not guidance_target_names:
        return clone_subjects(candidate_subjects)

    preserved_subjects: List[Dict[str, Any]] = []
    for baseline_subject in baseline_subjects:
        if baseline_subject["name"] not in guidance_target_names:
            preserved_subjects.append(copy.deepcopy(baseline_subject))
            continue

        candidate_subject = find_matching_subject(baseline_subject, candidate_subjects)
        preserved_subjects.append(copy.deepcopy(candidate_subject or baseline_subject))

    return preserved_subjects


def clause_requests_guidance_move(
    clause: str,
    direction_tokens: Tuple[str, ...],
) -> bool:
    return clause_contains_any(clause, GUIDANCE_MOVE_ACTION_TOKENS) and clause_contains_any(
        clause,
        direction_tokens,
    )


def guidance_step_multiplier(clause: str) -> float:
    if clause_contains_any(clause, GUIDANCE_STRONG_TOKENS):
        return 1.45
    if clause_contains_any(clause, GUIDANCE_SLIGHT_TOKENS):
        return 0.7
    return 1.0


def extract_rotation_delta_from_clause(clause: str) -> float:
    normalized_clause = normalize_free_text(clause)
    if not normalized_clause or not clause_contains_any(clause, GUIDANCE_ROTATE_ACTION_TOKENS):
        return 0.0

    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:degrees?|degree|deg|°|度)", normalized_clause)
    if not match:
        return 0.0

    angle_deg = abs(to_float(match.group(1), 0.0))
    if angle_deg <= 0.0:
        return 0.0

    is_counterclockwise = any(token in normalized_clause for token in GUIDANCE_ROTATE_CCW_TOKENS)
    is_clockwise = (not is_counterclockwise) and any(
        token in normalized_clause for token in GUIDANCE_ROTATE_CW_TOKENS
    )
    if is_counterclockwise:
        return angle_deg
    if is_clockwise:
        return -angle_deg
    return 0.0


def clause_mentions_dimension(clause: str, dimension_tokens: Tuple[str, ...]) -> bool:
    return clause_contains_any(clause, dimension_tokens)


def clause_requests_dimension_increase(clause: str, dimension_tokens: Tuple[str, ...], direct_tokens: Tuple[str, ...]) -> bool:
    return clause_contains_any(clause, direct_tokens) or (
        clause_mentions_dimension(clause, dimension_tokens)
        and clause_contains_any(clause, ENLARGE_HINT_TOKENS)
    )


def clause_requests_dimension_decrease(clause: str, dimension_tokens: Tuple[str, ...], direct_tokens: Tuple[str, ...]) -> bool:
    return clause_contains_any(clause, direct_tokens) or (
        clause_mentions_dimension(clause, dimension_tokens)
        and clause_contains_any(clause, REDUCE_HINT_TOKENS)
    )


def clause_requests_specific_dimension_resize(clause: str) -> bool:
    return (
        clause_requests_dimension_increase(clause, GUIDANCE_WIDTH_TOKENS, GUIDANCE_WIDER_TOKENS)
        or clause_requests_dimension_decrease(clause, GUIDANCE_WIDTH_TOKENS, GUIDANCE_NARROWER_TOKENS)
        or clause_requests_dimension_increase(clause, GUIDANCE_DEPTH_TOKENS, GUIDANCE_DEEPER_TOKENS)
        or clause_requests_dimension_decrease(clause, GUIDANCE_DEPTH_TOKENS, GUIDANCE_SHALLOWER_TOKENS)
        or clause_requests_dimension_increase(clause, GUIDANCE_HEIGHT_TOKENS, GUIDANCE_TALLER_TOKENS)
        or clause_requests_dimension_decrease(clause, GUIDANCE_HEIGHT_TOKENS, GUIDANCE_SHORTER_TOKENS)
    )


def extract_guidance_numeric_value(clause: str, keywords: Tuple[str, ...]) -> Optional[float]:
    normalized_clause = normalize_free_text(clause)
    for keyword in keywords:
        normalized_keyword = normalize_free_text(keyword)
        match = re.search(
            rf"{re.escape(normalized_keyword)}\s*(?:to|=|:|为|到|设为|设置为|set to|set|equal to)\s*(-?\d+(?:\.\d+)?)",
            normalized_clause,
        )
        if match:
            return to_float(match.group(1), 0.0)
    return None


def extract_guidance_delta_value(clause: str, keywords: Tuple[str, ...]) -> Optional[float]:
    normalized_clause = normalize_free_text(clause)
    delta_capture = r"(-?[0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+(?:\.[0-9]+)?)?|half|one half|a half|one third|a third|one quarter|a quarter|twice|double|triple)"
    for keyword in keywords:
        normalized_keyword = normalize_free_text(keyword)
        patterns = (
            rf"{re.escape(normalized_keyword)}.*?(?:增大|增加|减少|减小|increase|decrease|add|subtract|plus|minus|by|提升|降低)\s*{delta_capture}",
            rf"(?:增大|增加|减少|减小|increase|decrease|add|subtract|plus|minus|提升|降低)\s*{re.escape(normalized_keyword)}\s*{delta_capture}",
            rf"{re.escape(normalized_keyword)}.*?{delta_capture}",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized_clause)
            if not match:
                continue
            delta = parse_guidance_multiplier_text(match.group(1))
            if delta is not None and delta > 0.0:
                return delta
    return None


def parse_guidance_multiplier_text(value_text: str) -> Optional[float]:
    normalized = normalize_free_text(value_text)
    if not normalized:
        return None

    text_mapping = {
        "half": 0.5,
        "one half": 0.5,
        "a half": 0.5,
        "one third": 1.0 / 3.0,
        "a third": 1.0 / 3.0,
        "one quarter": 0.25,
        "a quarter": 0.25,
        "twice": 2.0,
        "double": 2.0,
        "triple": 3.0,
    }
    if normalized in text_mapping:
        return text_mapping[normalized]

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


def extract_guidance_multiplier_value(clause: str, keywords: Tuple[str, ...]) -> Optional[float]:
    normalized_clause = normalize_free_text(clause)
    multiplier_capture = r"([0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+(?:\.[0-9]+)?)?|half|one half|a half|one third|a third|one quarter|a quarter|twice|double|triple)"

    for keyword in keywords:
        normalized_keyword = normalize_free_text(keyword)
        patterns = (
            rf"{re.escape(normalized_keyword)}.*?(?:现在的?|its current value|current value|current)\s*{multiplier_capture}\s*(?:倍|x|times?)?",
            rf"{re.escape(normalized_keyword)}.*?{multiplier_capture}\s*(?:倍|x|times?)\s*(?:of its current value|of current value)?",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized_clause)
            if not match:
                continue
            multiplier = parse_guidance_multiplier_text(match.group(1))
            if multiplier is not None and multiplier > 0.0:
                return multiplier
    return None


def extract_subject_numeric_dimension_multipliers(
    subject: Dict[str, Any],
    guidance_text: str,
) -> Dict[int, float]:
    dim_multipliers: Dict[int, float] = {}
    for clause in subject_relevant_clauses(guidance_text, subject):
        multiplier = extract_guidance_multiplier_value(
            clause,
            build_subject_aliases(subject),
        )
        if multiplier is None:
            continue

        touched = False
        if clause_requests_dimension_increase(clause, GUIDANCE_WIDTH_TOKENS, GUIDANCE_WIDER_TOKENS) or clause_requests_dimension_decrease(
            clause,
            GUIDANCE_WIDTH_TOKENS,
            GUIDANCE_NARROWER_TOKENS,
        ):
            dim_multipliers[0] = multiplier
            touched = True
        if clause_requests_dimension_increase(clause, GUIDANCE_DEPTH_TOKENS, GUIDANCE_DEEPER_TOKENS) or clause_requests_dimension_decrease(
            clause,
            GUIDANCE_DEPTH_TOKENS,
            GUIDANCE_SHALLOWER_TOKENS,
        ):
            dim_multipliers[1] = multiplier
            touched = True
        if clause_requests_dimension_increase(clause, GUIDANCE_HEIGHT_TOKENS, GUIDANCE_TALLER_TOKENS) or clause_requests_dimension_decrease(
            clause,
            GUIDANCE_HEIGHT_TOKENS,
            GUIDANCE_SHORTER_TOKENS,
        ):
            dim_multipliers[2] = multiplier
            touched = True
        if (
            not touched
            and clause_contains_any(clause, GUIDANCE_ALL_DIMENSION_TOKENS)
            and (
                clause_contains_any(clause, ENLARGE_HINT_TOKENS)
                or clause_contains_any(clause, REDUCE_HINT_TOKENS)
            )
        ):
            dim_multipliers[0] = multiplier
            dim_multipliers[1] = multiplier
            dim_multipliers[2] = multiplier

    return dim_multipliers


def compute_target_dims_from_guidance(
    baseline_subject: Dict[str, Any],
    guidance_text: str,
    min_dim_value: float = 1e-6,
) -> Tuple[List[float], bool]:
    baseline_dims = [float(value) for value in baseline_subject["dims"]]
    target_dims = list(baseline_dims)
    changed = False

    for clause in subject_relevant_clauses(guidance_text, baseline_subject):
        clause_numeric_multipliers = extract_subject_numeric_dimension_multipliers(baseline_subject, clause)
        for dim_index, multiplier in clause_numeric_multipliers.items():
            target_dims[dim_index] = round(max(baseline_dims[dim_index] * multiplier, min_dim_value), 6)
            changed = True

        if clause_requests_dimension_increase(clause, GUIDANCE_WIDTH_TOKENS, GUIDANCE_WIDER_TOKENS) and 0 not in clause_numeric_multipliers:
            target_dims[0] = round(max(baseline_dims[0] * 1.18, min_dim_value), 6)
            changed = True
        if clause_requests_dimension_decrease(clause, GUIDANCE_WIDTH_TOKENS, GUIDANCE_NARROWER_TOKENS) and 0 not in clause_numeric_multipliers:
            target_dims[0] = round(max(baseline_dims[0] * 0.86, min_dim_value), 6)
            changed = True

        if clause_requests_dimension_increase(clause, GUIDANCE_DEPTH_TOKENS, GUIDANCE_DEEPER_TOKENS) and 1 not in clause_numeric_multipliers:
            target_dims[1] = round(max(baseline_dims[1] * 1.16, min_dim_value), 6)
            changed = True
        if clause_requests_dimension_decrease(clause, GUIDANCE_DEPTH_TOKENS, GUIDANCE_SHALLOWER_TOKENS) and 1 not in clause_numeric_multipliers:
            target_dims[1] = round(max(baseline_dims[1] * 0.86, min_dim_value), 6)
            changed = True

        if clause_requests_dimension_increase(clause, GUIDANCE_HEIGHT_TOKENS, GUIDANCE_TALLER_TOKENS) and 2 not in clause_numeric_multipliers:
            target_dims[2] = round(max(baseline_dims[2] * 1.14, min_dim_value), 6)
            changed = True
        if clause_requests_dimension_decrease(clause, GUIDANCE_HEIGHT_TOKENS, GUIDANCE_SHORTER_TOKENS) and 2 not in clause_numeric_multipliers:
            target_dims[2] = round(max(baseline_dims[2] * 0.88, min_dim_value), 6)
            changed = True

        if clause_contains_any(clause, GUIDANCE_ALL_DIMENSION_TOKENS) and not clause_numeric_multipliers:
            clause_multiplier = subject_size_multiplier_from_guidance(baseline_subject, clause)
            if abs(clause_multiplier - 1.0) > 1e-6:
                target_dims = [
                    round(max(baseline_dims[idx] * clause_multiplier, min_dim_value), 6)
                    for idx in range(3)
                ]
                changed = True

    return target_dims, changed


def extract_reference_dimension_multiplier(
    clause: str,
    reference_subject: Dict[str, Any],
    dimension_tokens: Tuple[str, ...],
) -> Optional[float]:
    normalized_clause = normalize_free_text(clause)
    number_pattern = r"([0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9]+(?:\.[0-9]+)?)?)"
    dimension_pattern = "|".join(re.escape(token) for token in dimension_tokens)

    for alias in build_subject_aliases(reference_subject):
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
            multiplier = parse_guidance_multiplier_text(match.group(1))
            if multiplier is not None and multiplier > 0.0:
                return multiplier
    return None


def collect_relative_axis_move_instructions(
    subjects: List[Dict[str, Any]],
    guidance_text: str,
    movable_subject_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    instructions: List[Dict[str, Any]] = []
    clauses = split_text_clauses(guidance_text)
    if not clauses:
        return instructions

    for clause in clauses:
        for subject in subjects:
            if movable_subject_names is not None and subject["name"] not in movable_subject_names:
                continue
            if not clause_mentions_subject(clause, subject):
                continue

            for reference_subject in subjects:
                if reference_subject is subject:
                    continue
                if not clause_mentions_subject(clause, reference_subject):
                    continue

                depth_multiplier = extract_reference_dimension_multiplier(clause, reference_subject, REFERENCE_DEPTH_TOKENS)
                width_multiplier = extract_reference_dimension_multiplier(clause, reference_subject, REFERENCE_WIDTH_TOKENS)
                height_multiplier = extract_reference_dimension_multiplier(clause, reference_subject, REFERENCE_HEIGHT_TOKENS)

                if depth_multiplier is not None:
                    if clause_requests_guidance_move(clause, GUIDANCE_MOVE_FRONT_TOKENS):
                        instructions.append({"subject_name": subject["name"], "reference_name": reference_subject["name"], "axis": "scene_x", "dim_index": 1, "sign": 1.0, "multiplier": depth_multiplier})
                    if clause_requests_guidance_move(clause, GUIDANCE_MOVE_BACK_TOKENS):
                        instructions.append({"subject_name": subject["name"], "reference_name": reference_subject["name"], "axis": "scene_x", "dim_index": 1, "sign": -1.0, "multiplier": depth_multiplier})

                if width_multiplier is not None:
                    if clause_requests_guidance_move(clause, GUIDANCE_MOVE_RIGHT_TOKENS):
                        instructions.append({"subject_name": subject["name"], "reference_name": reference_subject["name"], "axis": "scene_y", "dim_index": 0, "sign": 1.0, "multiplier": width_multiplier})
                    if clause_requests_guidance_move(clause, GUIDANCE_MOVE_LEFT_TOKENS):
                        instructions.append({"subject_name": subject["name"], "reference_name": reference_subject["name"], "axis": "scene_y", "dim_index": 0, "sign": -1.0, "multiplier": width_multiplier})

                if height_multiplier is not None:
                    if clause_requests_guidance_move(clause, GUIDANCE_MOVE_UP_TOKENS):
                        instructions.append({"subject_name": subject["name"], "reference_name": reference_subject["name"], "axis": "scene_z", "dim_index": 2, "sign": 1.0, "multiplier": height_multiplier})
                    if clause_requests_guidance_move(clause, GUIDANCE_MOVE_DOWN_TOKENS):
                        instructions.append({"subject_name": subject["name"], "reference_name": reference_subject["name"], "axis": "scene_z", "dim_index": 2, "sign": -1.0, "multiplier": height_multiplier})

    return instructions


def apply_manual_numeric_dimension_guidance(
    subjects: List[Dict[str, Any]],
    baseline_subjects: List[Dict[str, Any]],
    guidance_text: str,
) -> None:
    for subject in subjects:
        baseline_subject = find_matching_subject(subject, baseline_subjects) or subject
        target_dims, changed = compute_target_dims_from_guidance(baseline_subject, guidance_text)
        if not changed:
            continue
        subject["dims"] = [round(max(value, 1e-6), 6) for value in target_dims]


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
        subject = find_matching_subject({"name": instruction["subject_name"]}, subjects)
        reference_subject = find_matching_subject({"name": instruction["reference_name"]}, subjects)
        if subject is None or reference_subject is None:
            continue
        delta = reference_subject["dims"][instruction["dim_index"]] * instruction["multiplier"] * instruction["sign"]
        key = (subject["name"], instruction["axis"])
        axis_totals[key] = axis_totals.get(key, 0.0) + delta

    for subject_name, axis_name in axis_totals:
        subject = find_matching_subject({"name": subject_name}, subjects)
        baseline_subject = find_matching_subject({"name": subject_name}, baseline_subjects)
        if subject is None or baseline_subject is None:
            continue
        subject[axis_name] = round(baseline_subject[axis_name] + axis_totals[(subject_name, axis_name)], 6)


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
    return sanitize_name(" ".join(tokens), default_name="")


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
    candidate = sanitize_name(base_name, default_name="object")
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
    updated_subjects = clone_subjects(subjects)
    affected_names: Set[str] = set()

    for clause in split_text_clauses(guidance_text):
        normalized_clause = normalize_guidance_clause_text(clause)
        if not normalized_clause:
            continue

        add_name = extract_subject_name_after_keywords(normalized_clause, ADD_OBJECT_KEYWORDS)
        if add_name:
            asset_type = match_asset_type(None, add_name, allowed_types)
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
            delete_norm = canonicalize_type(delete_name)
            next_subjects: List[Dict[str, Any]] = []
            removed_names: List[str] = []
            for subject in updated_subjects:
                subject_name_norm = canonicalize_type(subject["name"])
                subject_base_norm = re.sub(r"\s+\d+$", "", subject_name_norm).strip()
                subject_type_norm = canonicalize_type(subject["type"])
                if delete_norm and (
                    subject_name_norm == delete_norm
                    or subject_base_norm == delete_norm
                    or subject_type_norm == delete_norm
                ):
                    removed_names.append(subject["name"])
                    continue
                next_subjects.append(subject)
            if removed_names:
                updated_subjects = next_subjects
                affected_names.update(removed_names)

    return updated_subjects, affected_names


def apply_guidance_camera_directives(
    camera_data: Dict[str, float],
    guidance_text: str,
) -> Dict[str, float]:
    updated_camera = dict(camera_data)
    updated_camera.setdefault("camera_elevation_deg", DEFAULT_CAMERA_ELEVATION_DEG)
    updated_camera.setdefault("lens_mm", DEFAULT_LENS_MM)
    updated_camera.setdefault("global_scale", DEFAULT_GLOBAL_SCALE)

    for clause in split_text_clauses(guidance_text):
        normalized_clause = normalize_free_text(clause)

        targets_global_scale = any(
            token in normalized_clause
            for token in ("global scale", "global_scale", "all cubes", "all cube", "all objects", "整体", "全部cube", "所有cube")
        )
        if targets_global_scale:
            explicit_scale = extract_guidance_numeric_value(
                clause,
                ("global scale", "global_scale", "scale all cubes", "scale all cube", "scale all objects", "整体缩放"),
            )
            multiplier_scale = extract_guidance_multiplier_value(
                clause,
                ("global scale", "global_scale", "scale all cubes", "scale all cube", "scale all objects", "整体缩放"),
            )
            if multiplier_scale is not None:
                updated_camera["global_scale"] = round(
                    clamp(updated_camera["global_scale"] * multiplier_scale, 0.05, 10.0),
                    6,
                )
            elif explicit_scale is not None and explicit_scale > 0.0:
                updated_camera["global_scale"] = round(clamp(explicit_scale, 0.05, 10.0), 6)
            elif any(token in normalized_clause for token in ("scale up", "larger", "increase", "放大", "增大", "增加")):
                updated_camera["global_scale"] = round(clamp(updated_camera["global_scale"] * 1.1, 0.05, 10.0), 6)
            elif any(token in normalized_clause for token in ("scale down", "smaller", "decrease", "reduce", "缩小", "减小", "降低")):
                updated_camera["global_scale"] = round(clamp(updated_camera["global_scale"] * 0.9, 0.05, 10.0), 6)

        targets_elevation = any(
            token in normalized_clause
            for token in ("camera elevation", "camera angle", "higher camera", "lower camera", "higher view", "lower view", "俯仰", "仰角", "相机俯仰", "相机仰角", "相机视角", "视角高度")
        )
        if targets_elevation:
            explicit_elevation = extract_guidance_numeric_value(
                clause,
                ("camera elevation", "camera angle", "elevation", "camera_elevation", "相机俯仰角", "俯仰角", "仰角"),
            )
            multiplier_elevation = extract_guidance_multiplier_value(
                clause,
                ("camera elevation", "camera angle", "elevation", "camera_elevation", "相机俯仰角", "俯仰角", "仰角"),
            )
            if multiplier_elevation is not None:
                updated_camera["camera_elevation_deg"] = round(
                    clamp(
                        updated_camera["camera_elevation_deg"] * multiplier_elevation,
                        MIN_CAMERA_ELEVATION_DEG,
                        MAX_CAMERA_ELEVATION_DEG,
                    ),
                    6,
                )
            elif explicit_elevation is not None:
                updated_camera["camera_elevation_deg"] = round(
                    clamp(explicit_elevation, MIN_CAMERA_ELEVATION_DEG, MAX_CAMERA_ELEVATION_DEG),
                    6,
                )
            elif any(token in normalized_clause for token in ("higher", "increase", "raise", "more top", "更高", "增大", "提高", "增加")):
                updated_camera["camera_elevation_deg"] = round(
                    clamp(
                        updated_camera["camera_elevation_deg"] + 2.0,
                        MIN_CAMERA_ELEVATION_DEG,
                        MAX_CAMERA_ELEVATION_DEG,
                    ),
                    6,
                )
            elif any(token in normalized_clause for token in ("lower", "decrease", "reduce", "更低", "减小", "降低")):
                updated_camera["camera_elevation_deg"] = round(
                    clamp(
                        updated_camera["camera_elevation_deg"] - 2.0,
                        MIN_CAMERA_ELEVATION_DEG,
                        MAX_CAMERA_ELEVATION_DEG,
                    ),
                    6,
                )

        targets_lens = any(
            token in normalized_clause
            for token in ("camera lens", "lens", "zoom", "wider camera view", "wider view", "视野", "焦距", "相机焦距")
        )
        if targets_lens:
            explicit_lens = extract_guidance_numeric_value(
                clause,
                ("camera lens", "lens", "camera_lens", "相机焦距", "焦距"),
            )
            delta_lens = extract_guidance_delta_value(
                clause,
                ("camera lens", "lens", "camera_lens", "相机焦距", "焦距"),
            )
            multiplier_lens = extract_guidance_multiplier_value(
                clause,
                ("camera lens", "lens", "camera_lens", "相机焦距", "焦距"),
            )
            if multiplier_lens is not None:
                updated_camera["lens_mm"] = round(
                    clamp(updated_camera["lens_mm"] * multiplier_lens, MIN_CAMERA_LENS_MM, 200.0),
                    6,
                )
            elif explicit_lens is not None:
                updated_camera["lens_mm"] = round(clamp(explicit_lens, MIN_CAMERA_LENS_MM, 200.0), 6)
            elif delta_lens is not None:
                if any(
                    token in normalized_clause
                    for token in ("减小", "减少", "decrease", "reduce", "lower", "smaller", "zoom out", "拉远", "更广", "窄")
                ):
                    updated_camera["lens_mm"] = round(
                        clamp(updated_camera["lens_mm"] - delta_lens, MIN_CAMERA_LENS_MM, 200.0),
                        6,
                    )
                elif any(
                    token in normalized_clause
                    for token in ("增大", "增加", "increase", "add", "raise", "larger", "zoom in", "拉近", "更窄", "焦距更大")
                ):
                    updated_camera["lens_mm"] = round(
                        clamp(updated_camera["lens_mm"] + delta_lens, MIN_CAMERA_LENS_MM, 200.0),
                        6,
                    )
            elif any(token in normalized_clause for token in ("wider view", "zoom out", "decrease", "reduce", "更广", "拉远", "减小", "降低")):
                updated_camera["lens_mm"] = round(clamp(updated_camera["lens_mm"] - 5.0, MIN_CAMERA_LENS_MM, 200.0), 6)
            elif any(token in normalized_clause for token in ("zoom in", "increase", "larger lens", "拉近", "增大", "增加")):
                updated_camera["lens_mm"] = round(clamp(updated_camera["lens_mm"] + 5.0, MIN_CAMERA_LENS_MM, 200.0), 6)

    return updated_camera


def collect_camera_guidance_application_issues(
    original_camera: Dict[str, float],
    actual_camera: Dict[str, float],
    expected_camera: Dict[str, float],
) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    checks = (
        ("camera_elevation_deg", 1e-4),
        ("lens_mm", 1e-4),
        ("global_scale", 1e-4),
    )
    for key, tolerance in checks:
        original_value = to_float(original_camera.get(key), 0.0)
        expected_value = to_float(expected_camera.get(key), original_value)
        actual_value = to_float(actual_camera.get(key), original_value)
        if abs(expected_value - original_value) <= tolerance:
            continue
        if abs(actual_value - expected_value) > tolerance:
            issues.append(
                {
                    "category": "guidance_not_applied",
                    "object": "camera_data",
                    "details": f"额外指导要求 {key}={expected_value}，但当前为 {actual_value}。",
                }
            )
    return issues


def apply_guidance_directives(
    subjects: List[Dict[str, Any]],
    guidance_text: str,
    baseline_subjects: List[Dict[str, Any]],
    guidance_target_names: Set[str],
) -> None:
    if not guidance_text.strip() or not guidance_target_names:
        return

    for subject in subjects:
        if subject["name"] not in guidance_target_names:
            continue

        baseline_subject = find_matching_subject(subject, baseline_subjects) or subject
        baseline_dims = normalize_dims(baseline_subject.get("dims"), subject["dims"])

        for clause in subject_relevant_clauses(guidance_text, subject):
            move_scale = guidance_step_multiplier(clause)
            lateral_step = round(max(baseline_dims[0] * 0.55, 0.35) * move_scale, 6)
            depth_step = round(max(baseline_dims[1] * 0.75, 0.35) * move_scale, 6)

            if clause_requests_guidance_move(clause, GUIDANCE_MOVE_LEFT_TOKENS):
                subject["scene_y"] = round(subject["scene_y"] - lateral_step, 6)
            if clause_requests_guidance_move(clause, GUIDANCE_MOVE_RIGHT_TOKENS):
                subject["scene_y"] = round(subject["scene_y"] + lateral_step, 6)
            if clause_requests_guidance_move(clause, GUIDANCE_MOVE_FRONT_TOKENS):
                subject["scene_x"] = round(subject["scene_x"] + depth_step, 6)
            if clause_requests_guidance_move(clause, GUIDANCE_MOVE_BACK_TOKENS):
                subject["scene_x"] = round(subject["scene_x"] - depth_step, 6)
            vertical_step = round(max(baseline_dims[2] * 0.5, 0.25) * move_scale, 6)
            if clause_requests_guidance_move(clause, GUIDANCE_MOVE_UP_TOKENS):
                subject["scene_z"] = round(subject["scene_z"] + vertical_step, 6)
            if clause_requests_guidance_move(clause, GUIDANCE_MOVE_DOWN_TOKENS):
                subject["scene_z"] = round(subject["scene_z"] - vertical_step, 6)

            if clause_contains_any(clause, CENTER_HINT_TOKENS):
                subject["scene_y"] = round(subject["scene_y"] * 0.4, 6)

            if clause_requests_dimension_increase(clause, GUIDANCE_WIDTH_TOKENS, GUIDANCE_WIDER_TOKENS):
                subject["dims"][0] = round(max(subject["dims"][0], baseline_dims[0] * 1.18), 6)
            if clause_requests_dimension_decrease(clause, GUIDANCE_WIDTH_TOKENS, GUIDANCE_NARROWER_TOKENS):
                subject["dims"][0] = round(min(subject["dims"][0], baseline_dims[0] * 0.86), 6)
            if clause_requests_dimension_increase(clause, GUIDANCE_DEPTH_TOKENS, GUIDANCE_DEEPER_TOKENS):
                subject["dims"][1] = round(max(subject["dims"][1], baseline_dims[1] * 1.16), 6)
            if clause_requests_dimension_decrease(clause, GUIDANCE_DEPTH_TOKENS, GUIDANCE_SHALLOWER_TOKENS):
                subject["dims"][1] = round(min(subject["dims"][1], baseline_dims[1] * 0.86), 6)
            if clause_requests_dimension_increase(clause, GUIDANCE_HEIGHT_TOKENS, GUIDANCE_TALLER_TOKENS):
                subject["dims"][2] = round(max(subject["dims"][2], baseline_dims[2] * 1.14), 6)
            if clause_requests_dimension_decrease(clause, GUIDANCE_HEIGHT_TOKENS, GUIDANCE_SHORTER_TOKENS):
                subject["dims"][2] = round(min(subject["dims"][2], baseline_dims[2] * 0.88), 6)

            rotation_delta = extract_rotation_delta_from_clause(clause)
            if abs(rotation_delta) > 1e-6:
                subject["azimuth_deg"] = round(subject["azimuth_deg"] + rotation_delta, 6)


def build_guidance_preview_subjects(
    subjects: List[Dict[str, Any]],
    guidance_text: str,
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
    guidance_target_names: Set[str],
) -> List[Dict[str, Any]]:
    preview_subjects = clone_subjects(subjects)
    if not guidance_text.strip() or not guidance_target_names:
        return preview_subjects

    for subject in preview_subjects:
        if subject["name"] not in guidance_target_names:
            continue

        baseline_subject = find_matching_subject(subject, subjects) or subject
        baseline_dims = normalize_dims(
            baseline_subject.get("dims"),
            default_dims_map.get(subject["type"], [1.0, 1.0, 1.0]),
        )
        size_multiplier = subject_size_multiplier_from_guidance(subject, guidance_text)
        if abs(size_multiplier - 1.0) < 1e-6:
            subject["dims"] = [round(max(to_float(v, 0.01), 0.01), 6) for v in baseline_dims]
            continue

        subject["dims"] = [
            round(max(reference_dim * size_multiplier, 0.01), 6)
            for reference_dim in baseline_dims
        ]

    apply_guidance_directives(
        preview_subjects,
        guidance_text=guidance_text,
        baseline_subjects=subjects,
        guidance_target_names=guidance_target_names,
    )
    apply_manual_numeric_dimension_guidance(
        preview_subjects,
        baseline_subjects=subjects,
        guidance_text=guidance_text,
    )
    apply_manual_relation_guidance(
        preview_subjects,
        guidance_text=guidance_text,
        movable_subject_names=guidance_target_names or None,
    )
    apply_manual_relative_axis_movement_guidance(
        preview_subjects,
        baseline_subjects=subjects,
        guidance_text=guidance_text,
        movable_subject_names=guidance_target_names or None,
    )
    return preview_subjects


def collect_guidance_application_issues(
    original_subjects: List[Dict[str, Any]],
    actual_subjects: List[Dict[str, Any]],
    expected_subjects: List[Dict[str, Any]],
    guidance_target_names: Set[str],
) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = collect_structural_guidance_application_issues(
        original_subjects,
        actual_subjects,
        expected_subjects,
    )
    tolerance = 1e-4

    for subject_name in sorted(guidance_target_names):
        original_subject = find_matching_subject({"name": subject_name}, original_subjects)
        actual_subject = find_matching_subject({"name": subject_name}, actual_subjects)
        expected_subject = find_matching_subject({"name": subject_name}, expected_subjects)
        if original_subject is None or actual_subject is None or expected_subject is None:
            continue

        azimuth_expected_changed = abs(expected_subject["azimuth_deg"] - original_subject["azimuth_deg"]) > tolerance
        if azimuth_expected_changed and abs(actual_subject["azimuth_deg"] - expected_subject["azimuth_deg"]) > tolerance:
            issues.append(
                {
                    "category": "guidance_not_applied",
                    "object": subject_name,
                    "details": (
                        f"额外指导要求调整朝向，但当前 azimuth_deg={actual_subject['azimuth_deg']}，"
                        f"预期应为 {expected_subject['azimuth_deg']}。"
                    ),
                }
            )

        dims_expected_changed = any(
            abs(expected_subject["dims"][idx] - original_subject["dims"][idx]) > tolerance
            for idx in range(3)
        )
        if dims_expected_changed and any(
            abs(actual_subject["dims"][idx] - expected_subject["dims"][idx]) > tolerance
            for idx in range(3)
        ):
            issues.append(
                {
                    "category": "guidance_not_applied",
                    "object": subject_name,
                    "details": (
                        f"额外指导要求调整尺寸，但当前 dims={actual_subject['dims']}，"
                        f"预期应为 {expected_subject['dims']}。"
                    ),
                }
            )

        for axis_name in ("scene_x", "scene_y", "scene_z"):
            expected_delta = expected_subject[axis_name] - original_subject[axis_name]
            if abs(expected_delta) <= tolerance:
                continue
            actual_delta = actual_subject[axis_name] - original_subject[axis_name]
            if expected_delta > 0.0 and actual_delta <= tolerance:
                issues.append(
                    {
                        "category": "guidance_not_applied",
                        "object": subject_name,
                        "details": f"额外指导要求增大 {axis_name}，但当前未按要求调整。",
                    }
                )
            elif expected_delta < 0.0 and actual_delta >= -tolerance:
                issues.append(
                    {
                        "category": "guidance_not_applied",
                        "object": subject_name,
                        "details": f"额外指导要求减小 {axis_name}，但当前未按要求调整。",
                    }
                )

    return deduplicate_issues(issues)


def subject_has_extreme_size_outlier(
    dims: List[float],
    reference_dims: Optional[List[float]],
) -> bool:
    if not reference_dims:
        return False
    return dims_ratio_distance(dims, reference_dims) >= EXTREME_SIZE_NORMALIZE_RATIO


def enforce_realistic_sizes(
    subjects: List[Dict[str, Any]],
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
    guidance_text: str,
    original_subjects: Optional[List[Dict[str, Any]]] = None,
    movable_subject_names: Optional[Set[str]] = None,
) -> None:
    baseline_subjects = original_subjects or subjects
    for subject in subjects:
        if movable_subject_names is not None and subject["name"] not in movable_subject_names:
            continue
        baseline_subject = find_matching_subject(subject, baseline_subjects)
        baseline_dims = normalize_dims(
            (baseline_subject or subject).get("dims"),
            default_dims_map.get(subject["type"], [1.0, 1.0, 1.0]),
        )
        size_reference_dims = pick_reference_dims_for_size_check(
            subject,
            reference_dims_map,
            default_dims_map,
        )
        size_multiplier = subject_size_multiplier_from_guidance(subject, guidance_text)

        if abs(size_multiplier - 1.0) < 1e-6:
            if subject_has_extreme_size_outlier(baseline_dims, size_reference_dims):
                subject["dims"] = [
                    round(max(to_float(v, 0.01), 0.01), 6)
                    for v in (size_reference_dims or baseline_dims)
                ]
            else:
                subject["dims"] = [round(max(to_float(v, 0.01), 0.01), 6) for v in baseline_dims]
            continue

        subject["dims"] = [
            round(max(reference_dim * size_multiplier, 0.01), 6)
            for reference_dim in baseline_dims
        ]


def subject_has_elevation_hint(subject: Dict[str, Any], text: str) -> bool:
    return any(
        clause_contains_any(clause, ELEVATION_HINT_TOKENS)
        for clause in subject_relevant_clauses(text, subject)
    )


def clause_implies_direct_support(
    clause: str,
    subject: Dict[str, Any],
    support_subject: Dict[str, Any],
) -> bool:
    if clause_contains_any(clause, SUPPORT_DIRECT_HINT_TOKENS):
        return True

    if subject.get("type") in SUPPORT_SURFACE_TYPES:
        return False

    if not clause_contains_any(clause, SUPPORT_VERB_HINT_TOKENS):
        return False

    if not clause_contains_any(clause, SUPPORT_SURFACE_SLOT_TOKENS):
        return False

    subject_max_dim = max(normalize_dims(subject.get("dims"), [1.0, 1.0, 1.0]))
    support_max_dim = max(normalize_dims(support_subject.get("dims"), [1.0, 1.0, 1.0]))
    return subject_max_dim <= support_max_dim * 0.9


def build_support_map(
    subjects: List[Dict[str, Any]],
    text: str,
) -> Dict[str, Dict[str, Any]]:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return {}

    subject_lookup = {subject["name"]: subject for subject in subjects}
    resolved_names: Dict[str, Optional[str]] = {}

    def resolve(subject: Dict[str, Any]) -> Optional[str]:
        subject_name = subject["name"]
        if subject_name in resolved_names:
            return resolved_names[subject_name]

        resolved_names[subject_name] = None
        clauses = subject_relevant_clauses(normalized_text, subject)

        for clause in clauses:
            for other_subject in subjects:
                if other_subject["name"] == subject_name:
                    continue
                if other_subject.get("type") == subject.get("type"):
                    continue
                if other_subject["type"] not in SUPPORT_SURFACE_TYPES:
                    continue
                if clause_mentions_subject(clause, other_subject):
                    if not clause_implies_direct_support(clause, subject, other_subject):
                        continue
                    resolved_names[subject_name] = other_subject["name"]
                    return other_subject["name"]

        if subject.get("type") not in SUPPORT_SURFACE_TYPES:
            for clause in clauses:
                if not clause_contains_any(clause, SUPPORT_INHERIT_HINT_TOKENS):
                    continue
                for other_subject in subjects:
                    if other_subject["name"] == subject_name:
                        continue
                    if not clause_mentions_subject(clause, other_subject):
                        continue
                    inherited_support_name = resolve(other_subject)
                    if inherited_support_name:
                        resolved_names[subject_name] = inherited_support_name
                        return inherited_support_name

        return resolved_names[subject_name]

    for subject in subjects:
        resolve(subject)

    support_map: Dict[str, Dict[str, Any]] = {}
    for subject_name, support_name in resolved_names.items():
        if support_name and support_name in subject_lookup:
            support_map[subject_name] = subject_lookup[support_name]
    return support_map


def clamp_subject_within_support(subject: Dict[str, Any], support_subject: Dict[str, Any]) -> None:
    max_scene_x_offset = max((support_subject["dims"][1] - subject["dims"][1]) / 2.0 - 0.02, 0.0)
    max_scene_y_offset = max((support_subject["dims"][0] - subject["dims"][0]) / 2.0 - 0.02, 0.0)
    subject["scene_x"] = round(
        clamp(
            subject["scene_x"],
            support_subject["scene_x"] - max_scene_x_offset,
            support_subject["scene_x"] + max_scene_x_offset,
        ),
        6,
    )
    subject["scene_y"] = round(
        clamp(
            subject["scene_y"],
            support_subject["scene_y"] - max_scene_y_offset,
            support_subject["scene_y"] + max_scene_y_offset,
        ),
        6,
    )


def infer_support_surface_offset(
    clause: str,
    support_subject: Dict[str, Any],
    subject: Dict[str, Any],
) -> Optional[Tuple[float, float]]:
    normalized_clause = normalize_free_text(clause)
    max_scene_x_offset = max((support_subject["dims"][1] - subject["dims"][1]) / 2.0 - 0.02, 0.0)
    max_scene_y_offset = max((support_subject["dims"][0] - subject["dims"][0]) / 2.0 - 0.02, 0.0)

    x_sign = 0.0
    y_sign = 0.0
    matched = False

    if "back-center" in normalized_clause or "back center" in normalized_clause:
        x_sign = -1.0
        matched = True
    elif "front-center" in normalized_clause or "front center" in normalized_clause:
        x_sign = 1.0
        matched = True
    elif "back-left" in normalized_clause or "back left" in normalized_clause:
        x_sign = -1.0
        y_sign = -1.0
        matched = True
    elif "back-right" in normalized_clause or "back right" in normalized_clause:
        x_sign = -1.0
        y_sign = 1.0
        matched = True
    elif "front-left" in normalized_clause or "front left" in normalized_clause:
        x_sign = 1.0
        y_sign = -1.0
        matched = True
    elif "front-right" in normalized_clause or "front right" in normalized_clause:
        x_sign = 1.0
        y_sign = 1.0
        matched = True
    elif "center of" in normalized_clause or "center" in normalized_clause or "中央" in normalized_clause:
        matched = True

    if not matched:
        return None

    return (
        round(x_sign * max_scene_x_offset * 0.78, 6),
        round(y_sign * max_scene_y_offset * 0.78, 6),
    )


def enforce_support_surface_layout(
    subjects: List[Dict[str, Any]],
    text: str,
    support_map: Dict[str, Dict[str, Any]],
    movable_subject_names: Optional[Set[str]] = None,
) -> None:
    clauses = split_text_clauses(text)
    if not clauses:
        return

    for subject in subjects:
        if movable_subject_names is not None and subject["name"] not in movable_subject_names:
            continue
        support_subject = support_map.get(subject["name"])
        if support_subject is None:
            continue
        subject["scene_z"] = round(max(support_subject["scene_z"] + support_subject["dims"][2], 0.0), 6)

    for clause in clauses:
        for subject in subjects:
            if movable_subject_names is not None and subject["name"] not in movable_subject_names:
                continue
            support_subject = support_map.get(subject["name"])
            if support_subject is None:
                continue
            if not clause_mentions_subject(clause, subject):
                continue
            if not clause_mentions_subject(clause, support_subject):
                continue

            offset = infer_support_surface_offset(clause, support_subject, subject)
            if offset is None:
                continue

            subject["scene_x"] = round(support_subject["scene_x"] + offset[0], 6)
            subject["scene_y"] = round(support_subject["scene_y"] + offset[1], 6)

    for subject in subjects:
        if movable_subject_names is not None and subject["name"] not in movable_subject_names:
            continue
        support_subject = support_map.get(subject["name"])
        if support_subject is not None:
            clamp_subject_within_support(subject, support_subject)


def enforce_grounding(
    subjects: List[Dict[str, Any]],
    text: str,
    support_map: Optional[Dict[str, Dict[str, Any]]] = None,
    movable_subject_names: Optional[Set[str]] = None,
) -> None:
    resolved_support_map = support_map or build_support_map(subjects, text)
    for subject in subjects:
        if movable_subject_names is not None and subject["name"] not in movable_subject_names:
            continue
        support_subject = resolved_support_map.get(subject["name"])
        if support_subject is not None:
            subject["scene_z"] = round(max(support_subject["scene_z"] + support_subject["dims"][2], 0.0), 6)
            continue
        subject["scene_z"] = round(max(to_float(subject.get("scene_z"), 0.0), 0.0), 6)
        if not subject_has_elevation_hint(subject, text):
            subject["scene_z"] = 0.0


def lateral_gap(
    first_subject: Dict[str, Any],
    second_subject: Dict[str, Any],
    same_support_surface: bool = False,
) -> float:
    minimum_gap = 0.04 if same_support_surface else 0.8
    return round(max((first_subject["dims"][0] + second_subject["dims"][0]) * 0.45, minimum_gap), 6)


def longitudinal_gap(
    first_subject: Dict[str, Any],
    second_subject: Dict[str, Any],
    same_support_surface: bool = False,
) -> float:
    minimum_gap = 0.04 if same_support_surface else 1.0
    return round(max((first_subject["dims"][1] + second_subject["dims"][1]) * 0.6, minimum_gap), 6)


def guidance_requests_pair_relation(
    guidance_text: str,
    first_subject: Dict[str, Any],
    second_subject: Dict[str, Any],
    relation_pattern: str,
) -> bool:
    for clause in split_text_clauses(guidance_text):
        if ordered_relation_in_clause(clause, first_subject, second_subject, relation_pattern):
            return True
    return False


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
    clauses = split_text_clauses(guidance_text)
    if not clauses or not subjects:
        return

    for _ in range(4):
        changed = False
        support_map = build_support_map(subjects, guidance_text)

        for clause in clauses:
            for first_subject in subjects:
                if movable_subject_names is not None and first_subject["name"] not in movable_subject_names:
                    continue
                if not clause_mentions_subject(clause, first_subject):
                    continue

                if clause_contains_any(clause, CENTER_HINT_TOKENS):
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
                    if not clause_mentions_subject(clause, second_subject):
                        continue

                    has_front_back_relation = (
                        guidance_requests_pair_relation(guidance_text, first_subject, second_subject, RELATION_FRONT_PATTERN)
                        or guidance_requests_pair_relation(guidance_text, first_subject, second_subject, RELATION_BEHIND_PATTERN)
                    )
                    has_left_right_relation = (
                        guidance_requests_pair_relation(guidance_text, first_subject, second_subject, RELATION_LEFT_PATTERN)
                        or guidance_requests_pair_relation(guidance_text, first_subject, second_subject, RELATION_RIGHT_PATTERN)
                    )

                    first_support = support_map.get(first_subject["name"])
                    second_support = support_map.get(second_subject["name"])
                    same_support_surface = (
                        first_support is not None
                        and second_support is not None
                        and first_support["name"] == second_support["name"]
                    )
                    lateral = lateral_gap(first_subject, second_subject, same_support_surface=same_support_surface)
                    longitudinal = longitudinal_gap(first_subject, second_subject, same_support_surface=same_support_surface)

                    if ordered_relation_in_clause(clause, first_subject, second_subject, RELATION_LEFT_PATTERN):
                        if (not has_front_back_relation) and abs(first_subject["scene_x"] - second_subject["scene_x"]) > 1e-6:
                            first_subject["scene_x"] = round(second_subject["scene_x"], 6)
                            changed = True
                        target_y = round(second_subject["scene_y"] - lateral, 6)
                        if clause_requests_tight_contact(clause):
                            target_y = round(second_subject["scene_y"] - (second_subject["dims"][0] / 2.0) - (first_subject["dims"][0] / 2.0), 6)
                        if clause_requests_tight_contact(clause):
                            if abs(first_subject["scene_y"] - target_y) > 1e-6:
                                first_subject["scene_y"] = target_y
                                changed = True
                        elif first_subject["scene_y"] > target_y:
                            first_subject["scene_y"] = target_y
                            changed = True

                    if ordered_relation_in_clause(clause, first_subject, second_subject, RELATION_RIGHT_PATTERN):
                        if (not has_front_back_relation) and abs(first_subject["scene_x"] - second_subject["scene_x"]) > 1e-6:
                            first_subject["scene_x"] = round(second_subject["scene_x"], 6)
                            changed = True
                        target_y = round(second_subject["scene_y"] + lateral, 6)
                        if clause_requests_tight_contact(clause):
                            target_y = round(second_subject["scene_y"] + (second_subject["dims"][0] / 2.0) + (first_subject["dims"][0] / 2.0), 6)
                        if clause_requests_tight_contact(clause):
                            if abs(first_subject["scene_y"] - target_y) > 1e-6:
                                first_subject["scene_y"] = target_y
                                changed = True
                        elif first_subject["scene_y"] < target_y:
                            first_subject["scene_y"] = target_y
                            changed = True

                    if ordered_relation_in_clause(clause, first_subject, second_subject, RELATION_FRONT_PATTERN):
                        if (not has_left_right_relation) and abs(first_subject["scene_y"] - second_subject["scene_y"]) > 1e-6:
                            first_subject["scene_y"] = round(second_subject["scene_y"], 6)
                            changed = True
                        target_x = round(second_subject["scene_x"] + longitudinal, 6)
                        if clause_requests_tight_contact(clause):
                            target_x = round(second_subject["scene_x"] + (second_subject["dims"][1] / 2.0) + (first_subject["dims"][1] / 2.0), 6)
                        if clause_requests_tight_contact(clause):
                            if abs(first_subject["scene_x"] - target_x) > 1e-6:
                                first_subject["scene_x"] = target_x
                                changed = True
                        elif first_subject["scene_x"] < target_x:
                            first_subject["scene_x"] = target_x
                            changed = True

                    if ordered_relation_in_clause(clause, first_subject, second_subject, RELATION_BEHIND_PATTERN):
                        if (not has_left_right_relation) and abs(first_subject["scene_y"] - second_subject["scene_y"]) > 1e-6:
                            first_subject["scene_y"] = round(second_subject["scene_y"], 6)
                            changed = True
                        target_x = round(second_subject["scene_x"] - longitudinal, 6)
                        if clause_requests_tight_contact(clause):
                            target_x = round(second_subject["scene_x"] - (second_subject["dims"][1] / 2.0) - (first_subject["dims"][1] / 2.0), 6)
                        if clause_requests_tight_contact(clause):
                            if abs(first_subject["scene_x"] - target_x) > 1e-6:
                                first_subject["scene_x"] = target_x
                                changed = True
                        elif first_subject["scene_x"] > target_x:
                            first_subject["scene_x"] = target_x
                            changed = True

                    if ordered_relation_in_clause(clause, first_subject, second_subject, r"(?:above|higher than|高于|上方)"):
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

                    if ordered_relation_in_clause(clause, first_subject, second_subject, r"(?:below|beneath|lower than|低于|下方)"):
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
                        second_subject["type"] in SUPPORT_SURFACE_TYPES
                        and clause_implies_direct_support(clause, first_subject, second_subject)
                    ):
                        target_z = round(max(second_subject["scene_z"] + second_subject["dims"][2], 0.0), 6)
                        if not clause_requests_tight_contact(clause):
                            target_z = round(target_z + 0.05, 6)
                        if abs(first_subject["scene_z"] - target_z) > 1e-6:
                            first_subject["scene_z"] = target_z
                            changed = True

                        offset = infer_support_surface_offset(clause, second_subject, first_subject)
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
                        clamp_subject_within_support(first_subject, second_subject)

        if not changed:
            break


def collect_structural_guidance_application_issues(
    original_subjects: List[Dict[str, Any]],
    actual_subjects: List[Dict[str, Any]],
    expected_subjects: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    original_names = {subject["name"] for subject in original_subjects}
    actual_names = {subject["name"] for subject in actual_subjects}
    expected_names = {subject["name"] for subject in expected_subjects}

    for subject_name in sorted(expected_names - original_names):
        if subject_name not in actual_names:
            issues.append(
                {
                    "category": "guidance_not_applied",
                    "object": subject_name,
                    "details": f"额外指导要求新增对象 {subject_name}，但当前场景中不存在该对象。",
                }
            )

    for subject_name in sorted(original_names - expected_names):
        if subject_name in actual_names:
            issues.append(
                {
                    "category": "guidance_not_applied",
                    "object": subject_name,
                    "details": f"额外指导要求删除对象 {subject_name}，但当前场景中仍然存在该对象。",
                }
            )

    return deduplicate_issues(issues)


def enforce_text_relations(
    subjects: List[Dict[str, Any]],
    text: str,
    support_map: Optional[Dict[str, Dict[str, Any]]] = None,
    movable_subject_names: Optional[Set[str]] = None,
) -> None:
    clauses = split_text_clauses(text)
    if not clauses or len(subjects) <= 1:
        return

    resolved_support_map = support_map or build_support_map(subjects, text)

    for _ in range(4):
        changed = False

        for clause in clauses:
            for first_subject in subjects:
                if movable_subject_names is not None and first_subject["name"] not in movable_subject_names:
                    continue
                if not clause_mentions_subject(clause, first_subject):
                    continue

                if clause_contains_any(clause, CENTER_HINT_TOKENS):
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
                    if not clause_mentions_subject(clause, second_subject):
                        continue

                    first_support = resolved_support_map.get(first_subject["name"])
                    second_support = resolved_support_map.get(second_subject["name"])
                    if first_support is second_subject:
                        continue

                    same_support_surface = (
                        first_support is not None
                        and second_support is not None
                        and first_support["name"] == second_support["name"]
                    )

                    current_lateral_gap = lateral_gap(
                        first_subject,
                        second_subject,
                        same_support_surface=same_support_surface,
                    )
                    current_longitudinal_gap = longitudinal_gap(
                        first_subject,
                        second_subject,
                        same_support_surface=same_support_surface,
                    )

                    if ordered_relation_in_clause(
                        clause,
                        first_subject,
                        second_subject,
                        RELATION_LEFT_PATTERN,
                    ):
                        target_y = round(second_subject["scene_y"] - current_lateral_gap, 6)
                        if first_subject["scene_y"] > target_y:
                            first_subject["scene_y"] = target_y
                            changed = True

                    if ordered_relation_in_clause(
                        clause,
                        first_subject,
                        second_subject,
                        RELATION_RIGHT_PATTERN,
                    ):
                        target_y = round(second_subject["scene_y"] + current_lateral_gap, 6)
                        if first_subject["scene_y"] < target_y:
                            first_subject["scene_y"] = target_y
                            changed = True

                    if ordered_relation_in_clause(
                        clause,
                        first_subject,
                        second_subject,
                        RELATION_FRONT_PATTERN,
                    ):
                        target_x = round(second_subject["scene_x"] + current_longitudinal_gap, 6)
                        if first_subject["scene_x"] < target_x:
                            first_subject["scene_x"] = target_x
                            changed = True

                    if ordered_relation_in_clause(
                        clause,
                        first_subject,
                        second_subject,
                        RELATION_BEHIND_PATTERN,
                    ):
                        target_x = round(second_subject["scene_x"] - current_longitudinal_gap, 6)
                        if first_subject["scene_x"] > target_x:
                            first_subject["scene_x"] = target_x
                            changed = True

        if not changed:
            break


def apply_strict_subject_constraints(
    subjects: List[Dict[str, Any]],
    scene_text: str,
    guidance_text: str,
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
    original_subjects: Optional[List[Dict[str, Any]]] = None,
    guidance_target_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    constrained_subjects = clone_subjects(subjects)
    baseline_subjects = original_subjects or subjects
    resolved_guidance_target_names = guidance_target_names or collect_subject_names_mentioned_in_text(
        baseline_subjects,
        guidance_text,
    )
    movable_subject_names = resolved_guidance_target_names or None
    if resolved_guidance_target_names:
        constrained_subjects = merge_subjects_by_guidance_scope(
            constrained_subjects,
            baseline_subjects,
            resolved_guidance_target_names,
        )

    combined_text = scene_text.strip()
    if guidance_text.strip():
        combined_text = f"{combined_text}\n{guidance_text.strip()}".strip()

    enforce_realistic_sizes(
        constrained_subjects,
        reference_dims_map,
        default_dims_map,
        guidance_text,
        original_subjects=baseline_subjects,
        movable_subject_names=movable_subject_names,
    )

    for _ in range(2):
        support_map = build_support_map(constrained_subjects, combined_text)
        enforce_grounding(
            constrained_subjects,
            combined_text,
            support_map=support_map,
            movable_subject_names=movable_subject_names,
        )
        enforce_support_surface_layout(
            constrained_subjects,
            combined_text,
            support_map,
            movable_subject_names=movable_subject_names,
        )
        enforce_text_relations(
            constrained_subjects,
            combined_text,
            support_map=support_map,
            movable_subject_names=movable_subject_names,
        )
        resolve_collisions(
            constrained_subjects,
            support_map=support_map,
            max_rounds=24,
            movable_subject_names=movable_subject_names,
        )
        for subject in constrained_subjects:
            if movable_subject_names is not None and subject["name"] not in movable_subject_names:
                continue
            support_subject = support_map.get(subject["name"])
            if support_subject is None:
                continue
            subject["scene_z"] = round(max(support_subject["scene_z"] + support_subject["dims"][2], 0.0), 6)
            clamp_subject_within_support(subject, support_subject)

    apply_guidance_directives(
        constrained_subjects,
        guidance_text=guidance_text,
        baseline_subjects=baseline_subjects,
        guidance_target_names=resolved_guidance_target_names,
    )
    apply_manual_numeric_dimension_guidance(
        constrained_subjects,
        baseline_subjects=baseline_subjects,
        guidance_text=guidance_text,
    )
    final_support_map = build_support_map(constrained_subjects, combined_text)
    enforce_grounding(
        constrained_subjects,
        combined_text,
        support_map=final_support_map,
        movable_subject_names=movable_subject_names,
    )
    enforce_support_surface_layout(
        constrained_subjects,
        combined_text,
        final_support_map,
        movable_subject_names=movable_subject_names,
    )
    resolve_collisions(
        constrained_subjects,
        support_map=final_support_map,
        max_rounds=24,
        movable_subject_names=movable_subject_names,
    )
    for subject in constrained_subjects:
        if movable_subject_names is not None and subject["name"] not in movable_subject_names:
            continue
        support_subject = final_support_map.get(subject["name"])
        if support_subject is None:
            continue
        subject["scene_z"] = round(max(support_subject["scene_z"] + support_subject["dims"][2], 0.0), 6)
        clamp_subject_within_support(subject, support_subject)

    apply_manual_relation_guidance(
        constrained_subjects,
        guidance_text=guidance_text,
        movable_subject_names=movable_subject_names,
    )
    apply_manual_relative_axis_movement_guidance(
        constrained_subjects,
        baseline_subjects=baseline_subjects,
        guidance_text=guidance_text,
        movable_subject_names=movable_subject_names,
    )

    if resolved_guidance_target_names:
        constrained_subjects = preserve_non_guidance_subjects(
            constrained_subjects,
            baseline_subjects,
            resolved_guidance_target_names,
        )

    return constrained_subjects


def deduplicate_issues(issues: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduplicated: List[Dict[str, str]] = []

    for issue in issues:
        key = (
            str(issue.get("category", "")).strip(),
            str(issue.get("object", "")).strip(),
            str(issue.get("details", "")).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(issue)
    return deduplicated


def collect_blocking_issues(local_checks: Dict[str, Any]) -> List[Dict[str, str]]:
    return deduplicate_issues(list(local_checks.get("hard_issues", [])))


def extract_candidate_from_response(
    raw_response: Optional[Dict[str, Any]],
    fallback_camera: Dict[str, float],
    allowed_types: List[str],
    reference_dims_map: Dict[str, List[float]],
) -> Tuple[List[Dict[str, Any]], float, float]:
    payload = raw_response if isinstance(raw_response, dict) else {}
    subjects = normalize_subjects_from_model(
        payload.get("subjects"),
        allowed_types=allowed_types,
        reference_dims_map=reference_dims_map,
    )
    camera_elevation_deg = clamp(
        to_float(payload.get("camera_elevation_deg"), fallback_camera["camera_elevation_deg"]),
        MIN_CAMERA_ELEVATION_DEG,
        MAX_CAMERA_ELEVATION_DEG,
    )
    lens_mm = clamp(
        to_float(payload.get("camera_lens_mm"), fallback_camera["lens_mm"]),
        MIN_CAMERA_LENS_MM,
        200.0,
    )
    return subjects, camera_elevation_deg, lens_mm


def build_audit_prompt(
    scene_text: str,
    guidance_text: str,
    current_subjects: List[Dict[str, Any]],
    camera_data: Dict[str, float],
    local_checks: Dict[str, Any],
    allowed_types: List[str],
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
) -> str:
    payload = {
        "scene_text": scene_text,
        "extra_modification_guidance": guidance_text,
        "coordinate_system": {
            "scene_x": "larger means more in front / closer to camera",
            "scene_y": "smaller means more to the left, larger means more to the right",
            "scene_z": "ground offset; ground objects should usually be 0.0",
            "dims": "full size [width, depth, height], not half extents",
            "azimuth_deg": "object yaw in degrees; counterclockwise rotation is positive, clockwise rotation is negative",
        },
        "current_camera": camera_data,
        "current_subjects": current_subjects,
        "local_deterministic_checks": {
            "hard_issues": local_checks["hard_issues"],
            "soft_issues": local_checks["soft_issues"],
            "visibility": local_checks["visibility"],
        },
        "allowed_asset_types": sorted(allowed_types),
        "scaled_reference_dims_for_current_types": build_relevant_reference_dims(
            current_subjects,
            reference_dims_map,
        ),
        "asset_default_dims_for_current_types": build_relevant_reference_dims(
            current_subjects,
            default_dims_map,
        ),
    }

    return f"""
你是一个极其严格的 3D 场景审核与修正助手。

你的任务是检查“当前 pkl 场景”是否完全符合“空间布局文字”。
文字可能是中文也可能是英文。

判定标准必须全部满足，缺一不可：
1. pkl 中所有 cube 的相对位置关系必须和文字描述完全匹配。
2. pkl 中每个 cube 的位置在物理上必须合理，例如地面物体不能漂浮、不能埋到地下。
3. pkl 中每个 cube 的大小必须符合现实比例，不能出现明显失真，例如狗比汽车大。
3.1 如果“缩放参考尺寸”和“当前物体尺寸”差得特别大，不要机械套用参考尺寸；应优先参考当前图中已经合理的相对大小，只在明确要求改尺寸的物体上改。
4. 图片里必须能看到每个 cube 的完整结构，不能只露出一半。若当前相机或布局会裁边，就必须判为不合适。
5. 如果提供了额外修改指导意见，你必须把它视为必须执行的补充约束，例如位置调整、大小调整、居中与否等。

额外要求：
1. 只有在完全合格时，才能输出 verdict="accept"。
2. 如果任何一条不满足，就输出 verdict="reject"。
3. 如果 reject，必须给出完整的修正版 subjects 列表，而不是只改一部分。
4. 修正版应尽量少改，但不能牺牲正确性。
4.1 如果额外指导只点名了部分物体，则未点名物体默认保持当前 layout 和 dims 不变；优先通过执行指导和调整相机来满足要求。
4.2 如果额外指导要求旋转某个物体，必须通过修改该物体的 azimuth_deg 实现；逆时针旋转 N 度就是 azimuth_deg 加 N，顺时针旋转 N 度就是 azimuth_deg 减 N，不要为了“旋转”去改它的其他参数。
5. 除非文字明确要求物体位于空中/桌上/高处，否则 scene_z 必须为 0.0。
6. camera_lens_mm 可以调整，但请优先用 28-50mm 的合理值；camera_elevation_deg 请保持在 5-25 度左右。
7. 只保留文字中明确提到的物体，不能缺失，也不能额外添加无关物体。
8. 必须严格返回 JSON，不要输出 markdown，不要解释，不要前后加任何多余文字。

返回 JSON schema：
{{
  "verdict": "accept" | "reject",
  "summary": "一句话结论",
  "issues": [
    {{
      "category": "missing_object | extra_object | relation_mismatch | physical_implausibility | size_implausibility | frame_clipping | other",
      "object": "物体名或空字符串",
      "details": "问题说明"
    }}
  ],
  "camera_elevation_deg": 12.0,
  "camera_lens_mm": 50.0,
  "subjects": [
    {{
      "name": "物体名字",
      "type": "资产类型",
      "scene_x": 0.0,
      "scene_y": 0.0,
      "scene_z": 0.0,
      "azimuth_deg": 0.0,
      "dims": [1.0, 1.0, 1.0],
      "reason": "为什么这样改"
    }}
  ]
}}

如果 verdict="accept"：
- issues 必须为空列表
- subjects 必须为空列表
- camera_elevation_deg 和 camera_lens_mm 仍然要给出

以下是输入数据：
{json.dumps(payload, ensure_ascii=False, indent=2)}
    """.strip()


def build_fix_prompt(
    scene_text: str,
    guidance_text: str,
    current_subjects: List[Dict[str, Any]],
    camera_data: Dict[str, float],
    local_checks: Dict[str, Any],
    audit_issues: List[Dict[str, str]],
    allowed_types: List[str],
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
) -> str:
    payload = {
        "scene_text": scene_text,
        "extra_modification_guidance": guidance_text,
        "current_camera": camera_data,
        "current_subjects": current_subjects,
        "known_issues": {
            "hard_issues": local_checks["hard_issues"],
            "soft_issues": local_checks["soft_issues"],
            "audit_issues": audit_issues,
        },
        "allowed_asset_types": sorted(allowed_types),
        "scaled_reference_dims_for_all_types": reference_dims_map,
        "asset_default_dims_for_all_types": default_dims_map,
    }

    return f"""
你需要直接给出一个“严格合格”的修正版 3D 场景方案。

要求：
1. 根据文字描述保留且仅保留被提到的物体。
2. scene_x / scene_y 的空间关系必须和文字完全匹配。
3. ground 物体默认 scene_z=0.0。
4. dims 是完整尺寸 [width, depth, height]，必须符合现实世界比例。
4.1 如果某个物体的缩放参考尺寸与当前场景尺寸差距特别大，不要机械使用参考尺寸；优先保留当前图中合理的相对大小。
5. 所有 cube 都必须完整出现在图中，并且离边缘保留安全余量；必要时可以调整 camera_lens_mm、camera_elevation_deg 或整体布局尺度。
6. 如果提供了额外修改指导意见，必须忠实执行，不能忽略。
6.1 如果额外指导只点名了部分物体，则未点名物体默认保持当前 layout 和 dims 不变；优先调整被点名物体和相机，不要无故改其他物体。
6.2 如果额外指导要求旋转某个物体，必须通过修改该物体的 azimuth_deg 实现；逆时针旋转 N 度就是 azimuth_deg 加 N，顺时针旋转 N 度就是 azimuth_deg 减 N，不要为了“旋转”去改它的 scene_x / scene_y / scene_z / dims。
7. 不能输出“基本合理”或“差不多”，只能输出严格合格的方案。
8. 返回 JSON only。

输出 schema：
{{
  "camera_elevation_deg": 12.0,
  "camera_lens_mm": 50.0,
  "subjects": [
    {{
      "name": "物体名字",
      "type": "资产类型",
      "scene_x": 0.0,
      "scene_y": 0.0,
      "scene_z": 0.0,
      "azimuth_deg": 0.0,
      "dims": [1.0, 1.0, 1.0],
      "reason": "简短理由"
    }}
  ]
}}

输入：
{json.dumps(payload, ensure_ascii=False, indent=2)}
    """.strip()


def normalize_issue_entries(raw_issues: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_issues, list):
        return []

    normalized: List[Dict[str, str]] = []
    for item in raw_issues:
        if isinstance(item, dict):
            normalized.append(
                {
                    "category": str(item.get("category", "other")),
                    "object": str(item.get("object", "")),
                    "details": str(item.get("details", "")),
                }
            )
        else:
            normalized.append(
                {
                    "category": "other",
                    "object": "",
                    "details": str(item),
                }
            )
    return normalized


def normalize_subjects_from_model(
    raw_subjects: Any,
    allowed_types: List[str],
    reference_dims_map: Dict[str, List[float]],
    prefer_reference_dims: bool = False,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_subjects, list):
        return []

    subjects: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_subjects):
        if not isinstance(item, dict):
            continue

        name = sanitize_name(
            item.get("name") or item.get("description"),
            default_name=f"object_{idx + 1}",
        )
        asset_type = match_asset_type(item.get("type"), name, allowed_types)
        fallback_dims = reference_dims_map.get(asset_type, [1.0, 1.0, 1.0])
        if prefer_reference_dims and asset_type in reference_dims_map and asset_type != "Custom":
            dims = [float(v) for v in fallback_dims]
        else:
            dims = normalize_dims(item.get("dims"), fallback_dims)

        subjects.append(
            {
                "name": name,
                "type": asset_type,
                "dims": [round(v, 6) for v in dims],
                "scene_x": round(to_float(item.get("scene_x"), 0.0), 6),
                "scene_y": round(to_float(item.get("scene_y"), 0.0), 6),
                "scene_z": round(to_float(item.get("scene_z"), 0.0), 6),
                "azimuth_deg": round(to_float(item.get("azimuth_deg"), 0.0), 6),
                "reason": str(item.get("reason", "")).strip(),
            }
        )

    ensure_unique_names(subjects)
    return subjects


def build_local_fallback_subjects(
    scene_text: str,
    asset_dimensions: Dict[str, List[float]],
    allowed_types: List[str],
    reference_dims_map: Dict[str, List[float]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    try:
        from agent_text2pkl_v2 import build_local_plan_from_known_types
    except Exception:  # noqa: BLE001
        return [], {}

    local_plan = build_local_plan_from_known_types(scene_text, asset_dimensions)
    if not isinstance(local_plan, dict):
        return [], {}

    subjects = normalize_subjects_from_model(
        local_plan.get("subjects"),
        allowed_types=allowed_types,
        reference_dims_map=reference_dims_map,
        prefer_reference_dims=False,
    )
    if not subjects:
        return [], {}

    camera_data = {
        "camera_elevation_deg": clamp(
            to_float(local_plan.get("camera_elevation_deg"), DEFAULT_CAMERA_ELEVATION_DEG),
            MIN_CAMERA_ELEVATION_DEG,
            MAX_CAMERA_ELEVATION_DEG,
        ),
        "camera_lens_mm": DEFAULT_LENS_MM,
    }
    return subjects, camera_data


def filter_current_subjects_by_expected_counts(
    current_subjects: List[Dict[str, Any]],
    scene_text: str,
    asset_dimensions: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    try:
        from agent_text2pkl_v2 import build_local_plan_from_known_types
    except Exception:  # noqa: BLE001
        return clone_subjects(current_subjects)

    local_plan = build_local_plan_from_known_types(scene_text, asset_dimensions)
    if not isinstance(local_plan, dict):
        return clone_subjects(current_subjects)

    raw_subjects = local_plan.get("subjects")
    if not isinstance(raw_subjects, list):
        return clone_subjects(current_subjects)

    expected_counts: Dict[str, int] = {}
    allowed_types = list(asset_dimensions.keys())
    for raw_subject in raw_subjects:
        if not isinstance(raw_subject, dict):
            continue
        asset_type = match_asset_type(
            raw_subject.get("type"),
            str(raw_subject.get("name") or ""),
            allowed_types,
        )
        expected_counts[asset_type] = expected_counts.get(asset_type, 0) + 1

    if not expected_counts:
        return clone_subjects(current_subjects)

    used_counts: Dict[str, int] = {}
    filtered_subjects: List[Dict[str, Any]] = []
    for subject in current_subjects:
        asset_type = subject["type"]
        if used_counts.get(asset_type, 0) >= expected_counts.get(asset_type, 0):
            continue
        filtered_subjects.append(copy.deepcopy(subject))
        used_counts[asset_type] = used_counts.get(asset_type, 0) + 1

    return filtered_subjects or clone_subjects(current_subjects)


def subjects_vertical_overlap(first_subject: Dict[str, Any], second_subject: Dict[str, Any]) -> bool:
    first_bottom = to_float(first_subject.get("scene_z"), 0.0)
    first_top = first_bottom + max(to_float(first_subject["dims"][2], 0.01), 0.01)
    second_bottom = to_float(second_subject.get("scene_z"), 0.0)
    second_top = second_bottom + max(to_float(second_subject["dims"][2], 0.01), 0.01)
    return min(first_top, second_top) > max(first_bottom, second_bottom) + 0.02


def resolve_collisions(
    subjects: List[Dict[str, Any]],
    support_map: Optional[Dict[str, Dict[str, Any]]] = None,
    max_rounds: int = 16,
    movable_subject_names: Optional[Set[str]] = None,
) -> None:
    resolved_support_map = support_map or {}
    for _ in range(max_rounds):
        changed = False
        for i in range(len(subjects)):
            for j in range(i + 1, len(subjects)):
                first_subject = subjects[i]
                second_subject = subjects[j]

                if resolved_support_map.get(first_subject["name"]) is second_subject:
                    continue
                if resolved_support_map.get(second_subject["name"]) is first_subject:
                    continue
                if not subjects_vertical_overlap(first_subject, second_subject):
                    continue

                dx = second_subject["scene_x"] - first_subject["scene_x"]
                dy = second_subject["scene_y"] - first_subject["scene_y"]

                depth_gap = (first_subject["dims"][1] + second_subject["dims"][1]) * 0.55
                width_gap = (first_subject["dims"][0] + second_subject["dims"][0]) * 0.55

                overlap_x = abs(dx) < depth_gap
                overlap_y = abs(dy) < width_gap
                if not (overlap_x and overlap_y):
                    continue

                first_movable = (
                    movable_subject_names is None or first_subject["name"] in movable_subject_names
                )
                second_movable = (
                    movable_subject_names is None or second_subject["name"] in movable_subject_names
                )
                if not first_movable and not second_movable:
                    continue

                changed = True
                target_subject = second_subject if second_movable else first_subject
                sign_x = 1.0 if dx >= 0 else -1.0
                sign_y = 1.0 if dy >= 0 else -1.0
                if target_subject is first_subject:
                    sign_x *= -1.0
                    sign_y *= -1.0

                if abs(dx) >= abs(dy):
                    push = (depth_gap - abs(dx)) + 0.05
                    if abs(dx) < 1e-6:
                        sign_x = 1.0 if target_subject is second_subject else -1.0
                    target_subject["scene_x"] += push * sign_x
                else:
                    push = (width_gap - abs(dy)) + 0.05
                    if abs(dy) < 1e-6:
                        sign_y = 1.0 if target_subject is second_subject else -1.0
                    target_subject["scene_y"] += push * sign_y

        if not changed:
            break


def clone_subjects(subjects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return copy.deepcopy(subjects)


def apply_layout_transform(
    subjects: List[Dict[str, Any]],
    layout_scale: float,
    scene_x_shift: float,
    scene_y_shift: float,
    movable_subject_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    transformed = clone_subjects(subjects)
    if not transformed:
        return transformed

    movable_subjects = transformed
    if movable_subject_names is not None:
        movable_subjects = [
            subject for subject in transformed if subject["name"] in movable_subject_names
        ]
        if not movable_subjects:
            return transformed

    mean_x = sum(subject["scene_x"] for subject in movable_subjects) / len(movable_subjects)
    mean_y = sum(subject["scene_y"] for subject in movable_subjects) / len(movable_subjects)

    for subject in transformed:
        if movable_subject_names is not None and subject["name"] not in movable_subject_names:
            continue
        subject["scene_x"] = mean_x + (subject["scene_x"] - mean_x) * layout_scale + scene_x_shift
        subject["scene_y"] = mean_y + (subject["scene_y"] - mean_y) * layout_scale + scene_y_shift
        subject["scene_x"] = round(subject["scene_x"], 6)
        subject["scene_y"] = round(subject["scene_y"], 6)

    return transformed


def evaluate_fit_quality(
    subjects: List[Dict[str, Any]],
    camera_elevation_deg: float,
    lens_mm: float,
    margin: float,
) -> Dict[str, Any]:
    visibility = evaluate_scene_visibility(subjects, camera_elevation_deg, lens_mm, margin=margin)
    union_bbox = visibility["union_bbox"]
    if union_bbox is None:
        return {
            "passed": False,
            "overflow": 1e9,
            "visibility": visibility,
        }

    overflow = 0.0
    overflow += max(margin - union_bbox["min_u"], 0.0)
    overflow += max(union_bbox["max_u"] - (1.0 - margin), 0.0)
    overflow += max(margin - union_bbox["min_v"], 0.0)
    overflow += max(union_bbox["max_v"] - (1.0 - margin), 0.0)
    overflow += sum(
        1.0 for result in visibility["subjects"] if not result["fully_inside_margin"]
    )
    overflow += sum(
        5.0 for result in visibility["subjects"] if result["bbox"] is None
    )

    return {
        "passed": visibility["all_inside_margin"],
        "overflow": round(overflow, 9),
        "visibility": visibility,
    }


def optimize_scene_y_shift(
    subjects: List[Dict[str, Any]],
    camera_elevation_deg: float,
    lens_mm: float,
    layout_scale: float,
    scene_x_shift: float,
    movable_subject_names: Optional[Set[str]] = None,
) -> float:
    low = -10.0
    high = 10.0
    best_shift = 0.0
    best_error = float("inf")

    for _ in range(32):
        mid = (low + high) / 2.0
        transformed = apply_layout_transform(
            subjects,
            layout_scale=layout_scale,
            scene_x_shift=scene_x_shift,
            scene_y_shift=mid,
            movable_subject_names=movable_subject_names,
        )
        visibility = evaluate_scene_visibility(transformed, camera_elevation_deg, lens_mm, margin=0.0)
        union_bbox = visibility["union_bbox"]
        if union_bbox is None:
            return 0.0

        center_u = (union_bbox["min_u"] + union_bbox["max_u"]) / 2.0
        error = abs(center_u - 0.5)
        if error < best_error:
            best_error = error
            best_shift = mid

        if center_u < 0.5:
            low = mid
        else:
            high = mid

    return best_shift


def fit_scene_to_camera(
    subjects: List[Dict[str, Any]],
    camera_elevation_deg: float,
    lens_mm: float,
    movable_subject_names: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    if not subjects:
        return {
            "subjects": [],
            "camera_elevation_deg": camera_elevation_deg,
            "lens_mm": lens_mm,
            "passed": True,
            "visibility": evaluate_scene_visibility([], camera_elevation_deg, lens_mm, margin=VISIBILITY_MARGIN),
        }

    candidate_elevations = sorted(
        {
            round(clamp(camera_elevation_deg, 5.0, 25.0), 6),
            8.0,
            10.0,
            12.0,
            15.0,
            18.0,
            20.0,
        }
    )
    candidate_lenses = sorted(
        {
            round(clamp(lens_mm, MIN_CAMERA_LENS_MM, MAX_CAMERA_LENS_MM), 6),
            50.0,
            45.0,
            40.0,
            35.0,
            30.0,
            28.0,
            26.0,
            24.0,
            22.0,
            20.0,
            18.0,
        },
        reverse=True,
    )
    candidate_scales = [1.0, 0.94, 0.88, 0.82, 0.76, 0.7, 0.64, 0.58, 0.52]
    candidate_x_shifts = [0.6, 0.3, 0.0, -0.3, -0.6, -1.0, -1.4, -1.8, -2.2, -2.6, -3.0, -3.5, -4.0]

    best_solution: Optional[Dict[str, Any]] = None

    for candidate_elevation in candidate_elevations:
        for candidate_lens in candidate_lenses:
            for layout_scale in candidate_scales:
                for scene_x_shift in candidate_x_shifts:
                    scene_y_shift = optimize_scene_y_shift(
                        subjects,
                        camera_elevation_deg=candidate_elevation,
                        lens_mm=candidate_lens,
                        layout_scale=layout_scale,
                        scene_x_shift=scene_x_shift,
                        movable_subject_names=movable_subject_names,
                    )
                    transformed = apply_layout_transform(
                        subjects,
                        layout_scale=layout_scale,
                        scene_x_shift=scene_x_shift,
                        scene_y_shift=scene_y_shift,
                        movable_subject_names=movable_subject_names,
                    )
                    fit_quality = evaluate_fit_quality(
                        transformed,
                        camera_elevation_deg=candidate_elevation,
                        lens_mm=candidate_lens,
                        margin=VISIBILITY_MARGIN,
                    )

                    score = fit_quality["overflow"] * 10000.0
                    score += abs(candidate_lens - lens_mm) * 0.2
                    score += abs(candidate_elevation - camera_elevation_deg) * 0.25
                    score += abs(1.0 - layout_scale) * 8.0
                    score += abs(scene_x_shift) * 1.0
                    score += abs(scene_y_shift) * 0.35

                    candidate_solution = {
                        "subjects": transformed,
                        "camera_elevation_deg": round(candidate_elevation, 6),
                        "lens_mm": round(candidate_lens, 6),
                        "passed": fit_quality["passed"],
                        "score": round(score, 9),
                        "visibility": fit_quality["visibility"],
                        "transform": {
                            "layout_scale": round(layout_scale, 6),
                            "scene_x_shift": round(scene_x_shift, 6),
                            "scene_y_shift": round(scene_y_shift, 6),
                        },
                    }

                    if best_solution is None or candidate_solution["score"] < best_solution["score"]:
                        best_solution = candidate_solution

                    if fit_quality["passed"]:
                        return candidate_solution

    if best_solution is None:
        best_solution = {
            "subjects": clone_subjects(subjects),
            "camera_elevation_deg": round(camera_elevation_deg, 6),
            "lens_mm": round(lens_mm, 6),
            "passed": False,
            "score": float("inf"),
            "visibility": evaluate_scene_visibility(
                subjects,
                camera_elevation_deg,
                lens_mm,
                margin=VISIBILITY_MARGIN,
            ),
            "transform": {
                "layout_scale": 1.0,
                "scene_x_shift": 0.0,
                "scene_y_shift": 0.0,
            },
        }
    return best_solution


def build_scene_dict_from_subjects(
    base_scene_dict: Dict[str, Any],
    subjects: List[Dict[str, Any]],
    camera_elevation_deg: float,
    lens_mm: float,
    scene_text: str,
    global_scale: Optional[float] = None,
) -> Dict[str, Any]:
    scene_dict = copy.deepcopy(base_scene_dict)
    scene_dict["subjects_data"] = []

    for subject in subjects:
        subject_dict = {
            "name": subject["name"],
            "type": subject["type"],
            "dims": tuple(round(v, 6) for v in subject["dims"]),
            "x": [round(subject["scene_x"] - SCENE_X_SAVE_OFFSET, 6)],
            "y": [round(subject["scene_y"], 6)],
            "z": [round(subject["scene_z"], 6)],
            "azimuth": [math.radians(subject["azimuth_deg"])],
            "bbox": [(0, 0, 0, 0)],
        }
        scene_dict["subjects_data"].append(subject_dict)

    base_camera_data = scene_dict.get("camera_data")
    if not isinstance(base_camera_data, dict):
        base_camera_data = {}

    resolved_global_scale = (
        to_float(global_scale, DEFAULT_GLOBAL_SCALE)
        if global_scale is not None
        else to_float(base_camera_data.get("global_scale"), DEFAULT_GLOBAL_SCALE)
    )
    scene_dict["camera_data"] = {
        "camera_elevation": math.radians(
            clamp(camera_elevation_deg, MIN_CAMERA_ELEVATION_DEG, MAX_CAMERA_ELEVATION_DEG)
        ),
        "lens": clamp(lens_mm, MIN_CAMERA_LENS_MM, 200.0),
        "global_scale": round(clamp(resolved_global_scale, 0.05, 10.0), 6),
    }
    scene_dict["surrounding_prompt"] = str(scene_dict.get("surrounding_prompt") or scene_text)

    inference_params = scene_dict.get("inference_params")
    if not isinstance(inference_params, dict):
        inference_params = {}
    merged_inference_params = dict(DEFAULT_INFERENCE_PARAMS)
    merged_inference_params.update(inference_params)
    scene_dict["inference_params"] = merged_inference_params

    if "checkpoint" not in scene_dict:
        scene_dict["checkpoint"] = DEFAULT_TOP_LEVEL_CHECKPOINT

    return scene_dict


def save_scene_pkl(scene_dict: Dict[str, Any], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as handle:
        pickle.dump(scene_dict, handle)


def call_llm_for_json(prompt: str, api_key: str, model: str) -> Dict[str, Any]:
    api_response = call_llm(prompt, api_key=api_key, model=model)
    output_text = extract_output_text(api_response)
    return extract_json_object(output_text)


def safe_call_llm_for_json(
    prompt: str,
    api_key: str,
    model: str,
) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        return call_llm_for_json(prompt, api_key=api_key, model=model), None
    except Exception as exc:  # noqa: BLE001
        return {}, str(exc)


def request_correction_plan(
    scene_text: str,
    guidance_text: str,
    current_subjects: List[Dict[str, Any]],
    camera_data: Dict[str, float],
    local_checks: Dict[str, Any],
    audit_issues: List[Dict[str, str]],
    allowed_types: List[str],
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
    api_key: str,
    model: str,
) -> Dict[str, Any]:
    fix_prompt = build_fix_prompt(
        scene_text=scene_text,
        guidance_text=guidance_text,
        current_subjects=current_subjects,
        camera_data=camera_data,
        local_checks=local_checks,
        audit_issues=audit_issues,
        allowed_types=allowed_types,
        reference_dims_map=reference_dims_map,
        default_dims_map=default_dims_map,
    )
    return call_llm_for_json(fix_prompt, api_key=api_key, model=model)


def format_issue_lines(issues: List[Dict[str, str]]) -> List[str]:
    lines: List[str] = []
    for idx, issue in enumerate(issues, start=1):
        object_name = issue.get("object", "").strip()
        object_prefix = f"[{object_name}] " if object_name else ""
        lines.append(f"{idx}. {object_prefix}{issue.get('details', '').strip()}")
    return lines


def first_pkl_value(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return value[0]
    return value if value is not None else default


def values_differ(first_value: Any, second_value: Any, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(first_value) - float(second_value)) > tolerance
    except (TypeError, ValueError):
        return first_value != second_value


def format_pkl_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (list, tuple)):
        formatted_values = [format_pkl_value(item) for item in value]
        bracket_left, bracket_right = ("[", "]") if isinstance(value, list) else ("(", ")")
        return f"{bracket_left}{', '.join(formatted_values)}{bracket_right}"
    return str(value)


def describe_numeric_change(parameter_name: str, old_value: Any, new_value: Any) -> str:
    try:
        old_number = float(old_value)
        new_number = float(new_value)
    except (TypeError, ValueError):
        return "改变" if old_value != new_value else "未改变"

    delta = new_number - old_number
    if abs(delta) <= 1e-6:
        return "未改变"

    base_change = "变大" if delta > 0 else "变小"
    lower_name = parameter_name.lower()
    extra = ""
    if lower_name.endswith(".x") or ".x" in lower_name:
        extra = "，向前" if delta > 0 else "，向后"
    elif lower_name.endswith(".y") or ".y" in lower_name:
        extra = "，向右" if delta > 0 else "，向左"
    elif lower_name.endswith(".z") or ".z" in lower_name:
        extra = "，向上" if delta > 0 else "，向下"
    elif "azimuth" in lower_name:
        delta_deg = math.degrees(delta)
        direction = "逆时针" if delta > 0 else "顺时针"
        extra = f"，{direction} {abs(delta_deg):.2f}°"
    elif "camera_elevation" in lower_name:
        extra = "，相机更俯视" if delta > 0 else "，相机更接近水平"
    elif "global_scale" in lower_name:
        extra = "，整体放大" if delta > 0 else "，整体缩小"
    elif "lens" in lower_name:
        extra = "，视野变窄/更拉近" if delta > 0 else "，视野变广/更拉远"

    return f"{base_change}{extra}（delta={delta:.6f}）"


def subject_diff_key(subject: Dict[str, Any], index: int) -> Tuple[str, str, int]:
    return (
        canonicalize_type(str(subject.get("name") or "")),
        canonicalize_type(str(subject.get("type") or "")),
        index,
    )


def build_subject_diff_lookup(subjects: List[Any]) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    type_counts: Dict[Tuple[str, str], int] = {}
    for subject in subjects:
        if not isinstance(subject, dict):
            continue
        name = canonicalize_type(str(subject.get("name") or ""))
        asset_type = canonicalize_type(str(subject.get("type") or ""))
        count_key = (name, asset_type)
        occurrence = type_counts.get(count_key, 0)
        type_counts[count_key] = occurrence + 1
        lookup[(name, asset_type, occurrence)] = subject
    return lookup


def collect_scene_parameter_changes(
    original_scene_dict: Dict[str, Any],
    modified_scene_dict: Dict[str, Any],
) -> List[Tuple[str, str, str, str]]:
    changes: List[Tuple[str, str, str, str]] = []

    original_subjects = original_scene_dict.get("subjects_data", [])
    modified_subjects = modified_scene_dict.get("subjects_data", [])
    if not isinstance(original_subjects, list):
        original_subjects = []
    if not isinstance(modified_subjects, list):
        modified_subjects = []

    original_lookup = build_subject_diff_lookup(original_subjects)
    modified_lookup = build_subject_diff_lookup(modified_subjects)
    all_subject_keys = sorted(set(original_lookup) | set(modified_lookup))

    dim_names = ["width", "depth", "height"]
    scalar_fields = ("x", "y", "z", "azimuth")

    for key in all_subject_keys:
        original_subject = original_lookup.get(key)
        modified_subject = modified_lookup.get(key)
        display_name = key[0] or key[1] or f"subject_{key[2]}"
        if original_subject is None:
            changes.append((f"subjects_data[{display_name}]", "不存在", format_pkl_value(modified_subject), "新增"))
            continue
        if modified_subject is None:
            changes.append((f"subjects_data[{display_name}]", format_pkl_value(original_subject), "不存在", "删除"))
            continue

        for field in ("name", "type"):
            old_value = original_subject.get(field)
            new_value = modified_subject.get(field)
            if values_differ(old_value, new_value):
                parameter_name = f"{display_name}.{field}"
                changes.append((parameter_name, format_pkl_value(old_value), format_pkl_value(new_value), "改变"))

        old_dims = original_subject.get("dims", [])
        new_dims = modified_subject.get("dims", [])
        for idx, dim_name in enumerate(dim_names):
            old_value = old_dims[idx] if isinstance(old_dims, (list, tuple)) and len(old_dims) > idx else None
            new_value = new_dims[idx] if isinstance(new_dims, (list, tuple)) and len(new_dims) > idx else None
            if values_differ(old_value, new_value):
                parameter_name = f"{display_name}.dims[{idx}]/{dim_name}"
                changes.append((
                    parameter_name,
                    format_pkl_value(old_value),
                    format_pkl_value(new_value),
                    describe_numeric_change(parameter_name, old_value, new_value),
                ))

        for field in scalar_fields:
            old_value = first_pkl_value(original_subject.get(field))
            new_value = first_pkl_value(modified_subject.get(field))
            if values_differ(old_value, new_value):
                parameter_name = f"{display_name}.{field}"
                changes.append((
                    parameter_name,
                    format_pkl_value(old_value),
                    format_pkl_value(new_value),
                    describe_numeric_change(parameter_name, old_value, new_value),
                ))

    original_camera = original_scene_dict.get("camera_data", {})
    modified_camera = modified_scene_dict.get("camera_data", {})
    if not isinstance(original_camera, dict):
        original_camera = {}
    if not isinstance(modified_camera, dict):
        modified_camera = {}
    for field in ("camera_elevation", "lens", "global_scale"):
        old_value = original_camera.get(field)
        new_value = modified_camera.get(field)
        if values_differ(old_value, new_value):
            parameter_name = f"camera_data.{field}"
            changes.append((
                parameter_name,
                format_pkl_value(old_value),
                format_pkl_value(new_value),
                describe_numeric_change(parameter_name, old_value, new_value),
            ))

    return changes


def print_scene_parameter_changes(
    original_scene_dict: Dict[str, Any],
    modified_scene_dict: Dict[str, Any],
) -> None:
    changes = collect_scene_parameter_changes(original_scene_dict, modified_scene_dict)
    print("参数修改记录：")
    if not changes:
        print("无")
        return
    for parameter_name, old_value, new_value, change_description in changes:
        print(f"参数名字：{parameter_name}；原始参数:{old_value}；修改后的参数:{new_value}；变化：{change_description}")


def build_guidance_forced_scene(
    base_scene_dict: Dict[str, Any],
    current_subjects: List[Dict[str, Any]],
    current_camera: Dict[str, float],
    guidance_target_camera: Dict[str, float],
    scene_text: str,
    guidance_text: str,
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
    guidance_target_names: Set[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    allowed_types = sorted(set(reference_dims_map.keys()) | set(default_dims_map.keys()))
    baseline_subjects, structural_target_names = apply_structural_guidance(
        current_subjects,
        guidance_text=guidance_text,
        allowed_types=allowed_types,
        reference_dims_map=reference_dims_map,
        default_dims_map=default_dims_map,
    )
    resolved_guidance_target_names = set(guidance_target_names) | structural_target_names

    forced_subjects = clone_subjects(baseline_subjects)
    if resolved_guidance_target_names:
        forced_subjects = apply_strict_subject_constraints(
            forced_subjects,
            scene_text=scene_text,
            guidance_text=guidance_text,
            reference_dims_map=reference_dims_map,
            default_dims_map=default_dims_map,
            original_subjects=baseline_subjects,
            guidance_target_names=resolved_guidance_target_names,
        )
        forced_subjects = preserve_non_guidance_subjects(
            forced_subjects,
            baseline_subjects,
            resolved_guidance_target_names,
        )

    camera_elevation_guided = (
        abs(guidance_target_camera["camera_elevation_deg"] - current_camera["camera_elevation_deg"]) > 1e-4
    )
    lens_guided = abs(guidance_target_camera["lens_mm"] - current_camera["lens_mm"]) > 1e-4
    global_scale_guided = abs(guidance_target_camera["global_scale"] - current_camera["global_scale"]) > 1e-4

    forced_camera_elevation = (
        guidance_target_camera["camera_elevation_deg"]
        if camera_elevation_guided
        else current_camera["camera_elevation_deg"]
    )
    forced_lens = guidance_target_camera["lens_mm"] if lens_guided else current_camera["lens_mm"]
    forced_global_scale = (
        guidance_target_camera["global_scale"]
        if global_scale_guided
        else current_camera["global_scale"]
    )

    final_subjects = forced_subjects
    if resolved_guidance_target_names:
        final_subjects = preserve_non_guidance_subjects(
            final_subjects,
            baseline_subjects,
            resolved_guidance_target_names,
        )

    fit_visibility = evaluate_scene_visibility(
        final_subjects,
        forced_camera_elevation,
        forced_lens,
        margin=VISIBILITY_MARGIN,
    )
    fit_result = {
        "subjects": final_subjects,
        "camera_elevation_deg": round(forced_camera_elevation, 6),
        "lens_mm": round(forced_lens, 6),
        "global_scale": round(forced_global_scale, 6),
        "passed": fit_visibility["all_inside_margin"],
        "visibility": fit_visibility,
        "transform": {
            "layout_scale": 1.0,
            "scene_x_shift": 0.0,
            "scene_y_shift": 0.0,
        },
    }

    forced_scene_dict = build_scene_dict_from_subjects(
        base_scene_dict=base_scene_dict,
        subjects=final_subjects,
        camera_elevation_deg=fit_result["camera_elevation_deg"],
        lens_mm=fit_result["lens_mm"],
        scene_text=scene_text,
        global_scale=forced_global_scale,
    )
    return forced_scene_dict, fit_result


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("必须是正整数。") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数。")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "审核已有场景 pkl，并在不合适时自动生成修正版 pkl。"
        ),
        epilog=(
            '示例:\n'
            '  python3 agent_check_pkl.py '
            '--scene-text "A bulldozer is positioned at the center of the image." '
            '--scene-pkl inference/saved_scenes/example.pkl '
            '--output inference/saved_scenes/example_fixed.pkl '
            '--guidance "make the bulldozer slightly smaller and keep it centered"'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--scene-text",
        required=True,
        help="空间布局文字描述。直接在运行命令中传入，需用引号包裹。",
    )
    parser.add_argument(
        "--scene-pkl",
        required=True,
        help="待审核的原始场景 pkl 路径。",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="修正版 pkl 输出路径。",
    )
    parser.add_argument(
        "--guidance",
        default="",
        help="可选，额外修改指导意见；会被当成必须执行的补充约束，例如调整空间位置或相对大小。",
    )
    parser.add_argument(
        "extra_guidance",
        nargs="?",
        default="",
        help="可选，额外修改指导意见的简写位置参数。",
    )
    parser.add_argument(
        "--api-key",
        default="sk-4vGkTSWP4WmCIQaiY1fthMExZHJL5aDl2Ad8QmtZiGcJYJzS",
        help="可选，显式传入 API Key。未提供时会尝试读取环境变量或 chatgpt_api.py。",
    )
    parser.add_argument(
        "--model",
        default=API_MODEL,
        help=f"模型名，默认: {API_MODEL}",
    )
    parser.add_argument(
        "--max-repair-attempts",
        type=positive_int,
        default=MAX_REPAIR_ATTEMPTS,
        help=f"严格修正的最大轮数，默认: {MAX_REPAIR_ATTEMPTS}。",
    )
    parser.add_argument(
        "--print-audit-json",
        action="store_true",
        help="打印审核阶段的大模型原始 JSON。",
    )
    parser.add_argument(
        "--print-final-scene-json",
        action="store_true",
        help="如果发生修正，打印最终写入 pkl 的 scene_dict 预览。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene_text = args.scene_text.strip()
    if not scene_text:
        raise RuntimeError("场景文字为空。")
    guidance_text = (args.guidance.strip() or args.extra_guidance.strip())

    scene_pkl = args.scene_pkl.strip()
    if not scene_pkl:
        raise RuntimeError("pkl 路径为空。")
    if not os.path.isabs(scene_pkl):
        scene_pkl = os.path.join(REPO_ROOT, scene_pkl)
    if not os.path.exists(scene_pkl):
        raise RuntimeError(f"找不到 pkl 文件：{scene_pkl}")

    output_path = args.output.strip()
    if not output_path:
        raise RuntimeError("输出 pkl 路径为空。")
    if not os.path.isabs(output_path):
        output_path = os.path.join(REPO_ROOT, output_path)

    asset_dimensions = load_asset_dimensions(ASSET_DIMENSIONS_PATH)
    object_scales = load_object_scales(OBJECT_SCALES_PATH)
    default_dims_map = build_default_dims_map(asset_dimensions)
    reference_dims_map = build_reference_dims_map(asset_dimensions, object_scales)
    allowed_types = list(asset_dimensions.keys())

    scene_dict = load_scene_pkl(scene_pkl)
    current_subjects_raw = summarize_scene_subjects(scene_dict, allowed_types, reference_dims_map)
    current_camera = summarize_camera(scene_dict)
    guidance_target_camera = apply_guidance_camera_directives(current_camera, guidance_text)
    current_subjects, structural_target_names = apply_structural_guidance(
        current_subjects_raw,
        guidance_text=guidance_text,
        allowed_types=allowed_types,
        reference_dims_map=reference_dims_map,
        default_dims_map=default_dims_map,
    )
    guidance_target_names = collect_subject_names_mentioned_in_text(current_subjects, guidance_text)
    guidance_target_names.update(structural_target_names)
    guidance_preview_subjects = build_guidance_preview_subjects(
        current_subjects,
        guidance_text,
        reference_dims_map,
        default_dims_map,
        guidance_target_names,
    )
    guidance_pending_issues = deduplicate_issues(
        collect_guidance_application_issues(
            current_subjects_raw,
            current_subjects_raw,
            guidance_preview_subjects,
            guidance_target_names,
        )
        + collect_camera_guidance_application_issues(
            current_camera,
            current_camera,
            guidance_target_camera,
        )
    )
    local_checks = run_local_checks(
        current_subjects_raw,
        camera_elevation_deg=current_camera["camera_elevation_deg"],
        lens_mm=current_camera["lens_mm"],
        reference_dims_map=reference_dims_map,
        default_dims_map=default_dims_map,
        scene_text=scene_text,
    )

    try:
        api_key = load_api_key(args.api_key)
    except Exception as exc:  # noqa: BLE001
        api_key = ""
        print(f"警告：未能加载大模型 API Key，已自动切换到本地修正流程：{exc}")

    audit_prompt = build_audit_prompt(
        scene_text=scene_text,
        guidance_text=guidance_text,
        current_subjects=current_subjects,
        camera_data=current_camera,
        local_checks=local_checks,
        allowed_types=allowed_types,
        reference_dims_map=reference_dims_map,
        default_dims_map=default_dims_map,
    )
    audit_json, audit_error = safe_call_llm_for_json(audit_prompt, api_key=api_key, model=args.model)
    if audit_error:
        print(f"警告：审核阶段大模型调用失败，已继续执行本地修正：{audit_error}")

    if args.print_audit_json:
        print("===== LLM Audit JSON =====")
        print(json.dumps(audit_json, ensure_ascii=False, indent=2))

    verdict = str(audit_json.get("verdict", "")).strip().lower()
    summary = str(audit_json.get("summary", "")).strip()
    audit_issues = normalize_issue_entries(audit_json.get("issues"))
    blocking_issues = deduplicate_issues(collect_blocking_issues(local_checks) + guidance_pending_issues)
    should_accept = verdict == "accept" and not audit_issues and not blocking_issues

    if should_accept:
        save_scene_pkl(scene_dict, output_path)
        print("结论：当前 pkl 合适。")
        if summary:
            print(f"说明：{summary}")
        if guidance_text:
            print(f"已严格核对额外指导意见：{guidance_text}")
        print(f"已输出 pkl：{output_path}")
        return

    print("结论：当前 pkl 不合适。")
    if summary:
        print(f"说明：{summary}")
    if guidance_text:
        print(f"额外修改指导：{guidance_text}")

    all_issue_lines = format_issue_lines(deduplicate_issues(audit_issues + blocking_issues))
    if all_issue_lines:
        print("存在的问题：")
        for line in all_issue_lines:
            print(line)

    working_subjects = current_subjects
    working_camera = guidance_target_camera
    working_checks = local_checks
    latest_response = audit_json
    latest_audit_issues = audit_issues
    latest_summary = summary

    final_scene_dict: Optional[Dict[str, Any]] = None
    final_fit_result: Optional[Dict[str, Any]] = None
    final_checks: Dict[str, Any] = local_checks
    final_summary = summary
    best_scene_dict: Dict[str, Any] = scene_dict
    best_fit_result: Dict[str, Any] = {
        "subjects": clone_subjects(current_subjects),
        "camera_elevation_deg": current_camera["camera_elevation_deg"],
        "lens_mm": current_camera["lens_mm"],
        "global_scale": current_camera["global_scale"],
        "transform": {
            "layout_scale": 1.0,
            "scene_x_shift": 0.0,
            "scene_y_shift": 0.0,
        },
    }
    best_checks = local_checks
    best_summary = summary
    best_issue_score = len(blocking_issues) * 100 + len(audit_issues)
    best_issues = deduplicate_issues(audit_issues + blocking_issues)

    max_repair_attempts = args.max_repair_attempts

    for attempt in range(1, max_repair_attempts + 1):
        print(f"开始第 {attempt} 轮严格修正...")

        if attempt == 1 and guidance_target_names:
            candidate_subjects = clone_subjects(working_subjects)
            candidate_camera_elevation_deg = working_camera["camera_elevation_deg"]
            candidate_lens_mm = working_camera["lens_mm"]
            print("提示：首轮先基于输入 pkl 强制执行 --guidance。")
        else:
            candidate_subjects, candidate_camera_elevation_deg, candidate_lens_mm = extract_candidate_from_response(
                latest_response,
                fallback_camera=working_camera,
                allowed_types=allowed_types,
                reference_dims_map=default_dims_map,
            )

        if not candidate_subjects:
            if api_key:
                try:
                    fix_json = request_correction_plan(
                        scene_text=scene_text,
                        guidance_text=guidance_text,
                        current_subjects=working_subjects,
                        camera_data=working_camera,
                        local_checks=working_checks,
                        audit_issues=latest_audit_issues,
                        allowed_types=allowed_types,
                        reference_dims_map=reference_dims_map,
                        default_dims_map=default_dims_map,
                        api_key=api_key,
                        model=args.model,
                    )
                except Exception as exc:  # noqa: BLE001
                    fix_json = {}
                    print(f"警告：修正阶段大模型调用失败，已继续执行本地修正：{exc}")

                candidate_subjects, candidate_camera_elevation_deg, candidate_lens_mm = extract_candidate_from_response(
                    fix_json,
                    fallback_camera=working_camera,
                    allowed_types=allowed_types,
                    reference_dims_map=default_dims_map,
                )
                latest_response = fix_json

        if not candidate_subjects and not guidance_target_names:
            filtered_subjects = filter_current_subjects_by_expected_counts(
                working_subjects,
                scene_text=scene_text,
                asset_dimensions=asset_dimensions,
            )
            if filtered_subjects:
                candidate_subjects = filtered_subjects
                candidate_camera_elevation_deg = working_camera["camera_elevation_deg"]
                candidate_lens_mm = working_camera["lens_mm"]
                print("提示：本轮优先沿用当前场景并移除文字未提到的多余物体。")

        if not candidate_subjects:
            if guidance_target_names:
                candidate_subjects = clone_subjects(working_subjects)
                candidate_camera_elevation_deg = working_camera["camera_elevation_deg"]
                candidate_lens_mm = working_camera["lens_mm"]
                print("提示：本轮未获得可用修正版，已仅基于当前 pkl 对 --guidance 提到的 cube 做规则修改。")
            else:
                fallback_scene_text = scene_text
                if guidance_text:
                    fallback_scene_text = f"{scene_text}. {guidance_text}"
                candidate_subjects, local_camera = build_local_fallback_subjects(
                    scene_text=fallback_scene_text,
                    asset_dimensions=asset_dimensions,
                    allowed_types=allowed_types,
                    reference_dims_map=default_dims_map,
                )
                if not candidate_subjects:
                    raise RuntimeError("大模型没有返回可用修正版，本地 fallback 也无法构造严格合格的场景。")
                candidate_camera_elevation_deg = clamp(
                    to_float(local_camera.get("camera_elevation_deg"), working_camera["camera_elevation_deg"]),
                    0.0,
                    90.0,
                )
                candidate_lens_mm = clamp(
                    to_float(local_camera.get("camera_lens_mm"), working_camera["lens_mm"]),
                    MIN_CAMERA_LENS_MM,
                    200.0,
                )
                print("警告：本轮大模型未返回可用修正版，已回退到本地已知类型布局规则。")

        candidate_subjects = preserve_non_guidance_subjects(
            candidate_subjects,
            working_subjects,
            guidance_target_names,
        )

        candidate_subjects = apply_strict_subject_constraints(
            candidate_subjects,
            scene_text=scene_text,
            guidance_text=guidance_text,
            reference_dims_map=reference_dims_map,
            default_dims_map=default_dims_map,
            original_subjects=working_subjects,
            guidance_target_names=guidance_target_names,
        )
        camera_elevation_guided = (
            abs(guidance_target_camera["camera_elevation_deg"] - current_camera["camera_elevation_deg"]) > 1e-4
        )
        lens_guided = abs(guidance_target_camera["lens_mm"] - current_camera["lens_mm"]) > 1e-4
        global_scale_guided = abs(guidance_target_camera["global_scale"] - current_camera["global_scale"]) > 1e-4

        if camera_elevation_guided:
            candidate_camera_elevation_deg = guidance_target_camera["camera_elevation_deg"]
        if lens_guided:
            candidate_lens_mm = guidance_target_camera["lens_mm"]
        candidate_global_scale = (
            guidance_target_camera["global_scale"]
            if global_scale_guided
            else working_camera.get("global_scale", current_camera["global_scale"])
        )

        fit_result = fit_scene_to_camera(
            candidate_subjects,
            camera_elevation_deg=candidate_camera_elevation_deg,
            lens_mm=candidate_lens_mm,
            movable_subject_names=guidance_target_names or None,
        )
        if camera_elevation_guided:
            fit_result["camera_elevation_deg"] = guidance_target_camera["camera_elevation_deg"]
        if lens_guided:
            fit_result["lens_mm"] = guidance_target_camera["lens_mm"]
        fit_result["global_scale"] = candidate_global_scale

        candidate_scene_dict = build_scene_dict_from_subjects(
            base_scene_dict=scene_dict,
            subjects=fit_result["subjects"],
            camera_elevation_deg=fit_result["camera_elevation_deg"],
            lens_mm=fit_result["lens_mm"],
            scene_text=scene_text,
            global_scale=candidate_global_scale,
        )
        candidate_subjects_summary = summarize_scene_subjects(
            candidate_scene_dict,
            allowed_types,
            reference_dims_map,
        )
        candidate_camera = summarize_camera(candidate_scene_dict)
        candidate_checks = run_local_checks(
            candidate_subjects_summary,
            camera_elevation_deg=candidate_camera["camera_elevation_deg"],
            lens_mm=candidate_camera["lens_mm"],
            reference_dims_map=reference_dims_map,
            default_dims_map=default_dims_map,
            scene_text=scene_text,
        )
        candidate_guidance_issues = deduplicate_issues(
            collect_guidance_application_issues(
                current_subjects_raw,
                candidate_subjects_summary,
                guidance_preview_subjects,
                guidance_target_names,
            )
            + collect_camera_guidance_application_issues(
                current_camera,
                candidate_camera,
                guidance_target_camera,
            )
        )

        candidate_blocking_issues = deduplicate_issues(
            collect_blocking_issues(candidate_checks) + candidate_guidance_issues
        )
        candidate_issue_score = len(candidate_blocking_issues) * 100
        if candidate_issue_score <= best_issue_score:
            best_scene_dict = candidate_scene_dict
            best_fit_result = fit_result
            best_checks = candidate_checks
            best_summary = latest_summary
            best_issue_score = candidate_issue_score
            best_issues = deduplicate_issues(candidate_blocking_issues)

        if candidate_blocking_issues:
            working_subjects = candidate_subjects_summary
            working_camera = candidate_camera
            working_checks = candidate_checks
            latest_response = {}
            latest_audit_issues = candidate_blocking_issues
            latest_summary = "本地严格校验未通过"
            continue

        final_audit_prompt = build_audit_prompt(
            scene_text=scene_text,
            guidance_text=guidance_text,
            current_subjects=candidate_subjects_summary,
            camera_data=candidate_camera,
            local_checks=candidate_checks,
            allowed_types=allowed_types,
            reference_dims_map=reference_dims_map,
            default_dims_map=default_dims_map,
        )
        final_audit_json, final_audit_error = safe_call_llm_for_json(
            final_audit_prompt,
            api_key=api_key,
            model=args.model,
        )
        if final_audit_error:
            print(f"警告：复审阶段大模型调用失败，本轮不能视为通过 agent 评估：{final_audit_error}")
            working_subjects = candidate_subjects_summary
            working_camera = candidate_camera
            working_checks = candidate_checks
            latest_response = {}
            latest_audit_issues = [
                {
                    "category": "final_audit_failed",
                    "object": "",
                    "details": str(final_audit_error),
                }
            ]
            latest_summary = "复审阶段大模型调用失败，未通过 agent 评估"
            continue

        if args.print_audit_json:
            print(f"===== Repair Round {attempt} Audit JSON =====")
            print(json.dumps(final_audit_json, ensure_ascii=False, indent=2))

        final_verdict = str(final_audit_json.get("verdict", "")).strip().lower()
        final_audit_issues = normalize_issue_entries(final_audit_json.get("issues"))
        latest_summary = str(final_audit_json.get("summary", "")).strip()
        candidate_issue_score = len(candidate_blocking_issues) * 100 + len(final_audit_issues)
        if candidate_issue_score <= best_issue_score:
            best_scene_dict = candidate_scene_dict
            best_fit_result = fit_result
            best_checks = candidate_checks
            best_summary = latest_summary
            best_issue_score = candidate_issue_score
            best_issues = deduplicate_issues(candidate_blocking_issues + final_audit_issues)

        if final_verdict == "accept" and not final_audit_issues:
            final_scene_dict = candidate_scene_dict
            final_fit_result = fit_result
            final_checks = candidate_checks
            final_summary = latest_summary
            break

        working_subjects = candidate_subjects_summary
        working_camera = candidate_camera
        working_checks = candidate_checks
        latest_response = final_audit_json
        latest_audit_issues = final_audit_issues

    if final_scene_dict is None or final_fit_result is None:
        remaining_issue_lines = format_issue_lines(best_issues)
        print("警告：在设定的修正轮数内未获得 LLM accept；将按 --guidance 规则强制生成并保存 pkl。")
        if remaining_issue_lines:
            print("LLM/本地剩余问题（已不阻止按 guidance 保存）：")
            for line in remaining_issue_lines[:8]:
                print(line)
        final_scene_dict, final_fit_result = build_guidance_forced_scene(
            base_scene_dict=scene_dict,
            current_subjects=current_subjects,
            current_camera=current_camera,
            guidance_target_camera=guidance_target_camera,
            scene_text=scene_text,
            guidance_text=guidance_text,
            reference_dims_map=reference_dims_map,
            default_dims_map=default_dims_map,
            guidance_target_names=guidance_target_names,
        )
        final_checks = run_local_checks(
            summarize_scene_subjects(final_scene_dict, allowed_types, reference_dims_map),
            camera_elevation_deg=summarize_camera(final_scene_dict)["camera_elevation_deg"],
            lens_mm=summarize_camera(final_scene_dict)["lens_mm"],
            reference_dims_map=reference_dims_map,
            default_dims_map=default_dims_map,
            scene_text=scene_text,
        )
        final_summary = "已按 --guidance 规则确定性修改并保存；未修改 --guidance 未提到的 cube。"

    save_scene_pkl(final_scene_dict, output_path)

    print(f"已保存修正版 pkl：{output_path}")
    print_scene_parameter_changes(scene_dict, final_scene_dict)

    if final_summary:
        print(f"最终说明：{final_summary}")

    if final_fit_result.get("transform"):
        transform = final_fit_result["transform"]
        print(
            "自动入框调整："
            f"lens={final_fit_result['lens_mm']:.2f}mm, "
            f"layout_scale={transform['layout_scale']:.3f}, "
            f"scene_x_shift={transform['scene_x_shift']:.3f}, "
            f"scene_y_shift={transform['scene_y_shift']:.3f}"
        )

    if args.print_final_scene_json:
        print("===== Final Scene Dict =====")
        print(json.dumps(final_scene_dict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
