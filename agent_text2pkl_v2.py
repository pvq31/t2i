#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从英文场景文字生成与 inference/saved_scenes/example0.pkl 同结构的场景文件。

设计目标：
1. 调用与 chatgpt_api.py 相同风格的 API（api.chatanywhere.tech /v1/responses）。
2. 让大模型从文字中提取物体类型、名称和空间布局。
3. 本地把模型输出稳定地转换成 example0.pkl 对应的 scene_dict 结构。

输出文件的名字和路径 搜索“输出文件的名字和路径”

输出的顶层结构：
{
  "subjects_data": [...],
  "camera_data": {...},
  "surrounding_prompt": "...",
  "inference_params": {...},
  "checkpoint": "seethrough3d_release/seethrough3d_release"
}
"""

from __future__ import annotations

import argparse
import ast
import difflib
import http.client
import json
import math
import os
import pickle
import re
import statistics
from typing import Any, Dict, List, Optional


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSET_DIMENSIONS_PATH = os.path.join(REPO_ROOT, "inference", "asset_dimensions.json")
DEFAULT_OUTPUT_PATH = os.path.join(
    REPO_ROOT, "inference", "saved_scenes", "example_test.pkl"
)

API_HOST = "api.chatanywhere.tech"
API_PATH = "/v1/responses"
API_MODEL = "gpt-5.2"

SCENE_X_SAVE_OFFSET = 6.0
DEFAULT_CAMERA_ELEVATION_DEG = 12.0
DEFAULT_LENS_MM = 50.0
DEFAULT_GLOBAL_SCALE = 1.0
CUSTOM_ASSET_TYPE = "Custom"
FUZZY_TYPE_MATCH_CUTOFF = 0.92

DEFAULT_INFERENCE_PARAMS = {
    "height": 1024,
    "width": 1024,
    "seed": 42,
    "guidance_scale": 3.5,
    "num_inference_steps": 25,
    "checkpoint": "rgb__finetune_1024/epoch-1__checkpoint-5000",
}
DEFAULT_TOP_LEVEL_CHECKPOINT = "seethrough3d_release/seethrough3d_release"

DEFAULT_TYPE_SIZE_SCALE = {
    "bear": 0.8,
    "cat": 0.35,
    "chair": 0.4,
    "cow": 0.8,
    "crow": 0.25,
    "deer": 0.8,
    "dog": 0.6,
    "elephant": 1.0,
    "flamingo": 0.35,
    "fox": 0.55,
    "giraffe": 0.9,
    "goat": 0.65,
    "hen": 0.25,
    "horse": 0.85,
    "lion": 0.8,
    "office chair": 0.4,
    "pigeon": 0.25,
    "pig": 0.6,
    "rabbit": 0.35,
    "sheep": 0.65,
    "shoe": 0.25,
    "sparrow": 0.18,
    "table": 0.8,
    "teddy": 0.3,
    "tiger": 0.8,
    "wolf": 0.65,
    "zebra": 0.8,
}

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
    "kitten": "cat",
    "motorcycle": "motorbike",
    "person": "man",
    "pickup": "pickup truck",
    "pickuptruck": "pickup truck",
    "puppy": "dog",
    "road bike": "bicycle",
    "scooter": "scooter",
    "sports car": "coupe",
    "table": "table",
    "truck": "pickup truck",
    "van": "van",
}

OBJECT_EXTRACTION_STOP_TOKENS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "in",
    "on",
    "at",
    "to",
    "from",
    "by",
    "near",
    "behind",
    "between",
    "along",
    "outside",
    "inside",
    "with",
    "holding",
    "showing",
    "resting",
    "positioned",
    "placed",
    "draped",
    "floating",
    "standing",
    "seen",
    "can",
    "not",
    "far",
    "while",
    "where",
    "which",
    "that",
}
GENERIC_OBJECT_PHRASES = {
    "image",
    "center",
    "middle",
    "left side",
    "right side",
    "back left",
    "back right",
    "front left",
    "front right",
    "distance",
    "street view",
    "outside",
    "inside",
}
GENERIC_OBJECT_HEAD_TOKENS = {
    "air",
    "atmosphere",
    "center",
    "composition",
    "daylight",
    "detail",
    "details",
    "distance",
    "floor",
    "image",
    "inside",
    "lake",
    "lakeshore",
    "light",
    "lighting",
    "middle",
    "outside",
    "road",
    "room",
    "scene",
    "shadow",
    "shadows",
    "shore",
    "street",
    "view",
    "water",
}
GENERIC_OBJECT_START_TOKENS = {
    "back",
    "center",
    "distance",
    "front",
    "image",
    "left",
    "middle",
    "right",
    "scene",
}
INVALID_OBJECT_EDGE_TOKENS = {
    "and",
    "back",
    "center",
    "front",
    "left",
    "middle",
    "of",
    "or",
    "right",
    "side",
    "there",
}
GENERIC_LOCATION_TOKENS = {
    "ahead",
    "air",
    "atmosphere",
    "background",
    "center",
    "curb",
    "direction",
    "distance",
    "floor",
    "ground",
    "image",
    "lane",
    "light",
    "lighting",
    "middle",
    "road",
    "roadside",
    "scene",
    "shadow",
    "shadows",
    "side",
    "street",
    "view",
}
GENERIC_GROUP_NOUNS = {
    "animals",
    "birds",
    "cars",
    "objects",
    "people",
    "vehicles",
}
DEFAULT_LOCAL_SURROUNDING_PROMPT = "highly realistic outdoor scene."
DEFAULT_VEHICLE_AZIMUTH_DEG = -90.0
VEHICLE_ASSET_TYPES = {
    "bugatti",
    "bulldozer",
    "bus",
    "coupe",
    "ferrari",
    "helicopter",
    "jeep",
    "lamborghini",
    "mclaren",
    "motorbike",
    "pickup truck",
    "scooter",
    "sedan",
    "suv",
    "tractor",
    "van",
    "vw beetle",
}
BIKE_LIKE_ASSET_TYPES = {
    "bicycle",
    "motorbike",
    "scooter",
}
AIRBORNE_HINT_TOKENS = {
    "air",
    "above",
    "flying",
    "hovering",
    "sky",
}
TABLETOP_OBJECT_KEYWORDS = {
    "cup",
    "mug",
    "latte",
    "coffee",
    "plate",
    "croissant",
    "book",
    "notebook",
    "menu",
    "bowl",
    "glass",
}
CHAIR_ATTACHED_OBJECT_KEYWORDS = {
    "scarf",
    "coat",
    "bag",
    "jacket",
}
CUSTOM_DIMENSION_PRESETS = [
    (("cup", "mug", "latte", "coffee", "teacup", "glass"), [0.12, 0.12, 0.16]),
    (("plate", "dish", "saucer"), [0.24, 0.24, 0.03]),
    (("croissant", "bread", "pastry"), [0.18, 0.10, 0.06]),
    (("book", "notebook", "menu"), [0.24, 0.18, 0.03]),
    (("scarf", "cloth", "shawl"), [0.55, 0.18, 0.03]),
    (("window",), [1.60, 0.12, 1.40]),
    (("boat", "canoe"), [2.60, 1.10, 0.90]),
    (("reeds", "reed", "grass"), [0.70, 0.35, 1.10]),
    (("lily pads", "lily pad", "lily"), [0.40, 0.40, 0.03]),
    (("cabin", "hut", "house"), [3.50, 3.50, 3.00]),
]


def load_asset_dimensions(path: str) -> Dict[str, List[float]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(k): [float(x) for x in v] for k, v in data.items()}


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


def build_prompt(scene_text: str, allowed_types: List[str]) -> str:
    allowed_types_text = ", ".join([CUSTOM_ASSET_TYPE] + sorted(allowed_types))
    return f"""
You are a structured 3D scene planner.
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
      "name": "natural language object name",
      "type": "one allowed asset type, a close synonym, or Custom",
      "scene_x": 0.0,
      "scene_y": 0.0,
      "scene_z": 0.0,
      "azimuth_deg": 0.0,
      "size_scale": 1.0,
      "dims": [1.0, 1.0, 1.0]
    }}
  ]
}}

Rules:
1. Extract every explicitly mentioned object.
2. scene_x / scene_y use the editable layout coordinate system BEFORE pickle export:
   - larger scene_x = more in front of / closer to camera
   - smaller scene_x = more behind / farther from camera
   - smaller scene_y = more to the left
   - larger scene_y = more to the right
3. Place a main reference object near scene_x=0, scene_y=0.
4. Keep the layout compact and reasonable. Nearby objects should usually differ by 1.0 to 2.5 units.
5. Use scene_z=0.0 for ground objects unless the text clearly says an object is elevated.
6. size_scale should usually be 1.0. Use smaller values like 0.5-0.9 for words such as puppy or kitten.
   Typical examples: chair 0.35-0.5, cat 0.3-0.45, dog 0.45-0.7, sedan 1.0, bicycle 0.9-1.1.
7. Choose a valid asset type whenever possible. If the text uses a synonym like "car" or "person", map it to the closest allowed asset type.
8. If an object is explicitly mentioned but is not representable by the allowed asset types, you must still keep it in subjects and set "type" to "Custom".
9. Never drop an explicitly mentioned object only because its asset type is unsupported.
10. For Custom objects, estimate a realistic size from the other known objects in the scene. Provide dims when possible.
11. surrounding_prompt should describe only the environment/background style, not the object list again.

Example:
Input:
"There is a sedan. A chair is to the left of the sedan, and a bicycle is in front of it."
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


def sanitize_name(name: Any, default_name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return default_name
    return re.sub(r"\s+", " ", text)


def canonicalize_type(raw_type: str) -> str:
    text = re.sub(r"\s+", " ", raw_type.strip().lower())
    text = text.replace("_", " ").replace("-", " ")
    return text.strip()


def resolve_allowed_type_candidate(
    normalized: str, allowed_lookup: Dict[str, str]
) -> Optional[str]:
    if not normalized:
        return None
    if normalized == canonicalize_type(CUSTOM_ASSET_TYPE):
        return CUSTOM_ASSET_TYPE
    if normalized in allowed_lookup:
        return allowed_lookup[normalized]
    alias = ASSET_TYPE_ALIASES.get(normalized)
    if alias:
        alias_norm = canonicalize_type(alias)
        if alias_norm in allowed_lookup:
            return allowed_lookup[alias_norm]
    return None


def build_asset_type_terms(allowed_types: List[str]) -> Dict[str, set[str]]:
    terms: Dict[str, set[str]] = {}
    for allowed_type in allowed_types:
        terms[allowed_type] = {canonicalize_type(allowed_type)}

    for alias, target in ASSET_TYPE_ALIASES.items():
        target_norm = canonicalize_type(target)
        for allowed_type in allowed_types:
            if canonicalize_type(allowed_type) == target_norm:
                terms[allowed_type].add(canonicalize_type(alias))

    return terms


def phrase_in_text(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    return re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", text) is not None


def normalize_name_variants(raw_name: str) -> List[str]:
    normalized_name = canonicalize_type(raw_name)
    if not normalized_name:
        return []

    variants = {normalized_name}
    tokens = []
    for token in normalized_name.split():
        if len(token) > 3 and token.endswith("s"):
            tokens.append(token[:-1])
        else:
            tokens.append(token)
    singularized = " ".join(tokens).strip()
    if singularized:
        variants.add(singularized)
    return list(variants)


def name_supports_asset_type(
    raw_name: str, asset_type: str, asset_type_terms: Dict[str, set[str]]
) -> bool:
    if not raw_name:
        return False
    terms = asset_type_terms.get(asset_type, {canonicalize_type(asset_type)})
    for variant in normalize_name_variants(raw_name):
        for term in terms:
            if phrase_in_text(variant, term):
                return True
    return False


def normalize_optional_dims(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None

    dims: List[float] = []
    for idx in range(3):
        try:
            current = float(value[idx])
        except (TypeError, ValueError):
            return None
        if current <= 0:
            return None
        dims.append(current)
    return dims


def has_explicit_size(item: Dict[str, Any]) -> bool:
    if normalize_optional_dims(item.get("dims")) is not None:
        return True
    return item.get("size_scale") is not None


def find_preset_dims_for_custom(name: str) -> Optional[List[float]]:
    lowered = canonicalize_type(name)
    for keywords, dims in CUSTOM_DIMENSION_PRESETS:
        for keyword in keywords:
            if phrase_in_text(lowered, canonicalize_type(keyword)):
                return list(dims)
    return None


def compute_scene_scale(
    known_entries: List[Dict[str, Any]],
    asset_dimensions: Dict[str, List[float]],
) -> float:
    ratios: List[float] = []
    for entry in known_entries:
        asset_type = entry["type"]
        if asset_type == CUSTOM_ASSET_TYPE:
            continue
        base_dims = asset_dimensions.get(asset_type)
        if not base_dims:
            continue
        base_max = max(base_dims)
        current_max = max(entry["dims"])
        if base_max > 1e-6 and current_max > 1e-6:
            ratios.append(current_max / base_max)

    if not ratios:
        return 1.0
    return clamp(float(statistics.median(ratios)), 0.25, 4.0)


def infer_custom_dims(
    name: str,
    scene_scale: float,
    requested_size_scale: Optional[float] = None,
) -> List[float]:
    preset = find_preset_dims_for_custom(name)
    if preset is None:
        lowered = name.lower()
        if "small" in lowered or "little" in lowered or "tiny" in lowered:
            preset = [0.35, 0.35, 0.35]
        elif "large" in lowered or "big" in lowered:
            preset = [1.40, 1.40, 1.40]
        else:
            preset = [0.70, 0.70, 0.70]

    effective_scale = scene_scale
    if requested_size_scale is not None:
        effective_scale *= requested_size_scale

    return [round(dim * effective_scale, 6) for dim in preset]


def extract_explicit_object_mentions(scene_text: str) -> List[str]:
    normalized_text = re.sub(r"\s+", " ", scene_text.strip().lower())
    if not normalized_text:
        return []

    mentions: List[str] = []
    article_pattern = re.compile(r"\b(?:a|an|the|some|several|one|two|three|four|five)\b")
    token_pattern = re.compile(r"[a-zA-Z]+(?:-[a-zA-Z]+)?")

    for match in article_pattern.finditer(normalized_text):
        tail = normalized_text[match.end() :]
        tokens = token_pattern.findall(tail)
        phrase_tokens: List[str] = []

        for token in tokens:
            if token in {"a", "an", "the", "some", "several", "one", "two", "three", "four", "five"}:
                break
            if token in OBJECT_EXTRACTION_STOP_TOKENS and not (token == "of" and phrase_tokens):
                break
            phrase_tokens.append(token)
            if len(phrase_tokens) >= 6 and token != "of":
                break

        if not phrase_tokens:
            continue

        phrase = " ".join(phrase_tokens).strip()
        if not phrase or phrase in GENERIC_OBJECT_PHRASES:
            continue
        if phrase_tokens[0] in GENERIC_OBJECT_START_TOKENS:
            continue
        if phrase_tokens[-1] in GENERIC_OBJECT_HEAD_TOKENS:
            continue
        if phrase_tokens[-1] in INVALID_OBJECT_EDGE_TOKENS:
            continue
        if any(token == "there" for token in phrase_tokens):
            continue
        if phrase in mentions:
            continue
        mentions.append(phrase)

    return mentions


def is_generic_non_object_phrase(phrase: str) -> bool:
    normalized = canonicalize_type(phrase)
    if not normalized:
        return True
    if normalized in GENERIC_OBJECT_PHRASES or normalized in GENERIC_GROUP_NOUNS:
        return True

    tokens = normalized.split()
    if not tokens:
        return True
    if tokens[0] in GENERIC_OBJECT_START_TOKENS and len(tokens) == 1:
        return True
    if tokens[-1] in GENERIC_OBJECT_HEAD_TOKENS:
        return True
    if tokens[-1] in INVALID_OBJECT_EDGE_TOKENS:
        return True
    if any(token in GENERIC_LOCATION_TOKENS for token in tokens):
        return True
    return False


def plan_has_valid_subjects(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return False
    subjects = plan.get("subjects")
    return isinstance(subjects, list) and any(isinstance(item, dict) for item in subjects)


def infer_default_azimuth_deg(asset_type: str) -> float:
    if asset_type in BIKE_LIKE_ASSET_TYPES:
        return -120.0
    if asset_type in VEHICLE_ASSET_TYPES:
        return DEFAULT_VEHICLE_AZIMUTH_DEG
    return 0.0


def get_subject_context(scene_text: str, subject_name: str) -> str:
    normalized_text = re.sub(r"\s+", " ", scene_text.strip().lower())
    subject_norm = canonicalize_type(subject_name)
    if not normalized_text or not subject_norm:
        return normalized_text

    match = re.search(rf"(?<![a-z]){re.escape(subject_norm)}(?![a-z])", normalized_text)
    if match is None:
        return normalized_text

    start = 0
    end = len(normalized_text)
    for delimiter in ".:;!?":
        prev_idx = normalized_text.rfind(delimiter, 0, match.start())
        if prev_idx >= 0:
            start = max(start, prev_idx + 1)
    next_candidates = [normalized_text.find(delimiter, match.end()) for delimiter in ".:;!?"]
    next_candidates = [idx for idx in next_candidates if idx >= 0]
    if next_candidates:
        end = min(next_candidates)
    return normalized_text[start:end].strip()


def find_reference_entry_from_context(
    context: str,
    current_name: str,
    existing_entries: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    current_norm = canonicalize_type(current_name)
    for entry in reversed(existing_entries):
        entry_name = canonicalize_type(entry["name"])
        entry_type = canonicalize_type(entry["type"])
        if current_norm and (entry_name == current_norm or entry_type == current_norm):
            continue
        if entry_name and phrase_in_text(context, entry_name):
            return entry
        if entry_type and phrase_in_text(context, entry_type):
            return entry
    if existing_entries:
        return existing_entries[0]
    return None


def build_known_type_subjects_from_text(
    scene_text: str,
    asset_dimensions: Dict[str, List[float]],
) -> Optional[List[Dict[str, Any]]]:
    allowed_types = list(asset_dimensions.keys())
    raw_mentions = extract_explicit_object_mentions(scene_text)
    if not raw_mentions:
        return None

    supported: List[Dict[str, Any]] = []
    for mention in raw_mentions:
        if is_generic_non_object_phrase(mention):
            continue
        asset_type = match_asset_type(None, mention, allowed_types)
        if asset_type == CUSTOM_ASSET_TYPE:
            return None
        if any(subject_name_matches_mention(item["name"], mention) for item in supported):
            continue
        supported.append(
            {
                "name": sanitize_name(mention, default_name=asset_type),
                "type": asset_type,
            }
        )

    if not supported:
        return None
    return supported


def infer_local_known_type_layout(
    subject: Dict[str, Any],
    existing_entries: List[Dict[str, Any]],
    scene_text: str,
    asset_dimensions: Dict[str, List[float]],
    subject_index: int,
) -> Dict[str, Any]:
    asset_type = subject["type"]
    dims = [round(float(v), 6) for v in asset_dimensions.get(asset_type, [1.0, 1.0, 1.0])]
    default_position = estimate_missing_entry_position(subject["name"], existing_entries, subject_index)
    context = get_subject_context(scene_text, subject["name"])
    reference = find_reference_entry_from_context(context, subject["name"], existing_entries)

    azimuth_deg = infer_default_azimuth_deg(asset_type)
    if reference is not None:
        longitudinal_gap = round((reference["dims"][1] + dims[1]) * 0.6, 6)
        lateral_gap = round((reference["dims"][0] + dims[0]) * 0.45, 6)

        if "same direction" in context:
            azimuth_deg = reference["azimuth_deg"]

        if "front-right" in context or "front right" in context:
            default_position["scene_x"] = round(reference["scene_x"] + longitudinal_gap, 6)
            default_position["scene_y"] = round(reference["scene_y"] + max(lateral_gap, 0.8), 6)
        elif "front-left" in context or "front left" in context:
            default_position["scene_x"] = round(reference["scene_x"] + longitudinal_gap, 6)
            default_position["scene_y"] = round(reference["scene_y"] - max(lateral_gap, 0.8), 6)
        elif "behind" in context:
            default_position["scene_x"] = round(reference["scene_x"] - longitudinal_gap, 6)
            default_position["scene_y"] = round(reference["scene_y"], 6)
        elif "in front of" in context or "ahead" in context:
            default_position["scene_x"] = round(reference["scene_x"] + longitudinal_gap, 6)

        if "left of" in context:
            default_position["scene_y"] = round(reference["scene_y"] - max(lateral_gap, 0.8), 6)
        elif "right of" in context:
            default_position["scene_y"] = round(reference["scene_y"] + max(lateral_gap, 0.8), 6)

        if "closer to the " in context:
            default_position["scene_x"] = round(reference["scene_x"] + max(longitudinal_gap * 0.8, 0.9), 6)
            default_position["scene_y"] = round(reference["scene_y"] + 0.25 * subject_index, 6)

        if "nearer to the curb" in context or "near the curb" in context:
            default_position["scene_y"] = round(reference["scene_y"] + max(lateral_gap, 1.0), 6)

        if any(token in context for token in AIRBORNE_HINT_TOKENS):
            default_position["scene_z"] = round(max(reference["dims"][2] * 1.5, dims[2] * 1.5, 0.8), 6)
            default_position["scene_x"] = round(reference["scene_x"] + 0.15, 6)
            default_position["scene_y"] = round(reference["scene_y"], 6)

    default_position["scene_x"] = round(clamp(default_position["scene_x"], -6.0, 6.0), 6)
    default_position["scene_y"] = round(clamp(default_position["scene_y"], -6.0, 6.0), 6)
    default_position["scene_z"] = round(max(default_position["scene_z"], 0.0), 6)

    return {
        "name": subject["type"],
        "type": asset_type,
        "scene_x": default_position["scene_x"],
        "scene_y": default_position["scene_y"],
        "scene_z": default_position["scene_z"],
        "azimuth_deg": azimuth_deg,
        "size_scale": 1.0,
        "dims": dims,
    }


def build_local_plan_from_known_types(
    scene_text: str,
    asset_dimensions: Dict[str, List[float]],
) -> Optional[Dict[str, Any]]:
    supported_subjects = build_known_type_subjects_from_text(scene_text, asset_dimensions)
    if not supported_subjects:
        return None

    local_subjects: List[Dict[str, Any]] = []
    local_entries: List[Dict[str, Any]] = []

    for index, subject in enumerate(supported_subjects):
        if index == 0:
            asset_type = subject["type"]
            dims = [round(float(v), 6) for v in asset_dimensions.get(asset_type, [1.0, 1.0, 1.0])]
            local_subject = {
                "name": asset_type,
                "type": asset_type,
                "scene_x": 0.0,
                "scene_y": 0.0,
                "scene_z": 0.0,
                "azimuth_deg": infer_default_azimuth_deg(asset_type),
                "size_scale": 1.0,
                "dims": dims,
            }
        else:
            local_subject = infer_local_known_type_layout(
                subject=subject,
                existing_entries=local_entries,
                scene_text=scene_text,
                asset_dimensions=asset_dimensions,
                subject_index=index,
            )

        local_subjects.append(local_subject)
        local_entries.append(
            {
                "name": local_subject["name"],
                "type": local_subject["type"],
                "scene_x": local_subject["scene_x"],
                "scene_y": local_subject["scene_y"],
                "scene_z": local_subject["scene_z"],
                "azimuth_deg": local_subject["azimuth_deg"],
                "dims": list(local_subject["dims"]),
            }
        )

    return {
        "surrounding_prompt": DEFAULT_LOCAL_SURROUNDING_PROMPT,
        "camera_elevation_deg": DEFAULT_CAMERA_ELEVATION_DEG,
        "subjects": local_subjects,
    }


def subject_name_matches_mention(subject_name: str, mention: str) -> bool:
    subject_norm = canonicalize_type(subject_name)
    mention_norm = canonicalize_type(mention)

    if not subject_norm or not mention_norm:
        return False
    if subject_norm == mention_norm:
        return True
    if subject_norm in mention_norm or mention_norm in subject_norm:
        return True

    subject_tokens = set(subject_norm.split())
    mention_tokens = set(mention_norm.split())
    if mention_tokens and mention_tokens.issubset(subject_tokens):
        return True
    if subject_tokens and subject_tokens.issubset(mention_tokens):
        return True

    return difflib.SequenceMatcher(None, subject_norm, mention_norm).ratio() >= 0.8


def find_missing_mentions(
    explicit_mentions: List[str],
    subject_names: List[str],
) -> List[str]:
    missing: List[str] = []
    for mention in explicit_mentions:
        if any(subject_name_matches_mention(subject_name, mention) for subject_name in subject_names):
            continue
        missing.append(mention)
    return missing


def build_refinement_prompt(
    scene_text: str,
    initial_plan: Dict[str, Any],
    explicit_mentions: List[str],
    allowed_types: List[str],
) -> str:
    return f"""
You are auditing and repairing a structured 3D scene plan.
Return JSON only. Do not output markdown. Do not add explanations.

Allowed asset types:
[{", ".join([CUSTOM_ASSET_TYPE] + sorted(allowed_types))}]

Original scene text:
\"\"\"{scene_text}\"\"\"

Explicit object mentions that must all appear in the final subjects list:
{json.dumps(explicit_mentions, ensure_ascii=False)}

Current plan:
{json.dumps(initial_plan, ensure_ascii=False, indent=2)}

Return the full corrected plan in exactly this schema:
{{
  "surrounding_prompt": "a short background/environment prompt",
  "camera_elevation_deg": 12.0,
  "subjects": [
    {{
      "name": "natural language object name",
      "type": "one allowed asset type, a close synonym, or Custom",
      "scene_x": 0.0,
      "scene_y": 0.0,
      "scene_z": 0.0,
      "azimuth_deg": 0.0,
      "size_scale": 1.0,
      "dims": [1.0, 1.0, 1.0]
    }}
  ]
}}

Rules:
1. Every explicitly mentioned object must appear in subjects.
2. If an object cannot be represented by an allowed asset type, set type to Custom instead of dropping it.
3. Do not merge different objects into one subject.
4. Preserve valid existing objects and only repair omissions or unreasonable sizes/positions.
5. For Custom objects, provide realistic approximate dims when possible. Estimate them relative to the known typed objects in the same scene.
6. Keep the scene compact and physically plausible.
7. surrounding_prompt should describe background/style only.
    """.strip()


def call_llm_for_plan(prompt: str, api_key: str, model: str) -> Dict[str, Any]:
    api_response = call_llm(prompt, api_key=api_key, model=model)
    output_text = extract_output_text(api_response)
    return extract_json_object(output_text)


def infer_default_size_scale(name: str, asset_type: str) -> float:
    lowered_name = name.lower()
    if "puppy" in lowered_name or "kitten" in lowered_name:
        return 0.65
    if "small" in lowered_name or "little" in lowered_name:
        return 0.8
    if "large" in lowered_name or "big" in lowered_name:
        return 1.2
    if asset_type in {"cat", "dog"} and ("baby" in lowered_name):
        return 0.65
    return DEFAULT_TYPE_SIZE_SCALE.get(asset_type, 1.0)


def match_asset_type(
    raw_type: Any, raw_name: str, allowed_types: List[str]
) -> str:
    allowed_lookup = {canonicalize_type(t): t for t in allowed_types}
    asset_type_terms = build_asset_type_terms(allowed_types)

    name_candidates: List[str] = []
    if raw_name:
        name_candidates.append(raw_name)
        name_candidates.extend(re.split(r"[^a-zA-Z]+", raw_name))

    for candidate in name_candidates:
        resolved = resolve_allowed_type_candidate(canonicalize_type(candidate), allowed_lookup)
        if resolved is not None:
            return resolved

    normalized_raw_type = canonicalize_type(str(raw_type)) if raw_type is not None else ""
    resolved_raw_type = resolve_allowed_type_candidate(normalized_raw_type, allowed_lookup)
    if resolved_raw_type == CUSTOM_ASSET_TYPE:
        return CUSTOM_ASSET_TYPE
    if resolved_raw_type is not None:
        if not raw_name or name_supports_asset_type(raw_name, resolved_raw_type, asset_type_terms):
            return resolved_raw_type

    if normalized_raw_type:
        search_space = list(allowed_lookup.keys()) + list(ASSET_TYPE_ALIASES.keys())
        matches = difflib.get_close_matches(
            normalized_raw_type,
            search_space,
            n=1,
            cutoff=FUZZY_TYPE_MATCH_CUTOFF,
        )
        if matches:
            resolved_fuzzy = resolve_allowed_type_candidate(matches[0], allowed_lookup)
            if resolved_fuzzy is not None:
                if not raw_name or name_supports_asset_type(raw_name, resolved_fuzzy, asset_type_terms):
                    return resolved_fuzzy

    return CUSTOM_ASSET_TYPE


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def ensure_unique_names(subjects: List[Dict[str, Any]]) -> None:
    counts: Dict[str, int] = {}
    for subject in subjects:
        name = subject["name"]
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > 1:
            subject["name"] = f"{name} {counts[name]}"


def find_first_entry_by_type(
    entries: List[Dict[str, Any]],
    asset_type: str,
) -> Optional[Dict[str, Any]]:
    for entry in entries:
        if entry["type"] == asset_type:
            return entry
    return None


def estimate_missing_entry_position(
    name: str,
    entries: List[Dict[str, Any]],
    missing_index: int,
) -> Dict[str, float]:
    lowered = canonicalize_type(name)
    table_entry = find_first_entry_by_type(entries, "table")
    chair_entry = find_first_entry_by_type(entries, "chair")
    base_entry = entries[0] if entries else None

    if table_entry and any(keyword in lowered for keyword in TABLETOP_OBJECT_KEYWORDS):
        offset = 0.18 * ((missing_index % 3) - 1)
        return {
            "scene_x": round(table_entry["scene_x"] + offset, 6),
            "scene_y": round(table_entry["scene_y"] + 0.18 * (missing_index // 3), 6),
            "scene_z": round(max(table_entry["dims"][2], 0.05), 6),
            "azimuth_deg": 0.0,
        }

    if chair_entry and any(keyword in lowered for keyword in CHAIR_ATTACHED_OBJECT_KEYWORDS):
        return {
            "scene_x": round(chair_entry["scene_x"] - 0.15, 6),
            "scene_y": round(chair_entry["scene_y"], 6),
            "scene_z": round(chair_entry["dims"][2] * 0.65, 6),
            "azimuth_deg": 0.0,
        }

    if base_entry is None:
        return {
            "scene_x": 0.0,
            "scene_y": 0.0,
            "scene_z": 0.0,
            "azimuth_deg": 0.0,
        }

    angle = (missing_index % 6) * (math.pi / 3.0)
    radius = 0.9 + 0.25 * (missing_index // 6)
    return {
        "scene_x": round(clamp(base_entry["scene_x"] + math.cos(angle) * radius, -6.0, 6.0), 6),
        "scene_y": round(clamp(base_entry["scene_y"] + math.sin(angle) * radius, -6.0, 6.0), 6),
        "scene_z": 0.0,
        "azimuth_deg": 0.0,
    }


def add_missing_subject_entries(
    entries: List[Dict[str, Any]],
    explicit_mentions: List[str],
    asset_dimensions: Dict[str, List[float]],
) -> None:
    allowed_types = list(asset_dimensions.keys())
    scene_scale = compute_scene_scale(entries, asset_dimensions)
    added_count = 0
    for mention in explicit_mentions:
        if is_generic_non_object_phrase(mention):
            continue
        if any(subject_name_matches_mention(entry["name"], mention) for entry in entries):
            continue
        asset_type = match_asset_type(None, mention, allowed_types)
        if asset_type == CUSTOM_ASSET_TYPE:
            dims = infer_custom_dims(mention, scene_scale)
        else:
            dims = [round(float(v) * scene_scale, 6) for v in asset_dimensions.get(asset_type, [1.0, 1.0, 1.0])]

        position = estimate_missing_entry_position(mention, entries, added_count)
        entries.append(
            {
                "name": mention,
                "type": asset_type,
                "scene_x": position["scene_x"],
                "scene_y": position["scene_y"],
                "scene_z": position["scene_z"],
                "azimuth_deg": position["azimuth_deg"],
                "dims": dims,
            }
        )
        added_count += 1


def build_subject_entries(
    plan: Dict[str, Any],
    asset_dimensions: Dict[str, List[float]],
    scene_text: str,
) -> List[Dict[str, Any]]:
    raw_subjects = plan.get("subjects")
    if not isinstance(raw_subjects, list) or not raw_subjects:
        raise ValueError("模型没有返回 subjects 列表，或 subjects 为空。")

    allowed_types = list(asset_dimensions.keys())
    normalized_items: List[Dict[str, Any]] = []

    for idx, item in enumerate(raw_subjects):
        if not isinstance(item, dict):
            continue

        name = sanitize_name(
            item.get("name") or item.get("description"),
            default_name=f"object_{idx + 1}",
        )
        asset_type = match_asset_type(item.get("type"), name, allowed_types)

        scene_x = to_float(item.get("scene_x"), 0.0)
        scene_y = to_float(item.get("scene_y"), 0.0)
        scene_z = to_float(item.get("scene_z"), 0.0)
        azimuth_deg = to_float(item.get("azimuth_deg"), 0.0)
        explicit_dims = normalize_optional_dims(item.get("dims"))

        default_scale = infer_default_size_scale(name, asset_type)
        raw_size_scale = item.get("size_scale")
        size_scale = to_float(raw_size_scale, default_scale)
        size_scale = clamp(size_scale, 0.2, 3.0)

        normalized_items.append(
            {
                "name": name,
                "type": asset_type,
                "scene_x": scene_x,
                "scene_y": scene_y,
                "scene_z": scene_z,
                "azimuth_deg": azimuth_deg,
                "explicit_dims": explicit_dims,
                "size_scale": size_scale,
                "raw_size_scale": raw_size_scale,
            }
        )

    if not normalized_items:
        raise ValueError("模型返回了 subjects，但无法解析出有效物体。")

    entries: List[Dict[str, Any]] = []
    for item in normalized_items:
        if item["type"] == CUSTOM_ASSET_TYPE:
            continue

        if item["explicit_dims"] is not None:
            dims = item["explicit_dims"]
        else:
            base_dims = asset_dimensions.get(item["type"], [1.0, 1.0, 1.0])
            dims = [round(float(v) * item["size_scale"], 6) for v in base_dims]

        entries.append(
            {
                "name": item["name"],
                "type": item["type"],
                "scene_x": item["scene_x"],
                "scene_y": item["scene_y"],
                "scene_z": item["scene_z"],
                "azimuth_deg": item["azimuth_deg"],
                "dims": dims,
            }
        )

    scene_scale = compute_scene_scale(entries, asset_dimensions)
    for item in normalized_items:
        if item["type"] != CUSTOM_ASSET_TYPE:
            continue

        if item["explicit_dims"] is not None:
            dims = item["explicit_dims"]
        else:
            requested_size_scale = (
                item["size_scale"] if item["raw_size_scale"] is not None else None
            )
            dims = infer_custom_dims(
                item["name"],
                scene_scale=scene_scale,
                requested_size_scale=requested_size_scale,
            )

        entries.append(
            {
                "name": item["name"],
                "type": item["type"],
                "scene_x": item["scene_x"],
                "scene_y": item["scene_y"],
                "scene_z": item["scene_z"],
                "azimuth_deg": item["azimuth_deg"],
                "dims": dims,
            }
        )

    add_missing_subject_entries(
        entries,
        explicit_mentions=extract_explicit_object_mentions(scene_text),
        asset_dimensions=asset_dimensions,
    )

    ensure_unique_names(entries)
    resolve_collisions(entries)
    return entries


def resolve_collisions(entries: List[Dict[str, Any]], max_rounds: int = 16) -> None:
    """
    轻量级碰撞消解：
    如果模型把两个物体摆得几乎重叠，就做小幅平移，避免生成完全堆叠的 cuboid。
    """
    for _ in range(max_rounds):
        changed = False
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                first = entries[i]
                second = entries[j]

                dx = second["scene_x"] - first["scene_x"]
                dy = second["scene_y"] - first["scene_y"]

                depth_gap = (first["dims"][1] + second["dims"][1]) * 0.55
                width_gap = (first["dims"][0] + second["dims"][0]) * 0.55

                overlap_x = abs(dx) < depth_gap
                overlap_y = abs(dy) < width_gap
                if not (overlap_x and overlap_y):
                    continue

                changed = True
                prefer_x = abs(dx) >= abs(dy)
                if prefer_x:
                    push = (depth_gap - abs(dx)) + 0.05
                    sign = 1.0 if dx >= 0 else -1.0
                    if abs(dx) < 1e-6:
                        sign = 1.0
                    second["scene_x"] += push * sign
                else:
                    push = (width_gap - abs(dy)) + 0.05
                    sign = 1.0 if dy >= 0 else -1.0
                    if abs(dy) < 1e-6:
                        sign = 1.0
                    second["scene_y"] += push * sign

                second["scene_x"] = clamp(second["scene_x"], -6.0, 6.0)
                second["scene_y"] = clamp(second["scene_y"], -6.0, 6.0)

        if not changed:
            break


def build_scene_dict_from_plan(
    plan: Dict[str, Any], asset_dimensions: Dict[str, List[float]], scene_text: str
) -> Dict[str, Any]:
    entries = build_subject_entries(plan, asset_dimensions, scene_text)

    subjects_data = []
    for entry in entries:
        subject_dict = {
            "name": entry["name"],
            "type": entry["type"],
            "dims": tuple(entry["dims"]),
            "x": [round(entry["scene_x"] - SCENE_X_SAVE_OFFSET, 6)],
            "y": [round(entry["scene_y"], 6)],
            "z": [round(entry["scene_z"], 6)],
            "azimuth": [math.radians(entry["azimuth_deg"])],
            "bbox": [(0, 0, 0, 0)],
        }
        subjects_data.append(subject_dict)

    surrounding_prompt = str(plan.get("surrounding_prompt") or "").strip()
    if not surrounding_prompt:
        surrounding_prompt = "highly realistic."

    camera_elevation_deg = clamp(
        to_float(plan.get("camera_elevation_deg"), DEFAULT_CAMERA_ELEVATION_DEG),
        0.0,
        90.0,
    )

    return {
        "subjects_data": subjects_data,
        "camera_data": {
            "camera_elevation": math.radians(camera_elevation_deg),
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


def save_scene_pkl(scene_dict: Dict[str, Any], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as handle:
        pickle.dump(strip_meta(scene_dict), handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="调用大模型，把英文场景文字转换成 example0.pkl 同结构的场景 pkl。",
        epilog=(
            '示例:\n'
            '  python agent_text2pkl0.py '
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


def main() -> None:
    args = parse_args()

    scene_text = args.scene_text.strip()
    if not scene_text:
        raise RuntimeError("场景描述为空。")

    asset_dimensions = load_asset_dimensions(ASSET_DIMENSIONS_PATH)
    plan = build_local_plan_from_known_types(scene_text, asset_dimensions)
    if plan is None:
        api_key = load_api_key(args.api_key)
        prompt = build_prompt(scene_text, list(asset_dimensions.keys()))
        plan = call_llm_for_plan(prompt, api_key=api_key, model=args.model)

        explicit_mentions = extract_explicit_object_mentions(scene_text)
        plan_subjects = plan.get("subjects") if isinstance(plan, dict) else None
        current_subject_names: List[str] = []
        custom_without_size = False
        if isinstance(plan_subjects, list):
            for item in plan_subjects:
                if not isinstance(item, dict):
                    continue
                name = sanitize_name(
                    item.get("name") or item.get("description"),
                    default_name="object",
                )
                current_subject_names.append(name)
                asset_type = match_asset_type(
                    item.get("type"),
                    name,
                    list(asset_dimensions.keys()),
                )
                if asset_type == CUSTOM_ASSET_TYPE and not has_explicit_size(item):
                    custom_without_size = True

        missing_mentions = find_missing_mentions(explicit_mentions, current_subject_names)
        if missing_mentions or custom_without_size:
            refinement_prompt = build_refinement_prompt(
                scene_text=scene_text,
                initial_plan=plan,
                explicit_mentions=explicit_mentions,
                allowed_types=list(asset_dimensions.keys()),
            )
            plan = call_llm_for_plan(refinement_prompt, api_key=api_key, model=args.model)

        if not plan_has_valid_subjects(plan):
            local_fallback = build_local_plan_from_known_types(scene_text, asset_dimensions)
            if local_fallback is not None:
                plan = local_fallback

    scene_dict = build_scene_dict_from_plan(plan, asset_dimensions, scene_text)

    if args.print_plan:
        print("===== Scene Plan =====")
        print(json.dumps(plan, ensure_ascii=False, indent=2))

    if args.print_scene_json:
        print("===== Final Scene Dict =====")
        print(json.dumps(strip_meta(scene_dict), ensure_ascii=False, indent=2))

    save_scene_pkl(scene_dict, args.output)
    print(f"场景文件已保存到: {args.output}")


if __name__ == "__main__":
    main()
