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
    allowed_types_text = ", ".join(sorted(allowed_types))
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
      "type": "one allowed asset type or a close synonym",
      "scene_x": 0.0,
      "scene_y": 0.0,
      "scene_z": 0.0,
      "azimuth_deg": 0.0,
      "size_scale": 1.0
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
8. surrounding_prompt should describe only the environment/background style, not the object list again.

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
    candidates = []

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
        if alias and canonicalize_type(alias) in allowed_lookup:
            return allowed_lookup[canonicalize_type(alias)]

    return "Custom"


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


def build_subject_entries(
    plan: Dict[str, Any], asset_dimensions: Dict[str, List[float]]
) -> List[Dict[str, Any]]:
    raw_subjects = plan.get("subjects")
    if not isinstance(raw_subjects, list) or not raw_subjects:
        raise ValueError("模型没有返回 subjects 列表，或 subjects 为空。")

    allowed_types = list(asset_dimensions.keys())
    entries: List[Dict[str, Any]] = []

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

        default_scale = infer_default_size_scale(name, asset_type)
        size_scale = to_float(item.get("size_scale"), default_scale)
        size_scale = clamp(size_scale, 0.2, 3.0)

        base_dims = asset_dimensions.get(asset_type, [1.0, 1.0, 1.0])
        dims = [round(float(v) * size_scale, 6) for v in base_dims]

        entries.append(
            {
                "name": name,
                "type": asset_type,
                "scene_x": scene_x,
                "scene_y": scene_y,
                "scene_z": scene_z,
                "azimuth_deg": azimuth_deg,
                "dims": dims,
            }
        )

    if not entries:
        raise ValueError("模型返回了 subjects，但无法解析出有效物体。")

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
    entries = build_subject_entries(plan, asset_dimensions)

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
        default="sk-4vGkTSWP4WmCIQaiY1fthMExZHJL5aDl2Ad8QmtZiGcJYJzS",
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
    api_key = load_api_key(args.api_key)
    prompt = build_prompt(scene_text, list(asset_dimensions.keys()))

    api_response = call_llm(prompt, api_key=api_key, model=args.model)
    output_text = extract_output_text(api_response)
    plan = extract_json_object(output_text)
    scene_dict = build_scene_dict_from_plan(plan, asset_dimensions, scene_text)

    if args.print_plan:
        print("===== LLM Scene Plan =====")
        print(json.dumps(plan, ensure_ascii=False, indent=2))

    if args.print_scene_json:
        print("===== Final Scene Dict =====")
        print(json.dumps(strip_meta(scene_dict), ensure_ascii=False, indent=2))

    save_scene_pkl(scene_dict, args.output)
    print(f"场景文件已保存到: {args.output}")


if __name__ == "__main__":
    main()
