#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审核并修正已有的场景 PKL。

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
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSET_DIMENSIONS_PATH = os.path.join(REPO_ROOT, "inference", "asset_dimensions.json")
OBJECT_SCALES_PATH = os.path.join(REPO_ROOT, "inference", "object_scales.py")

API_HOST = "api.chatanywhere.tech"
API_PATH = "/v1/responses"
API_MODEL = "gpt-5.2"

SCENE_X_SAVE_OFFSET = 6.0
DEFAULT_CAMERA_ELEVATION_DEG = 12.0
DEFAULT_LENS_MM = 50.0
DEFAULT_GLOBAL_SCALE = 1.0
DEFAULT_REFERENCE_DIM_SCALE = 2.25

CAMERA_CENTER_X = -6.0
CAMERA_RADIUS = 6.0
CAMERA_SENSOR_MM = 36.0
VISIBILITY_MARGIN = 0.02
MIN_CAMERA_LENS_MM = 20.0
MAX_CAMERA_LENS_MM = 60.0

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
        "camera_elevation_deg": round(clamp(camera_elevation_deg, 0.0, 90.0), 6),
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
) -> Dict[str, Any]:
    hard_issues: List[Dict[str, str]] = []
    soft_issues: List[Dict[str, str]] = []

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
        if any(not math.isfinite(v) or v <= 0.0 for v in dims):
            hard_issues.append(
                {
                    "category": "invalid_dims",
                    "object": subject["name"],
                    "details": f"dims 非法: {dims}",
                }
            )

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

        reference_dims = reference_dims_map.get(subject["type"])
        if reference_dims:
            ratios = [
                dims[idx] / reference_dims[idx] if reference_dims[idx] > 1e-6 else 1.0
                for idx in range(3)
            ]
            if any(ratio < 0.35 or ratio > 2.8 for ratio in ratios):
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
        first_reference = reference_dims_map.get(first_subject["type"])
        if not first_reference:
            continue
        first_scalar = max(first_reference)
        first_actual = max(first_subject["dims"])

        for second_subject in subjects[idx + 1 :]:
            second_reference = reference_dims_map.get(second_subject["type"])
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


def build_audit_prompt(
    scene_text: str,
    current_subjects: List[Dict[str, Any]],
    camera_data: Dict[str, float],
    local_checks: Dict[str, Any],
    allowed_types: List[str],
    reference_dims_map: Dict[str, List[float]],
) -> str:
    payload = {
        "scene_text": scene_text,
        "coordinate_system": {
            "scene_x": "larger means more in front / closer to camera",
            "scene_y": "smaller means more to the left, larger means more to the right",
            "scene_z": "ground offset; ground objects should usually be 0.0",
            "dims": "full size [width, depth, height], not half extents",
        },
        "current_camera": camera_data,
        "current_subjects": current_subjects,
        "local_deterministic_checks": {
            "hard_issues": local_checks["hard_issues"],
            "soft_issues": local_checks["soft_issues"],
            "visibility": local_checks["visibility"],
        },
        "allowed_asset_types": sorted(allowed_types),
        "reference_dims_for_current_types": build_relevant_reference_dims(
            current_subjects,
            reference_dims_map,
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
4. 图片里必须能看到每个 cube 的完整结构，不能只露出一半。若当前相机或布局会裁边，就必须判为不合适。

额外要求：
1. 只有在完全合格时，才能输出 verdict="accept"。
2. 如果任何一条不满足，就输出 verdict="reject"。
3. 如果 reject，必须给出完整的修正版 subjects 列表，而不是只改一部分。
4. 修正版应尽量少改，但不能牺牲正确性。
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
    current_subjects: List[Dict[str, Any]],
    camera_data: Dict[str, float],
    local_checks: Dict[str, Any],
    allowed_types: List[str],
    reference_dims_map: Dict[str, List[float]],
) -> str:
    payload = {
        "scene_text": scene_text,
        "current_camera": camera_data,
        "current_subjects": current_subjects,
        "known_issues": {
            "hard_issues": local_checks["hard_issues"],
            "soft_issues": local_checks["soft_issues"],
        },
        "allowed_asset_types": sorted(allowed_types),
        "reference_dims_for_all_types": reference_dims_map,
    }

    return f"""
你需要直接给出一个“严格合格”的修正版 3D 场景方案。

要求：
1. 根据文字描述保留且仅保留被提到的物体。
2. scene_x / scene_y 的空间关系必须和文字完全匹配。
3. ground 物体默认 scene_z=0.0。
4. dims 是完整尺寸 [width, depth, height]，必须符合现实世界比例。
5. 所有 cube 都必须完整出现在图中，必要时可以适度调整 camera_lens_mm。
6. 返回 JSON only。

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
        prefer_reference_dims=True,
    )
    if not subjects:
        return [], {}

    camera_data = {
        "camera_elevation_deg": clamp(
            to_float(local_plan.get("camera_elevation_deg"), DEFAULT_CAMERA_ELEVATION_DEG),
            0.0,
            90.0,
        ),
        "camera_lens_mm": DEFAULT_LENS_MM,
    }
    return subjects, camera_data


def resolve_collisions(subjects: List[Dict[str, Any]], max_rounds: int = 16) -> None:
    for _ in range(max_rounds):
        changed = False
        for i in range(len(subjects)):
            for j in range(i + 1, len(subjects)):
                first_subject = subjects[i]
                second_subject = subjects[j]

                dx = second_subject["scene_x"] - first_subject["scene_x"]
                dy = second_subject["scene_y"] - first_subject["scene_y"]

                depth_gap = (first_subject["dims"][1] + second_subject["dims"][1]) * 0.55
                width_gap = (first_subject["dims"][0] + second_subject["dims"][0]) * 0.55

                overlap_x = abs(dx) < depth_gap
                overlap_y = abs(dy) < width_gap
                if not (overlap_x and overlap_y):
                    continue

                changed = True
                if abs(dx) >= abs(dy):
                    push = (depth_gap - abs(dx)) + 0.05
                    sign = 1.0 if dx >= 0 else -1.0
                    if abs(dx) < 1e-6:
                        sign = 1.0
                    second_subject["scene_x"] += push * sign
                else:
                    push = (width_gap - abs(dy)) + 0.05
                    sign = 1.0 if dy >= 0 else -1.0
                    if abs(dy) < 1e-6:
                        sign = 1.0
                    second_subject["scene_y"] += push * sign

        if not changed:
            break


def clone_subjects(subjects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return copy.deepcopy(subjects)


def apply_layout_transform(
    subjects: List[Dict[str, Any]],
    layout_scale: float,
    scene_x_shift: float,
    scene_y_shift: float,
) -> List[Dict[str, Any]]:
    transformed = clone_subjects(subjects)
    if not transformed:
        return transformed

    mean_x = sum(subject["scene_x"] for subject in transformed) / len(transformed)
    mean_y = sum(subject["scene_y"] for subject in transformed) / len(transformed)

    for subject in transformed:
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
) -> Dict[str, Any]:
    if not subjects:
        return {
            "subjects": [],
            "camera_elevation_deg": camera_elevation_deg,
            "lens_mm": lens_mm,
            "passed": True,
            "visibility": evaluate_scene_visibility([], camera_elevation_deg, lens_mm, margin=VISIBILITY_MARGIN),
        }

    candidate_lenses = sorted(
        {
            round(clamp(lens_mm, MIN_CAMERA_LENS_MM, MAX_CAMERA_LENS_MM), 6),
            50.0,
            45.0,
            40.0,
            35.0,
            30.0,
            28.0,
            24.0,
            20.0,
        },
        reverse=True,
    )
    candidate_scales = [1.0, 0.94, 0.88, 0.82, 0.76]
    candidate_x_shifts = [0.0, -0.4, -0.8, -1.2, -1.6, -2.0, -2.5, -3.0]

    best_solution: Optional[Dict[str, Any]] = None

    for candidate_lens in candidate_lenses:
        for layout_scale in candidate_scales:
            for scene_x_shift in candidate_x_shifts:
                scene_y_shift = optimize_scene_y_shift(
                    subjects,
                    camera_elevation_deg=camera_elevation_deg,
                    lens_mm=candidate_lens,
                    layout_scale=layout_scale,
                    scene_x_shift=scene_x_shift,
                )
                transformed = apply_layout_transform(
                    subjects,
                    layout_scale=layout_scale,
                    scene_x_shift=scene_x_shift,
                    scene_y_shift=scene_y_shift,
                )
                fit_quality = evaluate_fit_quality(
                    transformed,
                    camera_elevation_deg=camera_elevation_deg,
                    lens_mm=candidate_lens,
                    margin=VISIBILITY_MARGIN,
                )

                score = fit_quality["overflow"] * 10000.0
                score += abs(candidate_lens - lens_mm) * 0.2
                score += abs(1.0 - layout_scale) * 8.0
                score += abs(scene_x_shift) * 1.0
                score += abs(scene_y_shift) * 0.35

                candidate_solution = {
                    "subjects": transformed,
                    "camera_elevation_deg": round(camera_elevation_deg, 6),
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

    scene_dict["camera_data"] = {
        "camera_elevation": math.radians(clamp(camera_elevation_deg, 0.0, 90.0)),
        "lens": clamp(lens_mm, MIN_CAMERA_LENS_MM, 200.0),
        "global_scale": to_float(base_camera_data.get("global_scale"), DEFAULT_GLOBAL_SCALE),
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


def request_correction_plan(
    scene_text: str,
    current_subjects: List[Dict[str, Any]],
    camera_data: Dict[str, float],
    local_checks: Dict[str, Any],
    allowed_types: List[str],
    reference_dims_map: Dict[str, List[float]],
    api_key: str,
    model: str,
) -> Dict[str, Any]:
    fix_prompt = build_fix_prompt(
        scene_text=scene_text,
        current_subjects=current_subjects,
        camera_data=camera_data,
        local_checks=local_checks,
        allowed_types=allowed_types,
        reference_dims_map=reference_dims_map,
    )
    return call_llm_for_json(fix_prompt, api_key=api_key, model=model)


def format_issue_lines(issues: List[Dict[str, str]]) -> List[str]:
    lines: List[str] = []
    for idx, issue in enumerate(issues, start=1):
        object_name = issue.get("object", "").strip()
        object_prefix = f"[{object_name}] " if object_name else ""
        lines.append(f"{idx}. {object_prefix}{issue.get('details', '').strip()}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "审核已有场景 pkl，并在不合适时自动生成修正版 pkl。"
        ),
        epilog=(
            '示例:\n'
            '  python agent_check_pkl0.py '
            '--scene-text "A bulldozer is positioned at the center of the image." '
            '--scene-pkl inference/saved_scenes/example.pkl '
            '--output inference/saved_scenes/example_fixed.pkl'
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

    scene_pkl = args.scene_pkl.strip()
    if not scene_pkl:
        raise RuntimeError("pkl 路径为空。")
    if not os.path.isabs(scene_pkl):
        scene_pkl = os.path.join(REPO_ROOT, scene_pkl)
    if not os.path.exists(scene_pkl):
        raise RuntimeError(f"找不到 pkl 文件：{scene_pkl}")

    asset_dimensions = load_asset_dimensions(ASSET_DIMENSIONS_PATH)
    object_scales = load_object_scales(OBJECT_SCALES_PATH)
    reference_dims_map = build_reference_dims_map(asset_dimensions, object_scales)
    allowed_types = list(asset_dimensions.keys())

    scene_dict = load_scene_pkl(scene_pkl)
    current_subjects = summarize_scene_subjects(scene_dict, allowed_types, reference_dims_map)
    current_camera = summarize_camera(scene_dict)
    local_checks = run_local_checks(
        current_subjects,
        camera_elevation_deg=current_camera["camera_elevation_deg"],
        lens_mm=current_camera["lens_mm"],
        reference_dims_map=reference_dims_map,
    )

    api_key = load_api_key(args.api_key)
    audit_prompt = build_audit_prompt(
        scene_text=scene_text,
        current_subjects=current_subjects,
        camera_data=current_camera,
        local_checks=local_checks,
        allowed_types=allowed_types,
        reference_dims_map=reference_dims_map,
    )
    audit_json = call_llm_for_json(audit_prompt, api_key=api_key, model=args.model)

    if args.print_audit_json:
        print("===== LLM Audit JSON =====")
        print(json.dumps(audit_json, ensure_ascii=False, indent=2))

    verdict = str(audit_json.get("verdict", "")).strip().lower()
    summary = str(audit_json.get("summary", "")).strip()
    audit_issues = normalize_issue_entries(audit_json.get("issues"))
    hard_issues = list(local_checks["hard_issues"])
    should_accept = verdict == "accept" and not audit_issues and not hard_issues

    if should_accept:
        print("结论：当前 pkl 合适。")
        if summary:
            print(f"说明：{summary}")
        return

    corrected_subjects = normalize_subjects_from_model(
        audit_json.get("subjects"),
        allowed_types=allowed_types,
        reference_dims_map=reference_dims_map,
    )

    if not corrected_subjects:
        fix_json = request_correction_plan(
            scene_text=scene_text,
            current_subjects=current_subjects,
            camera_data=current_camera,
            local_checks=local_checks,
            allowed_types=allowed_types,
            reference_dims_map=reference_dims_map,
            api_key=api_key,
            model=args.model,
        )
        corrected_subjects = normalize_subjects_from_model(
            fix_json.get("subjects"),
            allowed_types=allowed_types,
            reference_dims_map=reference_dims_map,
        )
        if not corrected_subjects:
            corrected_subjects, local_camera = build_local_fallback_subjects(
                scene_text=scene_text,
                asset_dimensions=asset_dimensions,
                allowed_types=allowed_types,
                reference_dims_map=reference_dims_map,
            )
            if not corrected_subjects:
                raise RuntimeError("大模型没有返回可用的修正版 subjects，本地 fallback 也无法生成修正版。")
            corrected_camera_elevation_deg = clamp(
                to_float(
                    local_camera.get("camera_elevation_deg"),
                    current_camera["camera_elevation_deg"],
                ),
                0.0,
                90.0,
            )
            corrected_lens_mm = clamp(
                to_float(local_camera.get("camera_lens_mm"), current_camera["lens_mm"]),
                MIN_CAMERA_LENS_MM,
                200.0,
            )
            print("警告：大模型未返回可用修正版，已回退到本地已知类型修正规则。")
        else:
            corrected_camera_elevation_deg = clamp(
                to_float(
                    fix_json.get("camera_elevation_deg"),
                    current_camera["camera_elevation_deg"],
                ),
                0.0,
                90.0,
            )
            corrected_lens_mm = clamp(
                to_float(fix_json.get("camera_lens_mm"), current_camera["lens_mm"]),
                MIN_CAMERA_LENS_MM,
                200.0,
            )
    else:
        corrected_camera_elevation_deg = clamp(
            to_float(
                audit_json.get("camera_elevation_deg"),
                current_camera["camera_elevation_deg"],
            ),
            0.0,
            90.0,
        )
        corrected_lens_mm = clamp(
            to_float(audit_json.get("camera_lens_mm"), current_camera["lens_mm"]),
            MIN_CAMERA_LENS_MM,
            200.0,
        )

    resolve_collisions(corrected_subjects)
    fit_result = fit_scene_to_camera(
        corrected_subjects,
        camera_elevation_deg=corrected_camera_elevation_deg,
        lens_mm=corrected_lens_mm,
    )
    fitted_subjects = fit_result["subjects"]

    final_scene_dict = build_scene_dict_from_subjects(
        base_scene_dict=scene_dict,
        subjects=fitted_subjects,
        camera_elevation_deg=fit_result["camera_elevation_deg"],
        lens_mm=fit_result["lens_mm"],
        scene_text=scene_text,
    )
    final_subjects = summarize_scene_subjects(final_scene_dict, allowed_types, reference_dims_map)
    final_camera = summarize_camera(final_scene_dict)
    final_checks = run_local_checks(
        final_subjects,
        camera_elevation_deg=final_camera["camera_elevation_deg"],
        lens_mm=final_camera["lens_mm"],
        reference_dims_map=reference_dims_map,
    )

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(REPO_ROOT, output_path)
    save_scene_pkl(final_scene_dict, output_path)

    print("结论：当前 pkl 不合适。")
    if summary:
        print(f"说明：{summary}")

    all_issue_lines = format_issue_lines(audit_issues + hard_issues)
    if all_issue_lines:
        print("存在的问题：")
        for line in all_issue_lines:
            print(line)

    print(f"已保存修正版 pkl：{output_path}")

    if fit_result.get("transform"):
        transform = fit_result["transform"]
        print(
            "自动入框调整："
            f"lens={fit_result['lens_mm']:.2f}mm, "
            f"layout_scale={transform['layout_scale']:.3f}, "
            f"scene_x_shift={transform['scene_x_shift']:.3f}, "
            f"scene_y_shift={transform['scene_y_shift']:.3f}"
        )

    if final_checks["hard_issues"]:
        print("警告：修正版仍存在本地硬性问题，请人工复核。")
        for line in format_issue_lines(final_checks["hard_issues"]):
            print(line)

    if args.print_final_scene_json:
        print("===== Final Scene Dict =====")
        print(json.dumps(final_scene_dict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
