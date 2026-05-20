#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅做“按 guidance 修正 pkl”：
1) 输入 scene_text + scene_pkl + guidance
2) 严格按 agent_check_pkl_v5.py 的 guidance 规则改动
3) 输出修正后的 pkl
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, List, Set

import agent_check_pkl_v5 as core


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


def strict_apply_guidance(
    scene_text: str,
    guidance_text: str,
    scene_dict: Dict[str, Any],
    allowed_types: List[str],
    reference_dims_map: Dict[str, List[float]],
    default_dims_map: Dict[str, List[float]],
) -> Dict[str, Any]:
    current_subjects = core.summarize_scene_subjects(scene_dict, allowed_types, reference_dims_map)
    current_camera = core.summarize_camera(scene_dict)
    guidance_target_camera = core.apply_guidance_camera_directives(current_camera, guidance_text)
    guidance_target_names: Set[str] = strict_collect_guidance_target_names(current_subjects, guidance_text)
    if not guidance_target_names:
        guidance_target_names = core.collect_subject_names_mentioned_in_text(current_subjects, guidance_text)

    forced_scene_dict, _ = core.build_guidance_forced_scene(
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

    actual_subjects = core.summarize_scene_subjects(forced_scene_dict, allowed_types, reference_dims_map)
    actual_camera = core.summarize_camera(forced_scene_dict)
    expected_subjects = core.build_guidance_preview_subjects(
        current_subjects,
        guidance_text,
        reference_dims_map,
        default_dims_map,
        guidance_target_names,
    )

    subject_issues = core.collect_guidance_application_issues(
        original_subjects=current_subjects,
        actual_subjects=actual_subjects,
        expected_subjects=expected_subjects,
        guidance_target_names=guidance_target_names,
    )
    camera_issues = core.collect_camera_guidance_application_issues(
        original_camera=current_camera,
        actual_camera=actual_camera,
        expected_camera=guidance_target_camera,
    )
    unresolved = core.deduplicate_issues(subject_issues + camera_issues)
    if unresolved:
        issue_lines = core.format_issue_lines(unresolved)
        lines_preview = "\n".join(issue_lines[:8])
        raise RuntimeError(
            "guidance 未被完整执行到位，已拒绝输出结果。"
            + ("\n" + lines_preview if lines_preview else "")
        )
    return forced_scene_dict


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
