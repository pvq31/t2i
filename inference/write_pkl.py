#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference/write_pkl.py

用途：
  把 `inference/read_pkl2.json` 中的内容，**完全忠实**地写回成 pkl 文件。

要求 / 保证：
  - 不增加任何字段（例如 `source_pkl`、`raw_scene_dict` 等）；
  - 不删除任何字段；
  - 不修改字段名、嵌套结构或数值；
  - 输出的 pkl 结构应与 `inference/saved_scenes/example2.pkl` 一致，
    即顶层就是一个 scene_dict，内部包含：
      - subjects_data
      - camera_data
      - surrounding_prompt
      - inference_params
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict





def load_raw_scene_dict_from_json(json_path: str) -> Dict[str, Any]:
    """
    从 `inference/read_pkl2.json` 中读取原始的 scene_dict。

    参数
    ----
    json_path : str
        JSON 文件路径，一般为 `inference/read_pkl2.json`。

    返回
    ----
    dict
        直接可用于 pickle.dump 的 scene_dict。
    """
    with open(json_path, "r", encoding="utf-8") as handle:
        data: Dict[str, Any] = json.load(handle)

    # 这里非常关键：
    #   - read_pkl2.json 中的内容本身就应该是“原始 scene_dict”的 JSON 版本；
    #   - 因此我们**不做任何包装或解包**，直接返回整个对象。
    # 也就是说，JSON 顶层结构是什么，写回 pkl 的顶层结构就是什么。
    return data


def save_scene_dict_to_pkl(scene_dict: Dict[str, Any], pkl_path: str) -> None:
    """
    把 scene_dict 原样保存为 pkl 文件。

    注意：
      - 不会改动任何字段；不做坐标偏移、角度转换等操作。
      - 目标是让输出 pkl 的内部结构与 JSON / example2.pkl 完全一致。
    """
    output_dir = os.path.dirname(pkl_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(pkl_path, "wb") as handle:
        pickle.dump(scene_dict, handle)


def main() -> None:
    """脚本入口：从 read_pkl2.json 读入，再写出 pkl 文件。"""

    READ_JSON_PATH = "inference/saved_scenes/harness_test5.4_6.json"
    OUTPUT_PKL_PATH = "inference/saved_scenes/harness_test5.4_6.pkl"
    print(f"读取 JSON: {READ_JSON_PATH}")
    scene_dict = load_raw_scene_dict_from_json(READ_JSON_PATH)

    print(f"写出 PKL: {OUTPUT_PKL_PATH}")
    save_scene_dict_to_pkl(scene_dict, OUTPUT_PKL_PATH)

    print("完成")


if __name__ == "__main__":
    main()

