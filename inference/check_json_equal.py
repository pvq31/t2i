#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference/check_json_equal.py

用途：
  对比两个 JSON 文件（这里固定为：
    - inference/read_pkl.json
    - inference/read_pkl2.2.json
 ），检查它们是否**完全一致**。

说明：
  - 如果完全一致，打印一行提示。
  - 如果不一致，打印出所有差异的“路径 + 两边的值”，方便人工排查。
"""

from __future__ import annotations

import json
from typing import Any, List





def _load_json(path: str) -> Any:
    """读取 JSON 文件并返回对应的 Python 对象。"""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _diff(a: Any, b: Any, path: str = "") -> List[str]:
    """
    递归比较两个 Python 对象（由 JSON 解析而来）。

    参数
    ----
    a, b : Any
        来自两个 JSON 的任意值（dict / list / 基本类型）。
    path : str
        当前字段所在的“路径”，例如：
          root.raw_scene_dict.subjects_data[0].name

    返回
    ----
    List[str]
        所有差异的描述字符串列表。
    """
    diffs: List[str] = []

    # 类型不同，直接认为不相等
    if type(a) is not type(b):
        diffs.append(
            f"{path or 'root'} 类型不同: {type(a).__name__} != {type(b).__name__}  (左={a!r}, 右={b!r})"
        )
        return diffs

    # 字典：比较 key 集合与各字段值
    if isinstance(a, dict):
        keys_a = set(a.keys())
        keys_b = set(b.keys())

        # 找到只在一侧存在的键
        only_in_a = keys_a - keys_b
        only_in_b = keys_b - keys_a
        for key in sorted(only_in_a):
            diffs.append(f"{path or 'root'}.{key} 仅出现在左侧 JSON 中，右侧缺失。")
        for key in sorted(only_in_b):
            diffs.append(f"{path or 'root'}.{key} 仅出现在右侧 JSON 中，左侧缺失。")

        # 对共同存在的键递归比较
        for key in sorted(keys_a & keys_b):
            child_path = f"{path}.{key}" if path else key
            diffs.extend(_diff(a[key], b[key], child_path))
        return diffs

    # 列表：先比较长度，再逐项比较
    if isinstance(a, list):
        if len(a) != len(b):
            diffs.append(
                f"{path or 'root'} 列表长度不同: 左={len(a)} 右={len(b)}"
            )

        # 对齐较短长度逐项比较
        min_len = min(len(a), len(b))
        for idx in range(min_len):
            child_path = f"{path}[{idx}]" if path else f"[{idx}]"
            diffs.extend(_diff(a[idx], b[idx], child_path))

        # 如果有多余的元素，也记录一下（内容整体提示）
        if len(a) > len(b):
            for idx in range(min_len, len(a)):
                diffs.append(
                    f"{path or 'root'}[{idx}] 仅出现在左侧 JSON 中，值={a[idx]!r}"
                )
        elif len(b) > len(a):
            for idx in range(min_len, len(b)):
                diffs.append(
                    f"{path or 'root'}[{idx}] 仅出现在右侧 JSON 中，值={b[idx]!r}"
                )
        return diffs

    # 基本类型：直接比较值
    if a != b:
        diffs.append(
            f"{path or 'root'} 值不同: 左={a!r} 右={b!r}"
        )
    return diffs


def main() -> None:
    JSON_PATH_1 = "inference/read_pkl2.json"
    JSON_PATH_2 = "inference/read_pkl2.2.json"

    """脚本入口：对比两个固定路径的 JSON 文件。"""
    print(f"读取 JSON 1: {JSON_PATH_1}")
    data1 = _load_json(JSON_PATH_1)

    print(f"读取 JSON 2: {JSON_PATH_2}")
    data2 = _load_json(JSON_PATH_2)

    print("\n开始比较两个 JSON 是否完全一致...\n")
    diffs = _diff(data1, data2, path="root")

    if not diffs:
        print("✅ 两个 JSON 文件完全一致。")
    else:
        print("❌ 两个 JSON 文件存在差异，具体如下：\n")
        for item in diffs:
            print("- " + item)


if __name__ == "__main__":
    main()

