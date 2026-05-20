#!/usr/bin/env python3
"""原样读取 infer2.py 使用的场景 PKL，并把内容保存到 JSON 文件。

重要要求：
  - **完全忠实**：不新增字段、不删除字段、不改字段名；
  - JSON 中的结构应当和原始 pkl 里的对象一模一样。

因此，本文件：
  - 只负责把 pickle.load 得到的对象做类型转换（主要是 numpy -> Python 原生），
  - 然后直接 json.dump 保存出去；
  - 不再添加诸如 "source_pkl" 之类的额外包装字段。
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict

try:
    import numpy as np
except ModuleNotFoundError:
    np = None





def to_builtin(value: Any) -> Any:
    """把 pkl 中可能出现的 numpy 类型递归转换成 Python 原生类型。

    注意：
    这里的目的只是为了让内容能被 JSON 序列化，
    不会修改字段含义，不会改坐标，不会改角度。
    """
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
    """直接读取原始 pkl 文件。

    这里严格对应 `inference/app.py:433` 附近写入的 scene_dict，
    即原样读取以下结构：
    - subjects_data
    - camera_data
    - surrounding_prompt
    - inference_params
    """
    try:
        with open(pkl_path, "rb") as handle:
            return pickle.load(handle)
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            raise RuntimeError("读取该 pkl 需要 numpy，请在项目环境中运行。") from exc
        raise



def export_pkl_contents(pkl_path: str) -> Dict[str, Any]:
    """原样导出 pkl 内容，对外表现为一个“纯 scene_dict”。

    这里不会做任何“纠正”或“恢复”操作，例如：
    - 不会对 x 加 6.0
    - 不会把弧度转成角度
    - 不会重命名字段

    只做一件事：
    把 pickle 里的内容读取出来，并转换成可保存为 JSON 的原生类型。
    """
    # 这里非常关键：只对 numpy 等类型做“无损转换”，
    # 不增加任何多余的键值对。返回值就是原始 scene_dict
    # 的一个“JSON 兼容版”。
    raw_scene = load_raw_pkl(pkl_path)
    return to_builtin(raw_scene)



def save_exported_json(exported: Dict[str, Any], output_json_path: str) -> None:
    """把导出的原始内容保存到 JSON 文件。

    注意：
      - 传入的 exported 就是原始 pkl 的内容（仅做了类型转换），
      - 不会在这里再加任何额外字段。
    """
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(exported, handle, ensure_ascii=False, indent=2)



def main() -> None:
    """脚本入口。

    默认读取 `inference/saved_scenes/example2.pkl`，
    并把原样导出的内容保存到 `inference/read_pkl.json`。

    导出的 JSON 结构会与原始 pkl 内部对象一一对应，
    不会有 "source_pkl"、"raw_scene_dict" 等包装字段。
    """


    pkl_path = "inference/gradio_result_files/good_e_test4.12_6_fixed3.2.pkl"
    output_json_path =  "inference/gradio_result_fist4.12_6_fixed3.2.json"

    exported = export_pkl_contents(pkl_path)
    save_exported_json(exported, output_json_path)

    print(f"已读取 PKL: {pkl_path}")
    print(f"已保存原始导出结果到: {output_json_path}")
    print(json.dumps(exported, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
