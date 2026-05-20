#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按顺序执行 agent_text2pkl_v1.py、重复多次 agent_check_pkl.py、"
            "最后执行 infer2_v2.py。"
        )
    )
    parser.add_argument(
        "--scene-text",
        required=True,
        help="输入给 agent_text2pkl_v1.py 和 agent_check_pkl.py 的场景文字。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="agent_text2pkl_v1.py 的输出 pkl 路径。",
    )
    parser.add_argument(
        "--check-runs",
        type=int,
        required=True,
        help="agent_check_pkl.py 的运行次数。可为 0。",
    )
    parser.add_argument(
        "--placeholder-prompt",
        required=True,
        help="infer2_v2.py 的 placeholder prompt，必须包含 PLACEHOLDER。",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="可选，同时透传给 agent_text2pkl_v1.py 和 agent_check_pkl.py。",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="可选，同时透传给 agent_text2pkl_v1.py 和 agent_check_pkl.py。",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="运行整条链路使用的 Python 解释器。默认是当前解释器。",
    )
    parser.add_argument(
        "--infer-image-size",
        type=int,
        default=None,
        help="可选，透传给 infer2_v2.py 的 --image-size。",
    )
    parser.add_argument(
        "--infer-guidance-scale",
        type=float,
        default=None,
        help="可选，透传给 infer2_v2.py 的 --guidance-scale。",
    )
    parser.add_argument(
        "--infer-lora-weight",
        type=float,
        default=1.0,
        help="可选，透传给 infer2_v2.py 的 --lora-weight。",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def normalize_executable(executable: str) -> str:
    path = Path(executable)
    if path.is_absolute():
        return executable
    if "/" in executable or executable.startswith("."):
        return str((REPO_ROOT / executable).resolve())
    return executable


def derive_fixed_output(raw_output: Path) -> Path:
    suffix = raw_output.suffix if raw_output.suffix else ".pkl"
    stem = raw_output.stem if raw_output.suffix else raw_output.name
    if stem.endswith("_fixed"):
        return raw_output if raw_output.suffix else raw_output.with_suffix(".pkl")
    return raw_output.with_name(f"{stem}_fixed{suffix}")


def run_command(cmd: list[str]) -> None:
    print(f"\n$ {shlex.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    args = parse_args()

    if args.check_runs < 0:
        raise ValueError("--check-runs 不能小于 0。")
    if "PLACEHOLDER" not in args.placeholder_prompt:
        raise ValueError("--placeholder-prompt 必须包含 PLACEHOLDER。")

    python_bin = normalize_executable(args.python_bin)

    raw_output = resolve_path(args.output)
    fixed_output = derive_fixed_output(raw_output)

    text2pkl_script = REPO_ROOT / "agent_text2pkl_v1.py"
    check_pkl_script = REPO_ROOT / "agent_check_pkl.py"
    infer2_script = REPO_ROOT / "infer2_v2.py"

    raw_output.parent.mkdir(parents=True, exist_ok=True)

    text2pkl_cmd = [
        python_bin,
        str(text2pkl_script),
        "--output",
        str(raw_output),
        "--scene-text",
        args.scene_text,
    ]
    if args.api_key:
        text2pkl_cmd.extend(["--api-key", args.api_key])
    if args.model:
        text2pkl_cmd.extend(["--model", args.model])

    run_command(text2pkl_cmd)

    current_scene_pkl = raw_output
    for idx in range(args.check_runs):
        check_cmd = [
            python_bin,
            str(check_pkl_script),
            "--scene-text",
            args.scene_text,
            "--scene-pkl",
            str(current_scene_pkl),
            "--output",
            str(fixed_output),
        ]
        if args.api_key:
            check_cmd.extend(["--api-key", args.api_key])
        if args.model:
            check_cmd.extend(["--model", args.model])

        print(
            f"Running agent_check_pkl.py [{idx + 1}/{args.check_runs}]",
            flush=True,
        )
        run_command(check_cmd)
        current_scene_pkl = fixed_output

    infer_cmd = [
        python_bin,
        str(infer2_script),
        "--scene-pkls",
        str(current_scene_pkl),
        "--placeholder-prompt",
        args.placeholder_prompt,
        "--lora-weight",
        str(args.infer_lora_weight),
    ]
    if args.infer_image_size is not None:
        infer_cmd.extend(["--image-size", str(args.infer_image_size)])
    if args.infer_guidance_scale is not None:
        infer_cmd.extend(["--guidance-scale", str(args.infer_guidance_scale)])
    run_command(infer_cmd)

    print("Pipeline finished.", flush=True)
    print(f"Raw scene pkl: {raw_output}", flush=True)
    print(f"Fixed scene pkl: {fixed_output}", flush=True)
    print(f"Inference input pkl: {current_scene_pkl}", flush=True)


if __name__ == "__main__":
    main()
