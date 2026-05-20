#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automated plan -> execute -> verify -> repair loop for scene pkls."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent
FINAL_REPAIR_EXTRA_PASSES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic harness loop for text-to-pkl scenes.")
    parser.add_argument("--scene-text", required=True, help="Scene text prompt.")
    parser.add_argument("--output", required=True, help="Final pkl output path.")
    parser.add_argument("--placeholder-prompt", default="", help="Optional infer2.py placeholder prompt. If omitted, infer2 is skipped.")
    parser.add_argument("--objects", default="", help="Optional object manifest shorthand, e.g. table:1,bowl:1.")
    parser.add_argument("--object-manifest", default="", help="Optional explicit object_manifest.json.")
    parser.add_argument("--max-rounds", type=int, default=4, help="Maximum verify/repair rounds.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable.")
    parser.add_argument("--run-dir", default="", help="Optional run directory.")
    parser.add_argument("--api-key", default="", help="Optional API key passed to agent_text2pkl_v5.py.")
    parser.add_argument("--model", default="", help="Optional model passed to agent_text2pkl_v5.py.")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def run_command(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + shlex.join(cmd) + "\n")
        handle.flush()
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, stdout=handle, stderr=subprocess.STDOUT)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sidecar(path: Path, suffix: str) -> Path:
    return path.with_suffix(f".{suffix}") if path.suffix == ".pkl" else Path(str(path) + f".{suffix}")


def repair_plan_action_count(validation: Dict[str, Any], repair_plan_path: Path) -> int:
    harness_payload = validation.get("harness", {})
    repair_plan = harness_payload.get("repair_plan", {})
    action_count = repair_plan.get("action_count")
    if action_count is not None:
        return int(action_count)
    if repair_plan_path.exists():
        return int(load_json(repair_plan_path).get("action_count", 0))
    return int(harness_payload.get("repair_action_count", 0) or 0)


def repair_actions_applied(actions_path: Path) -> int:
    if not actions_path.exists():
        return 0
    with actions_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_opinion_command(
    args: argparse.Namespace,
    scene_pkl: Path,
    json_output: Path,
    repair_plan_output: Path,
    manifest_source_pkl: Path,
) -> List[str]:
    opinion_cmd = [
        args.python_bin,
        str(REPO_ROOT / "agent_opinion.py"),
        "--scene-text",
        args.scene_text,
        "--scene-pkl",
        str(scene_pkl),
        "--json-output",
        str(json_output),
        "--repair-plan-output",
        str(repair_plan_output),
    ]
    manifest_path = sidecar(manifest_source_pkl, "object_manifest.json")
    predicates_path = sidecar(manifest_source_pkl, "predicates.json")
    if manifest_path.exists():
        opinion_cmd.extend(["--object-manifest", str(manifest_path)])
    elif args.object_manifest:
        opinion_cmd.extend(["--object-manifest", str(resolve_path(args.object_manifest))])
    if predicates_path.exists():
        opinion_cmd.extend(["--predicates-json", str(predicates_path)])
    if args.objects:
        opinion_cmd.extend(["--objects", args.objects])
    return opinion_cmd


def print_round_evaluation(round_idx: int, validation: Dict[str, Any], repair_plan_path: Path) -> None:
    print(f"\n===== Harness round {round_idx:02d} evaluation =====")
    print(f"overall_pass: {bool(validation.get('overall_pass', False))}")

    criteria = validation.get("criteria", {})
    if criteria:
        print("criteria:")
        for criterion_name, criterion_payload in criteria.items():
            criterion_pass = bool(criterion_payload.get("pass", False))
            issues = criterion_payload.get("issues", []) or []
            print(f"- {criterion_name}: {'PASS' if criterion_pass else 'FAIL'} ({len(issues)} issue(s))")
            for issue in issues:
                category = str(issue.get("category", "")).strip()
                obj = str(issue.get("object", "")).strip()
                details = str(issue.get("details") or issue.get("expected") or "").strip()
                issue_text = " ".join(part for part in (f"[{category}]" if category else "", obj, details) if part)
                print(f"  - {issue_text}")

    suggestions_cn = validation.get("suggestions_cn") or validation.get("suggestions") or []
    suggestions_en = validation.get("suggestions_en") or []
    print("suggestions_cn:")
    if suggestions_cn:
        for idx, suggestion in enumerate(suggestions_cn, start=1):
            print(f"{idx}. {suggestion}")
    else:
        print("无")

    print("suggestions_en:")
    if suggestions_en:
        for idx, suggestion in enumerate(suggestions_en, start=1):
            print(f"{idx}. {suggestion}")
    else:
        print("None")

    harness_payload = validation.get("harness", {})
    harness_validation = harness_payload.get("validation", {})
    if harness_validation:
        print(f"harness_validation_pass: {bool(harness_validation.get('overall_pass', False))}")
        issue_counts = harness_validation.get("issue_counts", {})
        if issue_counts:
            print(f"harness_issue_counts: {issue_counts}")

    repair_plan = harness_payload.get("repair_plan", {})
    action_count = repair_plan.get("action_count")
    if action_count is None:
        action_count = harness_payload.get("repair_action_count", 0)
    print(f"repair_plan: {repair_plan_path}")
    print(f"repair_action_count: {action_count}")


def main() -> None:
    args = parse_args()
    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be >= 1")

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = resolve_path(args.run_dir) if args.run_dir else REPO_ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_scene_harness")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input.txt").write_text(args.scene_text + "\n", encoding="utf-8")

    raw_pkl = run_dir / "round_00_scene.pkl"
    text_cmd = [
        args.python_bin,
        str(REPO_ROOT / "agent_text2pkl_v5.py"),
        "--scene-text",
        args.scene_text,
        "--output",
        str(raw_pkl),
    ]
    if args.objects:
        text_cmd.extend(["--objects", args.objects])
    if args.object_manifest:
        text_cmd.extend(["--object-manifest", str(resolve_path(args.object_manifest))])
    if args.api_key:
        text_cmd.extend(["--api-key", args.api_key])
    if args.model:
        text_cmd.extend(["--model", args.model])
    run_command(text_cmd, run_dir / "commands.log")

    current_pkl = raw_pkl
    final_validation: Dict[str, Any] = {}
    stalled_reason = ""
    for round_idx in range(args.max_rounds):
        round_dir = run_dir / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        validation_path = round_dir / "validation.json"
        repair_plan_path = round_dir / "repair_plan.json"
        opinion_cmd = build_opinion_command(args, current_pkl, validation_path, repair_plan_path, current_pkl)
        run_command(opinion_cmd, run_dir / "commands.log")
        final_validation = load_json(validation_path)
        print_round_evaluation(round_idx, final_validation, repair_plan_path)
        if final_validation.get("overall_pass"):
            output_path.write_bytes(current_pkl.read_bytes())
            break
        if round_idx == args.max_rounds - 1:
            current_repair_plan_path = repair_plan_path
            for final_pass_idx in range(FINAL_REPAIR_EXTRA_PASSES):
                planned_actions = repair_plan_action_count(final_validation, current_repair_plan_path)
                if planned_actions <= 0:
                    break
                final_repaired_pkl = run_dir / f"round_{round_idx + 1 + final_pass_idx:02d}_scene.pkl"
                reverse_cmd = [
                    args.python_bin,
                    str(REPO_ROOT / "agent_reverse.py"),
                    "--scene-text",
                    args.scene_text,
                    "--scene-pkl",
                    str(current_pkl),
                    "--repair-plan",
                    str(current_repair_plan_path),
                    "--output",
                    str(final_repaired_pkl),
                ]
                run_command(reverse_cmd, run_dir / "commands.log")
                applied_actions_path = sidecar(final_repaired_pkl, "repair_actions.jsonl")
                applied_actions = repair_actions_applied(applied_actions_path)
                same_pkl = current_pkl.read_bytes() == final_repaired_pkl.read_bytes()
                current_pkl = final_repaired_pkl
                if planned_actions > 0 and (applied_actions == 0 or same_pkl):
                    stalled_reason = (
                        f"final repair stalled after max round {round_idx:02d}, pass {final_pass_idx + 1}: "
                        f"planned_actions={planned_actions}, applied_actions={applied_actions}, same_pkl={same_pkl}"
                    )
                    print(stalled_reason)
                    break
                final_validation_path = round_dir / f"final_validation_after_repair_{final_pass_idx + 1}.json"
                final_repair_plan_path = round_dir / f"final_repair_plan_after_repair_{final_pass_idx + 1}.json"
                final_opinion_cmd = build_opinion_command(
                    args,
                    current_pkl,
                    final_validation_path,
                    final_repair_plan_path,
                    raw_pkl,
                )
                run_command(final_opinion_cmd, run_dir / "commands.log")
                final_validation = load_json(final_validation_path)
                print_round_evaluation(round_idx + 1, final_validation, final_repair_plan_path)
                if final_validation.get("overall_pass"):
                    output_path.write_bytes(current_pkl.read_bytes())
                    break
                current_repair_plan_path = final_repair_plan_path
            break
        repaired_pkl = run_dir / f"round_{round_idx + 1:02d}_scene.pkl"
        reverse_cmd = [
            args.python_bin,
            str(REPO_ROOT / "agent_reverse.py"),
            "--scene-text",
            args.scene_text,
            "--scene-pkl",
            str(current_pkl),
            "--repair-plan",
            str(repair_plan_path),
            "--output",
            str(repaired_pkl),
        ]
        run_command(reverse_cmd, run_dir / "commands.log")
        applied_actions_path = sidecar(repaired_pkl, "repair_actions.jsonl")
        planned_actions = repair_plan_action_count(final_validation, repair_plan_path)
        applied_actions = repair_actions_applied(applied_actions_path)
        same_pkl = current_pkl.read_bytes() == repaired_pkl.read_bytes()
        if planned_actions > 0 and (applied_actions == 0 or same_pkl):
            stalled_reason = (
                f"repair stalled at round {round_idx:02d}: planned_actions={planned_actions}, "
                f"applied_actions={applied_actions}, same_pkl={same_pkl}"
            )
            print(stalled_reason)
            break
        current_pkl = repaired_pkl

    summary = {
        "run_dir": str(run_dir),
        "final_pkl": str(output_path if output_path.exists() else current_pkl),
        "overall_pass": bool(final_validation.get("overall_pass", False)),
        "validation": final_validation,
    }
    if stalled_reason:
        summary["stalled"] = True
        summary["stalled_reason"] = stalled_reason
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not summary["overall_pass"]:
        print(f"Harness failed after {args.max_rounds} rounds. See {run_dir / 'summary.json'}")
        return

    print(f"Harness passed. Final pkl: {output_path}")
    print(f"Run dir: {run_dir}")
    if args.placeholder_prompt.strip():
        infer_cmd = [
            args.python_bin,
            str(REPO_ROOT / "infer2.py"),
            "--scene-pkls",
            str(output_path),
            "--placeholder-prompt",
            args.placeholder_prompt,
        ]
        run_command(infer_cmd, run_dir / "commands.log")


if __name__ == "__main__":
    main()
