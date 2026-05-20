#!/usr/bin/env python3

import argparse
import base64
import contextlib
import io
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image
import tempfile


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.join(REPO_ROOT, "inference")
DEFAULT_SAVE_ROOT = os.path.join(INFERENCE_DIR, "gradio_result_files")
DEFAULT_SCENE_ROOT = os.path.join(INFERENCE_DIR, "saved_scenes")
DEFAULT_SERVER_URLS = {
    "cv": "http://127.0.0.1:5001",
    "final": "http://127.0.0.1:5002",
    "paper": "http://127.0.0.1:5004",
}


def load_config_defaults() -> Tuple[str, str, Dict[str, str]]:
    save_root = DEFAULT_SAVE_ROOT
    scene_root = DEFAULT_SCENE_ROOT
    server_urls = DEFAULT_SERVER_URLS.copy()

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            import inference.config as config

        save_root = getattr(config, "GRADIO_FILES_DIR", save_root)
        scene_root = getattr(config, "SAVED_SCENES_DIR", scene_root)
        server_urls["cv"] = getattr(config, "BLENDER_CV_SERVER_URL", server_urls["cv"])
        server_urls["final"] = getattr(config, "BLENDER_FINAL_SERVER_URL", server_urls["final"])
    except Exception:
        pass

    return save_root, scene_root, server_urls


CONFIG_SAVE_ROOT, CONFIG_SCENE_ROOT, CONFIG_SERVER_URLS = load_config_defaults()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="输入场景 pkl，渲染并保存对应的 cube layout 图。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "scene_pkl_positional",
        nargs="?",
        default=None,
        help="位置参数写法的场景 pkl 路径，例如 inference/saved_scenes/test4.6/example_test4.6_shiyan1_v1.pkl",
    )
    parser.add_argument(
        "output_positional",
        nargs="?",
        default=None,
        help="位置参数写法的输出图片路径。",
    )
    parser.add_argument(
        "--scene_pkl",
        "--scene-pkl",
        dest="scene_pkl",
        default=None,
        help="场景 pkl 路径，例如 inference/saved_scenes/test4.6/example_test4.6_shiyan1_v1.pkl",
    )
    parser.add_argument(
        "-output",
        "--output",
        "-o",
        dest="output",
        default=None,
        help="输出图片路径；如果传目录，就自动使用 cube_xxx.jpg 文件名；不传则自动保存到 inference/gradio_result_files 下。",
    )
    parser.add_argument(
        "--render-mode",
        choices=["cv", "final", "paper"],
        default="final",
        help="渲染模式。要得到和 infer2_v2.py 一样的条件图，使用 final。",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="可选，手动指定 Blender render server 地址。",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="如果后端没有启动，则直接报错，不自动拉起 Blender render server。",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=45.0,
        help="自动启动 Blender render server 时，等待健康检查通过的最长秒数。",
    )
    args = parser.parse_args()

    args.scene_pkl = (
        args.scene_pkl
        or args.scene_pkl_positional
        or "inference/saved_scenes/test4.6/example_test4.6_2_fixed.pkl"
    )
    args.output = args.output or args.output_positional

    return args


def normalize_scene_pickle_path(scene_pkl: str) -> str:
    candidate_paths = []
    if os.path.isabs(scene_pkl):
        candidate_paths.append(scene_pkl)
    else:
        candidate_paths.append(os.path.abspath(scene_pkl))
        candidate_paths.append(os.path.join(REPO_ROOT, scene_pkl))
        candidate_paths.append(os.path.join(CONFIG_SCENE_ROOT, scene_pkl))

    checked_paths = []
    for candidate_path in candidate_paths:
        normalized_path = os.path.abspath(candidate_path)
        if normalized_path in checked_paths:
            continue
        checked_paths.append(normalized_path)
        if os.path.isfile(normalized_path):
            return normalized_path

    raise FileNotFoundError(f"找不到场景 pkl: {scene_pkl}")


def build_cube_layout_filename(scene_pkl_path: str) -> str:
    pkl_basename = os.path.splitext(os.path.basename(scene_pkl_path))[0]
    if pkl_basename.startswith("cube_"):
        cube_basename = pkl_basename
    elif pkl_basename.startswith("example_"):
        cube_basename = f"cube_{pkl_basename[len('example_'):] }"
    else:
        cube_basename = f"cube_{pkl_basename}"
    return f"{cube_basename}.jpg"


def resolve_output_path(scene_pkl_path: str, output_arg: Optional[str]) -> str:
    default_filename = build_cube_layout_filename(scene_pkl_path)

    if output_arg:
        absolute_output = os.path.abspath(output_arg)
        output_ext = os.path.splitext(absolute_output)[1].lower()
        if output_arg.endswith(os.sep) or os.path.isdir(absolute_output) or output_ext == "":
            return os.path.join(absolute_output, default_filename)
        return absolute_output

    pkl_stem = os.path.splitext(os.path.basename(scene_pkl_path))[0]
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    save_folder_name = f"{pkl_stem}_{timestamp}"

    relative_dir = ""
    try:
        relative_pkl_path = os.path.relpath(scene_pkl_path, CONFIG_SCENE_ROOT)
        if not relative_pkl_path.startswith(".."):
            relative_dir = os.path.dirname(relative_pkl_path)
    except ValueError:
        relative_dir = ""

    save_dir = os.path.join(CONFIG_SAVE_ROOT, relative_dir, save_folder_name)
    return os.path.join(save_dir, default_filename)


def load_scene_dict(scene_pkl_path: str) -> Dict[str, Any]:
    with open(scene_pkl_path, "rb") as file_obj:
        scene_dict = pickle.load(file_obj)

    if not isinstance(scene_dict, dict):
        raise ValueError(f"pkl 内容不是 dict: {type(scene_dict)}")
    return scene_dict


def first_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple)):
        if not value:
            return float(default)
        return float(value[0])
    return float(value)


def convert_scene_to_render_payload(scene_dict: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_subjects = scene_dict.get("subjects_data", [])
    if not raw_subjects:
        raise ValueError("场景里没有 subjects_data，无法渲染 cube 图。")

    subjects_data: List[Dict[str, Any]] = []
    for subject_index, subject_dict in enumerate(raw_subjects):
        dims = subject_dict.get("dims", (1.0, 1.0, 1.0))
        if len(dims) != 3:
            raise ValueError(f"第 {subject_index} 个物体的 dims 不是长度为 3 的数组: {dims}")

        render_subject = {
            "subject_name": str(subject_dict.get("name", f"cuboid_{subject_index}")),
            "x": first_float(subject_dict.get("x", [0.0])) + 6.0,
            "y": first_float(subject_dict.get("y", [0.0])),
            "z": first_float(subject_dict.get("z", [0.0])),
            "azimuth": math.degrees(first_float(subject_dict.get("azimuth", [0.0]))),
            "width": float(dims[0]),
            "depth": float(dims[1]),
            "height": float(dims[2]),
        }
        subjects_data.append(render_subject)

    raw_camera = scene_dict.get("camera_data", {})
    camera_data = {
        "camera_elevation": float(raw_camera.get("camera_elevation", math.radians(30.0))),
        "lens": float(raw_camera.get("lens", 50.0)),
        "global_scale": float(raw_camera.get("global_scale", 1.0)),
    }
    return subjects_data, camera_data


def is_server_healthy(server_url: str, timeout: float = 2.0) -> bool:
    try:
        response = requests.get(f"{server_url}/health", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def can_import_bpy(python_bin: str) -> bool:
    try:
        result = subprocess.run(
            [python_bin, "-c", "import bpy"],
            cwd=INFERENCE_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def build_blender_server_command(render_mode: str, server_host: str, server_port: int) -> Optional[List[str]]:
    blender_bin = os.environ.get("BLENDER_BIN") or shutil.which("blender")
    if not blender_bin:
        return None

    script_path = os.path.join(INFERENCE_DIR, "blender_server.py")
    python_expr = (
        "import os, sys, runpy; "
        f"repo_root = {REPO_ROOT!r}; "
        f"script = {script_path!r}; "
        "sys.path.insert(0, repo_root); "
        "sys.path.insert(0, os.path.dirname(script)); "
        f"sys.argv = [script, '--mode', {render_mode!r}, '--port', {str(server_port)!r}, '--host', {server_host!r}]; "
        "runpy.run_path(script, run_name='__main__')"
    )

    return [
        blender_bin,
        "--background",
        "--python-use-system-env",
        "--python-expr",
        python_expr,
    ]


def start_render_server(render_mode: str, server_url: str) -> subprocess.Popen:
    parsed_url = urlparse(server_url)
    server_host = parsed_url.hostname or "127.0.0.1"
    server_port = parsed_url.port
    if server_port is None:
        raise ValueError(f"无法从 server url 解析端口: {server_url}")

    python_bin = os.environ.get("BLENDER_SERVER_PYTHON", sys.executable)
    if can_import_bpy(python_bin):
        command = [
            python_bin,
            "blender_server.py",
            "--mode",
            render_mode,
            "--port",
            str(server_port),
            "--host",
            server_host,
        ]
    else:
        command = build_blender_server_command(render_mode, server_host, server_port)
        if command is None:
            raise RuntimeError(
                f"当前 Python ({python_bin}) 无法导入 bpy，且系统里找不到 blender 可执行文件。"
            )

    return subprocess.Popen(
        command,
        cwd=INFERENCE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_for_server(server_url: str, server_process: subprocess.Popen, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if is_server_healthy(server_url):
            return
        if server_process.poll() is not None:
            process_output = ""
            if server_process.stdout is not None:
                process_output = server_process.stdout.read().strip()
            raise RuntimeError(
                f"Blender render server 启动失败，退出码: {server_process.returncode}\n{process_output}"
            )
        time.sleep(1.0)

    raise TimeoutError(f"等待 Blender render server 就绪超时: {server_url}")


def stop_server(server_process: Optional[subprocess.Popen]) -> None:
    if server_process is None:
        return

    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()
        server_process.wait(timeout=5)


def render_cube_image(server_url: str, subjects_data: List[Dict[str, Any]], camera_data: Dict[str, Any]) -> Image.Image:
    request_data = {
        "subjects_data": subjects_data,
        "camera_data": camera_data,
        "num_samples": 1,
    }
    response = requests.post(
        f"{server_url}/render",
        json=request_data,
        timeout=120,
    )
    response.raise_for_status()

    result = response.json()
    if not result.get("success"):
        raise RuntimeError(f"渲染失败: {result.get('error_message', 'Unknown error')}")

    image_bytes = base64.b64decode(result["image_base64"])
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def render_cube_image_direct(scene_pkl_path: str, output_path: str, render_mode: str) -> Image.Image:
    blender_bin = os.environ.get("BLENDER_BIN") or shutil.which("blender")
    if not blender_bin:
        raise RuntimeError("找不到 blender 可执行文件，无法直接渲染。")

    blender_script = """
import math
import os
import pickle
import sys

repo_root = os.environ['PKL2FIG_REPO_ROOT']
inference_dir = os.environ['PKL2FIG_INFERENCE_DIR']
scene_path = os.environ['PKL2FIG_SCENE_PATH']
output_path = os.environ['PKL2FIG_OUTPUT_PATH']
render_mode = os.environ['PKL2FIG_RENDER_MODE']

sys.path.insert(0, repo_root)
sys.path.insert(0, inference_dir)

import blender_backend
from blender_backend import BlenderCuboidRenderer


def _set_bsdf_emission(bsdf, color):
    rgba = tuple(color[:4]) if len(color) >= 4 else tuple(color[:3]) + (1.0,)
    if 'Emission Color' in bsdf.inputs:
        bsdf.inputs['Emission Color'].default_value = rgba
    elif 'Emission' in bsdf.inputs:
        bsdf.inputs['Emission'].default_value = rgba
    if 'Emission Strength' in bsdf.inputs:
        bsdf.inputs['Emission Strength'].default_value = 1.0


def _get_primitive_object_translucent_compat(base_color=(0.0, 1.0, 0.0), edge_color=None, face_opacity=0.025):
    bpy = blender_backend.bpy
    mathutils = blender_backend.mathutils
    adjust_color_brightness = blender_backend.adjust_color_brightness

    bpy.ops.object.empty_add(type='PLAIN_AXES')
    empty_object = bpy.data.objects.new('Empty', None)
    before_objs = set(bpy.data.objects)
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0, 0))
    after_objs = set(bpy.data.objects)
    diff_objs = after_objs - before_objs

    obj = None
    for blender_obj in diff_objs:
        obj = blender_obj
        obj.parent = empty_object
        world_matrix = obj.matrix_world
        obj.matrix_world = world_matrix

    if obj:
        brightness_factors = [0.30, 0.30, 0.30, 0.30, 1.00, 0.30]
        colors = [adjust_color_brightness(base_color, factor) for factor in brightness_factors]

        for color_index, color in enumerate(colors):
            material = bpy.data.materials.new(name=f'FaceColor_{color_index}')
            material.use_nodes = True
            obj.data.materials.append(material)

            nodes = material.node_tree.nodes
            links = material.node_tree.links
            nodes.clear()

            bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf.location = (0, 0)
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Alpha'].default_value = face_opacity
            _set_bsdf_emission(bsdf, color)

            material_output = nodes.new(type='ShaderNodeOutputMaterial')
            material_output.location = (200, 0)
            links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

            material.blend_method = 'BLEND'
            if hasattr(material, 'show_transparent_back'):
                material.show_transparent_back = False

        if len(obj.data.polygons) == len(colors):
            for color_index, polygon in enumerate(obj.data.polygons):
                polygon.material_index = color_index

    bbox_corners = []
    bpy.context.view_layer.update()
    for child in empty_object.children:
        for corner in child.bound_box:
            world_corner = child.matrix_world @ mathutils.Vector(corner)
            bbox_corners.append(world_corner)

    if not bbox_corners:
        return 0, empty_object

    max_z = max(corner.z for corner in bbox_corners)
    return max_z, empty_object


def _get_primitive_object_translucent_rgb_compat(base_color=(0.0, 1.0, 0.0), edge_color=None, face_opacity=0.025):
    bpy = blender_backend.bpy
    mathutils = blender_backend.mathutils
    adjust_color_brightness = blender_backend.adjust_color_brightness

    bpy.ops.object.empty_add(type='PLAIN_AXES')
    empty_object = bpy.data.objects.new('Empty', None)
    before_objs = set(bpy.data.objects)
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0, 0))
    after_objs = set(bpy.data.objects)
    diff_objs = after_objs - before_objs

    obj = None
    for blender_obj in diff_objs:
        obj = blender_obj
        obj.parent = empty_object
        world_matrix = obj.matrix_world
        obj.matrix_world = world_matrix

    if obj:
        brightness_factors = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
        red = (1.0, 0.0, 0.0, 1.0)
        green = (0.0, 1.0, 0.0, 1.0)
        blue = (0.0, 0.0, 1.0, 1.0)
        colors = [adjust_color_brightness(green, factor) for factor in brightness_factors[:4]]
        colors += [adjust_color_brightness(blue, brightness_factors[4])]
        colors += [adjust_color_brightness(red, brightness_factors[5])]
        colors = [colors[-2], colors[-1], colors[0], colors[1], colors[2], colors[3]]

        for color_index, color in enumerate(colors):
            material = bpy.data.materials.new(name=f'FaceColor_{color_index}')
            material.use_nodes = True
            obj.data.materials.append(material)

            nodes = material.node_tree.nodes
            links = material.node_tree.links
            nodes.clear()

            bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf.location = (0, 0)
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Alpha'].default_value = face_opacity
            _set_bsdf_emission(bsdf, color)

            material_output = nodes.new(type='ShaderNodeOutputMaterial')
            material_output.location = (200, 0)
            links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

            material.blend_method = 'BLEND'
            if hasattr(material, 'show_transparent_back'):
                material.show_transparent_back = False

        if len(obj.data.polygons) == len(colors):
            for color_index, polygon in enumerate(obj.data.polygons):
                polygon.material_index = color_index

        edge_material = bpy.data.materials.new(name='EdgeDelimiterMaterial')
        edge_material.use_nodes = True
        nodes = edge_material.node_tree.nodes
        links = edge_material.node_tree.links
        nodes.clear()

        if edge_color is None:
            edge_color = adjust_color_brightness(base_color, 0.10)

        edge_emission_node = nodes.new(type='ShaderNodeEmission')
        edge_emission_node.inputs['Color'].default_value = edge_color
        edge_output_node = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(edge_emission_node.outputs['Emission'], edge_output_node.inputs['Surface'])

        obj.data.materials.append(edge_material)
        wire_mod = obj.modifiers.new(name='EdgeDelimiter', type='WIREFRAME')
        wire_mod.thickness = 0.01
        wire_mod.use_replace = False
        wire_mod.material_offset = len(obj.data.materials) - 1

    bbox_corners = []
    bpy.context.view_layer.update()
    for child in empty_object.children:
        for corner in child.bound_box:
            world_corner = child.matrix_world @ mathutils.Vector(corner)
            bbox_corners.append(world_corner)

    if not bbox_corners:
        return 0, empty_object

    max_z = max(corner.z for corner in bbox_corners)
    return max_z, empty_object


blender_backend.get_primitive_object_translucent = _get_primitive_object_translucent_compat
blender_backend.get_primitive_object_translucent_rgb = _get_primitive_object_translucent_rgb_compat

with open(scene_path, 'rb') as file_obj:
    scene_dict = pickle.load(file_obj)

raw_subjects = scene_dict.get('subjects_data', [])
if not raw_subjects:
    raise RuntimeError('No subjects_data in scene')

subjects = []
for subject_index, subject_dict in enumerate(raw_subjects):
    dims = subject_dict.get('dims', (1.0, 1.0, 1.0))
    x_values = subject_dict.get('x') or [0.0]
    y_values = subject_dict.get('y') or [0.0]
    z_values = subject_dict.get('z') or [0.0]
    azimuth_values = subject_dict.get('azimuth') or [0.0]
    subjects.append({
        'name': str(subject_dict.get('name', f'cuboid_{subject_index}')),
        'x': [float(x_values[0]) + 6.0],
        'y': [float(y_values[0])],
        'z': [float(z_values[0])],
        'dims': [float(dims[0]), float(dims[1]), float(dims[2])],
        'azimuth': [math.degrees(float(azimuth_values[0]))],
    })

camera_raw = scene_dict.get('camera_data', {})
camera = {
    'camera_elevation': float(camera_raw.get('camera_elevation', math.radians(30.0))),
    'lens': float(camera_raw.get('lens', 50.0)),
    'global_scale': float(camera_raw.get('global_scale', 1.0)),
}

output_dir = os.path.dirname(output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

render_engine = 'BLENDER_EEVEE_NEXT' if render_mode == 'cv' else 'CYCLES'
renderer = BlenderCuboidRenderer(render_engine)

if render_mode == 'cv':
    renderer.render_cv(subjects, camera, num_samples=1, output_path=output_path)
elif render_mode == 'paper':
    renderer.render_paper_figure(subjects, camera, num_samples=1, output_path=output_path)
else:
    renderer.render_final_representation(subjects, camera, num_samples=1, output_path=output_path)

print(output_path)
"""

    temp_script_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix="_pkl2fig_render.py", delete=False, dir="/tmp") as temp_file:
            temp_file.write(blender_script)
            temp_script_path = temp_file.name

        blender_env = os.environ.copy()
        blender_env["PKL2FIG_REPO_ROOT"] = REPO_ROOT
        blender_env["PKL2FIG_INFERENCE_DIR"] = INFERENCE_DIR
        blender_env["PKL2FIG_SCENE_PATH"] = scene_pkl_path
        blender_env["PKL2FIG_OUTPUT_PATH"] = output_path
        blender_env["PKL2FIG_RENDER_MODE"] = render_mode

        result = subprocess.run(
            [
                blender_bin,
                "--background",
                "--python-use-system-env",
                "--python",
                temp_script_path,
            ],
            cwd=INFERENCE_DIR,
            env=blender_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
            check=False,
        )
    finally:
        if temp_script_path and os.path.exists(temp_script_path):
            os.remove(temp_script_path)

    if result.returncode != 0:
        raise RuntimeError(f"Blender 直接渲染失败，退出码: {result.returncode}\n{result.stdout}")
    if not os.path.isfile(output_path):
        raise RuntimeError(f"Blender 已执行，但没有生成输出图片: {output_path}\n{result.stdout}")

    return Image.open(output_path).convert("RGB")


def main() -> None:
    args = parse_args()
    scene_pkl_path = normalize_scene_pickle_path(args.scene_pkl)
    output_path = resolve_output_path(scene_pkl_path, args.output)
    server_url = args.server_url or CONFIG_SERVER_URLS[args.render_mode]

    scene_dict = load_scene_dict(scene_pkl_path)
    subjects_data, camera_data = convert_scene_to_render_payload(scene_dict)

    started_server_process: Optional[subprocess.Popen] = None
    try:
        print(f"[Info] 加载场景: {scene_pkl_path}")
        print(f"[Info] 物体数量: {len(subjects_data)}")
        print(f"[Info] 渲染模式: {args.render_mode}")

        try:
            default_python_bin = os.environ.get("BLENDER_SERVER_PYTHON", sys.executable)
            if args.server_url is None and not can_import_bpy(default_python_bin):
                raise RuntimeError(
                    f"当前 Python ({default_python_bin}) 无法导入 bpy，直接使用 Blender 一次性渲染。"
                )

            if not is_server_healthy(server_url):
                if args.no_auto_start:
                    raise RuntimeError(f"Blender render server 未启动: {server_url}")
                print(f"[Info] 检测到后端未启动，正在拉起 {args.render_mode} render server: {server_url}")
                started_server_process = start_render_server(args.render_mode, server_url)
                wait_for_server(server_url, started_server_process, args.startup_timeout)

            image = render_cube_image(server_url, subjects_data, camera_data)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            image.save(output_path)
            print(f"[Done] cube 图已保存到: {output_path}")
        except Exception as render_error:
            print(f"[Warn] 通过 render server 渲染失败，改为 Blender 直接渲染: {render_error}")
            stop_server(started_server_process)
            started_server_process = None
            render_cube_image_direct(scene_pkl_path, output_path, args.render_mode)
            print(f"[Done] cube 图已保存到: {output_path}")
    finally:
        stop_server(started_server_process)


if __name__ == "__main__":
    main()
