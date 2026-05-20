#!/usr/bin/env python3

import argparse
import base64
import colorsys
import contextlib
import io
import json
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
from PIL import Image, ImageDraw
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

CUBE_COLOR_HEX_SEQUENCE = [
    # 这里按 pkl 中 cube 的顺序配置颜色；可以直接改成任意 #RRGGBB 颜色。
    # 例如：第 1 个 cube 想改成白色，就把 "#FF0000" 改成 "#FFFFFF"。
    ("红", "#FF7F7F"),  # 第 1 个 cube
    ("橙", "#FFDF7F"),  # 第 2 个 cube
    ("黄", "#FFFF7F"),  # 第 3 个 cube
    ("绿", "#7FD7A7"),  # 第 4 个 cube
    ("青", "#97DFD9"),  # 第 5 个 cube
    ("蓝", "#7FB7DF"),  # 第 6 个 cube
    ("紫", "#B797CF"),  # 第 7 个 cube
]

SUPPLEMENTAL_CUBE_COLOR_HEX_SEQUENCE = [
    # 超过 7 个 cube 时从这里继续取颜色，也可以按需改成 #RRGGBB。
    ("粉", "#F7C9CE"),
    ("棕", "#944A00"),
    ("橄榄", "#738C00"),
    ("深青", "#00738C"),
    ("靛", "#40008C"),
    ("玫红", "#D90073"),
    ("浅绿", "#73E633"),
]


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    normalized_color = hex_color.strip().lstrip("#")
    if len(normalized_color) == 3:
        normalized_color = "".join(channel * 2 for channel in normalized_color)
    if len(normalized_color) != 6:
        raise ValueError(f"颜色必须是 #RRGGBB 或 #RGB 格式: {hex_color}")

    try:
        red = int(normalized_color[0:2], 16) / 255.0
        green = int(normalized_color[2:4], 16) / 255.0
        blue = int(normalized_color[4:6], 16) / 255.0
    except ValueError as error:
        raise ValueError(f"颜色包含非法十六进制字符: {hex_color}") from error

    return (red, green, blue, alpha)


CUBE_COLOR_SEQUENCE = [
    (color_name, hex_to_rgba(hex_color))
    for color_name, hex_color in CUBE_COLOR_HEX_SEQUENCE
]

SUPPLEMENTAL_CUBE_COLORS = [
    (color_name, hex_to_rgba(hex_color))
    for color_name, hex_color in SUPPLEMENTAL_CUBE_COLOR_HEX_SEQUENCE
]


def get_cube_color(subject_index: int) -> Tuple[str, Tuple[float, float, float, float]]:
    fixed_colors = CUBE_COLOR_SEQUENCE + SUPPLEMENTAL_CUBE_COLORS
    if subject_index < len(fixed_colors):
        return fixed_colors[subject_index]

    used_colors = {color for _, color in fixed_colors}
    generated_index = subject_index - len(fixed_colors)
    golden_ratio_conjugate = 0.618033988749895
    for attempt in range(360):
        hue = (0.11 + (generated_index + attempt / 360.0) * golden_ratio_conjugate) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
        rgba = tuple(round(channel, 4) for channel in (*rgb, 1.0))
        if rgba not in used_colors:
            return f"自动色{generated_index + 1}", rgba

    hue = (0.11 + subject_index * golden_ratio_conjugate) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
    return f"自动色{generated_index + 1}", tuple(round(channel, 4) for channel in (*rgb, 1.0))


def build_cube_color_assignments(raw_subjects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    assignments: List[Dict[str, Any]] = []
    for subject_index, subject_dict in enumerate(raw_subjects):
        color_name, rgba = get_cube_color(subject_index)
        rgb_255 = tuple(round(channel * 255) for channel in rgba[:3])
        assignments.append(
            {
                "index": subject_index + 1,
                "subject_name": str(subject_dict.get("name", f"cuboid_{subject_index}")),
                "color_name": color_name,
                "rgba": rgba,
                "rgb255": rgb_255,
                "hex": "#{:02X}{:02X}{:02X}".format(*rgb_255),
            }
        )
    return assignments


def print_cube_color_assignments(assignments: List[Dict[str, Any]]) -> None:
    print("[Info] cube 颜色顺序:")
    for assignment in assignments:
        rgba_text = ", ".join(f"{channel:.4g}" for channel in assignment["rgba"])
        print(
            f"  {assignment['index']}. {assignment['subject_name']}: "
            f"{assignment['color_name']} {assignment['hex']} rgba=({rgba_text})"
        )


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



def render_cube_image_pil(
    subjects_data: List[Dict[str, Any]],
    camera_data: Dict[str, Any],
    color_assignments: List[Dict[str, Any]],
    output_path: str,
    image_size: int = 1024,
) -> Image.Image:
    def dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def normalize(vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
        length = math.sqrt(dot(vector, vector))
        if length == 0.0:
            return (0.0, 0.0, 0.0)
        return (vector[0] / length, vector[1] / length, vector[2] / length)

    def rotate_xy(x: float, y: float, azimuth: float) -> Tuple[float, float]:
        cos_azimuth = math.cos(azimuth)
        sin_azimuth = math.sin(azimuth)
        return (
            x * cos_azimuth - y * sin_azimuth,
            x * sin_azimuth + y * cos_azimuth,
        )

    elevation = float(camera_data.get("camera_elevation", math.radians(30.0)))
    lens = float(camera_data.get("lens", 50.0))
    camera_location = (
        6.0 * math.cos(elevation) - 6.0,
        0.0,
        6.0 * math.sin(elevation),
    )
    forward = normalize((-1.0, 0.0, -math.tan(elevation)))
    right = normalize(cross(forward, (0.0, 0.0, 1.0)))
    up = normalize(cross(right, forward))
    sensor_width = 36.0
    focal_scale = image_size * lens / sensor_width

    def project(point: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        relative = (
            point[0] - camera_location[0],
            point[1] - camera_location[1],
            point[2] - camera_location[2],
        )
        depth = dot(relative, forward)
        if depth <= 1e-4:
            return None
        image_x = image_size * 0.5 + dot(relative, right) * focal_scale / depth
        image_y = image_size * 0.5 - dot(relative, up) * focal_scale / depth
        return image_x, image_y, depth

    render_scale = 2
    cube_face_alpha = 128
    cube_edge_alpha = 190
    canvas_size = image_size * render_scale
    image = Image.new("RGBA", (canvas_size, canvas_size), (12, 12, 12, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    def scaled_polygon(points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        return [(round(x * render_scale), round(y * render_scale)) for x, y in points]

    ground_corners = [
        (-18.0, -10.0, 0.0),
        (4.0, -10.0, 0.0),
        (4.0, 10.0, 0.0),
        (-18.0, 10.0, 0.0),
    ]
    projected_ground = [project(corner) for corner in ground_corners]
    if all(point is not None for point in projected_ground):
        draw.polygon(
            scaled_polygon([(point[0], point[1]) for point in projected_ground if point is not None]),
            fill=(28, 28, 28, 255),
        )

    faces = []
    face_indices = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    for subject_index, subject_data in enumerate(subjects_data):
        global_scale = float(camera_data.get("global_scale", 1.0))
        width = float(subject_data["width"]) * global_scale
        depth = float(subject_data["depth"]) * global_scale
        height = float(subject_data["height"]) * global_scale
        center_x = float(subject_data["x"]) - 6.0
        center_y = float(subject_data["y"])
        center_z = float(subject_data["z"]) + height / 2.0
        azimuth = math.radians(float(subject_data["azimuth"]))
        color = color_assignments[subject_index]["rgb255"]

        local_vertices = [
            (-width / 2.0, -depth / 2.0, -height / 2.0),
            (width / 2.0, -depth / 2.0, -height / 2.0),
            (width / 2.0, depth / 2.0, -height / 2.0),
            (-width / 2.0, depth / 2.0, -height / 2.0),
            (-width / 2.0, -depth / 2.0, height / 2.0),
            (width / 2.0, -depth / 2.0, height / 2.0),
            (width / 2.0, depth / 2.0, height / 2.0),
            (-width / 2.0, depth / 2.0, height / 2.0),
        ]
        world_vertices = []
        for local_x, local_y, local_z in local_vertices:
            rotated_x, rotated_y = rotate_xy(local_x, local_y, azimuth)
            world_vertices.append((center_x + rotated_x, center_y + rotated_y, center_z + local_z))

        projected_vertices = [project(vertex) for vertex in world_vertices]
        for face in face_indices:
            face_points = [projected_vertices[index] for index in face]
            if any(point is None for point in face_points):
                continue
            avg_depth = sum(point[2] for point in face_points if point is not None) / len(face)
            polygon = [(point[0], point[1]) for point in face_points if point is not None]
            faces.append((avg_depth, polygon, color))

    for _, polygon, color in sorted(faces, key=lambda item: item[0], reverse=True):
        scaled = scaled_polygon(polygon)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay, "RGBA")
        overlay_draw.polygon(scaled, fill=(*color, cube_face_alpha))
        overlay_draw.line(
            scaled + [scaled[0]],
            fill=(0, 0, 0, cube_edge_alpha),
            width=2 * render_scale,
            joint="curve",
        )
        image = Image.alpha_composite(image, overlay)

    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS).convert("RGB")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    image.save(output_path)
    return image


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


def render_cube_image_direct(
    scene_pkl_path: str,
    output_path: str,
    render_mode: str,
    color_assignments: Optional[List[Dict[str, Any]]] = None,
) -> Image.Image:
    blender_bin = os.environ.get("BLENDER_BIN") or shutil.which("blender")
    if not blender_bin:
        raise RuntimeError("找不到 blender 可执行文件，无法直接渲染。")

    scene_dict = load_scene_dict(scene_pkl_path)
    subjects_data, camera_data = convert_scene_to_render_payload(scene_dict)
    if color_assignments is None:
        color_assignments = build_cube_color_assignments(scene_dict.get("subjects_data", []))

    cube_colors = [assignment["rgba"] for assignment in color_assignments]

    blender_script = """
import json
import math
import os

import bpy
import mathutils

subjects_data = json.loads(os.environ['PKL2FIG_SUBJECTS_DATA'])
camera_data = json.loads(os.environ['PKL2FIG_CAMERA_DATA'])
cube_colors = [tuple(color) for color in json.loads(os.environ['PKL2FIG_CUBE_COLORS'])]
output_path = os.environ['PKL2FIG_OUTPUT_PATH']
render_mode = os.environ['PKL2FIG_RENDER_MODE']


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)
    for light in list(bpy.data.lights):
        if light.users == 0:
            bpy.data.lights.remove(light)
    for camera in list(bpy.data.cameras):
        if camera.users == 0:
            bpy.data.cameras.remove(camera)


def make_opaque_material(name, color, emission_strength=1.0):
    rgba = tuple(color[:4]) if len(color) >= 4 else tuple(color[:3]) + (1.0,)
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = rgba
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs['Color'].default_value = rgba
    emission.inputs['Strength'].default_value = emission_strength
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(emission.outputs['Emission'], output_node.inputs['Surface'])

    material.blend_method = 'OPAQUE'
    material.use_screen_refraction = False
    return material


def add_plane():
    bpy.ops.mesh.primitive_plane_add(size=25.0, location=(-6.0, 0.0, 0.0))
    plane = bpy.context.object
    plane.name = 'GroundPlane'
    material = make_opaque_material('GroundDarkMaterial', (0.08, 0.08, 0.08, 1.0), 0.7)
    plane.data.materials.append(material)
    return plane


def add_lights():
    lights = [
        ('KeyArea', 'AREA', (-1.5, -4.5, 7.0), 450.0, 5.0),
        ('FillArea', 'AREA', (-4.5, 4.0, 5.0), 120.0, 7.0),
        ('TopPoint', 'POINT', (-6.0, 0.0, 7.5), 120.0, None),
    ]
    for name, light_type, location, energy, size in lights:
        bpy.ops.object.light_add(type=light_type, location=location)
        light = bpy.context.object
        light.name = name
        light.data.energy = energy
        if size is not None and hasattr(light.data, 'size'):
            light.data.size = size
        light.data.use_shadow = True


def setup_camera():
    elevation = float(camera_data.get('camera_elevation', math.radians(30.0)))
    lens = float(camera_data.get('lens', 50.0))
    radius = 6.0
    center = -6.0

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.location = mathutils.Vector((radius * math.cos(elevation) + center, 0.0, radius * math.sin(elevation)))
    direction = mathutils.Vector((-1.0, 0.0, -math.tan(elevation)))
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    camera.data.lens = lens
    camera.data.clip_end = 1000.0
    bpy.context.scene.camera = camera


def add_cubes():
    global_scale = float(camera_data.get('global_scale', 1.0))
    center_x = -6.0
    for subject_index, subject_data in enumerate(subjects_data):
        color = cube_colors[subject_index]
        width = float(subject_data['width']) * global_scale
        depth = float(subject_data['depth']) * global_scale
        height = float(subject_data['height']) * global_scale
        x = float(subject_data['x']) + center_x
        y = float(subject_data['y'])
        z = float(subject_data['z']) + height / 2.0
        azimuth = math.radians(float(subject_data['azimuth']))

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, z), rotation=(0.0, 0.0, azimuth))
        cube = bpy.context.object
        cube.name = str(subject_data.get('subject_name', f'cuboid_{subject_index}'))
        cube.scale = (width, depth, height)
        cube.data.materials.append(make_opaque_material(f'CubeColor_{subject_index}', color, 1.4))
        for polygon in cube.data.polygons:
            polygon.material_index = 0


def configure_render():
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE' if render_mode == 'cv' else 'CYCLES'
    if scene.render.engine == 'CYCLES':
        scene.cycles.samples = 32
        scene.cycles.use_denoising = True
    scene.use_nodes = False
    scene.render.use_compositing = False
    scene.render.film_transparent = False
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.filepath = output_path
    try:
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'Medium High Contrast'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass
    if scene.world is None:
        scene.world = bpy.data.worlds.new('World')
    scene.world.color = (0.1, 0.1, 0.1)


clear_scene()
configure_render()
add_plane()
add_lights()
add_cubes()
setup_camera()
bpy.context.view_layer.update()

output_dir = os.path.dirname(output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

bpy.ops.render.render(write_still=True)
print(output_path)
"""

    temp_script_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix="_pkl2fig_render.py", delete=False, dir="/tmp") as temp_file:
            temp_file.write(blender_script)
            temp_script_path = temp_file.name

        blender_env = os.environ.copy()
        for env_name in (
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONNOUSERSITE",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "CONDA_PROMPT_MODIFIER",
        ):
            blender_env.pop(env_name, None)
        blender_env["PKL2FIG_OUTPUT_PATH"] = output_path
        blender_env["PKL2FIG_RENDER_MODE"] = render_mode
        blender_env["PKL2FIG_SUBJECTS_DATA"] = json.dumps(subjects_data)
        blender_env["PKL2FIG_CAMERA_DATA"] = json.dumps(camera_data)
        blender_env["PKL2FIG_CUBE_COLORS"] = json.dumps(cube_colors)

        result = subprocess.run(
            [
                blender_bin,
                "--background",
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
    color_assignments = build_cube_color_assignments(scene_dict.get("subjects_data", []))

    started_server_process: Optional[subprocess.Popen] = None
    try:
        print(f"[Info] 加载场景: {scene_pkl_path}")
        print(f"[Info] 物体数量: {len(subjects_data)}")
        print(f"[Info] 渲染模式: {args.render_mode}")
        print_cube_color_assignments(color_assignments)

        if args.render_mode == "final":
            print("[Info] final 模式使用 PIL 直接投影渲染，保证 50% 透明且六面同色。")
            render_cube_image_pil(subjects_data, camera_data, color_assignments, output_path)
            print(f"[Done] cube 图已保存到: {output_path}")
            return

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
            render_cube_image_direct(scene_pkl_path, output_path, args.render_mode, color_assignments)
            print(f"[Done] cube 图已保存到: {output_path}")
    finally:
        stop_server(started_server_process)


if __name__ == "__main__":
    main()
