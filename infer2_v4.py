#!/usr/bin/env python
# coding: utf-8

# # Inference Notebook
# This notebook runs the backend without the Gradio UI.
# It loads scenes from pickle files, renders backends locally, and runs the inference.

# In[1]:


import argparse
import os
os.environ["GRADIO_TMP_DIR"]="./inference/gradio_tmp" 
os.environ["GRADIO_TEMP_DIR"]="./inference/gradio_tmp" 
os.environ["TEMPDIR"]="./inference/gradio_tmp" 
os.environ["TMP_DIR"]="./inference/gradio_tmp" 
os.environ["TEMP_DIR"]="./inference/gradio_tmp" 
os.environ["TMPDIR"]="./inference/gradio_tmp" 
os.environ["CUDA_VISIBLE_DEVICES"]="1," 
import sys
import json
import time
import subprocess
import math
import tempfile
import types
from PIL import Image, ImageDraw, ImageFilter
from typing import Any, Dict, List, Optional, Tuple

try:
    from IPython.display import display
except ModuleNotFoundError:
    def display(obj):
        print(obj)
from datetime import datetime, timezone, timedelta  # 为了生成北京时间的时间戳
from urllib.parse import urlparse

from pkl2fig_v3 import (
    build_cube_color_assignments,
    convert_scene_to_render_payload,
    load_scene_dict,
    render_cube_image_pil,
)


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PKL2FIG_LAYOUT_IMAGE_SIZE = 1024
V3_COND_SIZE = 512
V3_MASK_SIZE = 64
V3_MAX_REPAIR_ATTEMPTS = 3
V3_LAYOUT_BBOX_TOLERANCE = 0.35
V3_MIN_MASK_COVERAGE = 0.18
V4_DEFAULT_CLIP_MODEL = os.environ.get("V4_CLIP_MODEL", "openai/clip-vit-large-patch14")
V4_SEMANTIC_MIN_SCORE = 0.18
V4_SEMANTIC_SCORE_MARGIN = 0.05
V4_INPAINT_STEPS = 14
V4_INPAINT_STRENGTH = 0.45
V4_INPAINT_GUIDANCE = 5.0
V4_INPAINT_PADDING = 32
# 下面对应的参数是（lora_weight，guidance_scale，seed）
# lora_weight      = 听 layout / 空间条件的程度，太高会限制图片发挥
# guidance_scale   = 听文字 prompt 的程度
V4_FIXED_INFERENCE_PARAMS = (
    (1.0, 3.5, 42),
    (1.0, 3.5, 43),
    (1.0, 3.5, 44),
    (1.8, 2.8, 42),
    (1.8, 2.8, 43),
    (1.8, 2.8, 44),
)
initialize_inference_engine = None
run_inference_from_gradio = None
get_inference_engine = None
SceneManager = None
get_call_ids_from_placeholder_prompt_flux = None
tokenizer = None
config = None
make_train_dataset = None
collate_fn = None
torch = None
_v4_clip_bundle = None
_v4_clip_error = None
_v4_inpaint_pipe = None
_v4_inpaint_error = None

V4_OBJECT_ALIASES = {
    "cat": ["cat", "domestic cat", "house cat", "feline"],
    "dog": ["dog", "domestic dog", "puppy", "canine"],
    "sofa": ["sofa", "couch", "settee"],
    "lamp": ["lamp", "floor lamp", "table lamp", "light fixture"],
    "bookshelf": ["bookshelf", "bookcase", "shelf", "book shelf"],
    "chair": ["chair", "armchair", "office chair"],
    "table": ["table", "desk", "coffee table"],
    "bed": ["bed"],
    "cabinet": ["cabinet", "cupboard"],
    "plant": ["plant", "potted plant"],
}
V4_COMMON_CANDIDATES = [
    "cat",
    "dog",
    "sofa",
    "lamp",
    "bookshelf",
    "chair",
    "table",
    "bed",
    "cabinet",
    "plant",
]


def load_runtime_dependencies() -> None:
    global initialize_inference_engine
    global run_inference_from_gradio
    global get_inference_engine
    global SceneManager
    global get_call_ids_from_placeholder_prompt_flux
    global tokenizer
    global config
    global make_train_dataset
    global collate_fn
    global torch

    if initialize_inference_engine is not None:
        return

    import importlib

    backend_v2_module = importlib.import_module("inference.infer_backend_v2")
    sys.modules["inference.infer_backend"] = backend_v2_module

    from inference.infer_backend_v2 import (
        initialize_inference_engine as _initialize_inference_engine,
        run_inference_from_gradio as _run_inference_from_gradio,
        get_inference_engine as _get_inference_engine,
    )
    from inference.app import (
        SceneManager as _SceneManager,
        get_call_ids_from_placeholder_prompt_flux as _get_call_ids_from_placeholder_prompt_flux,
        tokenizer as _tokenizer,
    )
    import inference.config as _config
    from train.src.jsonl_datasets import (
        make_train_dataset as _make_train_dataset,
        collate_fn as _collate_fn,
    )
    import torch as _torch

    initialize_inference_engine = _initialize_inference_engine
    run_inference_from_gradio = _run_inference_from_gradio
    get_inference_engine = _get_inference_engine
    SceneManager = _SceneManager
    get_call_ids_from_placeholder_prompt_flux = _get_call_ids_from_placeholder_prompt_flux
    tokenizer = _tokenizer
    config = _config
    make_train_dataset = _make_train_dataset
    collate_fn = _collate_fn
    torch = _torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载一个或多个场景 pkl 进行推理，并通过命令行传入 scene pkl 和 placeholder prompt。",
        epilog=(
            '示例:\n'
            '  python infer20.py '
            '--scene-pkls inference/saved_scenes/example_test3.31_1_fixed.pkl '
            '--placeholder-prompt "A surreal yet highly realistic high-quality photo PLACEHOLDER, clear luminous lighting, crisp air, a quiet and dreamy atmosphere, rich details, and strong visual contrast."'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--scene-pkls",
        nargs="+",
        required=True,
        help="一个或多个场景 pkl 路径。",
    )
    parser.add_argument(
        "--placeholder-prompt",
        required=True,
        help="送入 FLUX 的 prompt 模板，必须包含 PLACEHOLDER。",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="可选，覆盖输出分辨率。建议 512 更利于严格位姿控制。",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="可选，覆盖 guidance scale。较低值通常更利于布局遵循。",
    )
    parser.add_argument(
        "--lora-weight",
        type=float,
        default=1.0,
        help="可选，控制空间 LoRA 强度。大于 1.0 通常会增强布局约束。",
    )
    parser.add_argument(
        "--v3-cond-size",
        type=int,
        default=V3_COND_SIZE,
        help="infer2_v3 专用条件图分辨率。1024 会比原 512 保留更多 layout 细节。",
    )
    parser.add_argument(
        "--v3-mask-size",
        type=int,
        default=V3_MASK_SIZE,
        help="infer2_v3 专用 attention mask 分辨率。64 比原 32 更细。",
    )
    parser.add_argument(
        "--v3-max-attempts",
        type=int,
        default=V3_MAX_REPAIR_ATTEMPTS,
        help="生成后 layout 检查失败时的最大重试次数。",
    )
    parser.add_argument(
        "--v3-max-lora-weight",
        type=float,
        default=1.8,
        help="自动重试时允许使用的最大 LoRA/layout 条件权重。",
    )
    parser.add_argument(
        "--v4-disable-semantic-check",
        action="store_true",
        help="关闭 infer2_v4 的 per-mask 物体类型语义检查。",
    )
    parser.add_argument(
        "--v4-disable-inpaint-repair",
        action="store_true",
        help="关闭 infer2_v4 的局部 inpaint 语义修复。",
    )
    parser.add_argument(
        "--v4-clip-model",
        default=V4_DEFAULT_CLIP_MODEL,
        help="infer2_v4 语义检查使用的 CLIP 模型路径或名称。",
    )
    parser.add_argument(
        "--v4-semantic-min-score",
        type=float,
        default=V4_SEMANTIC_MIN_SCORE,
        help="CLIP 区域分类接受目标类别的最低概率。",
    )
    parser.add_argument(
        "--v4-semantic-score-margin",
        type=float,
        default=V4_SEMANTIC_SCORE_MARGIN,
        help="CLIP 目标类别需要领先第二名的最低概率差。",
    )
    parser.add_argument(
        "--v4-max-inpaint-repairs",
        type=int,
        default=1,
        help="每张图最多执行多少轮局部 inpaint 修复。0 表示不修复。",
    )
    parser.add_argument(
        "--v4-inpaint-steps",
        type=int,
        default=V4_INPAINT_STEPS,
        help="局部 inpaint 修复的扩散步数，越小越快。",
    )
    parser.add_argument(
        "--v4-inpaint-strength",
        type=float,
        default=V4_INPAINT_STRENGTH,
        help="局部 inpaint 修复强度。较低值更保留原图风格和布局。",
    )
    parser.add_argument(
        "--v4-inpaint-guidance",
        type=float,
        default=V4_INPAINT_GUIDANCE,
        help="局部 inpaint 修复 guidance scale。",
    )
    parser.add_argument(
        "--v4-inpaint-padding",
        type=int,
        default=V4_INPAINT_PADDING,
        help="局部 inpaint mask crop padding 像素。",
    )
    return parser.parse_args()


def normalize_scene_pickle_paths(scene_pickle_paths: List[str]) -> List[str]:
    normalized_paths = []
    for pkl_path in scene_pickle_paths:
        if os.path.isabs(pkl_path):
            normalized_paths.append(pkl_path)
        else:
            normalized_paths.append(os.path.join(REPO_ROOT, pkl_path))
    return normalized_paths


def format_generation_param_tag(lora_weight: float, guidance_scale: float, seed: int) -> str:
    def fmt(value: float) -> str:
        text = f"{float(value):.1f}"
        return text.replace("-", "m").replace(".", "p")

    return f"lora{fmt(lora_weight)}_guidance{fmt(guidance_scale)}_seed{int(seed)}"


def build_placeholder_text(subject_descriptions: List[str]) -> str:
    placeholder_text = ""
    for subject in subject_descriptions[:-1]:
        placeholder_text += f"<placeholder> {subject} and "
    for subject in subject_descriptions[-1:]:
        placeholder_text += f"<placeholder> {subject}"
    return placeholder_text.strip()


def build_placeholder_token_prompt(subject_descriptions: List[str], placeholder_prompt: str) -> str:
    placeholder_text = build_placeholder_text(subject_descriptions)
    return placeholder_prompt.replace("PLACEHOLDER", placeholder_text)


def build_v4_identity_prompt(subject_descriptions: List[str], placeholder_prompt: str) -> str:
    if "required object identities from the scene file" in placeholder_prompt.lower():
        return placeholder_prompt

    counts: Dict[str, int] = {}
    for subject in subject_descriptions:
        label = normalize_object_label(subject)
        counts[label] = counts.get(label, 0) + 1
    object_list = ", ".join(f"{count} x {label}" for label, count in counts.items())
    identity_clause = (
        " Required object identities from the scene file: exactly "
        f"{object_list}. Each listed object must appear in its own projected cube/mask; "
        "do not replace one object type with another, do not duplicate cats or dogs, "
        "and do not add extra animals outside their assigned cubes."
    )
    return placeholder_prompt.rstrip() + identity_clause


def build_cube_layout_filename(pkl_path: str) -> str:
    pkl_basename = os.path.splitext(os.path.basename(pkl_path))[0]
    if pkl_basename.startswith("cube_"):
        cube_basename = pkl_basename
    elif pkl_basename.startswith("example_"):
        cube_basename = f"cube_{pkl_basename[len('example_'):]}"
    else:
        cube_basename = f"cube_{pkl_basename}"
    return f"{cube_basename}.jpg"


def start_blender_backends() -> subprocess.Popen:
    cv_port = urlparse(config.BLENDER_CV_SERVER_URL).port
    final_port = urlparse(config.BLENDER_FINAL_SERVER_URL).port
    seg_port = urlparse(config.BLENDER_SEGMASK_SERVER_URL).port

    print(f"Starting Blender backends on ports: CV={cv_port}, Final={final_port}, Seg={seg_port}, Paper=5004...")
    blender_process = subprocess.Popen(
        ["bash", "./launch_blender_backend.sh", str(cv_port), str(final_port), str(seg_port), "5004"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.join(REPO_ROOT, "inference")
    )

    print("Waiting 15 seconds for servers to initialize...")
    time.sleep(15)
    return blender_process


# ### 3. Initialize Model

# In[10]:


# ### 4. Scene Processing Logic

# In[14]:


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
    length = math.sqrt(_dot(vector, vector))
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _rotate_xy(x: float, y: float, azimuth: float) -> Tuple[float, float]:
    cos_azimuth = math.cos(azimuth)
    sin_azimuth = math.sin(azimuth)
    return (
        x * cos_azimuth - y * sin_azimuth,
        x * sin_azimuth + y * cos_azimuth,
    )


def _project_pkl2fig_point(
    point: Tuple[float, float, float],
    camera_data: Dict[str, Any],
    image_size: int,
) -> Optional[Tuple[float, float, float]]:
    elevation = float(camera_data.get("camera_elevation", math.radians(30.0)))
    lens = float(camera_data.get("lens", 50.0))
    camera_location = (
        6.0 * math.cos(elevation) - 6.0,
        0.0,
        6.0 * math.sin(elevation),
    )
    forward = _normalize((-1.0, 0.0, -math.tan(elevation)))
    right = _normalize(_cross(forward, (0.0, 0.0, 1.0)))
    up = _normalize(_cross(right, forward))
    focal_scale = image_size * lens / 36.0

    relative = (
        point[0] - camera_location[0],
        point[1] - camera_location[1],
        point[2] - camera_location[2],
    )
    depth = _dot(relative, forward)
    if depth <= 1e-4:
        return None
    image_x = image_size * 0.5 + _dot(relative, right) * focal_scale / depth
    image_y = image_size * 0.5 - _dot(relative, up) * focal_scale / depth
    return image_x, image_y, depth


def _pkl2fig_subject_vertices(
    subject_data: Dict[str, Any],
    camera_data: Dict[str, Any],
) -> List[Tuple[float, float, float]]:
    global_scale = float(camera_data.get("global_scale", 1.0))
    width = float(subject_data["width"]) * global_scale
    depth = float(subject_data["depth"]) * global_scale
    height = float(subject_data["height"]) * global_scale
    center_x = float(subject_data["x"]) - 6.0
    center_y = float(subject_data["y"])
    center_z = float(subject_data["z"]) + height / 2.0
    azimuth = math.radians(float(subject_data["azimuth"]))

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
        rotated_x, rotated_y = _rotate_xy(local_x, local_y, azimuth)
        world_vertices.append((center_x + rotated_x, center_y + rotated_y, center_z + local_z))
    return world_vertices


def _compute_pkl2fig_centering_shift(
    subjects_data: List[Dict[str, Any]],
    camera_data: Dict[str, Any],
    image_size: int,
    render_scale: int,
) -> Tuple[int, int]:
    canvas_size = image_size * render_scale
    cubes_layer = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(cubes_layer)
    face_indices = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    for subject_data in subjects_data:
        vertices = _pkl2fig_subject_vertices(subject_data, camera_data)
        projected_vertices = [
            _project_pkl2fig_point(vertex, camera_data, image_size)
            for vertex in vertices
        ]
        for face in face_indices:
            face_points = [projected_vertices[index] for index in face]
            if any(point is None for point in face_points):
                continue
            polygon = [
                (round(point[0] * render_scale), round(point[1] * render_scale))
                for point in face_points
                if point is not None
            ]
            draw.polygon(polygon, fill=255)
            draw.line(
                polygon + [polygon[0]],
                fill=255,
                width=2 * render_scale,
                joint="curve",
            )

    bbox = cubes_layer.getbbox()
    if bbox is None:
        return 0, 0

    component_center_x = (bbox[0] + bbox[2]) / 2.0
    component_center_y = (bbox[1] + bbox[3]) / 2.0
    target_center = canvas_size / 2.0
    return round(target_center - component_center_x), round(target_center - component_center_y)


def compute_pkl2fig_centering_shift_pixels(
    subjects_data: List[Dict[str, Any]],
    camera_data: Dict[str, Any],
    image_size: int = PKL2FIG_LAYOUT_IMAGE_SIZE,
) -> Tuple[int, int]:
    render_scale = 2
    shift_x, shift_y = _compute_pkl2fig_centering_shift(
        subjects_data,
        camera_data,
        image_size,
        render_scale,
    )
    return round(shift_x / render_scale), round(shift_y / render_scale)


def apply_image_space_shift(image: Image.Image, shift_x: int, shift_y: int) -> Image.Image:
    if shift_x == 0 and shift_y == 0:
        return image

    source = image.convert("RGB")
    shifted = Image.new("RGB", source.size, (255, 255, 255))
    shifted.paste(source, (shift_x, shift_y))
    return shifted


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _map_point_to_rgb255(x: float, y: float, z: float) -> Tuple[int, int, int]:
    x_min, x_max = -12.0, -1.0
    y_min_at_xmin, y_max_at_xmin = -4.5, 4.5
    y_min_at_xmax, y_max_at_xmax = -0.5, 0.5
    z_min, z_max = 0.0, 2.50

    x_norm = _clamp01((x - x_min) / (x_max - x_min))
    y_min = y_min_at_xmin + x_norm * (y_min_at_xmax - y_min_at_xmin)
    y_max = y_max_at_xmin + x_norm * (y_max_at_xmax - y_max_at_xmin)
    y_norm = _clamp01((y - y_min) / (y_max - y_min)) if y_max != y_min else 0.5
    z_norm = _clamp01((z - z_min) / (z_max - z_min))
    return (
        round(x_norm * 255),
        round(y_norm * 255),
        round(z_norm * 255),
    )


def render_pkl2fig_styled_condition_image(
    subjects_data: List[Dict[str, Any]],
    camera_data: Dict[str, Any],
    background_img: Image.Image,
    output_path: str,
    image_size: int = PKL2FIG_LAYOUT_IMAGE_SIZE,
) -> Image.Image:
    render_scale = 2
    canvas_size = image_size * render_scale
    shift_x, shift_y = _compute_pkl2fig_centering_shift(
        subjects_data,
        camera_data,
        image_size,
        render_scale,
    )
    face_indices = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    faces = []
    for subject_data in subjects_data:
        vertices = _pkl2fig_subject_vertices(subject_data, camera_data)
        projected_vertices = [
            _project_pkl2fig_point(vertex, camera_data, image_size)
            for vertex in vertices
        ]
        world_x = float(subject_data["x"]) - 6.0
        world_y = float(subject_data["y"])
        world_z = float(subject_data["z"])
        color = _map_point_to_rgb255(world_x, world_y, world_z)

        for face in face_indices:
            face_points = [projected_vertices[index] for index in face]
            if any(point is None for point in face_points):
                continue
            avg_depth = sum(point[2] for point in face_points if point is not None) / len(face)
            polygon = [
                (point[0], point[1])
                for point in face_points
                if point is not None
            ]
            faces.append((avg_depth, polygon, color))

    cube_layer = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(cube_layer, "RGBA")
    for _, polygon, color in sorted(faces, key=lambda item: item[0], reverse=True):
        scaled = [
            (round(x * render_scale) + shift_x, round(y * render_scale) + shift_y)
            for x, y in polygon
        ]
        draw.polygon(scaled, fill=(*color, 120))
        draw.line(
            scaled + [scaled[0]],
            fill=(245, 245, 245, 210),
            width=2 * render_scale,
            joint="curve",
        )

    cube_layer = cube_layer.resize((image_size, image_size), Image.Resampling.LANCZOS)
    background = background_img.convert("RGBA").resize((image_size, image_size), Image.Resampling.BILINEAR)
    image = Image.alpha_composite(background, cube_layer).convert("RGB")
    image.save(output_path)
    return image


def render_pkl2fig_layout_from_pkl(pkl_path: str, output_path: str) -> Tuple[Image.Image, List[Dict[str, Any]], Dict[str, Any]]:
    scene_dict = load_scene_dict(pkl_path)
    subjects_data, camera_data = convert_scene_to_render_payload(scene_dict)
    color_assignments = build_cube_color_assignments(scene_dict.get("subjects_data", []))
    final_img = render_cube_image_pil(
        subjects_data,
        camera_data,
        color_assignments,
        output_path,
        image_size=PKL2FIG_LAYOUT_IMAGE_SIZE,
    )
    return final_img, subjects_data, camera_data


def render_pkl2fig_segmasks(
    subjects_data: List[Dict[str, Any]],
    camera_data: Dict[str, Any],
    output_dir: str,
    image_size: int = PKL2FIG_LAYOUT_IMAGE_SIZE,
) -> List[str]:
    render_scale = 2
    canvas_size = image_size * render_scale
    shift_x, shift_y = _compute_pkl2fig_centering_shift(
        subjects_data,
        camera_data,
        image_size,
        render_scale,
    )
    face_indices = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    segmask_paths = []
    for subject_idx, subject_data in enumerate(subjects_data):
        faces = []
        vertices = _pkl2fig_subject_vertices(subject_data, camera_data)
        projected_vertices = [
            _project_pkl2fig_point(vertex, camera_data, image_size)
            for vertex in vertices
        ]

        for face in face_indices:
            face_points = [projected_vertices[index] for index in face]
            if any(point is None for point in face_points):
                continue
            avg_depth = sum(point[2] for point in face_points if point is not None) / len(face)
            polygon = [
                (point[0], point[1])
                for point in face_points
                if point is not None
            ]
            faces.append((avg_depth, polygon))

        mask = Image.new("L", (canvas_size, canvas_size), 0)
        draw = ImageDraw.Draw(mask)
        for _, polygon in sorted(faces, key=lambda item: item[0], reverse=True):
            scaled = [
                (round(x * render_scale) + shift_x, round(y * render_scale) + shift_y)
                for x, y in polygon
            ]
            draw.polygon(scaled, fill=255)
            draw.line(
                scaled + [scaled[0]],
                fill=255,
                width=2 * render_scale,
                joint="curve",
            )

        mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
        segmask_path = os.path.join(output_dir, f"main__segmask_{str(subject_idx).zfill(3)}__{1.00:.2f}.png")
        mask.save(segmask_path)
        segmask_paths.append(segmask_path)

    return segmask_paths


def render_pkl2fig_segmasks_for_attention(
    subjects_data: List[Dict[str, Any]],
    camera_data: Dict[str, Any],
    mask_size: int,
) -> List[Image.Image]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        mask_paths = render_pkl2fig_segmasks(
            subjects_data,
            camera_data,
            tmp_dir,
            image_size=PKL2FIG_LAYOUT_IMAGE_SIZE,
        )
        masks = []
        for mask_path in mask_paths:
            mask_img = Image.open(mask_path).convert("L")
            mask_img = mask_img.resize((mask_size, mask_size), Image.Resampling.NEAREST)
            masks.append(mask_img.copy())
        return masks


def build_v3_prompt(placeholder_prompt: str) -> str:
    layout_clause = (
        " The final objects must tightly match the provided cube layout: "
        "each object should stay inside its corresponding projected cube footprint, "
        "with matching relative size, height, width, and screen position."
    )
    if "cube layout" in placeholder_prompt.lower():
        return placeholder_prompt
    return placeholder_prompt.rstrip() + layout_clause


def build_v4_prompt(subject_descriptions: List[str], placeholder_prompt: str) -> str:
    return build_v4_identity_prompt(subject_descriptions, build_v3_prompt(placeholder_prompt))


class V3InferenceArgs:
    def __init__(self, jsonl_path: str, pretrained_model_name: str, cond_size: int):
        self.current_train_data_dir = jsonl_path
        self.inference_embeds_dir = ""
        self.pretrained_model_name_or_path = pretrained_model_name
        self.subject_column = None
        self.spatial_column = "cv"
        self.target_column = "target"
        self.caption_column = "PLACEHOLDER_prompts"
        self.cond_size = cond_size
        self.noise_size = 512
        self.revision = None
        self.variant = None
        self.max_sequence_length = 512


def run_v3_layout_inference(
    jsonl_path: str,
    checkpoint_name: str,
    height: int,
    width: int,
    seed: int,
    guidance_scale: float,
    num_inference_steps: int,
    lora_weight: float,
    cond_size: int,
    mask_size: int,
    attention_masks: List[Image.Image],
) -> Tuple[bool, Optional[Image.Image], str]:
    engine = get_inference_engine()
    try:
        if getattr(engine, "_v3_loaded_cond_size", None) != cond_size or getattr(engine, "_v3_loaded_lora_weight", None) != lora_weight:
            engine.current_lora_path = None
            original_load_lora = engine.load_lora

            def load_lora_with_v3_cond_size(self, checkpoint_name_inner, lora_weights=[1.0]):
                from train.src.lora_helper import set_single_lora
                lora_path = "/ptpool/users/qzy/code/seethrough3d/inference/checkpoints/seethrough3d_release/lora.safetensors"
                if not os.path.exists(lora_path):
                    raise FileNotFoundError(f"LoRA checkpoint not found at: {lora_path}")
                set_single_lora(
                    self.pipe.transformer,
                    lora_path,
                    lora_weights=lora_weights,
                    cond_size=cond_size,
                )
                self.current_lora_path = lora_path

            engine.load_lora = types.MethodType(load_lora_with_v3_cond_size, engine)
            try:
                engine.load_lora(checkpoint_name, lora_weights=[lora_weight])
            finally:
                engine.load_lora = original_load_lora
            engine._v3_loaded_cond_size = cond_size
            engine._v3_loaded_lora_weight = lora_weight
        else:
            print(f"LoRA already loaded for infer2_v3 cond_size={cond_size}, weight={lora_weight}")
        inference_args = V3InferenceArgs(jsonl_path, engine.base_model_path, cond_size)
        inference_dataset = make_train_dataset(inference_args, engine.tokenizers, accelerator=None, noise_size=512)
        inference_dataloader = torch.utils.data.DataLoader(
            inference_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        batch = next(iter(inference_dataloader))
        caption = batch["prompts"][0] if isinstance(batch["prompts"], list) else batch["prompts"]
        call_ids = batch["call_ids"]
        spatial_imgs = engine.tensor_to_image_list(batch["cond_pixel_values"])

        mask_tensors = []
        token_mask_size = cond_size // 16
        for mask_img in attention_masks:
            token_mask = mask_img.resize((token_mask_size, token_mask_size), Image.Resampling.NEAREST)
            mask_tensor = torch.as_tensor(list(token_mask.getdata()), dtype=torch.uint8).reshape(token_mask_size, token_mask_size)
            mask_tensor = (mask_tensor > 128).to(torch.uint8)
            mask_tensors.append(mask_tensor)
        cuboids_segmasks = torch.stack(mask_tensors, dim=0).unsqueeze(0)

        print("\n" + "=" * 60)
        print("Running infer2_v3 layout-strengthened inference:")
        print(f"  Prompt: {caption}")
        print(f"  Cond Size: {cond_size}")
        print(f"  Input Mask Size: {mask_size}x{mask_size}")
        print(f"  Attention Token Mask Size: {token_mask_size}x{token_mask_size}")
        print(f"  Guidance Scale: {guidance_scale}")
        print(f"  LoRA Weight: {lora_weight}")
        print(f"  Seed: {seed}, Steps: {num_inference_steps}")
        print("=" * 60 + "\n")

        image = engine.pipe(
            prompt=caption,
            height=int(height),
            width=int(width),
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
            subject_images=[],
            spatial_images=spatial_imgs,
            cond_size=cond_size,
            call_ids=call_ids,
            cuboids_segmasks=cuboids_segmasks,
        ).images[0]
        engine.clear_cache()
        torch.cuda.empty_cache()
        return True, image, "infer2_v3 layout-strengthened inference complete"
    except Exception as error:
        import traceback
        traceback.print_exc()
        return False, None, f"infer2_v3 inference failed: {error}"


def layout_reference_bboxes_from_masks(segmask_paths: List[str], image_size: int) -> List[Dict[str, Any]]:
    boxes = []
    for mask_path in segmask_paths:
        mask_img = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
        bbox = mask_img.getbbox()
        if bbox is None:
            boxes.append({"bbox": None, "area": 0})
        else:
            area = sum(1 for pixel in mask_img.getdata() if pixel > 128)
            boxes.append({"bbox": bbox, "area": area})
    return boxes


def estimate_generated_bboxes(
    generated_image: Image.Image,
    segmask_paths: List[str],
    image_size: int,
) -> List[Dict[str, Any]]:
    image = generated_image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    boxes = []
    for mask_path in segmask_paths:
        mask_img = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
        ref_bbox = mask_img.getbbox()
        if ref_bbox is None:
            boxes.append({"bbox": None, "area": 0})
            continue

        expanded = (
            max(0, ref_bbox[0] - 24),
            max(0, ref_bbox[1] - 24),
            min(image_size, ref_bbox[2] + 24),
            min(image_size, ref_bbox[3] + 24),
        )
        crop = image.crop(expanded)
        mask_crop = mask_img.crop(expanded)
        pixels = list(crop.getdata())
        mask_pixels = list(mask_crop.getdata())
        if not pixels:
            boxes.append({"bbox": None, "area": 0})
            continue

        bg_samples = [
            pixels[index]
            for index, mask_value in enumerate(mask_pixels)
            if mask_value <= 16
        ]
        if not bg_samples:
            bg_samples = pixels
        bg = tuple(sum(channel) / len(bg_samples) for channel in zip(*bg_samples))

        xs = []
        ys = []
        width = expanded[2] - expanded[0]
        for index, pixel in enumerate(pixels):
            distance = sum(abs(float(pixel[channel]) - bg[channel]) for channel in range(3))
            if distance > 36:
                x = index % width
                y = index // width
                xs.append(expanded[0] + x)
                ys.append(expanded[1] + y)

        if not xs:
            boxes.append({"bbox": None, "area": 0})
        else:
            boxes.append({"bbox": (min(xs), min(ys), max(xs) + 1, max(ys) + 1), "area": len(xs)})
    return boxes


def bbox_area(bbox: Optional[Tuple[int, int, int, int]]) -> int:
    if bbox is None:
        return 0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def normalize_object_label(label: str) -> str:
    normalized = str(label or "").strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    for suffix in (" object", " cube"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
    for prefix in ("a ", "an ", "the "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
    for canonical, aliases in V4_OBJECT_ALIASES.items():
        alias_set = {canonical, *aliases}
        if normalized in alias_set:
            return canonical
    return normalized


def build_mask_label_map(subject_descriptions: List[str], segmask_paths: List[str]) -> List[Dict[str, Any]]:
    mapping = []
    for index, mask_path in enumerate(segmask_paths):
        label = subject_descriptions[index] if index < len(subject_descriptions) else f"subject_{index}"
        mapping.append(
            {
                "index": index,
                "mask_path": mask_path,
                "label": label,
                "canonical_label": normalize_object_label(label),
            }
        )
    return mapping


def _candidate_labels_for_target(target_label: str, all_labels: List[str]) -> List[str]:
    target = normalize_object_label(target_label)
    candidates = []
    for label in [target, *all_labels, *V4_COMMON_CANDIDATES]:
        canonical = normalize_object_label(label)
        if canonical and canonical not in candidates:
            candidates.append(canonical)
    return candidates


def _text_prompts_for_label(label: str) -> List[str]:
    canonical = normalize_object_label(label)
    aliases = V4_OBJECT_ALIASES.get(canonical, [canonical])
    prompts = []
    for alias in aliases:
        prompts.append(f"a photo of a {alias}")
        prompts.append(f"a realistic {alias}")
    return prompts


def _load_v4_clip_bundle(model_name: str):
    global _v4_clip_bundle
    global _v4_clip_error
    if _v4_clip_bundle is not None:
        return _v4_clip_bundle
    if _v4_clip_error is not None:
        raise RuntimeError(_v4_clip_error)

    try:
        from transformers import CLIPModel, CLIPProcessor

        requested_device = os.environ.get("V4_CLIP_DEVICE", "cpu").strip().lower()
        if requested_device == "cuda" and torch is not None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.to(device)
        model.eval()
        _v4_clip_bundle = {
            "model": model,
            "processor": processor,
            "device": device,
            "model_name": model_name,
        }
        return _v4_clip_bundle
    except Exception as error:
        _v4_clip_error = f"v4 CLIP semantic verifier unavailable: {error}"
        raise RuntimeError(_v4_clip_error) from error


def crop_masked_region_for_semantics(
    image: Image.Image,
    mask_path: str,
    image_size: int,
    padding: int = 20,
) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int, int, int]]]:
    resized_image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    mask_img = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
    bbox = mask_img.getbbox()
    if bbox is None:
        return None, None

    crop_box = (
        max(0, bbox[0] - padding),
        max(0, bbox[1] - padding),
        min(image_size, bbox[2] + padding),
        min(image_size, bbox[3] + padding),
    )
    crop = resized_image.crop(crop_box)
    crop_mask = mask_img.crop(crop_box)
    white = Image.new("RGB", crop.size, (244, 244, 244))
    crop = Image.composite(crop, white, crop_mask)
    return crop, crop_box


def classify_crop_with_clip(
    crop: Image.Image,
    target_label: str,
    all_labels: List[str],
    clip_model_name: str,
) -> Dict[str, Any]:
    bundle = _load_v4_clip_bundle(clip_model_name)
    model = bundle["model"]
    processor = bundle["processor"]
    device = bundle["device"]

    candidate_labels = _candidate_labels_for_target(target_label, all_labels)
    text_prompts = []
    prompt_to_label = []
    for label in candidate_labels:
        for prompt in _text_prompts_for_label(label):
            text_prompts.append(prompt)
            prompt_to_label.append(label)

    inputs = processor(
        text=text_prompts,
        images=crop,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].detach().float().cpu().tolist()

    label_scores: Dict[str, float] = {}
    for label, score in zip(prompt_to_label, probs):
        label_scores[label] = max(label_scores.get(label, 0.0), float(score))
    ranked = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
    target = normalize_object_label(target_label)
    target_score = label_scores.get(target, 0.0)
    top_label, top_score = ranked[0] if ranked else ("", 0.0)
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    return {
        "target_label": target,
        "top_label": top_label,
        "top_score": top_score,
        "target_score": target_score,
        "second_score": second_score if top_label == target else top_score,
        "scores": dict(ranked[:8]),
    }


def evaluate_generated_semantics(
    generated_image: Image.Image,
    mask_label_map: List[Dict[str, Any]],
    image_size: int,
    clip_model_name: str,
    min_score: float,
    score_margin: float,
) -> Dict[str, Any]:
    all_labels = [item["canonical_label"] for item in mask_label_map]
    issues = []
    results = []
    verifier_error = None

    try:
        _load_v4_clip_bundle(clip_model_name)
    except Exception as error:
        verifier_error = str(error)
        return {
            "pass": False,
            "available": False,
            "error": verifier_error,
            "issues": [
                {
                    "reason": "semantic_verifier_unavailable",
                    "message": verifier_error,
                }
            ],
            "results": [],
        }

    for item in mask_label_map:
        crop, crop_box = crop_masked_region_for_semantics(
            generated_image,
            item["mask_path"],
            image_size,
        )
        if crop is None:
            issue = {
                "index": item["index"],
                "subject": item["label"],
                "target_label": item["canonical_label"],
                "reason": "empty_mask",
            }
            issues.append(issue)
            results.append(issue)
            continue

        classification = classify_crop_with_clip(
            crop,
            item["canonical_label"],
            all_labels,
            clip_model_name,
        )
        top_matches = classification["top_label"] == item["canonical_label"]
        target_score = float(classification["target_score"])
        top_score = float(classification["top_score"])
        margin = target_score - float(classification["second_score"])
        passed = top_matches and target_score >= min_score and margin >= score_margin
        result = {
            "index": item["index"],
            "subject": item["label"],
            "target_label": item["canonical_label"],
            "crop_box": crop_box,
            "pass": passed,
            "top_label": classification["top_label"],
            "top_score": top_score,
            "target_score": target_score,
            "score_margin": margin,
            "scores": classification["scores"],
        }
        results.append(result)
        if not passed:
            issue = dict(result)
            issue["reason"] = "semantic_mismatch"
            issues.append(issue)

    return {
        "pass": len(issues) == 0,
        "available": True,
        "issues": issues,
        "results": results,
    }


def _load_v4_inpaint_pipeline(base_model_path: str):
    global _v4_inpaint_pipe
    global _v4_inpaint_error
    if _v4_inpaint_pipe is not None:
        return _v4_inpaint_pipe
    if _v4_inpaint_error is not None:
        raise RuntimeError(_v4_inpaint_error)

    try:
        from diffusers import (
            FluxInpaintPipeline,
            FluxTransformer2DModel as DiffusersFluxTransformer2DModel,
        )

        engine = get_inference_engine()
        device = getattr(engine, "device", "cuda")
        dtype = torch.bfloat16 if torch is not None and torch.cuda.is_available() else torch.float32
        print("Loading infer2_v4 FluxInpaintPipeline transformer lazily for semantic repair...")
        inpaint_transformer = DiffusersFluxTransformer2DModel.from_pretrained(
            base_model_path,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        _v4_inpaint_pipe = FluxInpaintPipeline(
            scheduler=engine.pipe.scheduler,
            vae=engine.pipe.vae,
            text_encoder=engine.pipe.text_encoder,
            tokenizer=engine.pipe.tokenizer,
            text_encoder_2=engine.pipe.text_encoder_2,
            tokenizer_2=engine.pipe.tokenizer_2,
            transformer=inpaint_transformer,
        )
        _v4_inpaint_pipe.to(device)
        return _v4_inpaint_pipe
    except Exception as error:
        _v4_inpaint_error = f"v4 Flux inpaint repair unavailable: {error}"
        raise RuntimeError(_v4_inpaint_error) from error


def prepare_inpaint_mask(
    mask_path: str,
    image_size: int,
    expand_px: int = 6,
    blur_px: float = 1.0,
) -> Image.Image:
    mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
    mask = mask.point(lambda value: 255 if value > 128 else 0)
    if expand_px > 0:
        filter_size = max(3, int(expand_px) | 1)
        mask = mask.filter(ImageFilter.MaxFilter(filter_size))
    if blur_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(float(blur_px)))
    return mask


def build_v4_inpaint_prompt(target_label: str, placeholder_prompt: str) -> str:
    label = normalize_object_label(target_label)
    label_phrase = {
        "cat": "a single realistic domestic cat, clearly feline, pointed ears, cat face, cat body",
        "dog": "a single realistic domestic dog, clearly canine, dog face, dog body",
        "sofa": "a realistic sofa",
        "lamp": "a realistic lamp",
        "bookshelf": "a realistic bookshelf with shelves and books",
    }.get(label, f"a realistic {label}")
    scene_style = placeholder_prompt.replace("PLACEHOLDER", label_phrase)
    return (
        f"{label_phrase}. Match the existing room lighting, camera perspective, color tone, "
        f"object scale, and the exact masked region. {scene_style}"
    )


def inpaint_single_object_region(
    image: Image.Image,
    mask_path: str,
    target_label: str,
    placeholder_prompt: str,
    base_model_path: str,
    image_size: int,
    seed: int,
    steps: int,
    strength: float,
    guidance_scale: float,
    padding: int,
) -> Image.Image:
    pipe = _load_v4_inpaint_pipeline(base_model_path)
    source = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    mask = prepare_inpaint_mask(mask_path, image_size)
    prompt = build_v4_inpaint_prompt(target_label, placeholder_prompt)
    generator = torch.Generator("cpu").manual_seed(int(seed))
    repaired = pipe(
        prompt=prompt,
        image=source,
        mask_image=mask,
        height=image_size,
        width=image_size,
        padding_mask_crop=max(0, int(padding)),
        strength=max(0.05, min(0.95, float(strength))),
        num_inference_steps=max(1, int(steps)),
        guidance_scale=float(guidance_scale),
        generator=generator,
        max_sequence_length=512,
    ).images[0].convert("RGB")

    # Keep all pixels outside the object mask from the original image.
    composite_mask = prepare_inpaint_mask(mask_path, image_size, expand_px=3, blur_px=1.5)
    return Image.composite(repaired, source, composite_mask)


def repair_semantic_mismatches_with_inpaint(
    image: Image.Image,
    semantic_report: Dict[str, Any],
    mask_label_map: List[Dict[str, Any]],
    placeholder_prompt: str,
    base_model_path: str,
    image_size: int,
    seed: int,
    steps: int,
    strength: float,
    guidance_scale: float,
    padding: int,
    max_rounds: int,
    root_save_dir: str,
    clip_model_name: str,
    min_score: float,
    score_margin: float,
) -> Tuple[Image.Image, Dict[str, Any]]:
    repair_log: Dict[str, Any] = {
        "enabled": True,
        "rounds": [],
        "initial_semantic_pass": bool(semantic_report.get("pass")),
        "final_semantic_pass": bool(semantic_report.get("pass")),
        "error": None,
    }
    current_image = image
    current_report = semantic_report
    map_by_index = {int(item["index"]): item for item in mask_label_map}

    if current_report.get("pass") or max_rounds <= 0:
        return current_image, repair_log

    for repair_round in range(max_rounds):
        issues = [
            issue
            for issue in current_report.get("issues", [])
            if issue.get("reason") == "semantic_mismatch"
        ]
        if not issues:
            break

        round_record = {
            "round": repair_round + 1,
            "repairs": [],
        }
        try:
            for issue_idx, issue in enumerate(issues):
                item = map_by_index.get(int(issue.get("index", -1)))
                if item is None:
                    continue
                print(
                    "infer2_v4 semantic repair:",
                    f"mask {item['index']} {item['label']} top={issue.get('top_label')}",
                )
                current_image = inpaint_single_object_region(
                    current_image,
                    item["mask_path"],
                    item["canonical_label"],
                    placeholder_prompt,
                    base_model_path,
                    image_size,
                    seed + 1000 + repair_round * 97 + issue_idx,
                    steps,
                    strength,
                    guidance_scale,
                    padding,
                )
                repair_path = os.path.join(
                    root_save_dir,
                    f"v4_inpaint_round_{repair_round + 1:02d}_mask_{item['index']:03d}_{item['canonical_label']}.jpg",
                )
                current_image.save(repair_path)
                round_record["repairs"].append(
                    {
                        "index": item["index"],
                        "subject": item["label"],
                        "target_label": item["canonical_label"],
                        "repair_path": repair_path,
                        "previous_top_label": issue.get("top_label"),
                    }
                )

            current_report = evaluate_generated_semantics(
                current_image,
                mask_label_map,
                image_size,
                clip_model_name,
                min_score,
                score_margin,
            )
            round_record["semantic_report_after_round"] = current_report
            repair_log["rounds"].append(round_record)
            repair_log["final_semantic_pass"] = bool(current_report.get("pass"))
            if current_report.get("pass"):
                break
        except Exception as error:
            repair_log["error"] = str(error)
            repair_log["rounds"].append(round_record)
            break

    repair_log["final_semantic_report"] = current_report
    return current_image, repair_log


def evaluate_generated_layout(
    generated_image: Image.Image,
    segmask_paths: List[str],
    subject_descriptions: List[str],
    image_size: int,
) -> Dict[str, Any]:
    reference_boxes = layout_reference_bboxes_from_masks(segmask_paths, image_size)
    generated_boxes = estimate_generated_bboxes(generated_image, segmask_paths, image_size)
    issues = []

    for index, (reference, generated) in enumerate(zip(reference_boxes, generated_boxes)):
        name = subject_descriptions[index] if index < len(subject_descriptions) else f"subject_{index}"
        reference_area = max(1, reference["area"] or bbox_area(reference["bbox"]))
        generated_area = generated["area"] or bbox_area(generated["bbox"])
        coverage = generated_area / float(reference_area)

        ref_bbox = reference["bbox"]
        gen_bbox = generated["bbox"]
        center_error = 0.0
        if ref_bbox is None or gen_bbox is None:
            issues.append(
                {
                    "subject": name,
                    "reason": "missing_generated_bbox",
                    "coverage": coverage,
                    "reference_bbox": ref_bbox,
                    "generated_bbox": gen_bbox,
                }
            )
            continue

        ref_cx = (ref_bbox[0] + ref_bbox[2]) / 2.0
        ref_cy = (ref_bbox[1] + ref_bbox[3]) / 2.0
        gen_cx = (gen_bbox[0] + gen_bbox[2]) / 2.0
        gen_cy = (gen_bbox[1] + gen_bbox[3]) / 2.0
        ref_diag = max(1.0, math.sqrt(bbox_area(ref_bbox)))
        center_error = math.sqrt((ref_cx - gen_cx) ** 2 + (ref_cy - gen_cy) ** 2) / ref_diag

        if coverage < V3_MIN_MASK_COVERAGE or center_error > V3_LAYOUT_BBOX_TOLERANCE:
            issues.append(
                {
                    "subject": name,
                    "reason": "layout_deviation",
                    "coverage": coverage,
                    "center_error": center_error,
                    "reference_bbox": ref_bbox,
                    "generated_bbox": gen_bbox,
                }
            )

    return {
        "pass": len(issues) == 0,
        "issues": issues,
        "reference_boxes": reference_boxes,
        "generated_boxes": generated_boxes,
    }


def process_scene(
    pkl_path,
    checkpoint_name,
    placeholder_prompt,
    image_size_override=None,
    guidance_scale_override=None,
    lora_weight=1.0,
    v3_cond_size=V3_COND_SIZE,
    v3_mask_size=V3_MASK_SIZE,
    v3_max_attempts=V3_MAX_REPAIR_ATTEMPTS,
    v3_max_lora_weight=1.8,
    v4_semantic_check=True,
    v4_inpaint_repair=True,
    v4_clip_model=V4_DEFAULT_CLIP_MODEL,
    v4_semantic_min_score=V4_SEMANTIC_MIN_SCORE,
    v4_semantic_score_margin=V4_SEMANTIC_SCORE_MARGIN,
    v4_max_inpaint_repairs=1,
    v4_inpaint_steps=V4_INPAINT_STEPS,
    v4_inpaint_strength=V4_INPAINT_STRENGTH,
    v4_inpaint_guidance=V4_INPAINT_GUIDANCE,
    v4_inpaint_padding=V4_INPAINT_PADDING,
):
    print(f"\n{'='*80}\nProcessing scene from: {pkl_path}\n{'='*80}")

    # Load Scene
    scene_manager = SceneManager()
    success, num_objects, error = scene_manager.load_scene_from_pkl(pkl_path)
    if not success:
        print(f"Error loading scene: {error}")
        # return None  # 原代码
        return None, None, None

    print(f"PKL 文件: {pkl_path}")
    print(f"PKL 里有 {num_objects} 个 cube")
    print(f"scene_manager.objects 实际长度 = {len(scene_manager.objects)}")


    # Prepare prompts and placeholders 原代码开始
    surrounding_prompt = scene_manager.surrounding_prompt
    print("surrounding_prompt:",surrounding_prompt)
    ### 原代码结束##3


    ###自己写的开始### 修改cube的名字
    # custom_subject_descriptions = [ # 每个cube的名字
    #     "a wooden bed",
    #     "a white nightstand",
    #     "a small table lamp",
    #     "a cat",
    #     "a toy"
    # ]

    # # cube 名字数量必须和cube数量一致
    # assert len(custom_subject_descriptions) == len(scene_manager.objects), \
    #     f"{len(custom_subject_descriptions)=} != {len(scene_manager.objects)=}"

    # for obj, desc in zip(scene_manager.objects, custom_subject_descriptions):
    #     obj['description'] = desc
    ###自己写的结束###

    # 真正影响生图内容的是cube的名字，而不是prompt。cube名字即：obj['description']，是在pkl中定义的，具体代码是inference/app.py:433，搜索【写入pkl的代码】
    subject_descriptions = scene_manager.get_subject_descriptions_for_prompt()
    print("pkl中cube的名称:",subject_descriptions)

    placeholder_prompt = build_v4_prompt(subject_descriptions, placeholder_prompt)
    print("输入flux的prompt:",placeholder_prompt)

    subject_embeds = [] 
    for subject_desc in subject_descriptions: 
        input_ids = tokenizer.encode(subject_desc, return_tensors="pt", max_length=77)[0] 
        subject_embed = {"input_ids_t5": input_ids.tolist()} 
        subject_embeds.append(subject_embed)

    placeholder_token_prompt = build_placeholder_token_prompt(subject_descriptions, placeholder_prompt)

    # Get Call IDs
    call_ids = get_call_ids_from_placeholder_prompt_flux(
        prompt=placeholder_token_prompt, 
        subjects=subject_descriptions, 
        subjects_embeds=subject_embeds,
        debug=False
    )

    # Save Rendering Results for Inference
    # root_save_dir = config.GRADIO_FILES_DIR  # 原代码
    
    #### 自己写的开始####
    beijing_tz = timezone(timedelta(hours=8))
    now_bj = datetime.now(beijing_tz)
    # 以当前 pkl 文件名为前缀
    pkl_basename = os.path.splitext(os.path.basename(pkl_path))[0]
    # 按“月+日+几点几分”生成后缀，例如 0311_1423
    folder_suffix = now_bj.strftime("%m%d_%H%M")
    folder_name = f"{pkl_basename}_{folder_suffix}"
    # 文件夹仍然放在 inference 目录下
    # inference_dir = os.path.dirname(config.GRADIO_FILES_DIR)  # 原代码
    inference_dir = config.GRADIO_FILES_DIR
    print("inference_dir:",inference_dir)
    root_save_dir = os.path.join(inference_dir, folder_name)
    print("root_save_dir:",root_save_dir)

    os.makedirs(root_save_dir, exist_ok=True)
    # os.system(f"rm -f {root_save_dir}/*") 
    #### 自己写的结束####

    cube_layout_filename = build_cube_layout_filename(pkl_path)
    final_render_path = os.path.join(root_save_dir, cube_layout_filename)# 送入flux的cube layout图
    print("cube layout图(final_render_path):",final_render_path)
    pkl2fig_reference_path = os.path.join(root_save_dir, f"pkl2fig_reference_{cube_layout_filename}")
    print("Rendering pkl2fig_v3 PIL reference layout...")
    final_img, layout_subjects_data, layout_camera_data = render_pkl2fig_layout_from_pkl(
        pkl_path,
        pkl2fig_reference_path,
    )
    print("pkl2fig_v3 reference layout render received:", pkl2fig_reference_path)

    print("Rendering Blender final empty background for v2-style tone/light...")
    background_img = scene_manager.render_client._send_render_request(
        scene_manager.render_client.final_server_url,
        [],
        layout_camera_data,
    )
    if background_img is None:
        print("Error rendering final background (Ensure cycles server is running correctly).")
        return None, None, None
    print("Compositing pkl2fig-projected cube layer over Blender final background...")
    final_img = render_pkl2fig_styled_condition_image(
        layout_subjects_data,
        layout_camera_data,
        background_img,
        final_render_path,
    )
    print("Styled pkl2fig-aligned condition image saved.")

    print("Rendering pkl2fig-aligned segmentation masks...")
    segmask_paths = render_pkl2fig_segmasks(
        layout_subjects_data,
        layout_camera_data,
        root_save_dir,
    )
    print(f"pkl2fig-aligned segmask count: {len(segmask_paths)}")
    mask_label_map = build_mask_label_map(subject_descriptions, segmask_paths)
    mask_label_map_path = os.path.join(root_save_dir, "mask_label_map.json")
    with open(mask_label_map_path, "w", encoding="utf-8") as f_map:
        json.dump(mask_label_map, f_map, ensure_ascii=False, indent=2)
    print("infer2_v4 mask-label map:", mask_label_map_path)

    # 自己加的，同时把送入flux的文字 prompt 保存为 prompt.txt，便于复现和检查
    prompt_path = os.path.join(root_save_dir, "prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f_prompt:
        f_prompt.write(placeholder_prompt)

    # Create cuboids JSONL for dataloader，送入flux的集合
    jsonl = [{
        "cv": final_render_path,
        "target": final_render_path,
        "cuboids_segmasks": segmask_paths,
        "PLACEHOLDER_prompts": placeholder_prompt, 
        "subjects": subject_descriptions, 
        "call_ids": call_ids,
        "mask_label_map": mask_label_map,
    }] 

    jsonl_path = os.path.join(root_save_dir, "cuboids.jsonl")
    with open(jsonl_path, "w") as f: 
        json.dump(jsonl[0], f)

    # Fetch generation parameters
    params = scene_manager.inference_params
    image_size = image_size_override or params.get('height', 512)
    num_steps = params.get('num_inference_steps', 25)

    attention_masks = render_pkl2fig_segmasks_for_attention(
        layout_subjects_data,
        layout_camera_data,
        v3_mask_size,
    )

    print("\nRunning Diffusion Inference with v3 layout verification loop...")
    best_image = None
    best_report = None
    best_semantic_report = None
    best_repair_log = None
    attempt_records = []
    for attempt_idx, (attempt_lora_weight, attempt_guidance, attempt_seed) in enumerate(V4_FIXED_INFERENCE_PARAMS):
        param_tag = format_generation_param_tag(
            attempt_lora_weight,
            attempt_guidance,
            attempt_seed,
        )

        success, generated_image, msg = run_v3_layout_inference(
            jsonl_path=jsonl_path,
            checkpoint_name=checkpoint_name,
            height=image_size,
            width=image_size,
            seed=attempt_seed,
            guidance_scale=attempt_guidance,
            num_inference_steps=num_steps,
            lora_weight=attempt_lora_weight,
            cond_size=v3_cond_size,
            mask_size=v3_mask_size,
            attention_masks=attention_masks,
        )
        if not success:
            print(f"Inference attempt {attempt_idx + 1} failed: {msg}")
            attempt_records.append(
                {
                    "attempt": attempt_idx + 1,
                    "success": False,
                    "message": msg,
                    "seed": attempt_seed,
                    "guidance_scale": attempt_guidance,
                    "lora_weight": attempt_lora_weight,
                    "param_tag": param_tag,
                }
            )
            continue

        attempt_path = os.path.join(
            root_save_dir,
            f"attempt_{attempt_idx + 1:02d}_{param_tag}_{os.path.basename(pkl_path).replace('.pkl', '.jpg')}",
        )
        generated_image.save(attempt_path)
        report = evaluate_generated_layout(
            generated_image,
            segmask_paths,
            subject_descriptions,
            image_size,
        )
        report.update(
            {
                "attempt": attempt_idx + 1,
                "success": True,
                "seed": attempt_seed,
                "guidance_scale": attempt_guidance,
                "lora_weight": attempt_lora_weight,
                "param_tag": param_tag,
                "attempt_path": attempt_path,
            }
        )

        semantic_report = {
            "pass": True,
            "available": False,
            "skipped": True,
            "reason": "v4_semantic_check_disabled",
            "issues": [],
            "results": [],
        }
        repair_log = {
            "enabled": False,
            "initial_semantic_pass": True,
            "final_semantic_pass": True,
            "rounds": [],
        }
        if v4_semantic_check:
            semantic_report = evaluate_generated_semantics(
                generated_image,
                mask_label_map,
                image_size,
                v4_clip_model,
                v4_semantic_min_score,
                v4_semantic_score_margin,
            )
            print(
                f"Attempt {attempt_idx + 1} semantic pass: {semantic_report.get('pass')}, "
                f"issues: {len(semantic_report.get('issues', []))}"
            )
            if (
                report["pass"]
                and not semantic_report.get("pass")
                and semantic_report.get("available", False)
                and v4_inpaint_repair
                and int(v4_max_inpaint_repairs) > 0
            ):
                generated_image, repair_log = repair_semantic_mismatches_with_inpaint(
                    generated_image,
                    semantic_report,
                    mask_label_map,
                    placeholder_prompt,
                    config.PRETRAINED_MODEL_NAME_OR_PATH,
                    image_size,
                    attempt_seed,
                    v4_inpaint_steps,
                    v4_inpaint_strength,
                    v4_inpaint_guidance,
                    v4_inpaint_padding,
                    int(v4_max_inpaint_repairs),
                    root_save_dir,
                    v4_clip_model,
                    v4_semantic_min_score,
                    v4_semantic_score_margin,
                )
                semantic_report = repair_log.get("final_semantic_report", semantic_report)
                repaired_attempt_path = os.path.join(
                    root_save_dir,
                    f"attempt_{attempt_idx + 1:02d}_{param_tag}_v4_semantic_repaired_{os.path.basename(pkl_path).replace('.pkl', '.jpg')}",
                )
                generated_image.save(repaired_attempt_path)
                report = evaluate_generated_layout(
                    generated_image,
                    segmask_paths,
                    subject_descriptions,
                    image_size,
                )
                report.update(
                    {
                        "attempt": attempt_idx + 1,
                        "success": True,
                        "seed": attempt_seed,
                        "guidance_scale": attempt_guidance,
                        "lora_weight": attempt_lora_weight,
                        "param_tag": param_tag,
                        "attempt_path": repaired_attempt_path,
                        "semantic_repaired": True,
                    }
                )
            elif v4_semantic_check:
                repair_log = {
                    "enabled": bool(v4_inpaint_repair),
                    "initial_semantic_pass": bool(semantic_report.get("pass")),
                    "final_semantic_pass": bool(semantic_report.get("pass")),
                    "rounds": [],
                    "error": None,
                    "skipped_reason": (
                        None
                        if semantic_report.get("pass")
                        else "layout_failed_or_verifier_unavailable_or_repair_disabled"
                    ),
                }

        report["semantic_pass"] = bool(semantic_report.get("pass"))
        report["semantic_available"] = bool(semantic_report.get("available", False))
        report["semantic_issue_count"] = len(semantic_report.get("issues", []))
        report["semantic_repair_error"] = repair_log.get("error") if isinstance(repair_log, dict) else None
        attempt_records.append(report)
        best_image = generated_image
        best_report = report
        best_semantic_report = semantic_report
        best_repair_log = repair_log
        print(f"Attempt {attempt_idx + 1} layout pass: {report['pass']}, issues: {len(report['issues'])}")

    verification_path = os.path.join(root_save_dir, "layout_verification.json")
    with open(verification_path, "w", encoding="utf-8") as f_verify:
        json.dump(
            {
                "pass": bool(best_report and best_report.get("pass")),
                "attempts": attempt_records,
                "cond_size": v3_cond_size,
                "mask_size": v3_mask_size,
            },
            f_verify,
            ensure_ascii=False,
            indent=2,
        )

    semantic_verification_path = os.path.join(root_save_dir, "semantic_verification.json")
    with open(semantic_verification_path, "w", encoding="utf-8") as f_semantic:
        json.dump(
            {
                "pass": bool(best_semantic_report and best_semantic_report.get("pass")),
                "available": bool(best_semantic_report and best_semantic_report.get("available", False)),
                "enabled": bool(v4_semantic_check),
                "clip_model": v4_clip_model,
                "min_score": v4_semantic_min_score,
                "score_margin": v4_semantic_score_margin,
                "mask_label_map": mask_label_map,
                "best_report": best_semantic_report,
            },
            f_semantic,
            ensure_ascii=False,
            indent=2,
        )

    inpaint_repair_log_path = os.path.join(root_save_dir, "inpaint_repair_log.json")
    with open(inpaint_repair_log_path, "w", encoding="utf-8") as f_repair:
        json.dump(
            best_repair_log or {"enabled": bool(v4_inpaint_repair), "rounds": []},
            f_repair,
            ensure_ascii=False,
            indent=2,
        )

    if best_image is not None:
        print("Inference complete with v3 layout verification.")
        print("layout verification:", verification_path)
        print("semantic verification:", semantic_verification_path)
        print("inpaint repair log:", inpaint_repair_log_path)
        return best_image, root_save_dir, best_report

    print("All inference attempts failed.")
    return None, None, None


# ### 5. Visualizing OSCR 
# Inspect the rendered Blender image, segmentation masks, and prompt **before** running inference.

# In[15]:


def visualize_input_conditions(pkl_path, placeholder_prompt):
    """Render pkl2fig-aligned layout and segmentation masks for a given pickle and display."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # ── 1. Load scene ────────────────────────────────────────────────────────
    sm = SceneManager()
    success, num_objects, error = sm.load_scene_from_pkl(pkl_path)
    if not success:
        print(f"Error loading scene: {error}")
        return

    subject_descriptions = sm.get_subject_descriptions_for_prompt()

    # ── 2. Build the full placeholder prompt ─────────────────────────────────
    full_prompt = build_placeholder_token_prompt(subject_descriptions, placeholder_prompt)

    # ── 3. Render the same styled condition image and masks used for inference ─
    with tempfile.TemporaryDirectory() as tmp_dir:
        preview_path = os.path.join(tmp_dir, build_cube_layout_filename(pkl_path))
        pkl2fig_reference_path = os.path.join(tmp_dir, f"pkl2fig_reference_{build_cube_layout_filename(pkl_path)}")
        print("Rendering pkl2fig_v3 reference layout for visualization...")
        _, layout_subjects_data, layout_camera_data = render_pkl2fig_layout_from_pkl(
            pkl_path,
            pkl2fig_reference_path,
        )
        print("Rendering Blender final empty background for visualization...")
        background_img = sm.render_client._send_render_request(
            sm.render_client.final_server_url,
            [],
            layout_camera_data,
        )
        final_img = render_pkl2fig_styled_condition_image(
            layout_subjects_data,
            layout_camera_data,
            background_img,
            preview_path,
        )
        segmask_paths = render_pkl2fig_segmasks(
            layout_subjects_data,
            layout_camera_data,
            tmp_dir,
        )
        segmask_images = [Image.open(mask_path).copy() for mask_path in segmask_paths]

    n_masks = len(segmask_images)

    # ── 4. Figure layout ──────────────────────────────────────────────────────
    # Columns: render | mask_0 | mask_1 | … | mask_N
    n_cols = 1 + max(n_masks, 1)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(5 * n_cols, 6),
        gridspec_kw={"wspace": 0.04}
    )
    if n_cols == 1:
        axes = [axes]

    # Dark background
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#0d0d0d")

    TITLE_KW  = dict(color="#e0e0e0", fontsize=11, fontweight="bold", pad=8)
    BORDER_KW = dict(linewidth=2)

    # ── Blender render ────────────────────────────────────────────────────────
    ax_render = axes[0]
    ax_render.imshow(np.array(final_img))
    ax_render.set_title("OSCR", **TITLE_KW)
    for spine in ax_render.spines.values():
        spine.set_edgecolor("#4fc3f7")
        spine.set(**BORDER_KW)
    ax_render.set_xticks([])
    ax_render.set_yticks([])

    # ── Segmentation masks ────────────────────────────────────────────────────
    accent_colors = [
        "#ef5350", "#42a5f5", "#66bb6a", "#ffa726",
        "#ab47bc", "#26c6da", "#d4e157", "#ff7043",
    ]
    legend_patches = []

    for i, ax in enumerate(axes[1:]):
        if i < n_masks:
            mask_img  = segmask_images[i]
            mask_arr  = np.array(mask_img.convert("L")) if mask_img.mode != "L" else np.array(mask_img)
            color_hex = accent_colors[i % len(accent_colors)]
            color_rgb = tuple(int(color_hex.lstrip("#")[j:j+2], 16) / 255.0 for j in (0, 2, 4))

            # Colorize: accent colour on dark background
            colored = np.zeros((*mask_arr.shape, 3))
            alpha   = mask_arr / 255.0
            for c, val in enumerate(color_rgb):
                colored[:, :, c] = val * alpha + 0.08 * (1 - alpha)
            colored = np.clip(colored, 0, 1)

            ax.imshow(colored)
            obj_label = subject_descriptions[i] if i < len(subject_descriptions) else f"Object {i}"
            ax.set_title(f"Mask {i}: {obj_label}", **TITLE_KW)
            for spine in ax.spines.values():
                spine.set_edgecolor(color_hex)
                spine.set(**BORDER_KW)
            legend_patches.append(mpatches.Patch(facecolor=color_hex, label=obj_label))
        else:
            ax.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    # ── Prompt strip ─────────────────────────────────────────────────────────
    fig.text(
        0.5, -0.02,
        f'"  {full_prompt.replace("<placeholder>", "")}  "',
        ha="center", va="top",
        fontsize=10, color="#b0bec5", style="italic", wrap=True,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                  edgecolor="#37474f", linewidth=1.2)
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc="lower right",
            framealpha=0.25,
            facecolor="#1a1a2e",
            edgecolor="#37474f",
            labelcolor="#e0e0e0",
            fontsize=9,
            bbox_to_anchor=(0.99, 0.06)
        )

    # ── Super-title ───────────────────────────────────────────────────────────
    fig.suptitle(
        f"Input Conditions  ·  {num_objects} object(s)  ·  {os.path.basename(pkl_path)}",
        color="#ffffff", fontsize=13, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.show()
    print(f"\nFull prompt:\n  {full_prompt}\n")


def main():
    args = parse_args()
    if "PLACEHOLDER" not in args.placeholder_prompt:
        raise ValueError("`--placeholder-prompt` 必须包含 PLACEHOLDER。")

    scene_pickle_paths = normalize_scene_pickle_paths(args.scene_pkls)
    for pkl_path in scene_pickle_paths:
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"找不到场景 pkl: {pkl_path}")

    load_runtime_dependencies()
    blender_process = start_blender_backends()
    try:
        initialize_inference_engine(base_model_path=config.PRETRAINED_MODEL_NAME_OR_PATH)

        if scene_pickle_paths:
            visualize_input_conditions(scene_pickle_paths[0], args.placeholder_prompt)

        checkpoint = config.CHECKPOINT_NAMES[0]
        for pkl_path in scene_pickle_paths:
            image, root_save_dir, best_report = process_scene(
                pkl_path,
                checkpoint_name=checkpoint,
                placeholder_prompt=args.placeholder_prompt,
                image_size_override=args.image_size,
                guidance_scale_override=args.guidance_scale,
                lora_weight=args.lora_weight,
                v3_cond_size=args.v3_cond_size,
                v3_mask_size=args.v3_mask_size,
                v3_max_attempts=args.v3_max_attempts,
                v3_max_lora_weight=args.v3_max_lora_weight,
                v4_semantic_check=not args.v4_disable_semantic_check,
                v4_inpaint_repair=not args.v4_disable_inpaint_repair,
                v4_clip_model=args.v4_clip_model,
                v4_semantic_min_score=args.v4_semantic_min_score,
                v4_semantic_score_margin=args.v4_semantic_score_margin,
                v4_max_inpaint_repairs=args.v4_max_inpaint_repairs,
                v4_inpaint_steps=args.v4_inpaint_steps,
                v4_inpaint_strength=args.v4_inpaint_strength,
                v4_inpaint_guidance=args.v4_inpaint_guidance,
                v4_inpaint_padding=args.v4_inpaint_padding,
            )
            if image:
                print(f"Success for {pkl_path}:")
                display(image)
                param_tag = (
                    best_report.get("param_tag", "params_unknown")
                    if isinstance(best_report, dict)
                    else "params_unknown"
                )
                basename = os.path.basename(pkl_path).replace('.pkl', '.jpg')
                flux_fig_path = os.path.join(root_save_dir, f"best_{param_tag}_{basename}")
                print("flux生成的图片:", flux_fig_path)
                image.save(flux_fig_path)
            else:
                print(f"Skipping {pkl_path} due to error.\n")
    finally:
        print("Starting cleanup...")
        blender_process.terminate()
        blender_process.wait()


if __name__ == "__main__":
    main()
