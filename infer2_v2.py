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
import shutil
import subprocess
from PIL import Image
from typing import List

try:
    from IPython.display import display
except ModuleNotFoundError:
    def display(obj):
        print(obj)
from datetime import datetime, timezone, timedelta  # 为了生成北京时间的时间戳

from urllib.parse import urlparse


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
initialize_inference_engine = None
run_inference_from_gradio = None
SceneManager = None
get_call_ids_from_placeholder_prompt_flux = None
tokenizer = None
config = None


def load_runtime_dependencies() -> None:
    global initialize_inference_engine
    global run_inference_from_gradio
    global SceneManager
    global get_call_ids_from_placeholder_prompt_flux
    global tokenizer
    global config

    if initialize_inference_engine is not None:
        return

    import importlib

    backend_v2_module = importlib.import_module("inference.infer_backend_v2")
    sys.modules["inference.infer_backend"] = backend_v2_module

    from inference.infer_backend_v2 import (
        initialize_inference_engine as _initialize_inference_engine,
        run_inference_from_gradio as _run_inference_from_gradio,
    )
    from inference.app import (
        SceneManager as _SceneManager,
        get_call_ids_from_placeholder_prompt_flux as _get_call_ids_from_placeholder_prompt_flux,
        tokenizer as _tokenizer,
    )
    import inference.config as _config

    initialize_inference_engine = _initialize_inference_engine
    run_inference_from_gradio = _run_inference_from_gradio
    SceneManager = _SceneManager
    get_call_ids_from_placeholder_prompt_flux = _get_call_ids_from_placeholder_prompt_flux
    tokenizer = _tokenizer
    config = _config


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
    return parser.parse_args()


def normalize_scene_pickle_paths(scene_pickle_paths: List[str]) -> List[str]:
    normalized_paths = []
    for pkl_path in scene_pickle_paths:
        if os.path.isabs(pkl_path):
            normalized_paths.append(pkl_path)
        else:
            normalized_paths.append(os.path.join(REPO_ROOT, pkl_path))
    return normalized_paths


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


def process_scene(
    pkl_path,
    checkpoint_name,
    placeholder_prompt,
    image_size_override=None,
    guidance_scale_override=None,
    lora_weight=1.0,
):
    print(f"\n{'='*80}\nProcessing scene from: {pkl_path}\n{'='*80}")

    # Load Scene
    scene_manager = SceneManager()
    success, num_objects, error = scene_manager.load_scene_from_pkl(pkl_path)
    if not success:
        print(f"Error loading scene: {error}")
        # return None  # 原代码
        return None, None

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

    # Render Blender Views
    subjects_data, camera_data = scene_manager._convert_to_blender_format()

    print("Rendering final view...")
    final_img = scene_manager.render_client._send_render_request(
        scene_manager.render_client.final_server_url, 
        subjects_data, 
        camera_data
    )

    if final_img is None:
        print("Error rendering final view (Ensure cycles server is running correctly).")
        # return None  # 原代码
        return None, None
    else:
        print("CV Render received.")

    print("Rendering segmentation masks...")
    success, segmask_images, error_msg = scene_manager.render_client.render_segmasks(subjects_data, camera_data)
    if not success:
        print(f"Failed to render segmasks: {error_msg}")
        # Sometimes it depends on backend state, but let's continue if it fails? 
        # Better to return None as downstream needs them.
        # return None  # 原代码
        return None, None

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
    final_img.save(final_render_path)

    # 自己加的，同时把送入flux的文字 prompt 保存为 prompt.txt，便于复现和检查
    prompt_path = os.path.join(root_save_dir, "prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f_prompt:
        f_prompt.write(placeholder_prompt)

    for subject_idx in range(len(subject_descriptions)): 
        shutil.move(
            f"inference/{str(subject_idx).zfill(3)}_segmask_cv.png", # 送入flux的mask图,每个cube都有一个mask图
            os.path.join(root_save_dir, f"main__segmask_{str(subject_idx).zfill(3)}__{1.00:.2f}.png")
        ) 

    # Create cuboids JSONL for dataloader，送入flux的集合
    jsonl = [{
        "cv": final_render_path,
        "target": final_render_path,
        "cuboids_segmasks": [
            os.path.join(root_save_dir, f"main__segmask_{str(subject_idx).zfill(3)}__{1.00:.2f}.png") 
            for subject_idx in range(len(subject_descriptions))
        ],
        "PLACEHOLDER_prompts": placeholder_prompt, 
        "subjects": subject_descriptions, 
        "call_ids": call_ids, 
    }] 

    jsonl_path = os.path.join(root_save_dir, "cuboids.jsonl")
    with open(jsonl_path, "w") as f: 
        json.dump(jsonl[0], f)

    # Fetch generation parameters
    params = scene_manager.inference_params
    image_size = image_size_override or params.get('height', 512)
    seed = params.get('seed', 42)
    guidance_scale = guidance_scale_override if guidance_scale_override is not None else params.get('guidance_scale', 3.5)
    num_steps = params.get('num_inference_steps', 25)

    print("\nRunning Diffusion Inference...")
    success, generated_image, msg = run_inference_from_gradio(
        checkpoint_name=checkpoint_name,
        height=image_size,
        width=image_size,
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        jsonl_path=jsonl_path,
        lora_weight=lora_weight,
    )

    if success:
        print("Inference complete!")
        # return generated_image  # 原代码
        return generated_image, root_save_dir
    else:
        print(f"Inference failed: {msg}")
        return None, None


# ### 5. Visualizing OSCR 
# Inspect the rendered Blender image, segmentation masks, and prompt **before** running inference.

# In[15]:


def visualize_input_conditions(pkl_path, placeholder_prompt):
    """Render the Blender scene and segmentation masks for a given pickle and display."""
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

    # ── 3. Render via Blender ─────────────────────────────────────────────────
    subjects_data, camera_data = sm._convert_to_blender_format()

    print("Rendering Blender scene for visualization...")
    final_img = sm.render_client._send_render_request(
        sm.render_client.final_server_url, subjects_data, camera_data
    )
    if final_img is None:
        print("Could not render final view – is the Blender backend running?")
        return

    print("Rendering segmentation masks...")
    ok, segmask_images, err_msg = sm.render_client.render_segmasks(subjects_data, camera_data)
    if not ok:
        print(f"Could not render segmentation masks: {err_msg}")
        segmask_images = []

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
            image, root_save_dir = process_scene(
                pkl_path,
                checkpoint_name=checkpoint,
                placeholder_prompt=args.placeholder_prompt,
                image_size_override=args.image_size,
                guidance_scale_override=args.guidance_scale,
                lora_weight=args.lora_weight,
            )
            if image:
                print(f"Success for {pkl_path}:")
                display(image)
                basename = os.path.basename(pkl_path).replace('.pkl', '.jpg')
                flux_fig_path = os.path.join(root_save_dir, basename)
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
