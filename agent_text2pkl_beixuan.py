import http.client
import json
import os
import pickle
import sys
from typing import Any, Dict


API_HOST = "api.chatanywhere.tech"
API_PATH = "/v1/responses"
API_MODEL = "gpt-5.2"


def build_prompt(scene_text: str) -> str:
    """
    构造提示词：让大模型直接输出和 example0.pkl / example0.json 一致结构的 JSON。
    """
    return f"""
你是一个 3D 场景布局助手。现在有一段英文文字描述一个平面场景，请你把它转换成一个 JSON 对象，
结构必须严格符合下面的 Python 字典结构说明（字段名和层级必须完全一致）：

scene_dict = {{
  "subjects_data": [  # 场景中所有物体
    {{
      "name": "字符串，物体的自然语言描述，如 'sedan'",
      "type": "字符串，资产类型，如 'sedan'、'chair'、'dog' 等，必须能在资产库中出现",
      "dims": [宽度width, 深度depth, 高度height],  # 三个浮点数，单位任意但要相互合理
      "x": [x坐标],   # 物体中心在布局坐标系中的 X（远近方向），只要相对关系合理即可
      "y": [y坐标],   # 物体中心在布局坐标系中的 Y（左右方向）
      "z": [z坐标],   # 物体中心在布局坐标系中的 Z 基准（高度基准，可以先设为 0）
      "azimuth": [朝向角弧度],  # 水平朝向，单位是弧度，可简单设为 0 或一个合理的值
      "bbox": [[0, 0, 0, 0]]  # 始终用占位 [0,0,0,0]
    }},
    ...
  ],
  "camera_data": {{
    "camera_elevation": 浮点数弧度,  # 相机仰角，建议使用 0.2 左右
    "lens": 50.0,
    "global_scale": 1.0
  }},
  "surrounding_prompt": "字符串，对整体环境的自然语言描述，可以直接复用输入文本或其精炼版",
  "inference_params": {{
    "height": 1024,
    "width": 1024,
    "seed": 42,
    "guidance_scale": 3.5,
    "num_inference_steps": 25,
    "checkpoint": "rgb__finetune_1024/epoch-1__checkpoint-5000"
  }},
  "checkpoint": "seethrough3d_release/seethrough3d_release"
}}

要求：
1. 根据输入文字，识别出场景中有哪些物体（例如 sedan, chair, bicycle, dog, cat 等），为每个物体生成一条 subjects_data。
2. 物体的 type 要和 name 吻合，例如名字含有 "sedan" 则 type 用 "sedan"。
3. 根据文字中的空间关系（如 left of, right of, in front of, behind），给出一组自洽的 (x, y) 坐标：
   - 可以把某个主体（例如第一辆车）放在 x=-6, y=0 左右，
   - "in front of" 可以理解为 x 值比参照物更大一点（更靠近相机），
   - "behind" 可以理解为 x 值更小一点，
   - "to the left of" 可以理解为 y 值更小一点，
   - "to the right of" 可以理解为 y 值更大一点。
   只要整体相对关系和文本一致即可，不需要完全精准。
4. dims 请给出大致合理的尺寸，不需要和真实数据完全一致，只要大小比例看起来合理即可。
5. 严格输出一个 JSON 对象，不能有多余的解释文字，也不要使用 ``` 包裹。
6. 除非prompt明确说明某个物体位于高处，否则对于地面物体，请使用 z=0.0。

现在的场景描述如下（英文）：
\"\"\"{scene_text}\"\"\" 
    """.strip()


def call_llm(content: str) -> str:
    """
    调用大模型 API，返回模型生成的纯文本（期望是 JSON 字符串）。
    需要环境变量 CHATANYWHERE_API_KEY 提供 API Key。
    """
    # api_key = os.getenv("CHATANYWHERE_API_KEY", "").strip()
    api_key = "REPLACE_WITH_API_KEY"
    if not api_key:
        raise RuntimeError("环境变量 CHATANYWHERE_API_KEY 未设置，无法调用大模型 API。")

    payload = json.dumps(
        {
            "model": API_MODEL,
            "input": content,
        }
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    conn = http.client.HTTPSConnection(API_HOST)
    conn.request("POST", API_PATH, payload, headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    conn.close()

    try:
        resp = json.loads(data)
    except json.JSONDecodeError:
        # 如果服务端直接返回纯文本，就直接当成文本用
        return data

    # 兼容几种常见的 responses 格式，尽量提取出模型文本
    # 1) OpenAI Responses API 风格：resp["output"][0]["content"][0]["text"]
    if isinstance(resp, dict):
        out = resp.get("output")
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict):
                content_list = first.get("content")
                if isinstance(content_list, list) and content_list:
                    first_content = content_list[0]
                    if isinstance(first_content, dict) and "text" in first_content:
                        return first_content["text"]

    # 2) 简单的 { "text": "..." } / { "output_text": "..." }
    for key in ("text", "output_text", "message"):
        if isinstance(resp, dict) and key in resp and isinstance(resp[key], str):
            return resp[key]

    # 3) 回退：把整个 JSON 再转回字符串
    return json.dumps(resp, ensure_ascii=False)


def parse_scene_json(text: str) -> Dict[str, Any]:
    """
    从模型输出中提取 JSON，解析为 Python dict。
    """
    # 尝试找到第一个 '{' 和最后一个 '}'，避免前后有无关文字
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"无法从模型输出中提取 JSON：{text[:200]}...")

    json_str = text[start : end + 1]
    scene = json.loads(json_str)

    # 补全 / 修正关键字段，防止模型漏掉
    if "inference_params" not in scene or not isinstance(scene["inference_params"], dict):
        scene["inference_params"] = {}
    inf = scene["inference_params"]
    inf.setdefault("height", 1024)
    inf.setdefault("width", 1024)
    inf.setdefault("seed", 42)
    inf.setdefault("guidance_scale", 3.5)
    inf.setdefault("num_inference_steps", 25)
    inf.setdefault("checkpoint", "rgb__finetune_1024/epoch-1__checkpoint-5000")

    scene.setdefault("checkpoint", "seethrough3d_release/seethrough3d_release")

    # subjects_data 至少要是列表
    if "subjects_data" not in scene or not isinstance(scene["subjects_data"], list):
        scene["subjects_data"] = []

    # camera_data 基本字段
    if "camera_data" not in scene or not isinstance(scene["camera_data"], dict):
        scene["camera_data"] = {}
    cam = scene["camera_data"]
    cam.setdefault("camera_elevation", 0.2)
    cam.setdefault("lens", 50.0)
    cam.setdefault("global_scale", 1.0)

    # surrounding_prompt 如果缺失，就用原文本整体（调用方再覆盖）
    scene.setdefault("surrounding_prompt", "")

    return scene


def save_scene_pkl(scene: Dict[str, Any], output_path: str) -> None:
    """
    将场景字典保存为 pkl，结构与 example0.pkl 一致。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(scene, f)


def main() -> None:
    """
    用法：
        python shiyan.py "英文场景描述" [输出文件路径]

    如果未提供描述，则从标准输入读取一行。
    输出文件默认：inference/saved_scenes/generated_scene.pkl
    """
    if len(sys.argv) >= 2:
        scene_text = sys.argv[1]
    else:
        scene_text = input("请输入英文场景描述：").strip()

    if not scene_text:
        print("场景描述为空，退出。")
        return

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # 默认保存到仓库中的 inference/saved_scenes 目录
        repo_root = os.path.dirname(__file__)
        output_path = os.path.join(
            repo_root,
            "inference",
            "saved_scenes",
            "example_shiyan1.pkl",
        )

    prompt = build_prompt(scene_text)
    llm_output = call_llm(prompt)
    scene_dict = parse_scene_json(llm_output)

    # 如果 surrounding_prompt 被模型留空，则直接用原始文本
    if not scene_dict.get("surrounding_prompt"):
        scene_dict["surrounding_prompt"] = scene_text

    save_scene_pkl(scene_dict, output_path)
    print(f"场景已生成并保存为：{output_path}")


if __name__ == "__main__":
    main()

