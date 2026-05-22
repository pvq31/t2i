#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate one image through an OpenAI-compatible image endpoint."""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_HOST = "api.chatanywhere.tech"
DEFAULT_API_KEY = "REPLACE_WITH_API_KEY"
DEFAULT_OUTPUT = Path("methods/dalle3_sofa_pet_corner.png")
DEFAULT_MODEL = "gpt-image-1-mini"
DEFAULT_FALLBACK_MODELS = ["gpt-image-1-mini"]
DEFAULT_PROMPT = (
    "A sofa is in the center of the image. A cat is front-right of the sofa, "
    "and a dog is front-left of the sofa. A lamp is located to the left of the sofa, "
    "and a bookshelf is located to the right of the sofa. The scene is a quiet and "
    "comfortable pet resting corner, soft warm indoor lighting, lazy and relaxed home "
    "atmosphere, naturally arranged objects, rich details, and a warm realistic composition."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with an OpenAI-compatible image model.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt for image generation.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output image path.")
    parser.add_argument("--host", default=os.getenv("OPENAI_BASE_HOST", DEFAULT_HOST), help="API host.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY), help="API key.")
    parser.add_argument("--model", default=os.getenv("IMAGE_MODEL", DEFAULT_MODEL), help="Primary image model.")
    parser.add_argument(
        "--fallback-models",
        default=os.getenv("IMAGE_FALLBACK_MODELS", ",".join(DEFAULT_FALLBACK_MODELS)),
        help="Comma-separated fallback models used when the primary model is unavailable.",
    )
    parser.add_argument("--size", default="1024x1024", help="Image size, e.g. 1024x1024.")
    parser.add_argument("--quality", default="standard", choices=("standard", "hd"), help="DALL-E 3 quality.")
    return parser.parse_args()


class ImageAPIError(RuntimeError):
    def __init__(self, status: int, data: Dict[str, Any]) -> None:
        self.status = status
        self.data = data
        super().__init__(f"Image API failed with HTTP {status}: {json.dumps(data, ensure_ascii=False)}")


def request_image(args: argparse.Namespace, model: str) -> Dict[str, Any]:
    payload_dict = {
        "model": model,
        "prompt": args.prompt,
        "n": 1,
        "size": args.size,
    }
    if model == "dall-e-3-hd":
        payload_dict["quality"] = args.quality
    payload = json.dumps(payload_dict)
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    conn = http.client.HTTPSConnection(args.host, timeout=120)
    try:
        conn.request("POST", "/v1/images/generations", payload, headers)
        response = conn.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
    finally:
        conn.close()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API returned non-JSON response: {raw[:500]}") from exc

    if response.status >= 400:
        raise ImageAPIError(response.status, data)
    return data


def model_unavailable(exc: ImageAPIError) -> bool:
    error = exc.data.get("error")
    if not isinstance(error, dict):
        return False
    code = str(error.get("code", "")).lower()
    param = str(error.get("param", "")).lower()
    message = str(error.get("message", "")).lower()
    return param == "model" or code in {"invalid_value", "model_not_found"} or "model" in message and "does not exist" in message


def model_candidates(primary: str, fallbacks: str) -> List[str]:
    candidates: List[str] = []
    for model in [primary, *fallbacks.split(",")]:
        model = model.strip()
        if model and model not in candidates:
            candidates.append(model)
    return candidates


def request_image_with_fallback(args: argparse.Namespace) -> tuple[Dict[str, Any], str]:
    last_error: ImageAPIError | None = None
    for model in model_candidates(args.model, args.fallback_models):
        try:
            return request_image(args, model), model
        except ImageAPIError as exc:
            last_error = exc
            if not model_unavailable(exc):
                raise
            print(f"Model {model!r} is unavailable on {args.host}; trying fallback.", file=sys.stderr)
    if last_error is not None:
        raise last_error
    raise RuntimeError("No image model candidates were provided.")


def save_image(result: Dict[str, Any], output_path: Path) -> None:
    images = result.get("data")
    if not isinstance(images, list) or not images:
        raise RuntimeError(f"Image API response has no data[0]: {json.dumps(result, ensure_ascii=False)}")

    item = images[0]
    if not isinstance(item, dict):
        raise RuntimeError(f"Unexpected image item: {item!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if item.get("b64_json"):
        output_path.write_bytes(base64.b64decode(str(item["b64_json"])))
        return

    image_url = str(item.get("url") or "").strip()
    if not image_url:
        raise RuntimeError(f"Image API response has neither url nor b64_json: {json.dumps(item, ensure_ascii=False)}")
    with urllib.request.urlopen(image_url, timeout=120) as response:
        output_path.write_bytes(response.read())


def main() -> None:
    args = parse_args()
    if not args.api_key.strip():
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    output_path = Path(args.output)
    result, used_model = request_image_with_fallback(args)
    save_image(result, output_path)
    print(f"Saved image to: {output_path}")
    print(f"Used model: {used_model}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
