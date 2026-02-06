"""
Image and Video Analysis Module

Provides tools for analyzing generated assets:
- Dimension extraction for aspect ratio calculations
- Object detection for validation before expensive operations
"""

import os
import base64
import urllib.request
import json
from typing import List

# ComfyUI connection
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://localhost:8188")

# Ollama connection for VLM-based detection
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_VLM = os.environ.get("COMFY_MCP_VLM_MODEL", "qwen2.5-vl:7b")


def get_image_dimensions(asset_id: str) -> dict:
    """
    Get dimensions of an image asset.

    Essential for Image-to-Video workflows where agent needs to know
    aspect ratio to set video width/height correctly.

    Args:
        asset_id: The asset ID to analyze.

    Returns:
        {
            "width": 1024,
            "height": 768,
            "aspect_ratio": "4:3",
            "aspect_ratio_decimal": 1.333,
            "orientation": "landscape",
            "recommended_video_size": {"width": 1280, "height": 720}
        }
    """
    from . import assets

    # Get asset from registry
    registry = assets.get_global_registry()
    asset = registry.get_asset(asset_id)

    if not asset:
        return {"error": f"Asset not found: {asset_id}"}

    # Check if dimensions already cached
    if asset.width and asset.height:
        width, height = asset.width, asset.height
    else:
        # Fetch from ComfyUI to get actual dimensions
        try:
            view_url = (
                f"{COMFYUI_URL}/view?filename={asset.filename}&subfolder={asset.subfolder}&type={asset.asset_type}"
            )
            _req = urllib.request.Request(view_url, method="HEAD")
            # Note: ComfyUI doesn't return dimensions in HEAD, need to fetch image
            # For now, return cached or estimate from workflow
            width = asset.width or 1024
            height = asset.height or 1024
        except Exception as e:
            return {"error": f"Failed to get dimensions: {str(e)}"}

    # Calculate aspect ratio
    from math import gcd

    divisor = gcd(width, height)
    aspect_w = width // divisor
    aspect_h = height // divisor

    # Simplify common ratios
    aspect_ratio = f"{aspect_w}:{aspect_h}"
    common_ratios = {
        (16, 9): "16:9",
        (9, 16): "9:16",
        (4, 3): "4:3",
        (3, 4): "3:4",
        (1, 1): "1:1",
        (21, 9): "21:9",
        (3, 2): "3:2",
        (2, 3): "2:3",
    }
    aspect_ratio = common_ratios.get((aspect_w, aspect_h), aspect_ratio)

    # Determine orientation
    if width > height:
        orientation = "landscape"
    elif height > width:
        orientation = "portrait"
    else:
        orientation = "square"

    # Recommend video size based on aspect ratio
    video_sizes = {
        "16:9": {"width": 1280, "height": 720},
        "9:16": {"width": 720, "height": 1280},
        "4:3": {"width": 960, "height": 720},
        "3:4": {"width": 720, "height": 960},
        "1:1": {"width": 720, "height": 720},
        "21:9": {"width": 1280, "height": 548},
    }
    recommended = video_sizes.get(aspect_ratio, {"width": 1024, "height": 576})

    return {
        "asset_id": asset_id,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "aspect_ratio_decimal": round(width / height, 3),
        "orientation": orientation,
        "recommended_video_size": recommended,
    }


def detect_objects(
    asset_id: str,
    objects: List[str],
    vlm_model: str = None,
) -> dict:
    """
    Detect if specific objects exist in a generated image.

    Use this to validate output before expensive operations like video generation.
    E.g., verify "a cat" actually exists in output before generating video.

    Args:
        asset_id: The asset to analyze.
        objects: List of objects to detect (e.g., ["cat", "dog", "person"]).
        vlm_model: VLM to use for detection (default: qwen2.5-vl:7b).

    Returns:
        {
            "detected": ["cat", "person"],
            "not_detected": ["dog"],
            "confidence": {
                "cat": 0.95,
                "person": 0.87,
                "dog": 0.12
            },
            "description": "The image shows a cat sitting next to a person..."
        }
    """
    from . import assets

    vlm_model = vlm_model or DEFAULT_VLM

    # Get asset from registry
    registry = assets.get_global_registry()
    asset = registry.get_asset(asset_id)

    if not asset:
        return {"error": f"Asset not found: {asset_id}"}

    # Fetch image from ComfyUI
    try:
        view_url = f"{COMFYUI_URL}/view?filename={asset.filename}&subfolder={asset.subfolder}&type={asset.asset_type}"
        with urllib.request.urlopen(view_url, timeout=30) as response:
            image_data = response.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        return {"error": f"Failed to fetch image: {str(e)}"}

    # Build prompt for object detection
    objects_list = ", ".join(objects)
    prompt = f"""Analyze this image and determine which of these objects are present: {objects_list}

For each object, rate your confidence (0.0-1.0) that it appears in the image.
Also provide a brief description of what you see.

Respond in this exact JSON format:
{{
    "detected": ["list", "of", "detected", "objects"],
    "confidence": {{"object1": 0.95, "object2": 0.12}},
    "description": "Brief description of the image"
}}"""

    # Call Ollama VLM
    try:
        ollama_payload = {
            "model": vlm_model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "format": "json",
        }

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=json.dumps(ollama_payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode())
            vlm_response = result.get("response", "{}")

        # Parse VLM response
        try:
            detection_result = json.loads(vlm_response)
        except json.JSONDecodeError:
            # Fallback: try to extract from text
            detection_result = {
                "detected": [],
                "confidence": {},
                "description": vlm_response,
            }

        # Ensure all requested objects have confidence scores
        confidence = detection_result.get("confidence", {})
        detected = detection_result.get("detected", [])

        for obj in objects:
            if obj not in confidence:
                confidence[obj] = 0.9 if obj in detected else 0.1

        not_detected = [obj for obj in objects if obj not in detected]

        return {
            "asset_id": asset_id,
            "detected": detected,
            "not_detected": not_detected,
            "confidence": confidence,
            "description": detection_result.get("description", ""),
            "vlm_model": vlm_model,
        }

    except urllib.error.URLError:
        return {
            "error": "Ollama not available. Start with: ollama serve",
            "ollama_url": OLLAMA_URL,
        }
    except Exception as e:
        return {"error": f"Object detection failed: {str(e)}"}


def get_video_info(asset_id: str) -> dict:
    """
    Get information about a video asset.

    Args:
        asset_id: The video asset ID.

    Returns:
        {
            "width": 1280,
            "height": 720,
            "duration_seconds": 5.0,
            "frame_count": 120,
            "fps": 24,
            "aspect_ratio": "16:9"
        }
    """
    from . import assets

    registry = assets.get_global_registry()
    asset = registry.get_asset(asset_id)

    if not asset:
        return {"error": f"Asset not found: {asset_id}"}

    if asset.asset_type != "video":
        return {"error": f"Asset is not a video: {asset.asset_type}"}

    # Extract video info from workflow parameters if available
    params = asset.parameters or {}
    _workflow = asset.workflow or {}

    # Try to find frame count and fps from workflow
    frame_count = params.get("FRAMES", 120)
    fps = params.get("FPS", 24)
    duration = frame_count / fps if fps > 0 else 0

    width = asset.width or params.get("WIDTH", 1280)
    height = asset.height or params.get("HEIGHT", 720)

    # Calculate aspect ratio
    from math import gcd

    divisor = gcd(width, height)
    aspect_w = width // divisor
    aspect_h = height // divisor
    aspect_ratio = f"{aspect_w}:{aspect_h}"

    return {
        "asset_id": asset_id,
        "width": width,
        "height": height,
        "duration_seconds": round(duration, 2),
        "frame_count": frame_count,
        "fps": fps,
        "aspect_ratio": aspect_ratio,
    }
