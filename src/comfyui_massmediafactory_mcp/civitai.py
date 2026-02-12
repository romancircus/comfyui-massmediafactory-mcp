"""
CivitAI Workflow Discovery & Import

Discovers popular ComfyUI workflows from CivitAI's API and converts them
to the MassMediaFactory template format with {{PLACEHOLDER}} injection.

API docs: https://developer.civitai.com/docs/api/public-rest

Usage (CLI):
    mmf search-workflow "flux portrait"
    mmf import-workflow <civitai_url> --name my_template

Usage (Python):
    from .civitai import search_workflows, import_workflow
    results = search_workflows("flux portrait", limit=10)
    template = import_workflow(workflow_json, name="flux_portrait_custom")
"""

import json
import logging
import re
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Optional

logger = logging.getLogger("comfyui-mcp")

CIVITAI_API_BASE = "https://civitai.com/api/v1"

# Allowed domains for workflow fetching (SSRF prevention)
_ALLOWED_FETCH_DOMAINS = {
    "civitai.com",
    "www.civitai.com",
    "image.civitai.com",
    "raw.githubusercontent.com",
    "huggingface.co",
}

# Node input names that should become placeholders
_PLACEHOLDER_INPUTS = {
    # Prompts
    "text": "PROMPT",
    "text_positive": "PROMPT",
    "text_negative": "NEGATIVE_PROMPT",
    # Seeds
    "seed": "SEED",
    "noise_seed": "SEED",
    # Dimensions
    "width": "WIDTH",
    "height": "HEIGHT",
    # Frames
    "length": "FRAMES",
    "frame_length": "FRAMES",
    "num_frames": "FRAMES",
    # Sampling
    "steps": "STEPS",
    "cfg": "CFG",
    "denoise": "DENOISE",
    # Files
    "image": "IMAGE_PATH",
    "filename_prefix": "OUTPUT_PREFIX",
}

# Node classes that are text encoders (their "text" input is a prompt)
_TEXT_ENCODER_NODES = {
    "CLIPTextEncode",
    "CLIPTextEncodeSDXL",
    "WanVideoTextEncode",
    "LTXVGemmaEnhancePrompt",
    "TextEncodeQwenImageEdit",
    "TextEncodeQwenImageEditPlus",
}

# Node classes where "text" is NOT a prompt
_NON_PROMPT_TEXT_NODES = {
    "ShowText",
    "DisplayText",
    "StringConstant",
    "PrimitiveNode",
}


def search_workflows(
    query: str,
    limit: int = 10,
    sort: str = "Most Reactions",
    period: str = "Month",
) -> dict:
    """
    Search CivitAI for ComfyUI workflows.

    Args:
        query: Search query (e.g., "flux portrait", "wan video").
        limit: Max results to return.
        sort: Sort order — "Most Reactions", "Newest", "Most Comments".
        period: Time period — "Day", "Week", "Month", "Year", "AllTime".

    Returns:
        Dict with workflow metadata (not full workflows — those need import_workflow_from_url).
    """
    params = {
        "limit": min(limit, 20),
        "sort": sort,
        "period": period,
        "types": "Workflows",
        "query": query,
    }
    url = f"{CIVITAI_API_BASE}/images?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "MassMediaFactory/1.0")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        return {"error": f"CivitAI API error: {e}", "query": query}

    items = data.get("items", [])
    results = []
    for item in items:
        meta = item.get("meta", {}) or {}
        results.append(
            {
                "id": item.get("id"),
                "url": item.get("url"),
                "width": item.get("width"),
                "height": item.get("height"),
                "prompt": meta.get("prompt", "")[:200],
                "model": meta.get("Model", "unknown"),
                "sampler": meta.get("sampler", ""),
                "steps": meta.get("steps"),
                "cfg": meta.get("cfgScale"),
                "stats": item.get("stats", {}),
            }
        )

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "metadata": {
            "cursor": data.get("metadata", {}).get("nextCursor"),
        },
    }


def fetch_workflow_from_url(url: str) -> dict:
    """
    Fetch a ComfyUI workflow JSON from a CivitAI URL.

    Supports:
    - Direct JSON URLs (from allowed domains only)
    - CivitAI image/post pages (extracts workflow from metadata)

    Args:
        url: CivitAI URL or direct JSON URL.

    Returns:
        Raw workflow JSON dict, or error dict.
    """
    # Validate URL domain (SSRF prevention)
    domain_error = _validate_fetch_url(url)
    if domain_error:
        return {"error": domain_error}

    # Extract image ID from CivitAI URL patterns
    image_id = _extract_civitai_image_id(url)
    if image_id:
        return _fetch_workflow_by_image_id(image_id)

    # Try direct JSON fetch
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "MassMediaFactory/1.0")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            if isinstance(data, dict) and any(
                isinstance(v, dict) and "class_type" in v for v in data.values() if isinstance(v, dict)
            ):
                return {"workflow": data}
            return {"error": "URL does not contain a valid ComfyUI workflow"}
    except Exception as e:
        return {"error": f"Failed to fetch workflow: {e}"}


def convert_to_template(
    workflow: dict,
    name: str,
    model: str = "unknown",
    task: str = "unknown",
    description: str = "",
) -> dict:
    """
    Convert a raw ComfyUI workflow to MassMediaFactory template format.

    Replaces known parameter values with {{PLACEHOLDER}} tokens and adds _meta.

    Args:
        workflow: Raw ComfyUI API-format workflow JSON.
        name: Template name (e.g., "flux_portrait_custom").
        model: Model identifier (e.g., "flux", "wan", "ltx").
        task: Task type (e.g., "txt2img", "img2vid").
        description: Human-readable description.

    Returns:
        Template dict with _meta and placeholders injected.
    """
    import copy

    template = copy.deepcopy(workflow)
    placeholders_injected = []

    # Detect model type from nodes if not specified
    if model == "unknown":
        model = _detect_model_from_workflow(workflow)
    if task == "unknown":
        task = _detect_task_from_workflow(workflow)

    # Inject placeholders
    for node_id, node_data in template.items():
        if node_id.startswith("_") or not isinstance(node_data, dict):
            continue
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})

        for input_name, value in list(inputs.items()):
            # Skip connection references
            if isinstance(value, list):
                continue

            placeholder = _get_placeholder(class_type, input_name, value)
            if placeholder:
                inputs[input_name] = f"{{{{{placeholder}}}}}"
                placeholders_injected.append(
                    {
                        "node": node_id,
                        "class_type": class_type,
                        "input": input_name,
                        "placeholder": placeholder,
                        "original_value": value,
                    }
                )

    # Add _meta
    template["_meta"] = {
        "name": name,
        "model": model,
        "task": task,
        "description": description or f"Imported from CivitAI: {name}",
        "source": "civitai",
        "version": "1.0.0",
        "placeholders": {p["placeholder"]: p["original_value"] for p in placeholders_injected},
    }

    return {
        "template": template,
        "placeholders": placeholders_injected,
        "model_detected": model,
        "task_detected": task,
    }


# =============================================================================
# Internal Helpers
# =============================================================================


def _validate_fetch_url(url: str) -> Optional[str]:
    """Validate URL is from an allowed domain. Returns error string or None."""
    if len(url) > 2048:
        return "URL too long (max 2048 characters)"
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return "Invalid URL format"
    if parsed.scheme not in ("http", "https"):
        return f"Unsupported URL scheme: {parsed.scheme}"
    hostname = parsed.hostname
    if not hostname:
        return "URL has no hostname"
    hostname = hostname.lower()
    if hostname not in _ALLOWED_FETCH_DOMAINS:
        return f"Domain '{hostname}' not allowed. Allowed: {', '.join(sorted(_ALLOWED_FETCH_DOMAINS))}"
    return None


def _extract_civitai_image_id(url: str) -> Optional[str]:
    """Extract image ID from a CivitAI URL."""
    if len(url) > 2048:
        return None
    # Match patterns like civitai.com/images/12345 (bounded digit capture)
    match = re.search(r"civitai\.com/images/(\d{1,20})", url)
    if match:
        return match.group(1)
    return None


def _fetch_workflow_by_image_id(image_id: str) -> dict:
    """Fetch workflow metadata for a CivitAI image."""
    url = f"{CIVITAI_API_BASE}/images?imageId={image_id}"
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "MassMediaFactory/1.0")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        items = data.get("items", [])
        if not items:
            return {"error": f"Image {image_id} not found on CivitAI"}

        meta = items[0].get("meta", {}) or {}
        # CivitAI stores the ComfyUI workflow in meta.comfyWorkflow (if available)
        workflow = meta.get("comfyWorkflow")
        if workflow:
            if isinstance(workflow, str):
                workflow = json.loads(workflow)
            return {"workflow": workflow, "source": f"civitai:image:{image_id}"}

        return {
            "error": "No ComfyUI workflow attached to this image",
            "meta_available": list(meta.keys()),
        }
    except Exception as e:
        return {"error": f"Failed to fetch from CivitAI: {e}"}


def _get_placeholder(class_type: str, input_name: str, value: Any) -> Optional[str]:
    """Determine the placeholder name for a given input, or None if it shouldn't be parameterized."""
    input_lower = input_name.lower()

    # Text inputs on text encoder nodes -> PROMPT / NEGATIVE_PROMPT
    if input_lower == "text" and class_type in _TEXT_ENCODER_NODES:
        return "PROMPT"
    if input_lower == "text" and class_type in _NON_PROMPT_TEXT_NODES:
        return None

    # Direct mapping
    if input_lower in _PLACEHOLDER_INPUTS:
        return _PLACEHOLDER_INPUTS[input_lower]

    # Negative prompt detection by input name
    if "negative" in input_lower and isinstance(value, str) and len(value) > 5:
        return "NEGATIVE_PROMPT"

    return None


def _detect_model_from_workflow(workflow: dict) -> str:
    """Detect model type from node classes in workflow."""
    nodes = set()
    for node_id, node_data in workflow.items():
        if isinstance(node_data, dict) and "class_type" in node_data:
            nodes.add(node_data["class_type"])

    if "WanVideoModelLoader" in nodes or "WanVideoSampler" in nodes:
        return "wan"
    if "LTXVScheduler" in nodes or "LTXVConditioning" in nodes:
        return "ltx"
    if "HunyuanVideoModelLoader" in nodes:
        return "hunyuan"
    if "FluxGuidance" in nodes or "DualCLIPLoader" in nodes:
        return "flux"
    if "ModelSamplingAuraFlow" in nodes:
        return "qwen"
    if "CheckpointLoaderSimple" in nodes:
        return "sdxl"
    return "unknown"


def _detect_task_from_workflow(workflow: dict) -> str:
    """Detect task type from node classes in workflow."""
    nodes = set()
    for node_data in workflow.values():
        if isinstance(node_data, dict) and "class_type" in node_data:
            nodes.add(node_data["class_type"])

    has_video_output = any(n in nodes for n in ("SaveVideo", "VHS_VideoCombine", "CreateVideo"))
    has_image_input = "LoadImage" in nodes

    if has_video_output:
        if has_image_input:
            return "img2vid"
        return "txt2vid"
    if has_image_input:
        return "img2img"
    return "txt2img"
