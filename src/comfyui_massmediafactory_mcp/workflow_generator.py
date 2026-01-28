"""
Workflow Generator - Meta-Template System

Generates complete, validated ComfyUI workflow JSONs from high-level parameters.
Uses pre-validated skeleton templates with parameter injection and auto-correction.
"""

import json
import random
import copy
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from . import topology_validator
from . import reference_docs

# Module paths
MODULE_DIR = Path(__file__).parent
SKELETONS_DIR = MODULE_DIR.parent.parent / "docs" / "library" / "skeletons"

# Model to skeleton mapping
MODEL_SKELETON_MAP = {
    ("ltx", "t2v"): "ltx_video_t2v.json",
    ("ltx", "txt2vid"): "ltx_video_t2v.json",
    ("ltx", "text-to-video"): "ltx_video_t2v.json",
    ("ltx", "i2v"): "ltx_video_i2v.json",
    ("ltx", "img2vid"): "ltx_video_i2v.json",
    ("ltx", "image-to-video"): "ltx_video_i2v.json",
    ("ltx2", "t2v"): "ltx_video_t2v.json",
    ("ltx2", "i2v"): "ltx_video_i2v.json",
    ("flux", "t2i"): "flux_dev_t2i.json",
    ("flux", "txt2img"): "flux_dev_t2i.json",
    ("flux", "text-to-image"): "flux_dev_t2i.json",
    ("flux2", "t2i"): "flux_dev_t2i.json",
    ("wan", "t2v"): "wan_t2v.json",
    ("wan", "txt2vid"): "wan_t2v.json",
    ("wan26", "t2v"): "wan_t2v.json",
    ("qwen", "t2i"): "qwen_t2i.json",
    ("qwen", "txt2img"): "qwen_t2i.json",
}

# Default parameters by model
MODEL_DEFAULTS = {
    "ltx": {
        "width": 768,
        "height": 512,
        "frames": 97,
        "steps": 30,
        "cfg": 3.0,
    },
    "ltx2": {
        "width": 768,
        "height": 512,
        "frames": 97,
        "steps": 30,
        "cfg": 3.0,
    },
    "flux": {
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "guidance": 3.5,
    },
    "flux2": {
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "guidance": 3.5,
    },
    "wan": {
        "width": 832,
        "height": 480,
        "frames": 81,
        "steps": 30,
        "cfg": 5.0,
    },
    "wan26": {
        "width": 832,
        "height": 480,
        "frames": 81,
        "steps": 30,
        "cfg": 5.0,
    },
    "qwen": {
        "width": 1296,
        "height": 1296,
        "steps": 25,
        "cfg": 3.0,
    },
}


def load_skeleton(model: str, workflow_type: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Load a skeleton template for the given model and workflow type.

    Returns:
        (skeleton_dict, skeleton_name) or (None, error_message)
    """
    key = (model.lower(), workflow_type.lower())
    skeleton_name = MODEL_SKELETON_MAP.get(key)

    if not skeleton_name:
        available = list(set(MODEL_SKELETON_MAP.values()))
        return None, f"No skeleton for model={model}, type={workflow_type}. Available: {available}"

    skeleton_path = SKELETONS_DIR / skeleton_name
    if not skeleton_path.exists():
        return None, f"Skeleton file not found: {skeleton_path}"

    try:
        skeleton = json.loads(skeleton_path.read_text())
        return skeleton, skeleton_name
    except json.JSONDecodeError as e:
        return None, f"Invalid skeleton JSON: {e}"


def resolve_parameters(
    skeleton: dict,
    model: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = None,
    height: int = None,
    frames: int = None,
    seed: int = None,
    steps: int = None,
    cfg: float = None,
    guidance: float = None,
    **extra_params
) -> Dict[str, Any]:
    """
    Resolve final parameters from user input, skeleton defaults, and model defaults.

    Order of precedence:
    1. User-provided parameters
    2. Skeleton defaults
    3. Model defaults
    """
    # Get model defaults
    defaults = MODEL_DEFAULTS.get(model.lower(), {})

    # Get skeleton defaults
    skeleton_defaults = skeleton.get("defaults", {})

    # Build parameter dict
    params = {
        "PROMPT": prompt,
        "NEGATIVE": negative_prompt or "blurry, low quality, distorted",
    }

    # Seed
    if seed is not None and seed != -1:
        params["SEED"] = seed
    else:
        params["SEED"] = random.randint(0, 2**32 - 1)

    # Resolution
    params["WIDTH"] = width or defaults.get("width", 1024)
    params["HEIGHT"] = height or defaults.get("height", 1024)

    # Frames (for video)
    if frames is not None:
        params["FRAMES"] = frames
        params["LENGTH"] = frames
    elif "frames" in defaults:
        params["FRAMES"] = defaults["frames"]
        params["LENGTH"] = defaults["frames"]

    # Steps
    params["STEPS"] = steps or defaults.get("steps", 20)

    # CFG / Guidance
    if cfg is not None:
        params["CFG"] = cfg
    elif "cfg" in defaults:
        params["CFG"] = defaults["cfg"]

    if guidance is not None:
        params["GUIDANCE"] = guidance
    elif "guidance" in defaults:
        params["GUIDANCE"] = defaults["guidance"]

    # Add any extra params
    for key, value in extra_params.items():
        if value is not None:
            params[key.upper()] = value

    return params


def expand_skeleton_to_workflow(skeleton: dict, params: Dict[str, Any]) -> dict:
    """
    Convert skeleton format to ComfyUI API format with parameter injection.

    Skeleton format:
    {
        "nodes": [{"id": "1", "type": "UNETLoader", ...}],
        "connections": [{"from": "1.MODEL", "to": "2.model"}],
        "defaults": {"1.unet_name": "model.safetensors"}
    }

    API format:
    {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "..."}}
    }
    """
    workflow = {}
    nodes = skeleton.get("nodes", [])
    connections = skeleton.get("connections", [])
    defaults = skeleton.get("defaults", {})

    # Build connection lookup: "to" -> ("from_node", from_slot)
    connection_map = {}
    for conn in connections:
        from_part = conn.get("from", "")
        to_part = conn.get("to", "")

        # Parse from: "1.MODEL" -> node_id="1", output="MODEL"
        if "." in from_part:
            from_node, from_output = from_part.split(".", 1)
        else:
            from_node, from_output = from_part, "0"

        # Parse to: "2.model" -> node_id="2", input="model"
        if "." in to_part:
            to_node, to_input = to_part.split(".", 1)
        else:
            to_node, to_input = to_part, "input"

        # Determine output slot index
        output_slot = 0
        # Common output mappings
        output_mappings = {
            "MODEL": 0, "CLIP": 1, "VAE": 2,
            "CONDITIONING": 0, "CONDITIONING+": 0, "CONDITIONING-": 1,
            "LATENT": 0, "LATENT_DENOISED": 1,
            "IMAGE": 0, "MASK": 1,
            "SIGMAS": 0, "SAMPLER": 0,
        }
        if from_output.upper() in output_mappings:
            output_slot = output_mappings[from_output.upper()]
        elif from_output.isdigit():
            output_slot = int(from_output)

        connection_map[(to_node, to_input.lower())] = (from_node, output_slot)

    # Build workflow nodes
    for node in nodes:
        node_id = str(node.get("id"))
        class_type = node.get("type")

        inputs = {}

        # Add connections as inputs
        for (target_node, target_input), (source_node, source_slot) in connection_map.items():
            if target_node == node_id:
                inputs[target_input] = [str(source_node), source_slot]

        # Add defaults for this node
        for key, value in defaults.items():
            if key.startswith(f"{node_id}."):
                input_name = key.split(".", 1)[1]
                inputs[input_name] = value

        workflow[node_id] = {
            "class_type": class_type,
            "inputs": inputs
        }

    # Inject parameters (replace {{PLACEHOLDER}} values)
    workflow_str = json.dumps(workflow)
    for param_name, param_value in params.items():
        placeholder = f"{{{{{param_name}}}}}"
        if isinstance(param_value, str):
            workflow_str = workflow_str.replace(f'"{placeholder}"', json.dumps(param_value))
            workflow_str = workflow_str.replace(placeholder, param_value)
        else:
            workflow_str = workflow_str.replace(f'"{placeholder}"', str(param_value))
            workflow_str = workflow_str.replace(placeholder, str(param_value))

    return json.loads(workflow_str)


def auto_correct_params(params: Dict[str, Any], model: str) -> Tuple[Dict[str, Any], List[dict]]:
    """
    Auto-correct parameters based on model constraints.

    Returns:
        (corrected_params, list of corrections)
    """
    corrections = []
    corrected = copy.deepcopy(params)

    constraints = topology_validator.MODEL_CONSTRAINTS.get(model.lower(), {})
    divisor = constraints.get("resolution_divisor", 8)
    min_res = constraints.get("resolution_min", 256)
    max_res = constraints.get("resolution_max", 2048)

    # Correct resolution
    for dim in ["WIDTH", "HEIGHT"]:
        if dim in corrected:
            value = corrected[dim]
            if value % divisor != 0:
                new_value = (value // divisor) * divisor
                new_value = max(min_res, min(max_res, new_value))
                corrections.append({
                    "field": dim,
                    "from": value,
                    "to": new_value,
                    "reason": f"Rounded to nearest {divisor}"
                })
                corrected[dim] = new_value

    # Correct LTX frames
    if model.lower() in ["ltx", "ltx2"]:
        for dim in ["FRAMES", "LENGTH"]:
            if dim in corrected:
                value = corrected[dim]
                if (value - 1) % 8 != 0:
                    new_value = ((value - 1) // 8) * 8 + 1
                    if new_value < 9:
                        new_value = 9
                    corrections.append({
                        "field": dim,
                        "from": value,
                        "to": new_value,
                        "reason": "LTX requires 8n+1 frames"
                    })
                    corrected[dim] = new_value

    # Correct CFG
    if "CFG" in corrected:
        cfg_range = constraints.get("cfg_range")
        if cfg_range:
            min_cfg, max_cfg = cfg_range
            value = corrected["CFG"]
            if value < min_cfg or value > max_cfg:
                new_value = constraints.get("cfg_default", min_cfg)
                corrections.append({
                    "field": "CFG",
                    "from": value,
                    "to": new_value,
                    "reason": f"CFG clamped to model range [{min_cfg}, {max_cfg}]"
                })
                corrected["CFG"] = new_value

    return corrected, corrections


def generate_workflow(
    model: str,
    workflow_type: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = None,
    height: int = None,
    frames: int = None,
    seed: int = None,
    steps: int = None,
    cfg: float = None,
    guidance: float = None,
    auto_correct: bool = True,
    validate: bool = True,
    **extra_params
) -> dict:
    """
    Generate a complete, validated ComfyUI workflow from high-level parameters.

    Args:
        model: Model identifier ("ltx", "flux", "wan", "qwen")
        workflow_type: Type ("t2v", "i2v", "t2i")
        prompt: Generation prompt
        negative_prompt: Negative prompt (optional)
        width: Width in pixels (optional, uses model default)
        height: Height in pixels (optional, uses model default)
        frames: Frame count for video (optional)
        seed: Random seed (optional, generates random if not provided)
        steps: Sampling steps (optional)
        cfg: CFG scale (optional)
        guidance: FluxGuidance value for FLUX (optional)
        auto_correct: Auto-correct invalid parameters (default True)
        validate: Validate before returning (default True)
        **extra_params: Additional model-specific parameters

    Returns:
        {
            "workflow": {...},
            "parameters_used": {...},
            "auto_corrections": [...],
            "validation": {...},
            "skeleton_used": "..."
        }
    """
    # Load skeleton
    skeleton, skeleton_name = load_skeleton(model, workflow_type)
    if skeleton is None:
        return {"error": skeleton_name}

    # Resolve parameters
    params = resolve_parameters(
        skeleton, model, prompt, negative_prompt,
        width, height, frames, seed, steps, cfg, guidance,
        **extra_params
    )

    # Auto-correct if enabled
    corrections = []
    if auto_correct:
        params, corrections = auto_correct_params(params, model)

    # Expand skeleton to workflow
    workflow = expand_skeleton_to_workflow(skeleton, params)

    # Validate if enabled
    validation_result = {"valid": True, "errors": [], "warnings": []}
    if validate:
        validation_result = topology_validator.validate_topology(workflow, model)

        # If still invalid after auto-correct, return error
        if not validation_result.get("valid", True) and not auto_correct:
            return {
                "error": "Workflow validation failed",
                "validation": validation_result,
                "parameters_used": params,
                "skeleton_used": skeleton_name
            }

    return {
        "workflow": workflow,
        "parameters_used": params,
        "auto_corrections": corrections,
        "validation": validation_result,
        "skeleton_used": skeleton_name,
        "model": model,
        "workflow_type": workflow_type
    }


def list_supported_workflows() -> dict:
    """
    List all supported model/workflow_type combinations.

    Returns:
        {"workflows": [...], "count": N}
    """
    workflows = []
    seen = set()

    for (model, wf_type), skeleton_name in MODEL_SKELETON_MAP.items():
        key = f"{model}:{wf_type}"
        if key not in seen:
            seen.add(key)
            workflows.append({
                "model": model,
                "workflow_type": wf_type,
                "skeleton": skeleton_name,
                "defaults": MODEL_DEFAULTS.get(model, {})
            })

    return {"workflows": workflows, "count": len(workflows)}
