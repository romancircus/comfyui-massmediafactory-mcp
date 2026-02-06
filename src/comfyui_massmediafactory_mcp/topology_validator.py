"""
Workflow Topology Validator

Validates LLM-generated workflow JSON against PARAMETER_RULES constraints.
Catches common mistakes before execution.

NOTE: Validation constraints are now centralized in model_registry.py.
This module builds validation-specific views from that central registry.
"""

import json
from typing import Dict, List, Tuple, Any, Optional

from .model_registry import (
    MODEL_CONSTRAINTS as REGISTRY_CONSTRAINTS,
)


def _build_validation_constraints() -> Dict[str, Dict[str, Any]]:
    """Build validation-friendly constraints from model_registry.MODEL_CONSTRAINTS."""
    result = {}

    for model_key, pattern in REGISTRY_CONSTRAINTS.items():
        cfg = pattern.get("cfg", {})
        resolution = pattern.get("resolution", {})
        frames = pattern.get("frames", {})
        required = pattern.get("required_nodes", {})
        forbidden = pattern.get("forbidden_nodes", {})

        constraint = {
            "resolution_divisor": resolution.get("divisible_by", 8),
            "resolution_min": resolution.get("min", 256),
            "resolution_max": max(resolution.get("max", [2048, 2048]))
            if isinstance(resolution.get("max"), list)
            else resolution.get("max", 2048),
            "cfg_range": (cfg.get("min", 1.0), cfg.get("max", 15.0)) if cfg.get("min") else None,
            "cfg_default": cfg.get("default"),
            "requires_flux_guidance": cfg.get("via") == "FluxGuidance",
            "frame_formula": frames.get("formula") if frames else None,
            "sampler_type": required.get("sampler", "KSampler") if isinstance(required, dict) else "KSampler",
            "required_nodes": _flatten_required_nodes(required) if isinstance(required, dict) else required,
            "forbidden_nodes": list(forbidden.keys()) if isinstance(forbidden, dict) else forbidden,
            "scheduler_node": required.get("scheduler") if isinstance(required, dict) else None,
        }
        result[model_key] = constraint

        # Create aliases (ltx -> ltx2, flux -> flux2, wan -> wan26)
        base_name = model_key.rstrip("0123456789_")
        if base_name != model_key and base_name not in result:
            result[base_name] = constraint

    # Add sd15 model (not in registry as it's deprecated)
    if "sd15" not in result:
        result["sd15"] = {
            "resolution_divisor": 8,
            "resolution_min": 256,
            "resolution_max": 1024,
            "cfg_range": (5.0, 15.0),
            "cfg_default": 7.5,
            "sampler_type": "KSampler",
            "required_nodes": [],
            "forbidden_nodes": [],
        }

    return result


def _flatten_required_nodes(required: Dict[str, Any]) -> List[str]:
    """Flatten required_nodes dict to list of node class names."""
    nodes = []
    for value in required.values():
        if isinstance(value, list):
            nodes.extend(value)
        elif isinstance(value, str):
            nodes.append(value)
    return nodes


# Build validation constraints from model_registry (single source of truth)
MODEL_CONSTRAINTS = _build_validation_constraints()

# Video-unsafe samplers
VIDEO_UNSAFE_SAMPLERS = ["euler_ancestral", "dpmpp_2m_sde", "dpmpp_sde"]

# Type compatibility matrix - maps output types to compatible input names
TYPE_COMPATIBILITY = {
    "MODEL": ["model", "unet"],
    "CLIP": ["clip"],
    "VAE": ["vae"],
    "CONDITIONING": ["positive", "negative", "conditioning"],
    "LATENT": ["latent_image", "samples", "latent"],
    "IMAGE": ["image", "images", "pixels"],
    "SIGMAS": ["sigmas"],
    "SAMPLER": ["sampler"],
    "MASK": ["mask"],
    "NOISE": ["noise"],
    "GUIDER": ["guider"],
    "WANMODEL": ["wan_model"],
    "IMAGEEMBEDS": ["image_embeds"],
    "GEMMA_MODEL": ["gemma_model", "clip"],
}

# Node output types - maps node class to their output slot types
NODE_OUTPUT_TYPES = {
    # Loaders
    "CheckpointLoaderSimple": ["MODEL", "CLIP", "VAE"],
    "UNETLoader": ["MODEL"],
    "CLIPLoader": ["CLIP"],
    "DualCLIPLoader": ["CLIP"],
    "VAELoader": ["VAE"],
    "LoraLoader": ["MODEL", "CLIP"],
    "LoraLoaderModelOnly": ["MODEL"],
    "LTXVLoader": ["MODEL", "CLIP", "VAE"],
    "HunyuanVideoModelLoader": ["MODEL", "VAE"],
    "WanVideoModelLoader": ["WANVIDEOMODEL"],
    "WanVideoVAELoader": ["WANVAE"],
    "LoadWanVideoT5TextEncoder": ["WANTEXTENCODER"],
    "LoadWanVideoClipTextEncoder": ["CLIP_VISION"],
    "WanVideoTextEncode": ["WANVIDEOTEXTEMBEDS"],
    "WanVideoClipVisionEncode": ["WANVIDIMAGE_CLIPEMBEDS"],
    "WanVideoImageToVideoEncode": ["WANVIDIMAGE_EMBEDS"],
    "WanVideoEmptyEmbeds": ["WANVIDIMAGE_EMBEDS"],
    "LTXVGemmaCLIPModelLoader": ["GEMMA_MODEL"],
    # Encoding
    "CLIPTextEncode": ["CONDITIONING"],
    "FluxGuidance": ["CONDITIONING"],
    "LTXVConditioning": ["CONDITIONING", "CONDITIONING"],
    "LTXVGemmaEnhancePrompt": ["STRING"],
    # Latent
    "EmptyLatentImage": ["LATENT"],
    "EmptySD3LatentImage": ["LATENT"],
    "EmptyLTXVLatentVideo": ["LATENT"],
    "EmptyHunyuanLatentVideo": ["LATENT"],
    "EmptyWanLatentVideo": ["LATENT"],
    "LTXVImgToVideo": ["CONDITIONING", "CONDITIONING", "LATENT"],
    # Samplers
    "KSampler": ["LATENT"],
    "SamplerCustom": ["LATENT", "LATENT"],
    "SamplerCustomAdvanced": ["LATENT", "LATENT"],
    "HunyuanVideoSampler": ["LATENT"],
    "WanVideoSampler": ["LATENT", "LATENT"],
    "KSamplerSelect": ["SAMPLER"],
    "BasicScheduler": ["SIGMAS"],
    "LTXVScheduler": ["SIGMAS"],
    "RandomNoise": ["NOISE"],
    "BasicGuider": ["GUIDER"],
    # Decode/Encode
    "VAEDecode": ["IMAGE"],
    "VAEEncode": ["LATENT"],
    "HunyuanVideoVAEDecode": ["IMAGE"],
    "WanVideoDecode": ["IMAGE"],
    "HunyuanVideoImageEncode": ["IMAGE_EMBEDS"],
    # Image
    "LoadImage": ["IMAGE", "MASK"],
    "ImageScale": ["IMAGE"],
    "LTXVPreprocess": ["IMAGE"],
    # Output
    "SaveImage": [],
    "SaveAnimatedWEBP": [],
    "CreateVideo": ["VIDEO"],
    "SaveVideo": [],
}


def validate_connection_types(workflow: dict) -> List[str]:
    """
    Validate that all connections in the workflow are type-compatible.

    Checks that output types match expected input types.

    Args:
        workflow: The workflow JSON

    Returns:
        List of error messages (empty if all connections valid)
    """
    errors = []

    for node_id, node_data in workflow.items():
        if node_id.startswith("_"):
            continue
        if not isinstance(node_data, dict) or "class_type" not in node_data:
            continue

        _class_type = node_data.get("class_type")
        inputs = node_data.get("inputs", {})

        for input_name, input_value in inputs.items():
            # Check if this is a connection (list of [node_id, slot])
            if isinstance(input_value, list) and len(input_value) == 2:
                source_node_id, source_slot = input_value
                source_node_id = str(source_node_id)

                # Find source node
                source_node = workflow.get(source_node_id)
                if not source_node:
                    errors.append(f"Node {node_id}: Input '{input_name}' references non-existent node {source_node_id}")
                    continue

                source_class = source_node.get("class_type")
                if not source_class:
                    continue

                # Get output types for source node
                output_types = NODE_OUTPUT_TYPES.get(source_class, [])
                if not output_types:
                    # Unknown node type, skip validation
                    continue

                # Check slot index
                if source_slot >= len(output_types):
                    errors.append(
                        f"Node {node_id}: Input '{input_name}' references slot {source_slot} of {source_class}, but it only has {len(output_types)} outputs"
                    )
                    continue

                # Get the output type
                output_type = output_types[source_slot]

                # Check if input name is compatible with output type
                compatible_inputs = TYPE_COMPATIBILITY.get(output_type, [])
                input_name_lower = input_name.lower()

                # Check if input name matches any compatible input
                if compatible_inputs and input_name_lower not in compatible_inputs:
                    # Check if it's a known mismatch (not just an unknown input)
                    _known_input = False
                    for type_name, type_inputs in TYPE_COMPATIBILITY.items():
                        if input_name_lower in type_inputs:
                            _known_input = True
                            if type_name != output_type:
                                errors.append(
                                    f"Node {node_id}: Input '{input_name}' expects {type_name} but receives {output_type} from {source_class}[{source_slot}]"
                                )
                            break

    return errors


def detect_model_type(workflow: dict) -> Optional[str]:
    """
    Detect which model type a workflow is using based on nodes present.

    Args:
        workflow: The workflow JSON

    Returns:
        Model type string or None if unknown
    """
    nodes = set()
    for node_id, node_data in workflow.items():
        if node_id.startswith("_"):
            continue
        if isinstance(node_data, dict) and "class_type" in node_data:
            nodes.add(node_data["class_type"])

    # Check for model-specific nodes
    if "HunyuanVideoModelLoader" in nodes or "HunyuanVideoSampler" in nodes:
        return "hunyuan"
    if "LTXVScheduler" in nodes or "LTXVConditioning" in nodes:
        if "LTXVGemmaEnhancePrompt" in nodes:
            return "ltx_distilled"
        return "ltx"
    if "WanVideoModelLoader" in nodes or "WanVideoDecode" in nodes or "WanVideoSampler" in nodes:
        return "wan"
    if "DualCLIPLoader" in nodes or "FluxGuidance" in nodes:
        return "flux"
    if "ModelSamplingAuraFlow" in nodes:
        return "qwen"
    if "CheckpointLoaderSimple" in nodes:
        # Could be SDXL or SD1.5, default to SDXL
        return "sdxl"

    return None


def validate_resolution(width: int, height: int, model: str) -> Tuple[bool, List[str]]:
    """
    Validate resolution against model constraints.

    Returns:
        (is_valid, list of error messages)
    """
    constraints = MODEL_CONSTRAINTS.get(model, {})
    errors = []

    divisor = constraints.get("resolution_divisor", 8)
    min_res = constraints.get("resolution_min", 256)
    max_res = constraints.get("resolution_max", 2048)

    if width % divisor != 0:
        errors.append(f"Width {width} not divisible by {divisor}. Use {(width // divisor) * divisor}.")
    if height % divisor != 0:
        errors.append(f"Height {height} not divisible by {divisor}. Use {(height // divisor) * divisor}.")
    if width < min_res or width > max_res:
        errors.append(f"Width {width} outside valid range [{min_res}, {max_res}].")
    if height < min_res or height > max_res:
        errors.append(f"Height {height} outside valid range [{min_res}, {max_res}].")

    return len(errors) == 0, errors


def validate_ltx_frames(frames: int) -> Tuple[bool, List[str]]:
    """
    Validate LTX frame count (must be 8n+1).

    Returns:
        (is_valid, list of error messages)
    """
    if frames < 9:
        return False, [f"Frame count {frames} too low. Minimum is 9."]

    if (frames - 1) % 8 != 0:
        # Calculate nearest valid values
        lower = ((frames - 1) // 8) * 8 + 1
        upper = lower + 8
        return False, [f"Frame count {frames} invalid for LTX. Use {lower} or {upper} (8n+1 rule)."]

    return True, []


def validate_cfg(cfg: float, model: str) -> Tuple[bool, List[str]]:
    """
    Validate CFG value against model constraints.

    Returns:
        (is_valid, list of error messages)
    """
    constraints = MODEL_CONSTRAINTS.get(model, {})
    warnings = []

    if constraints.get("requires_flux_guidance"):
        warnings.append("FLUX uses FluxGuidance node instead of cfg parameter.")
        return True, warnings

    cfg_range = constraints.get("cfg_range")
    if cfg_range:
        min_cfg, max_cfg = cfg_range
        if cfg < min_cfg or cfg > max_cfg:
            default = constraints.get("cfg_default", min_cfg)
            warnings.append(f"CFG {cfg} outside recommended range [{min_cfg}, {max_cfg}]. Suggested: {default}.")

    return True, warnings


def validate_sampler(sampler_name: str, model: str, is_video: bool) -> Tuple[bool, List[str]]:
    """
    Validate sampler selection for model and task type.

    Returns:
        (is_valid, list of error messages)
    """
    errors = []

    if is_video and sampler_name in VIDEO_UNSAFE_SAMPLERS:
        errors.append(f"Sampler '{sampler_name}' causes flickering in video. Use 'euler' or 'res_multistep'.")

    return len(errors) == 0, errors


def validate_topology(workflow_json: str, model: str = None) -> dict:
    """
    Validate a workflow JSON against parameter rules.

    This is the main validation function that checks:
    1. Resolution divisibility
    2. Frame count (for LTX)
    3. CFG values
    4. Sampler type (SamplerCustom vs KSampler)
    5. Required nodes present
    6. Forbidden nodes absent
    7. Type-safe connections

    Args:
        workflow_json: The workflow JSON as a string
        model: Optional model type override (auto-detected if not provided)

    Returns:
        {
            "valid": True/False,
            "errors": [...],
            "warnings": [...],
            "suggestions": [...],
            "model_detected": "..."
        }
    """
    errors = []
    warnings = []
    suggestions = []

    # Parse JSON
    try:
        if isinstance(workflow_json, str):
            workflow = json.loads(workflow_json)
        else:
            workflow = workflow_json
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "errors": [f"Invalid JSON: {e}"],
            "warnings": [],
            "suggestions": ["Check JSON syntax"],
            "model_detected": None,
        }

    # Detect model type
    detected_model = detect_model_type(workflow)
    model = model or detected_model

    if not model:
        warnings.append("Could not detect model type. Using generic validation.")
        model = "sdxl"  # Default to SDXL rules

    constraints = MODEL_CONSTRAINTS.get(model, {})

    # Collect node info
    nodes_present = set()
    sampler_nodes = []
    latent_nodes = []
    _conditioning_nodes = []

    for node_id, node_data in workflow.items():
        if node_id.startswith("_"):
            continue
        if not isinstance(node_data, dict) or "class_type" not in node_data:
            continue

        class_type = node_data["class_type"]
        nodes_present.add(class_type)
        inputs = node_data.get("inputs", {})

        # Collect samplers
        if class_type in ["KSampler", "SamplerCustom"]:
            sampler_nodes.append((node_id, class_type, inputs))

        # Collect latent generators
        if "Latent" in class_type or "Empty" in class_type:
            latent_nodes.append((node_id, class_type, inputs))

        # Check resolution in latent nodes
        if "width" in inputs and "height" in inputs:
            try:
                width = int(inputs["width"]) if not isinstance(inputs["width"], list) else None
                height = int(inputs["height"]) if not isinstance(inputs["height"], list) else None
                if width and height:
                    valid, res_errors = validate_resolution(width, height, model)
                    errors.extend(res_errors)
            except (ValueError, TypeError):
                pass

        # Check frame count for LTX
        if class_type == "EmptyLTXVLatentVideo" and "length" in inputs:
            try:
                frames = int(inputs["length"]) if not isinstance(inputs["length"], list) else None
                if frames:
                    valid, frame_errors = validate_ltx_frames(frames)
                    errors.extend(frame_errors)
            except (ValueError, TypeError):
                pass

        # Check CFG in samplers
        if "cfg" in inputs and class_type in ["KSampler", "SamplerCustom"]:
            try:
                cfg = float(inputs["cfg"]) if not isinstance(inputs["cfg"], list) else None
                if cfg:
                    valid, cfg_warnings = validate_cfg(cfg, model)
                    warnings.extend(cfg_warnings)
            except (ValueError, TypeError):
                pass

        # Check sampler_name
        if "sampler_name" in inputs:
            sampler_name = inputs["sampler_name"]
            if not isinstance(sampler_name, list):
                is_video = model in ["ltx", "ltx_distilled", "wan"]
                valid, sampler_errors = validate_sampler(sampler_name, model, is_video)
                errors.extend(sampler_errors)

    # Check required nodes
    for required_node in constraints.get("required_nodes", []):
        if required_node not in nodes_present:
            errors.append(f"Missing required node: {required_node}")

    # Check forbidden nodes
    for forbidden_node in constraints.get("forbidden_nodes", []):
        if forbidden_node in nodes_present:
            replacement = constraints.get("sampler_type", "SamplerCustom")
            errors.append(f"Forbidden node: {forbidden_node}. Use {replacement} instead.")

    # Check sampler type for video models
    expected_sampler = constraints.get("sampler_type")
    if expected_sampler == "SamplerCustom":
        if "KSampler" in nodes_present and "SamplerCustom" not in nodes_present:
            errors.append(f"Using KSampler - should use SamplerCustom for {model}")

    # Check for FluxGuidance if FLUX
    if constraints.get("requires_flux_guidance"):
        if "FluxGuidance" not in nodes_present:
            errors.append("FLUX requires FluxGuidance node for guidance control")

    # Validate connection types
    connection_errors = validate_connection_types(workflow)
    errors.extend(connection_errors)

    # Generate suggestions
    if errors:
        suggestions.append("Review the pattern documentation with get_model_pattern()")
        suggestions.append("Use get_workflow_skeleton_json() for a working template")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "model_detected": detected_model,
        "model_used": model,
        "nodes_found": list(nodes_present),
    }


def auto_correct_parameters(workflow: dict, model: str = None) -> dict:
    """
    Auto-correct invalid parameters in a workflow.

    Returns:
        {
            "workflow": corrected_workflow,
            "corrections": [{"field": "...", "from": X, "to": Y, "reason": "..."}]
        }
    """
    import copy

    corrected = copy.deepcopy(workflow)
    corrections = []

    detected_model = detect_model_type(workflow)
    model = model or detected_model or "sdxl"
    constraints = MODEL_CONSTRAINTS.get(model, {})

    divisor = constraints.get("resolution_divisor", 8)

    for node_id, node_data in corrected.items():
        if node_id.startswith("_"):
            continue
        if not isinstance(node_data, dict) or "class_type" not in node_data:
            continue

        inputs = node_data.get("inputs", {})

        # Correct resolution
        for dim in ["width", "height"]:
            if dim in inputs and not isinstance(inputs[dim], list):
                try:
                    value = int(inputs[dim])
                    if value % divisor != 0:
                        corrected_value = (value // divisor) * divisor
                        if corrected_value < constraints.get("resolution_min", 256):
                            corrected_value = constraints.get("resolution_min", 256)
                        inputs[dim] = corrected_value
                        corrections.append(
                            {
                                "field": f"{node_id}.{dim}",
                                "from": value,
                                "to": corrected_value,
                                "reason": f"Rounded to nearest {divisor}",
                            }
                        )
                except (ValueError, TypeError):
                    pass

        # Correct LTX frames
        if node_data.get("class_type") == "EmptyLTXVLatentVideo" and "length" in inputs:
            if not isinstance(inputs["length"], list):
                try:
                    frames = int(inputs["length"])
                    if (frames - 1) % 8 != 0:
                        corrected_frames = ((frames - 1) // 8) * 8 + 1
                        if corrected_frames < 9:
                            corrected_frames = 9
                        inputs["length"] = corrected_frames
                        corrections.append(
                            {
                                "field": f"{node_id}.length",
                                "from": frames,
                                "to": corrected_frames,
                                "reason": "LTX requires 8n+1 frames",
                            }
                        )
                except (ValueError, TypeError):
                    pass

    return {"workflow": corrected, "corrections": corrections, "model": model}
