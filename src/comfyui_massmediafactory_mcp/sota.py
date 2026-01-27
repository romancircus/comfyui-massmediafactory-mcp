"""
SOTA Model Recommendations

Provides current State-of-the-Art model recommendations for ComfyUI workflows.
Integrates knowledge from sota-tracker to help select optimal models.

Last Updated: January 2026
"""

from typing import Optional, List, Dict
from .client import get_client
from .vram import MODEL_VRAM_ESTIMATES, detect_model_type


# Current SOTA models by category (January 2026)
# Keep in sync with sota-tracker MCP
SOTA_MODELS = {
    "image_gen": {
        "sota": [
            {
                "name": "Qwen-Image-2512",
                "comfyui_files": ["qwen_image_2512_fp8_e4m3fn.safetensors", "qwen_image_2512.safetensors"],
                "vram_fp16": 14,
                "vram_fp8": 7,
                "strengths": ["photorealistic", "text rendering", "faces", "high detail"],
                "best_for": ["portraits", "product photos", "text in images"],
            },
            {
                "name": "FLUX.2-dev",
                "comfyui_files": ["flux2-dev.safetensors", "flux2-dev-fp8.safetensors"],
                "vram_fp16": 24,
                "vram_fp8": 12,
                "strengths": ["text rendering", "coherent compositions", "prompt adherence"],
                "best_for": ["general purpose", "text-heavy images", "logos"],
            },
            {
                "name": "Z-Image-Turbo",
                "comfyui_files": ["z-image-turbo.safetensors"],
                "vram_fp16": 12,
                "vram_fp8": 6,
                "strengths": ["fast", "good quality", "efficient"],
                "best_for": ["rapid iteration", "drafts", "batch generation"],
            },
        ],
        "deprecated": ["FLUX.1-dev", "SD 1.5", "SD 2.0", "SDXL Base"],
    },
    "video_gen": {
        "sota": [
            {
                "name": "LTX-2 19B",
                "comfyui_files": ["ltx-2-19b-dev-fp8.safetensors", "ltx-2-19b.safetensors"],
                "vram_fp16": 20,
                "vram_fp8": 10,
                "strengths": ["native audio", "lip sync", "long duration"],
                "best_for": ["talking head", "music videos", "dialogue scenes"],
                "max_frames": 241,  # ~10 seconds at 24fps
            },
            {
                "name": "Wan 2.6",
                "comfyui_files": ["wan2.6.safetensors", "wan-2.6-i2v.safetensors"],
                "vram_fp16": 16,
                "vram_fp8": 8,
                "strengths": ["motion quality", "temporal consistency", "i2v"],
                "best_for": ["image-to-video", "animation", "smooth motion"],
                "max_frames": 121,
            },
            {
                "name": "HunyuanVideo 1.5",
                "comfyui_files": ["hunyuan-video-1.5.safetensors"],
                "vram_fp16": 24,
                "vram_fp8": 12,
                "strengths": ["high resolution", "cinematic quality"],
                "best_for": ["high-quality shorts", "cinematic content"],
                "max_frames": 129,
            },
        ],
        "deprecated": ["SVD", "AnimateDiff v2", "Wan 2.1"],
    },
    "controlnet": {
        "sota": [
            {
                "name": "Qwen ControlNet (Canny)",
                "comfyui_files": ["qwen_image_canny_diffsynth_controlnet.safetensors"],
                "vram": 2.5,
                "best_for": ["edge-guided generation", "structure preservation"],
            },
            {
                "name": "Qwen ControlNet (Depth)",
                "comfyui_files": ["qwen_image_depth_controlnet.safetensors"],
                "vram": 2.5,
                "best_for": ["depth-guided generation", "3D consistency"],
            },
        ],
    },
}

# Model selection matrix based on task and constraints
TASK_MODEL_MATRIX = {
    "portrait": {
        "recommended": "Qwen-Image-2512",
        "reason": "Best face quality and photorealism",
        "alternatives": ["FLUX.2-dev"],
    },
    "text_in_image": {
        "recommended": "FLUX.2-dev",
        "reason": "Superior text rendering capabilities",
        "alternatives": ["Qwen-Image-2512"],
    },
    "fast_iteration": {
        "recommended": "Z-Image-Turbo",
        "reason": "Fastest generation with good quality",
        "alternatives": ["FLUX.2-dev (with fewer steps)"],
    },
    "talking_head": {
        "recommended": "LTX-2 19B",
        "reason": "Native audio and lip sync support",
        "alternatives": ["Wan 2.6"],
    },
    "image_to_video": {
        "recommended": "Wan 2.6",
        "reason": "Best motion quality for i2v",
        "alternatives": ["LTX-2 19B", "HunyuanVideo 1.5"],
    },
    "cinematic_video": {
        "recommended": "HunyuanVideo 1.5",
        "reason": "Highest quality cinematic output",
        "alternatives": ["LTX-2 19B"],
    },
    "style_transfer": {
        "recommended": "Qwen-Image-2512 + ControlNet",
        "reason": "Best structure preservation with style change",
        "alternatives": ["FLUX.2-dev + IP-Adapter"],
    },
}


def get_sota_for_category(category: str) -> dict:
    """
    Get current SOTA models for a category.

    Args:
        category: One of "image_gen", "video_gen", "controlnet"

    Returns:
        SOTA models with details and deprecated list.
    """
    if category not in SOTA_MODELS:
        return {
            "error": f"Unknown category: {category}",
            "available_categories": list(SOTA_MODELS.keys()),
        }

    return {
        "category": category,
        "sota_models": SOTA_MODELS[category]["sota"],
        "deprecated": SOTA_MODELS[category].get("deprecated", []),
        "last_updated": "January 2026",
    }


def recommend_model_for_task(
    task: str,
    available_vram_gb: Optional[float] = None,
    prefer_speed: bool = False,
) -> dict:
    """
    Recommend the best model for a specific task.

    Args:
        task: The task type (portrait, text_in_image, talking_head, etc.)
        available_vram_gb: Available GPU memory (auto-detected if not provided)
        prefer_speed: If True, prioritize faster models

    Returns:
        Model recommendation with reasoning.
    """
    # Auto-detect VRAM if not provided
    if available_vram_gb is None:
        client = get_client()
        stats = client.get_system_stats()
        for device in stats.get("devices", []):
            if device.get("type") == "cuda":
                available_vram_gb = device.get("vram_free", 0) / (1024**3)
                break

    if available_vram_gb is None:
        available_vram_gb = 32.0  # Default assumption

    # Get task recommendation
    if task not in TASK_MODEL_MATRIX:
        return {
            "error": f"Unknown task: {task}",
            "available_tasks": list(TASK_MODEL_MATRIX.keys()),
        }

    task_info = TASK_MODEL_MATRIX[task]
    recommended_name = task_info["recommended"]

    # Find the model details
    recommended_model = None
    category = "image_gen" if "video" not in task.lower() else "video_gen"

    for model in SOTA_MODELS.get(category, {}).get("sota", []):
        if model["name"] == recommended_name or recommended_name in model["name"]:
            recommended_model = model
            break

    if recommended_model is None:
        return {
            "task": task,
            "recommended": recommended_name,
            "reason": task_info["reason"],
            "alternatives": task_info["alternatives"],
        }

    # Check VRAM fit
    vram_fp8 = recommended_model.get("vram_fp8", recommended_model.get("vram", 8))
    vram_fp16 = recommended_model.get("vram_fp16", recommended_model.get("vram", 16))

    precision = "fp8" if vram_fp8 <= available_vram_gb else "fp16"
    fits = (precision == "fp8" and vram_fp8 <= available_vram_gb) or \
           (precision == "fp16" and vram_fp16 <= available_vram_gb)

    return {
        "task": task,
        "recommended": recommended_name,
        "reason": task_info["reason"],
        "model_details": recommended_model,
        "suggested_precision": precision,
        "estimated_vram_gb": vram_fp8 if precision == "fp8" else vram_fp16,
        "available_vram_gb": round(available_vram_gb, 1),
        "will_fit": fits,
        "alternatives": task_info["alternatives"],
        "comfyui_files": recommended_model.get("comfyui_files", []),
    }


def check_model_is_sota(model_name: str) -> dict:
    """
    Check if a model is current SOTA or deprecated.

    Args:
        model_name: The model filename or name.

    Returns:
        Status (current/deprecated/unknown) with replacement suggestions.
    """
    name_lower = model_name.lower()

    # Check against deprecated list
    for category, info in SOTA_MODELS.items():
        for deprecated in info.get("deprecated", []):
            if deprecated.lower() in name_lower or name_lower in deprecated.lower():
                # Find replacement
                replacement = info["sota"][0] if info["sota"] else None
                return {
                    "model": model_name,
                    "status": "DEPRECATED",
                    "category": category,
                    "message": f"{deprecated} is outdated",
                    "replacement": replacement["name"] if replacement else None,
                    "replacement_files": replacement.get("comfyui_files", []) if replacement else [],
                }

    # Check against SOTA list
    for category, info in SOTA_MODELS.items():
        for sota_model in info.get("sota", []):
            for file in sota_model.get("comfyui_files", []):
                if file.lower() in name_lower or name_lower in file.lower():
                    return {
                        "model": model_name,
                        "status": "CURRENT",
                        "category": category,
                        "message": f"{sota_model['name']} is current SOTA",
                        "details": sota_model,
                    }
            if sota_model["name"].lower() in name_lower:
                return {
                    "model": model_name,
                    "status": "CURRENT",
                    "category": category,
                    "message": f"{sota_model['name']} is current SOTA",
                    "details": sota_model,
                }

    return {
        "model": model_name,
        "status": "UNKNOWN",
        "message": "Model not in SOTA database - may be fine-tune or community model",
    }


def get_optimal_settings(model_name: str) -> dict:
    """
    Get optimal ComfyUI settings for a model.

    Args:
        model_name: The model name or filename.

    Returns:
        Recommended settings (CFG, steps, sampler, scheduler).
    """
    model_type, precision = detect_model_type(model_name)

    # Default settings by model type
    OPTIMAL_SETTINGS = {
        "flux": {
            "cfg": 3.5,
            "steps": 20,
            "sampler": "euler",
            "scheduler": "simple",
            "guidance_node": "FluxGuidance",
            "notes": "Use FluxGuidance instead of CFG. Low guidance (3-4) works best.",
        },
        "flux-schnell": {
            "cfg": 3.5,
            "steps": 4,
            "sampler": "euler",
            "scheduler": "simple",
            "guidance_node": "FluxGuidance",
            "notes": "Schnell is optimized for 4 steps. Fast but slightly lower quality.",
        },
        "sdxl": {
            "cfg": 7.0,
            "steps": 30,
            "sampler": "dpmpp_2m_sde",
            "scheduler": "karras",
            "notes": "CFG 5-7 for realistic, 7-12 for stylized. Karras scheduler recommended.",
        },
        "qwen-image": {
            "cfg": 3.0,
            "steps": 25,
            "sampler": "euler",
            "scheduler": "simple",
            "shift": 3.1,
            "notes": "Use ModelSamplingAuraFlow with shift=3.1. Low CFG like Flux.",
        },
        "ltx": {
            "cfg": 3.0,
            "steps": 20,
            "sampler": "euler",
            "scheduler": "LTXVScheduler",
            "max_shift": 2.05,
            "base_shift": 0.95,
            "notes": "Use LTXVScheduler for best results. Native audio supported.",
        },
        "wan": {
            "cfg": 5.0,
            "steps": 30,
            "sampler": "euler",
            "scheduler": "normal",
            "notes": "Moderate CFG. Good for image-to-video with motion prompts.",
        },
        "hunyuan": {
            "cfg": 6.0,
            "steps": 30,
            "sampler": "euler",
            "scheduler": "normal",
            "notes": "Higher CFG for more prompt adherence. High quality but slower.",
        },
    }

    if model_type in OPTIMAL_SETTINGS:
        settings = OPTIMAL_SETTINGS[model_type].copy()
        settings["model_type"] = model_type
        settings["detected_precision"] = precision
        return settings

    # Generic defaults
    return {
        "model_type": model_type,
        "detected_precision": precision,
        "cfg": 7.0,
        "steps": 30,
        "sampler": "euler",
        "scheduler": "normal",
        "notes": "Using generic defaults. Check model documentation for optimal settings.",
    }


def get_available_sota_models() -> dict:
    """
    Check which SOTA models are installed in ComfyUI.

    Returns:
        Installed vs missing SOTA models.
    """
    client = get_client()

    # Get installed models
    checkpoints_result = client.get_object_info("CheckpointLoaderSimple")
    unets_result = client.get_object_info("UNETLoader")

    installed_checkpoints = set()
    installed_unets = set()

    try:
        installed_checkpoints = set(
            checkpoints_result["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        )
    except (KeyError, IndexError, TypeError):
        pass

    try:
        installed_unets = set(
            unets_result["UNETLoader"]["input"]["required"]["unet_name"][0]
        )
    except (KeyError, IndexError, TypeError):
        pass

    all_installed = installed_checkpoints | installed_unets

    # Check SOTA models
    installed_sota = []
    missing_sota = []

    for category, info in SOTA_MODELS.items():
        for model in info.get("sota", []):
            found = False
            found_file = None

            for file in model.get("comfyui_files", []):
                if file in all_installed:
                    found = True
                    found_file = file
                    break

            if found:
                installed_sota.append({
                    "name": model["name"],
                    "category": category,
                    "file": found_file,
                })
            else:
                missing_sota.append({
                    "name": model["name"],
                    "category": category,
                    "expected_files": model.get("comfyui_files", []),
                })

    return {
        "installed_sota": installed_sota,
        "missing_sota": missing_sota,
        "total_checkpoints": len(installed_checkpoints),
        "total_unets": len(installed_unets),
    }
