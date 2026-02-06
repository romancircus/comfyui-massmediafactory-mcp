"""
SOTA Model Recommendations

Provides current State-of-the-Art model recommendations for ComfyUI workflows.
Integrates knowledge from sota-tracker to help select optimal models.

Last Updated: February 2026
Cross-reference: ~/Applications/sota-tracker-mcp for live SOTA data
"""

from typing import Optional
from .client import get_client
from .vram import detect_model_type


# Current SOTA models by category (February 2026)
# Keep in sync with sota-tracker MCP
# Updated with RTX 5090 32GB optimization recommendations
SOTA_MODELS = {
    "image_gen": {
        "sota": [
            {
                "name": "FLUX.2-dev",
                "comfyui_files": ["flux2-dev.safetensors", "flux2-dev-fp8.safetensors"],
                "vram_fp16": 24,
                "vram_fp8": 12,
                "strengths": [
                    "photorealism",
                    "anatomy",
                    "lighting",
                    "prompt adherence",
                ],
                "best_for": ["portraits", "general purpose", "artistic styles"],
                "prompt_style": "natural_language",  # Descriptive, "The image showcases..."
                "loader": "UNETLoader",
                "guidance_node": "FluxGuidance",
                "optimal_cfg": 3.5,
                "optimal_steps": 20,
                "sampler": "euler",
                "scheduler": "simple",
                "native_res": 1024,
                "max_res": 2048,
                "rtx5090_precision": "fp16",  # 32GB can handle fp16 easily
            },
            {
                "name": "Qwen-Image-2512",
                "comfyui_files": [
                    "qwen_image_2512_fp16.safetensors",
                    "qwen_image_2512_fp8_e4m3fn.safetensors",
                ],
                "vram_fp16": 14,
                "vram_fp8": 7,
                "strengths": [
                    "text rendering",
                    "complex layouts",
                    "OCR-heavy tasks",
                    "UI design",
                ],
                "best_for": [
                    "posters",
                    "ui design",
                    "charts",
                    "text in images",
                    "logos",
                ],
                "prompt_style": "instructional",  # "Place text X at top left"
                "loader": "UNETLoader",
                "guidance_node": "ModelSamplingAuraFlow",
                "optimal_shift": 3.1,
                "optimal_cfg": 3.0,
                "optimal_steps": 25,
                "sampler": "euler",
                "scheduler": "simple",
                "native_res": 1296,
                "max_res": 2512,
                "rtx5090_precision": "fp16",  # 32GB can handle fp16 easily
            },
            {
                "name": "Z-Image-Turbo",
                "comfyui_files": ["z-image-turbo.safetensors"],
                "vram_fp16": 12,
                "vram_fp8": 6,
                "strengths": ["fast", "good quality", "efficient"],
                "best_for": ["rapid iteration", "drafts", "batch generation"],
                "prompt_style": "natural_language",
                "optimal_steps": 8,
                "rtx5090_precision": "fp16",
            },
        ],
        "deprecated": ["FLUX.1-dev", "SD 1.5", "SD 2.0", "SDXL Base"],
    },
    "video_gen": {
        "sota": [
            {
                "name": "LTX-2 19B",
                "comfyui_files": [
                    "ltx-2-19b-dev-fp8.safetensors",
                    "ltx-2-19b.safetensors",
                ],
                "vram_fp16": 38,  # Too large for RTX 5090!
                "vram_fp8": 19,
                "strengths": [
                    "native audio",
                    "lip sync",
                    "long duration",
                    "narrative coherence",
                ],
                "best_for": [
                    "talking head",
                    "music videos",
                    "dialogue scenes",
                    "audio-reactive",
                ],
                "prompt_style": "motion",  # Describe change, not static state
                "audio_support": True,
                "lip_sync": True,
                "scheduler": "LTXVScheduler",
                "max_shift": 2.05,
                "base_shift": 0.95,
                "optimal_cfg": 3.0,
                "optimal_steps": 20,
                "max_frames": 241,  # ~10 seconds at 24fps
                "native_res": "1280x720",
                "rtx5090_precision": "fp8",  # MUST use fp8 - fp16 causes OOM
                "rtx5090_note": "fp16 requires 38GB - use fp8 on RTX 5090",
            },
            {
                "name": "Wan 2.6",
                "comfyui_files": ["wan2.6_i2v_fp8.safetensors", "wan2.6.safetensors"],
                "vram_fp16": 32,  # Too tight for RTX 5090
                "vram_fp8": 16,
                "strengths": [
                    "motion quality",
                    "high-dynamic motion",
                    "physics simulation",
                    "cinematography",
                ],
                "best_for": [
                    "image-to-video",
                    "action sequences",
                    "b-roll",
                    "cinematic shots",
                ],
                "prompt_style": "motion",  # Describe change, not static state
                "audio_support": False,
                "motion_brush": True,
                "scheduler": "WanVideoScheduler",
                "optimal_cfg": 5.0,
                "optimal_steps": 30,
                "max_frames": 121,  # ~5 seconds at 24fps
                "native_res": "848x480",
                "rtx5090_precision": "fp8",  # Recommended for headroom
            },
            {
                "name": "HunyuanVideo 1.5",
                "comfyui_files": [
                    "hunyuan-video-1.5.safetensors",
                    "hunyuan-video-1.5-fp8.safetensors",
                ],
                "vram_fp16": 24,
                "vram_fp8": 12,
                "strengths": ["high resolution", "cinematic quality", "stable motion"],
                "best_for": ["high-quality shorts", "cinematic content"],
                "prompt_style": "motion",
                "optimal_cfg": 6.0,
                "optimal_steps": 30,
                "max_frames": 129,
                "rtx5090_precision": "fp16",  # Can fit in 32GB
            },
        ],
        "deprecated": ["SVD", "AnimateDiff v2", "Wan 2.1", "LTX-1"],
    },
    "controlnet": {
        "sota": [
            {
                "name": "Flux.2 Union ControlNet",
                "comfyui_files": ["Flux2UnionControlNet.safetensors"],
                "vram": 3.0,
                "best_for": ["multi-mode control", "edge + depth + pose combined"],
                "modes": {"canny": 0, "depth": 1, "pose": 2, "normal": 3},
                "note": "Replaces individual Canny/Depth models for Flux.2",
            },
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
    "ipadapter": {
        "sota": [
            {
                "name": "IP-Adapter FaceID Plus v2",
                "comfyui_files": [
                    "ip-adapter-faceid-plusv2_sd15.bin",
                    "ip-adapter-faceid-plusv2_sdxl.bin",
                ],
                "vram": 3.0,
                "best_for": [
                    "character consistency",
                    "face preservation",
                    "style transfer",
                ],
                "note": "Standard for character consistency across generations",
            },
            {
                "name": "IP-Adapter Flux",
                "comfyui_files": ["ip-adapter.bin"],
                "clip_vision": "google/siglip-so400m-patch14-384",
                "vram": 3.5,
                "best_for": ["style transfer", "reference-based generation"],
                "optimal_weight": 0.35,
            },
        ],
    },
}

# Model selection matrix based on task and constraints
# Updated for RTX 5090 32GB with January 2026 SOTA models
TASK_MODEL_MATRIX = {
    # === IMAGE GENERATION TASKS ===
    "portrait": {
        "recommended": "FLUX.2-dev",
        "reason": "Superior photorealism, anatomy, and lighting for faces",
        "alternatives": ["Qwen-Image-2512"],
        "precision": "fp16",
        "prompt_tip": "Use natural language: 'A close-up photograph of..., natural lighting'",
    },
    "text_in_image": {
        "recommended": "Qwen-Image-2512",
        "reason": "Best text rendering - can handle layout instructions",
        "alternatives": ["FLUX.2-dev"],
        "precision": "fp16",
        "prompt_tip": "Use instructional: 'Place text X at top center, bold font, red color'",
    },
    "poster_design": {
        "recommended": "Qwen-Image-2512",
        "reason": "Handles complex layouts and text positioning",
        "alternatives": ["FLUX.2-dev"],
        "precision": "fp16",
        "prompt_tip": "Be explicit about layout: 'Design a poster. Title at top, image in center...'",
    },
    "ui_design": {
        "recommended": "Qwen-Image-2512",
        "reason": "OCR-heavy tasks and clean layouts",
        "alternatives": [],
        "precision": "fp16",
        "prompt_tip": "Describe UI elements with positions and sizes",
    },
    "fast_iteration": {
        "recommended": "Z-Image-Turbo",
        "reason": "Fastest generation (8 steps) with good quality",
        "alternatives": ["FLUX.2-dev (with fewer steps)"],
        "precision": "fp16",
    },
    "artistic_style": {
        "recommended": "FLUX.2-dev",
        "reason": "Excellent prompt adherence for artistic styles",
        "alternatives": ["Qwen-Image-2512"],
        "precision": "fp16",
        "prompt_tip": "Describe style naturally: 'in the style of...', 'oil painting', etc.",
    },
    # === VIDEO GENERATION TASKS ===
    "talking_head": {
        "recommended": "LTX-2 19B",
        "reason": "Native audio input with lip sync support",
        "alternatives": ["Wan 2.6"],
        "precision": "fp8",  # MUST use fp8 on RTX 5090
        "prompt_tip": "Describe motion, not appearance: 'the person speaks, lips move naturally'",
        "requires_audio": True,
    },
    "music_video": {
        "recommended": "LTX-2 19B",
        "reason": "Audio-reactive generation syncs with music",
        "alternatives": ["Wan 2.6"],
        "precision": "fp8",
        "requires_audio": True,
    },
    "dialogue_scene": {
        "recommended": "LTX-2 19B",
        "reason": "Long coherence (10s) with audio sync",
        "alternatives": [],
        "precision": "fp8",
        "requires_audio": True,
    },
    "image_to_video": {
        "recommended": "Wan 2.6",
        "reason": "Best motion quality and physics for I2V",
        "alternatives": ["LTX-2 19B", "HunyuanVideo 1.5"],
        "precision": "fp8",
        "prompt_tip": "Describe the CHANGE: 'camera pans left, subject walks forward'",
    },
    "action_sequence": {
        "recommended": "Wan 2.6",
        "reason": "High-dynamic motion and physics simulation",
        "alternatives": ["LTX-2 19B"],
        "precision": "fp8",
        "prompt_tip": "Focus on movement: 'rapid acceleration, dust kicks up, camera follows'",
    },
    "cinematic_video": {
        "recommended": "HunyuanVideo 1.5",
        "reason": "Highest quality cinematic output",
        "alternatives": ["LTX-2 19B"],
        "precision": "fp16",  # HunyuanVideo fits in fp16
    },
    "b_roll": {
        "recommended": "Wan 2.6",
        "reason": "Smooth motion for background footage",
        "alternatives": ["LTX-2 19B"],
        "precision": "fp8",
    },
    # === SPECIALIZED TASKS ===
    "style_transfer": {
        "recommended": "Qwen-Image-2512 + ControlNet",
        "reason": "Best structure preservation with style change",
        "alternatives": ["FLUX.2-dev + IP-Adapter"],
        "precision": "fp16",
        "controlnet": "qwen_image_canny_diffsynth_controlnet.safetensors",
    },
    "character_consistency": {
        "recommended": "FLUX.2-dev + IP-Adapter FaceID",
        "reason": "Face preservation across generations",
        "alternatives": ["Qwen-Image-2512 + ControlNet"],
        "precision": "fp16",
        "ipadapter": "ip-adapter-faceid-plusv2",
    },
    "reference_style": {
        "recommended": "FLUX.2-dev + IP-Adapter Flux",
        "reason": "Style transfer without color bleed",
        "alternatives": [],
        "precision": "fp16",
        "optimal_weight": 0.35,
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
    fits = (precision == "fp8" and vram_fp8 <= available_vram_gb) or (
        precision == "fp16" and vram_fp16 <= available_vram_gb
    )

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
            "cfg": 4.0,  # Official Lightricks recommendation for I2V motion
            "steps": 25,
            "sampler": "res_2s",  # Official Lightricks I2V workflow sampler
            "scheduler": "LTXVScheduler",
            "max_shift": 2.05,
            "base_shift": 0.95,
            "strength": 0.6,  # I2V strength (identity preservation)
            "notes": "Use LTXVScheduler + res_2s sampler for I2V. CFG 4.0 for motion, 3.0 for T2V. Native audio supported.",
        },
        "ltx-2": {
            "cfg": 4.0,  # Official Lightricks recommendation for I2V motion
            "steps": 25,
            "sampler": "res_2s",  # Official Lightricks I2V workflow sampler
            "scheduler": "LTXVScheduler",
            "max_shift": 2.05,
            "base_shift": 0.95,
            "strength": 0.6,  # I2V strength (identity preservation)
            "notes": "LTX-2 19B with native audio. Use res_2s sampler + CFG 4.0 for I2V, CFG 3.0 for T2V.",
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
        installed_checkpoints = set(checkpoints_result["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0])
    except (KeyError, IndexError, TypeError):
        pass

    try:
        installed_unets = set(unets_result["UNETLoader"]["input"]["required"]["unet_name"][0])
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
                installed_sota.append(
                    {
                        "name": model["name"],
                        "category": category,
                        "file": found_file,
                    }
                )
            else:
                missing_sota.append(
                    {
                        "name": model["name"],
                        "category": category,
                        "expected_files": model.get("comfyui_files", []),
                    }
                )

    return {
        "installed_sota": installed_sota,
        "missing_sota": missing_sota,
        "total_checkpoints": len(installed_checkpoints),
        "total_unets": len(installed_unets),
    }
