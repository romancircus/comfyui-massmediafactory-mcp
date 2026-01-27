"""
VRAM Estimation

Estimate GPU memory usage for workflows before execution.
Helps prevent OOM errors and optimize model selection.
"""

import re
from typing import Optional, Tuple
from .client import get_client


# Known model VRAM requirements (in GB) at different precisions
# These are approximate and based on inference, not training
# Updated January 2026 with RTX 5090 32GB optimization recommendations
MODEL_VRAM_ESTIMATES = {
    # ==========================================================================
    # IMAGE MODELS
    # ==========================================================================
    # RTX 5090 32GB Strategy: Use fp16 for all image models (quality > speed)

    "flux": {
        "fp32": 48.0,
        "fp16": 24.0,
        "bf16": 24.0,
        "fp8": 12.0,
        "default": 24.0,
        "rtx5090_recommended": "fp16",
        "rtx5090_note": "32GB easily handles fp16 with 8GB headroom",
    },
    "flux2": {
        "fp32": 48.0,
        "fp16": 24.0,
        "bf16": 24.0,
        "fp8": 12.0,
        "default": 24.0,
        "rtx5090_recommended": "fp16",
        "rtx5090_note": "Use fp16 for best quality; fp8 only for speed iteration",
    },
    "flux-schnell": {
        "fp32": 48.0,
        "fp16": 24.0,
        "bf16": 24.0,
        "fp8": 12.0,
        "default": 24.0,
        "rtx5090_recommended": "fp16",
    },
    "sdxl": {
        "fp32": 14.0,
        "fp16": 7.0,
        "bf16": 7.0,
        "fp8": 4.0,
        "default": 7.0,
        "rtx5090_recommended": "fp16",
        "rtx5090_note": "Deprecated - use FLUX.2 or Qwen instead",
    },
    "sd15": {
        "fp32": 8.0,
        "fp16": 4.0,
        "bf16": 4.0,
        "fp8": 2.5,
        "default": 4.0,
        "rtx5090_recommended": "fp16",
        "rtx5090_note": "Deprecated - use FLUX.2 or Qwen instead",
    },
    "sd3": {
        "fp32": 24.0,
        "fp16": 12.0,
        "bf16": 12.0,
        "fp8": 6.0,
        "default": 12.0,
        "rtx5090_recommended": "fp16",
    },
    "qwen-image": {
        "fp32": 28.0,
        "fp16": 14.0,
        "bf16": 14.0,
        "fp8": 7.0,
        "default": 14.0,
        "rtx5090_recommended": "fp16",
        "rtx5090_note": "32GB handles fp16 easily; best for text rendering",
    },
    "z-image-turbo": {
        "fp32": 24.0,
        "fp16": 12.0,
        "bf16": 12.0,
        "fp8": 6.0,
        "default": 12.0,
        "rtx5090_recommended": "fp16",
        "rtx5090_note": "Fast iteration model - 8 steps",
    },

    # ==========================================================================
    # VIDEO MODELS
    # ==========================================================================
    # RTX 5090 32GB Strategy: Use fp8 for large video models (19B+)

    "ltx": {
        "fp32": 40.0,
        "fp16": 20.0,
        "bf16": 20.0,
        "fp8": 10.0,
        "default": 20.0,
        "rtx5090_recommended": "fp8",
    },
    "ltx-2": {
        "fp32": 76.0,  # 19B parameters
        "fp16": 38.0,  # TOO LARGE for RTX 5090!
        "bf16": 38.0,
        "fp8": 19.0,
        "default": 19.0,  # Default to fp8
        "rtx5090_recommended": "fp8",
        "rtx5090_note": "CRITICAL: fp16 requires 38GB - MUST use fp8 on RTX 5090",
        "rtx5090_warning": "OOM_IF_FP16",
    },
    "ltx-2-19b": {
        "fp32": 76.0,
        "fp16": 38.0,  # OOM on RTX 5090
        "bf16": 38.0,
        "fp8": 19.0,
        "default": 19.0,
        "rtx5090_recommended": "fp8",
        "rtx5090_note": "19B model - fp8 mandatory on 32GB cards",
        "rtx5090_warning": "OOM_IF_FP16",
    },
    "wan": {
        "fp32": 32.0,
        "fp16": 16.0,
        "bf16": 16.0,
        "fp8": 8.0,
        "default": 16.0,
        "rtx5090_recommended": "fp8",
        "rtx5090_note": "Use fp8 for headroom with frames/latents",
    },
    "wan2": {
        "fp32": 32.0,
        "fp16": 16.0,
        "bf16": 16.0,
        "fp8": 8.0,
        "default": 16.0,
        "rtx5090_recommended": "fp8",
    },
    "wan2.6": {
        "fp32": 64.0,
        "fp16": 32.0,  # Too tight on RTX 5090
        "bf16": 32.0,
        "fp8": 16.0,
        "default": 16.0,
        "rtx5090_recommended": "fp8",
        "rtx5090_note": "fp16 uses full 32GB - use fp8 for safety margin",
    },
    "hunyuan": {
        "fp32": 48.0,
        "fp16": 24.0,
        "bf16": 24.0,
        "fp8": 12.0,
        "default": 24.0,
        "rtx5090_recommended": "fp16",
        "rtx5090_note": "Can use fp16 on 32GB with 8GB headroom",
    },
    "hunyuan-video": {
        "fp32": 48.0,
        "fp16": 24.0,
        "bf16": 24.0,
        "fp8": 12.0,
        "default": 24.0,
        "rtx5090_recommended": "fp16",
    },
    "svd": {
        "fp32": 16.0,
        "fp16": 8.0,
        "bf16": 8.0,
        "fp8": 4.0,
        "default": 8.0,
        "rtx5090_recommended": "fp16",
        "rtx5090_note": "Deprecated - use Wan 2.6 or LTX-2 instead",
    },

    # Fallback
    "unknown": {
        "fp32": 16.0,
        "fp16": 8.0,
        "bf16": 8.0,
        "fp8": 4.0,
        "default": 8.0,
        "rtx5090_recommended": "fp16",
    },
}

# RTX 5090 Specific Constants
RTX_5090_VRAM_GB = 32.0
RTX_5090_SAFE_MARGIN_GB = 4.0  # Leave 4GB for system/latents
RTX_5090_USABLE_GB = RTX_5090_VRAM_GB - RTX_5090_SAFE_MARGIN_GB  # 28GB usable

# Additional VRAM for components
COMPONENT_VRAM = {
    "vae": 1.5,  # GB for VAE
    "clip": 2.0,  # GB for CLIP text encoder
    "t5": 8.0,   # GB for T5 text encoder (Flux, SD3)
    "controlnet": 2.5,  # GB per ControlNet
    "lora": 0.5,  # GB per LoRA (approximate)
    "ipadapter": 3.0,  # GB for IP-Adapter
}

# Resolution scaling factors (relative to 512x512)
def get_resolution_multiplier(width: int, height: int) -> float:
    """Get VRAM multiplier based on resolution."""
    base_pixels = 512 * 512
    actual_pixels = width * height
    # VRAM scales roughly linearly with pixel count for latents
    return max(1.0, (actual_pixels / base_pixels) ** 0.5)


def detect_model_type(model_name: str) -> Tuple[str, str]:
    """
    Detect model type and precision from filename.

    Returns:
        (model_type, precision)
    """
    name_lower = model_name.lower()

    # Detect precision
    precision = "default"
    if "fp32" in name_lower:
        precision = "fp32"
    elif "fp16" in name_lower:
        precision = "fp16"
    elif "bf16" in name_lower:
        precision = "bf16"
    elif "fp8" in name_lower or "e4m3" in name_lower or "e5m2" in name_lower:
        precision = "fp8"
    elif "gguf" in name_lower or "q4" in name_lower or "q8" in name_lower:
        precision = "fp8"  # Quantized models similar to fp8

    # Detect model type - order matters (more specific first)
    if "flux2" in name_lower or "flux-2" in name_lower or "flux.2" in name_lower:
        return "flux2", precision
    elif "flux" in name_lower:
        if "schnell" in name_lower:
            return "flux-schnell", precision
        return "flux", precision
    elif "sdxl" in name_lower or "sd_xl" in name_lower:
        return "sdxl", precision
    elif "sd3" in name_lower or "sd_3" in name_lower:
        return "sd3", precision
    elif "sd15" in name_lower or "sd_1" in name_lower or "v1-5" in name_lower:
        return "sd15", precision
    elif "qwen" in name_lower:
        return "qwen-image", precision
    elif "z-image" in name_lower or "z_image" in name_lower or "zimage" in name_lower:
        return "z-image-turbo", precision
    elif "ltx" in name_lower:
        if "19b" in name_lower or "ltx-2" in name_lower or "ltx2" in name_lower:
            return "ltx-2-19b", precision
        elif "2" in name_lower:
            return "ltx-2", precision
        return "ltx", precision
    elif "wan" in name_lower:
        if "2.6" in name_lower or "26" in name_lower:
            return "wan2.6", precision
        return "wan2", precision
    elif "hunyuan" in name_lower:
        if "video" in name_lower:
            return "hunyuan-video", precision
        return "hunyuan", precision
    elif "svd" in name_lower:
        return "svd", precision

    return "unknown", precision


def get_rtx5090_recommendation(model_name: str) -> dict:
    """
    Get RTX 5090 32GB specific recommendations for a model.

    Args:
        model_name: Model filename

    Returns:
        Recommendations including precision, will_fit, and notes.
    """
    model_type, detected_precision = detect_model_type(model_name)
    vram_table = MODEL_VRAM_ESTIMATES.get(model_type, MODEL_VRAM_ESTIMATES["unknown"])

    recommended_precision = vram_table.get("rtx5090_recommended", "fp16")
    recommended_vram = vram_table.get(recommended_precision, vram_table["default"])

    # Add overhead for VAE, CLIP, etc.
    total_estimated = recommended_vram + 4.0  # Conservative overhead

    will_fit = total_estimated <= RTX_5090_USABLE_GB
    has_warning = vram_table.get("rtx5090_warning") is not None

    return {
        "model": model_name,
        "model_type": model_type,
        "detected_precision": detected_precision,
        "recommended_precision": recommended_precision,
        "estimated_vram_gb": round(recommended_vram, 1),
        "total_with_overhead_gb": round(total_estimated, 1),
        "rtx5090_usable_gb": RTX_5090_USABLE_GB,
        "will_fit": will_fit,
        "note": vram_table.get("rtx5090_note", ""),
        "warning": vram_table.get("rtx5090_warning"),
        "is_video_model": model_type in ["ltx", "ltx-2", "ltx-2-19b", "wan", "wan2", "wan2.6", "hunyuan-video", "svd"],
    }


def get_pipeline_vram_strategy(image_model: str, video_model: str) -> dict:
    """
    Get VRAM management strategy for image → video pipelines on RTX 5090.

    Args:
        image_model: Image generation model name
        video_model: Video generation model name

    Returns:
        Strategy for managing VRAM between pipeline stages.
    """
    image_rec = get_rtx5090_recommendation(image_model)
    video_rec = get_rtx5090_recommendation(video_model)

    combined_vram = image_rec["total_with_overhead_gb"] + video_rec["total_with_overhead_gb"]
    can_stack = combined_vram <= RTX_5090_USABLE_GB

    return {
        "image_model": {
            "model": image_model,
            "precision": image_rec["recommended_precision"],
            "vram_gb": image_rec["estimated_vram_gb"],
        },
        "video_model": {
            "model": video_model,
            "precision": video_rec["recommended_precision"],
            "vram_gb": video_rec["estimated_vram_gb"],
        },
        "combined_vram_gb": round(combined_vram, 1),
        "can_run_simultaneously": can_stack,
        "strategy": "keep_loaded" if can_stack else "sequential_unload",
        "action": (
            "Models can remain loaded simultaneously"
            if can_stack
            else "MUST unload image model before loading video model. Use POST /free {unload_models: true}"
        ),
    }


def estimate_workflow_vram(workflow: dict) -> dict:
    """
    Estimate VRAM usage for a workflow.

    Args:
        workflow: ComfyUI workflow JSON

    Returns:
        VRAM estimate with breakdown.
    """
    total_vram = 0.0
    breakdown = []

    # Track what we've seen
    models_found = []
    has_vae = False
    has_clip = False
    has_t5 = False
    controlnets = 0
    loras = 0
    ipadapters = 0

    # Find resolution
    width = 1024
    height = 1024
    batch_size = 1

    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        # Detect models
        if class_type in ["CheckpointLoaderSimple", "UNETLoader"]:
            model_name = inputs.get("ckpt_name", inputs.get("unet_name", "unknown"))
            model_type, precision = detect_model_type(model_name)
            vram = MODEL_VRAM_ESTIMATES.get(model_type, MODEL_VRAM_ESTIMATES["unknown"])
            vram_gb = vram.get(precision, vram["default"])
            models_found.append({
                "name": model_name,
                "type": model_type,
                "precision": precision,
                "vram_gb": vram_gb,
            })
            total_vram += vram_gb
            breakdown.append(f"{model_type} ({precision}): {vram_gb:.1f}GB")

        # Detect VAE
        if class_type in ["VAELoader", "VAEDecode", "VAEEncode"]:
            has_vae = True

        # Detect CLIP
        if class_type in ["CLIPLoader", "CLIPTextEncode"]:
            has_clip = True

        # Detect T5 (Flux, SD3)
        if class_type in ["DualCLIPLoader"]:
            has_t5 = True
            has_clip = True

        # Detect ControlNet
        if "controlnet" in class_type.lower():
            controlnets += 1

        # Detect LoRA
        if "lora" in class_type.lower():
            loras += 1

        # Detect IP-Adapter
        if "ipadapter" in class_type.lower():
            ipadapters += 1

        # Detect resolution
        if class_type in ["EmptyLatentImage", "EmptySD3LatentImage"]:
            width = inputs.get("width", width)
            height = inputs.get("height", height)
            batch_size = inputs.get("batch_size", batch_size)

    # Add component VRAM
    if has_vae:
        total_vram += COMPONENT_VRAM["vae"]
        breakdown.append(f"VAE: {COMPONENT_VRAM['vae']:.1f}GB")

    if has_clip:
        total_vram += COMPONENT_VRAM["clip"]
        breakdown.append(f"CLIP: {COMPONENT_VRAM['clip']:.1f}GB")

    if has_t5:
        total_vram += COMPONENT_VRAM["t5"]
        breakdown.append(f"T5: {COMPONENT_VRAM['t5']:.1f}GB")

    if controlnets > 0:
        cn_vram = controlnets * COMPONENT_VRAM["controlnet"]
        total_vram += cn_vram
        breakdown.append(f"ControlNet x{controlnets}: {cn_vram:.1f}GB")

    if loras > 0:
        lora_vram = loras * COMPONENT_VRAM["lora"]
        total_vram += lora_vram
        breakdown.append(f"LoRA x{loras}: {lora_vram:.1f}GB")

    if ipadapters > 0:
        ip_vram = ipadapters * COMPONENT_VRAM["ipadapter"]
        total_vram += ip_vram
        breakdown.append(f"IP-Adapter x{ipadapters}: {ip_vram:.1f}GB")

    # Apply resolution multiplier
    res_multiplier = get_resolution_multiplier(width, height)
    if res_multiplier > 1.0:
        # Only apply to latent-related VRAM (roughly 30% of total)
        latent_overhead = (total_vram * 0.3) * (res_multiplier - 1.0)
        total_vram += latent_overhead
        breakdown.append(f"Resolution overhead ({width}x{height}): {latent_overhead:.1f}GB")

    # Batch size multiplier
    if batch_size > 1:
        batch_overhead = (total_vram * 0.2) * (batch_size - 1)
        total_vram += batch_overhead
        breakdown.append(f"Batch size x{batch_size}: {batch_overhead:.1f}GB")

    # Get current system stats
    client = get_client()
    stats = client.get_system_stats()

    available_vram = 0.0
    total_system_vram = 0.0

    if "devices" in stats:
        for device in stats["devices"]:
            if device.get("type") == "cuda":
                available_vram = device.get("vram_free", 0) / (1024**3)
                total_system_vram = device.get("vram_total", 0) / (1024**3)
                break

    will_fit = total_vram <= available_vram if available_vram > 0 else None

    return {
        "estimated_vram_gb": round(total_vram, 1),
        "available_vram_gb": round(available_vram, 1),
        "total_system_vram_gb": round(total_system_vram, 1),
        "will_fit": will_fit,
        "breakdown": breakdown,
        "models": models_found,
        "resolution": f"{width}x{height}",
        "batch_size": batch_size,
        "recommendation": get_recommendation(total_vram, available_vram, models_found),
    }


def get_recommendation(estimated: float, available: float, models: list) -> str:
    """Generate recommendation based on VRAM analysis."""
    if available == 0:
        return "Could not get VRAM info from ComfyUI"

    if estimated <= available:
        margin = available - estimated
        if margin > 5:
            return f"Good fit with {margin:.1f}GB headroom"
        else:
            return f"Should fit with {margin:.1f}GB margin (may be tight)"
    else:
        deficit = estimated - available
        suggestions = []

        # Suggest fp8 if not already using it
        for model in models:
            if model["precision"] != "fp8" and model["type"] != "unknown":
                fp8_vram = MODEL_VRAM_ESTIMATES.get(model["type"], {}).get("fp8", 0)
                current_vram = model["vram_gb"]
                if fp8_vram < current_vram:
                    savings = current_vram - fp8_vram
                    suggestions.append(f"Use fp8 {model['type']} to save ~{savings:.1f}GB")

        if suggestions:
            return f"Needs {deficit:.1f}GB more VRAM. " + " | ".join(suggestions[:2])
        else:
            return f"Needs {deficit:.1f}GB more VRAM. Consider smaller model or lower resolution."


def check_model_fits(model_name: str, precision: str = "default") -> dict:
    """
    Quick check if a specific model will fit in available VRAM.

    Args:
        model_name: Model filename
        precision: Precision to check (fp32, fp16, bf16, fp8, default)

    Returns:
        Whether model fits and recommendations.
    """
    model_type, detected_precision = detect_model_type(model_name)

    if precision == "default":
        precision = detected_precision

    vram_table = MODEL_VRAM_ESTIMATES.get(model_type, MODEL_VRAM_ESTIMATES["unknown"])
    estimated_vram = vram_table.get(precision, vram_table["default"])

    # Add typical overhead (VAE, CLIP, etc.)
    estimated_vram += 4.0  # Conservative overhead

    # Get available VRAM
    client = get_client()
    stats = client.get_system_stats()

    available_vram = 0.0
    for device in stats.get("devices", []):
        if device.get("type") == "cuda":
            available_vram = device.get("vram_free", 0) / (1024**3)
            break

    will_fit = estimated_vram <= available_vram

    return {
        "model": model_name,
        "model_type": model_type,
        "precision": precision,
        "estimated_vram_gb": round(estimated_vram, 1),
        "available_vram_gb": round(available_vram, 1),
        "will_fit": will_fit,
        "alternatives": get_alternatives(model_type, available_vram) if not will_fit else [],
    }


def get_alternatives(model_type: str, available_vram: float) -> list:
    """Get alternative models that fit in available VRAM."""
    alternatives = []

    vram_table = MODEL_VRAM_ESTIMATES.get(model_type, {})

    for precision in ["fp8", "bf16", "fp16", "fp32"]:
        if precision in vram_table:
            vram_needed = vram_table[precision] + 4.0  # Include overhead
            if vram_needed <= available_vram:
                alternatives.append({
                    "precision": precision,
                    "estimated_vram_gb": round(vram_needed, 1),
                })

    return alternatives
