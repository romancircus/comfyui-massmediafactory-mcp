"""
Hardware-Aware Optimization for ComfyUI Workflows

Connects vram.py, model_registry.py, and sota.py to produce optimal
workflow parameters for the detected GPU. Eliminates manual tuning
by auto-selecting load_device, quantization, and attention mode.

Key insight: RTX 5090 (32GB, compute 12.0) can keep WAN I2V 14B on GPU
with fp8_e4m3fn_scaled_fast, yielding 3-4x speedup over offload_device.
Templates default to safe/slow settings; this module overrides at runtime.
"""

import logging
from typing import Optional

from .client import get_client
from .vram import MODEL_VRAM_ESTIMATES

logger = logging.getLogger("comfyui-mcp.optimization")


# ============================================================================
# VRAM thresholds → load strategy
# ============================================================================

# Ordered from best to worst. First match where available >= threshold wins.
LOAD_STRATEGIES = [
    {
        "min_vram_gb": 28,
        "load_device": "main_device",
        "force_offload_sampler": False,
        "label": "full_gpu",
        "note": "Full GPU - model stays in VRAM, fastest",
    },
    {
        "min_vram_gb": 20,
        "load_device": "main_device",
        "force_offload_sampler": True,
        "label": "gpu_with_offload",
        "note": "Model on GPU but offloaded between steps",
    },
    {
        "min_vram_gb": 0,
        "load_device": "offload_device",
        "force_offload_sampler": True,
        "label": "cpu_offload",
        "note": "CPU offload - slowest but safest",
    },
]


# Compute capability → best quantization format
COMPUTE_QUANTIZATION = {
    (12, 0): "fp8_e4m3fn_scaled_fast",  # RTX 5090 - native fp8 matmul
    (8, 9): "fp8_e4m3fn_scaled_fast",  # RTX 4090 - fp8 matmul supported
    (8, 6): "fp8_e4m3fn_scaled",  # RTX 3090 - no fast path
    (7, 5): "fp8_e4m3fn",  # Older - basic fp8
}

# Timing estimates per model (seconds, for main_device + fp8_fast)
TIMING_ESTIMATES = {
    ("wan26", "i2v"): {"full_gpu": 150, "gpu_with_offload": 300, "cpu_offload": 510},
    ("wan26", "t2v"): {"full_gpu": 60, "gpu_with_offload": 120, "cpu_offload": 200},
    ("ltx2", "t2v"): {"full_gpu": 45, "gpu_with_offload": 90, "cpu_offload": 150},
    ("ltx2", "i2v"): {"full_gpu": 50, "gpu_with_offload": 100, "cpu_offload": 160},
    ("flux2", "t2i"): {"full_gpu": 15, "gpu_with_offload": 25, "cpu_offload": 45},
    ("qwen", "t2i"): {"full_gpu": 20, "gpu_with_offload": 35, "cpu_offload": 60},
    ("qwen_edit", "edit"): {"full_gpu": 25, "gpu_with_offload": 40, "cpu_offload": 70},
    ("sdxl", "t2i"): {"full_gpu": 8, "gpu_with_offload": 12, "cpu_offload": 25},
    ("hunyuan15", "t2v"): {"full_gpu": 120, "gpu_with_offload": 240, "cpu_offload": 480},
    ("hunyuan15", "i2v"): {"full_gpu": 130, "gpu_with_offload": 260, "cpu_offload": 500},
    ("z_turbo", "t2i"): {"full_gpu": 3, "gpu_with_offload": 5, "cpu_offload": 10},
    ("cogvideox_5b", "t2v"): {"full_gpu": 90, "gpu_with_offload": 180, "cpu_offload": 360},
}

# ============================================================================
# Node class → overridable hardware params
# ============================================================================
# Maps each ComfyUI node class to the abstract parameter names it accepts.
# Abstract params: load_device, quantization, attention_mode, force_offload
# The dict value maps abstract_param → actual node input name.

HARDWARE_PARAMS_MAP = {
    # --- WAN (WanVideoWrapper pipeline) ---
    "WanVideoModelLoader": {
        "load_device": "load_device",
        "quantization": "quantization",
        "attention_mode": "attention_mode",
    },
    "LoadWanVideoT5TextEncoder": {
        "load_device": "load_device",
    },
    "LoadWanVideoClipTextEncoder": {
        "load_device": "load_device",
    },
    "WanVideoSampler": {
        "force_offload": "force_offload",
    },
    "WanVideoTextEncode": {
        "force_offload": "force_offload",
    },
    "WanVideoClipVisionEncode": {
        "force_offload": "force_offload",
    },
    "WanVideoImageToVideoEncode": {
        "force_offload": "force_offload",
    },
    # --- WAN (native ComfyUI pipeline) ---
    "UNETLoader": {
        "load_device": "weight_dtype",  # UNETLoader uses weight_dtype for precision control
    },
    # --- LTX-2 ---
    "CheckpointLoaderSimple": {
        # No hardware params exposed directly — controlled by ComfyUI launch flags
    },
    # --- HunyuanVideo ---
    "HunyuanVideoModelLoader": {
        "load_device": "load_device",
        "quantization": "quantization",
    },
    "HunyuanVideoSampler": {
        "force_offload": "force_offload",
    },
    # --- CogVideoX ---
    "CogVideoLoader": {
        "quantization": "quantization",
    },
}

# VRAM key mapping: model registry key → vram.py key
_VRAM_KEYS = {
    "wan26": "wan2.6",
    "wan": "wan",
    "ltx2": "ltx-2",
    "flux2": "flux2",
    "qwen": "qwen-image",
    "qwen_edit": "qwen-image",
    "sdxl": "sdxl",
    "hunyuan15": "hunyuan-video",
    "z_turbo": "z-image-turbo",
    "cogvideox_5b": "unknown",
}


def detect_hardware() -> dict:
    """Detect GPU from ComfyUI /system_stats.

    Returns:
        {
            "name": "NVIDIA GeForce RTX 5090",
            "vram_total_gb": 32.0,
            "vram_free_gb": 30.6,
            "compute_capability": (12, 0),  # None if unknown
        }
    """
    try:
        client = get_client()
        stats = client.get_system_stats()
        if "error" in stats:
            return _fallback_hardware(f"ComfyUI error: {stats['error']}")

        devices = stats.get("devices", [])
        for device in devices:
            if device.get("type") == "cuda":
                name = device.get("name", "Unknown GPU")
                vram_total = device.get("vram_total", 0) / (1024**3)
                vram_free = device.get("vram_free", 0) / (1024**3)
                compute = _detect_compute_capability(name)
                return {
                    "name": name,
                    "vram_total_gb": round(vram_total, 1),
                    "vram_free_gb": round(vram_free, 1),
                    "compute_capability": compute,
                }

        return _fallback_hardware("No CUDA device found")
    except Exception as e:
        return _fallback_hardware(str(e))


def _fallback_hardware(reason: str) -> dict:
    """Return safe fallback when hardware detection fails."""
    logger.warning("Hardware detection failed: %s", reason)
    return {
        "name": "Unknown",
        "vram_total_gb": 0,
        "vram_free_gb": 0,
        "compute_capability": None,
        "_fallback": True,
        "_reason": reason,
    }


def _detect_compute_capability(gpu_name: str) -> Optional[tuple]:
    """Infer compute capability from GPU name."""
    name = gpu_name.lower()
    if "5090" in name or "5080" in name or "5070" in name:
        return (12, 0)  # Blackwell
    if "4090" in name or "4080" in name or "4070" in name:
        return (8, 9)  # Ada Lovelace
    if "3090" in name or "3080" in name or "3070" in name:
        return (8, 6)  # Ampere
    if "2080" in name or "2070" in name:
        return (7, 5)  # Turing
    return None


def _select_load_strategy(available_vram_gb: float) -> dict:
    """Select optimal load strategy based on available VRAM."""
    for strategy in LOAD_STRATEGIES:
        if available_vram_gb >= strategy["min_vram_gb"]:
            return strategy
    return LOAD_STRATEGIES[-1]  # Safest fallback


def _select_quantization(model: str, compute_capability: Optional[tuple]) -> str:
    """Select optimal quantization for model + GPU combo."""
    vram_key = _VRAM_KEYS.get(model, model)
    vram_table = MODEL_VRAM_ESTIMATES.get(vram_key, {})

    # If model recommends fp8 (large video models), use compute-optimal fp8
    recommended = vram_table.get("rtx5090_recommended", "fp16")
    if recommended == "fp8" and compute_capability:
        return COMPUTE_QUANTIZATION.get(compute_capability, "fp8_e4m3fn_scaled")

    # For fp16-recommended models, quantization is "disabled"
    return "disabled"


def get_optimal_workflow_params(
    model: str,
    task: str = "i2v",
    available_vram_gb: Optional[float] = None,
    compute_capability: Optional[tuple] = None,
) -> dict:
    """Return hardware-optimized parameters for a model+task combo.

    Queries vram.py and model_registry.py to produce a unified
    recommendation. Used by generate_workflow() and batch scripts.

    Args:
        model: Model key (wan26, ltx2, flux2, qwen, etc.)
        task: Task type (i2v, t2v, t2i, edit)
        available_vram_gb: Override VRAM (auto-detected if None)
        compute_capability: Override compute cap (auto-detected if None)

    Returns:
        {
            "hardware": {"gpu_name", "vram_gb", "compute", "profile_used"},
            "load_device": str,
            "quantization": str,
            "attention_mode": str,
            "force_offload_sampler": bool,
            "workflow_overrides": {node_id: {param: value}},
            "timing_estimate_seconds": int,
            "warnings": [str],
        }
    """
    warnings = []

    # Auto-detect hardware if not provided
    hw = detect_hardware()
    if available_vram_gb is None:
        available_vram_gb = hw.get("vram_free_gb", 0)
    if compute_capability is None:
        compute_capability = hw.get("compute_capability")
    gpu_name = hw.get("name", "Unknown") if not hw.get("_fallback") else "Unknown"

    # Select load strategy
    strategy = _select_load_strategy(available_vram_gb)

    # Select quantization
    quantization = _select_quantization(model, compute_capability)

    # Attention mode: sdpa is universally best for modern GPUs
    attention_mode = "sdpa"

    # Generate workflow overrides for ALL node types in the workflow
    workflow_overrides = _build_workflow_overrides(
        load_device=strategy["load_device"],
        quantization=quantization,
        attention_mode=attention_mode,
        force_offload_sampler=strategy["force_offload_sampler"],
    )

    # Timing estimate
    model_lower = model.lower()
    task_lower = task.lower()
    timing = TIMING_ESTIMATES.get(
        (model_lower, task_lower), {"full_gpu": 120, "gpu_with_offload": 240, "cpu_offload": 480}
    )
    estimated_seconds = timing.get(strategy["label"], timing.get("cpu_offload", 300))

    # Warnings
    if strategy["label"] == "cpu_offload":
        warnings.append(
            f"Running in degraded mode (offload_device). "
            f"Only {available_vram_gb:.1f}GB VRAM free. "
            f"Expected ~{estimated_seconds}s per generation."
        )
    if hw.get("_fallback"):
        warnings.append(f"Hardware detection failed: {hw.get('_reason', 'unknown')}")

    return {
        "hardware": {
            "gpu_name": gpu_name,
            "vram_total_gb": hw.get("vram_total_gb", 0),
            "vram_free_gb": available_vram_gb,
            "compute_capability": (
                f"{compute_capability[0]}.{compute_capability[1]}" if compute_capability else "unknown"
            ),
            "profile_used": strategy["label"],
        },
        "load_device": strategy["load_device"],
        "quantization": quantization,
        "attention_mode": attention_mode,
        "force_offload_sampler": strategy["force_offload_sampler"],
        "workflow_overrides": workflow_overrides,
        "timing_estimate_seconds": estimated_seconds,
        "warnings": warnings,
    }


def _build_workflow_overrides(
    load_device: str,
    quantization: str,
    attention_mode: str,
    force_offload_sampler: bool,
) -> dict:
    """Build node-level overrides dict keyed by class_type.

    Iterates HARDWARE_PARAMS_MAP to generate overrides for ALL node types
    that accept hardware parameters (WAN, LTX, FLUX, HunyuanVideo, CogVideoX, etc.).

    Returns: {"WanVideoModelLoader": {"load_device": "main_device", ...}, ...}
    """
    # Map abstract param names to their resolved values
    param_values = {
        "load_device": load_device,
        "quantization": quantization,
        "attention_mode": attention_mode,
        "force_offload": force_offload_sampler,
    }

    overrides = {}
    for class_type, param_map in HARDWARE_PARAMS_MAP.items():
        if not param_map:
            continue
        node_overrides = {}
        for abstract_param, node_input_name in param_map.items():
            if abstract_param in param_values:
                node_overrides[node_input_name] = param_values[abstract_param]
        if node_overrides:
            overrides[class_type] = node_overrides

    return overrides


def apply_hardware_overrides(workflow: dict, overrides: dict) -> dict:
    """Apply hardware overrides to a workflow in-place.

    Args:
        workflow: ComfyUI API format workflow (node_id → {class_type, inputs})
        overrides: {class_type → {param → value}} from get_optimal_workflow_params

    Returns:
        The modified workflow (same object, mutated).
    """
    for node_id, node_data in workflow.items():
        if node_id.startswith("_"):
            continue
        class_type = node_data.get("class_type", "")
        if class_type in overrides:
            for param, value in overrides[class_type].items():
                node_data.setdefault("inputs", {})[param] = value
    return workflow


def format_hardware_banner(params: dict) -> str:
    """Format a one-line hardware summary for script startup banners.

    Args:
        params: Result from get_optimal_workflow_params()

    Returns:
        Formatted banner string.
    """
    hw = params.get("hardware", {})
    gpu = hw.get("gpu_name", "Unknown")
    vram_free = hw.get("vram_free_gb", 0)
    vram_total = hw.get("vram_total_gb", 0)
    profile = hw.get("profile_used", "unknown")
    est = params.get("timing_estimate_seconds", 0)
    quant = params.get("quantization", "unknown")
    device = params.get("load_device", "unknown")

    lines = [
        f"GPU: {gpu} | {vram_free:.1f}/{vram_total:.1f}GB free",
        f"Mode: {device} + {quant} | Profile: {profile} | ~{est}s/video",
    ]

    if params.get("warnings"):
        for w in params["warnings"]:
            lines.append(f"WARNING: {w}")

    return "\n".join(lines)
