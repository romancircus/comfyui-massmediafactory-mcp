"""Centralized Error Handling with Actionable Guidance.

Legacy error formatting functions - now integrated with core.errors.

Transform cryptic error messages into actionable guidance for users.
"""

from typing import Dict, Any, Optional, List
import os
import sys
from pathlib import Path

# Add src to path for core module access
_src_path = str(Path(__file__).parent.parent.parent)
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from core import (
    format_model_not_found,
    format_custom_node_missing,
    format_template_metadata,
    format_template_parameter,
)


def format_model_not_found(
    model_name: str,
    model_type: str,
    available_models: List[str]
) -> Dict[str, Any]:
    """Format error for model not found with actionable suggestions.
    
    Returns:
        {
            "isError": True,
            "error": str,
            "suggestion": str,
            "available": List[str],
            "code": "MODEL_NOT_FOUND"
        }
    """
    return {
        "isError": True,
        "error": f"Model not found: {model_name} in models/{model_type}/",
        "suggestion": f"Run list_models('{model_type}') to see available models. "
                     f"Common issue: typo in model name or model not downloaded.",
        "available": available_models[:10] if available_models else [],
        "troubleshooting": f"Check ~/ComfyUI/models/{model_type}/ for the file.",
        "code": "MODEL_NOT_FOUND"
    }


def format_invalid_resolution(
    width: int,
    height: int,
    model: str,
    divisible_by: int,
    valid_examples: List[int]
) -> Dict[str, Any]:
    """Format error for invalid resolution with suggested valid sizes.
    
    Returns:
        {
            "isError": True,
            "error": str,
            "suggestion": str,
            "valid_sizes": List[Tuple[int, int]],
            "code": "INVALID_RESOLUTION"
        }
    """
    # Calculate nearest valid sizes
    valid_width = (width // divisible_by) * divisible_by
    valid_height = (height // divisible_by) * divisible_by
    
    if valid_width != width:
        valid_width += divisible_by
    if valid_height != height:
        valid_height += divisible_by
    
    return {
        "isError": True,
        "error": f"Resolution {width}x{height} invalid for {model}. "
                 f"Must be divisible by {divisible_by}.",
        "suggestion": f"Suggested valid sizes: {valid_width}x{valid_height} or use native resolution.",
        "valid_sizes": [
            (valid_width, valid_height),
            (valid_width + divisible_by, valid_height),
            (valid_width, valid_height + divisible_by)
        ],
        "troubleshooting": f"FLUX requires divisible by 16, SDXL/LTX by 8.",
        "code": "INVALID_RESOLUTION"
    }


def format_connection_failed(
    host: str,
    port: int,
    original_error: str
) -> Dict[str, Any]:
    """Format error for connection failure with troubleshooting steps.
    
    Returns:
        {
            "isError": True,
            "error": str,
            "suggestion": str,
            "troubleshooting": List[str],
            "code": "CONNECTION_FAILED"
        }
    """
    return {
        "isError": True,
        "error": f"Cannot connect to ComfyUI at http://{host}:{port}: {original_error}",
        "suggestion": "Is ComfyUI running? Check systemctl or start ComfyUI manually.",
        "troubleshooting": [
            f"1. Check ComfyUI status: sudo systemctl status comfyui",
            f"2. Test connection: curl http://{host}:{port}/system_stats",
            f"3. Verify Tailscale is connected (if using remote)",
            f"4. Check firewall rules: sudo ufw status"
        ],
        "code": "CONNECTION_FAILED"
    }


def format_missing_i2v_image(
    workflow_type: str,
    missing_param: str = "IMAGE_PATH"
) -> Dict[str, Any]:
    """Format error for missing I2V image parameter.
    
    Returns:
        {
            "isError": True,
            "error": str,
            "suggestion": str,
            "example": str,
            "code": "MISSING_I2V_IMAGE"
        }
    """
    return {
        "isError": True,
        "error": f"{workflow_type} workflow requires {missing_param} parameter.",
        "suggestion": "Upload image first, then provide the path in workflow parameters.",
        "example": f"1. upload_image('/path/to/image.png')\n"
                  f"2. Use returned filename in workflow: IMAGE_PATH='uploaded_image.png'",
        "troubleshooting": "I2V workflows need a reference image to animate.",
        "code": "MISSING_I2V_IMAGE"
    }


def format_vram_error(
    required_gb: float,
    available_gb: float,
    suggestions: List[str] = None
) -> Dict[str, Any]:
    """Format error for out-of-VRAM with specific suggestions.
    
    Returns:
        {
            "isError": True,
            "error": str,
            "suggestion": str,
            "actions": List[str],
            "code": "OUT_OF_VRAM"
        }
    """
    if suggestions is None:
        suggestions = [
            "Lower resolution (try 512x512 instead of 1024x1024)",
            "Reduce batch size to 1",
            "Use free_memory(unload_models=True) to clear VRAM",
            "Use FP8 or FP16 precision instead of FP32",
            "Close other applications using GPU"
        ]
    
    return {
        "isError": True,
        "error": f"Out of VRAM: {required_gb:.1f}GB needed, {available_gb:.1f}GB available.",
        "suggestion": f"Need {required_gb - available_gb:.1f}GB more VRAM to run this workflow.",
        "actions": suggestions,
        "troubleshooting": "VRAM usage scales with resolution. 2x resolution = 4x VRAM.",
        "code": "OUT_OF_VRAM"
    }


def format_workflow_validation_error(
    errors: List[str],
    warnings: List[str],
    model: str
) -> Dict[str, Any]:
    """Format workflow validation errors.
    
    Returns:
        {
            "isError": True,
            "error": str,
            "details": List[str],
            "warnings": List[str],
            "code": "WORKFLOW_VALIDATION"
        }
    """
    return {
        "isError": True,
        "error": f"Workflow validation failed for {model}.",
        "details": errors,
        "warnings": warnings,
        "suggestion": "Use get_model_constraints() to see valid parameters for this model.",
        "troubleshooting": "Check for forbidden nodes, resolution constraints, or CFG limits.",
        "code": "WORKFLOW_VALIDATION"
    }


def format_timeout_error(
    prompt_id: str,
    timeout_seconds: int,
    status: str
) -> Dict[str, Any]:
    """Format timeout error with suggestions.
    
    Returns:
        {
            "isError": True,
            "error": str,
            "suggestion": str,
            "code": "TIMEOUT"
        }
    """
    return {
        "isError": True,
        "error": f"Workflow {prompt_id} timed out after {timeout_seconds} seconds.",
        "suggestion": "Video generation may take 5-15 minutes. Try increasing timeout_seconds.",
        "current_status": status,
        "troubleshooting": [
            "Check if still running: get_workflow_status(prompt_id)",
            "Check ComfyUI logs: sudo journalctl -u comfyui -f",
            "For long generations, use timeout_seconds=1200 or more"
        ],
        "code": "TIMEOUT"
    }


def format_batch_error(
    failed_indices: List[int],
    total: int,
    error_messages: List[str]
) -> Dict[str, Any]:
    """Format batch execution error.
    
    Returns:
        {
            "isError": True,
            "error": str,
            "succeeded": int,
            "failed": int,
            "failed_indices": List[int],
            "code": "BATCH_PARTIAL_FAILURE"
        }
    """
    return {
        "isError": True,
        "error": f"Batch execution: {len(failed_indices)}/{total} jobs failed.",
        "succeeded": total - len(failed_indices),
        "failed": len(failed_indices),
        "failed_indices": failed_indices,
        "errors": error_messages[:3],  # First 3 errors
        "suggestion": "Check failed_indices for which jobs failed. Review error messages.",
        "troubleshooting": "Batch failures often due to VRAM or invalid parameters.",
        "code": "BATCH_PARTIAL_FAILURE"
    }


# Error code registry
ERROR_CODES = {
    "MODEL_NOT_FOUND": "Model file not found in expected location",
    "INVALID_RESOLUTION": "Resolution doesn't meet model requirements",
    "CONNECTION_FAILED": "Cannot connect to ComfyUI server",
    "MISSING_I2V_IMAGE": "Image-to-Video workflow missing reference image",
    "OUT_OF_VRAM": "GPU ran out of memory",
    "WORKFLOW_VALIDATION": "Workflow failed validation checks",
    "TIMEOUT": "Workflow execution exceeded time limit",
    "BATCH_PARTIAL_FAILURE": "Some jobs in batch failed",
    "INVALID_PARAMS": "Invalid parameter values provided",
    "COMFYUI_ERROR": "ComfyUI returned an error",
}


def get_error_help(error_code: str) -> str:
    """Get help text for an error code."""
    return ERROR_CODES.get(error_code, "Unknown error code")
