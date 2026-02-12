"""
LTX Planner Module - Family-Specific Workflow Routing

Provides explicit branch selection for LTX I2V/T2V variants with canonical node chains.
Implements explain trace for selected vs rejected branches.

FR Mapping: FR-2.3, FR-2.5, FR-2.6
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class LTXVariant(Enum):
    """LTX workflow variants supported by the planner."""

    T2V_STANDARD = "txt2vid"  # Standard T2V with AV pipeline
    T2V_DISTILLED = "txt2vid_distilled"  # Fast distilled variant
    T2V_ENHANCED = "txt2vid_enhanced"  # STG-enhanced quality
    I2V_STANDARD = "img2vid"  # Standard I2V
    I2V_DISTILLED = "i2v_distilled"  # Fast I2V distilled
    V2V = "vid2vid"  # Video-to-video with guide
    AUDIO_REACTIVE = "audio_reactive"  # Audio-reactive generation


class LTXSampler(Enum):
    """Sampler variants for LTX workflows."""

    SAMPLER_CUSTOM = "SamplerCustom"
    SAMPLER_CUSTOM_ADVANCED = "SamplerCustomAdvanced"


@dataclass
class BranchSelection:
    """Result of branch selection with explain trace."""

    selected_variant: LTXVariant
    selected_skeleton: str
    canonical_nodes: Dict[str, str]
    sampler_type: LTXSampler
    explain_trace: List[Dict[str, Any]] = field(default_factory=list)
    rejected_alternatives: List[Dict[str, str]] = field(default_factory=list)
    hints_matched: Dict[str, Any] = field(default_factory=dict)


# Canonical LTX node chains per variant
LTX_CANONICAL_NODES = {
    LTXVariant.T2V_STANDARD: {
        "loader": "CheckpointLoaderSimple",
        "text_encoder": "LTXAVTextEncoderLoader",
        "audio_vae": "LTXVAudioVAELoader",
        "conditioning": "LTXVConditioning",
        "latent_video": "EmptyLTXVLatentVideo",
        "latent_audio": "LTXVEmptyLatentAudio",
        "av_concat": "LTXVConcatAVLatent",
        "scheduler": "LTXVScheduler",
        "sampler": "SamplerCustomAdvanced",
        "guider": "CFGGuider",
        "av_separate": "LTXVSeparateAVLatent",
        "vae_decode": "VAEDecodeTiled",
        "audio_decode": "LTXVAudioVAEDecode",
        "output": "SaveVideo",
    },
    LTXVariant.T2V_DISTILLED: {
        "loader": "CheckpointLoaderSimple",
        "text_encoder": "LTXVGemmaCLIPModelLoader",
        "prompt_enhance": "LTXVGemmaEnhancePrompt",
        "conditioning": "LTXVConditioning",
        "latent_video": "EmptyLTXVLatentVideo",
        "scheduler": "LTXVScheduler",
        "sampler": "SamplerCustom",
        "vae_decode": "VAEDecode",
        "output": "VHS_VideoCombine",
    },
    LTXVariant.T2V_ENHANCED: {
        "loader": "CheckpointLoaderSimple",
        "stg_enhance": "LTXVApplySTG",
        "text_encoder": "LTXAVTextEncoderLoader",
        "audio_vae": "LTXVAudioVAELoader",
        "conditioning": "LTXVConditioning",
        "latent_video": "EmptyLTXVLatentVideo",
        "latent_audio": "LTXVEmptyLatentAudio",
        "av_concat": "LTXVConcatAVLatent",
        "scheduler": "LTXVScheduler",
        "sampler": "SamplerCustomAdvanced",
        "guider": "CFGGuider",
        "av_separate": "LTXVSeparateAVLatent",
        "vae_decode": "VAEDecodeTiled",
        "audio_decode": "LTXVAudioVAEDecode",
        "output": "SaveVideo",
    },
    LTXVariant.I2V_STANDARD: {
        "loader": "CheckpointLoaderSimple",
        "text_encoder": "LTXAVTextEncoderLoader",
        "audio_vae": "LTXVAudioVAELoader",
        "conditioning": "LTXVConditioning",
        "image_load": "LoadImage",
        "image_to_video": "LTXVImgToVideoAdvanced",
        "latent_audio": "LTXVEmptyLatentAudio",
        "av_concat": "LTXVConcatAVLatent",
        "scheduler": "LTXVScheduler",
        "sampler": "SamplerCustomAdvanced",
        "guider": "CFGGuider",
        "av_separate": "LTXVSeparateAVLatent",
        "vae_decode": "VAEDecodeTiled",
        "audio_decode": "LTXVAudioVAEDecode",
        "output": "SaveVideo",
    },
    LTXVariant.I2V_DISTILLED: {
        "loader": "CheckpointLoaderSimple",
        "text_encoder": "LTXVGemmaCLIPModelLoader",
        "prompt_enhance": "LTXVGemmaEnhancePrompt",
        "conditioning": "LTXVConditioning",
        "image_load": "LoadImage",
        "image_preprocess": "LTXVPreprocess",
        "image_to_video": "LTXVImgToVideo",
        "scheduler": "LTXVScheduler",
        "sampler": "SamplerCustom",
        "vae_decode": "VAEDecode",
        "output": "VHS_VideoCombine",
    },
    LTXVariant.V2V: {
        "loader": "CheckpointLoaderSimple",
        "text_encoder": "LTXAVTextEncoderLoader",
        "audio_vae": "LTXVAudioVAELoader",
        "conditioning": "LTXVConditioning",
        "latent_video": "EmptyLTXVLatentVideo",
        "latent_audio": "LTXVEmptyLatentAudio",
        "av_concat": "LTXVConcatAVLatent",
        "image_load": "LoadImage",
        "add_guide": "LTXVAddGuide",
        "scheduler": "LTXVScheduler",
        "sampler": "SamplerCustomAdvanced",
        "guider": "CFGGuider",
        "av_separate": "LTXVSeparateAVLatent",
        "vae_decode": "VAEDecodeTiled",
        "audio_decode": "LTXVAudioVAEDecode",
        "output": "SaveVideo",
    },
}


# Selection criteria hints
HINT_PRIORITY = {
    "speed": 10,  # Fast generation priority
    "quality": 9,  # Maximum quality
    "enhanced": 8,  # STG or other enhancements
    "distilled": 7,  # Use distilled model
    "audio": 6,  # Audio output needed
    "guide": 5,  # Video guide provided
    "v2v": 5,  # Video-to-video
    "i2v": 4,  # Image-to-video
    "t2v": 3,  # Text-to-video
    "low_vram": 2,  # VRAM constraints
}


def select_ltx_variant(
    task_type: str,
    hints: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
    available_vram_gb: float = 24.0,
) -> BranchSelection:
    """
    Select the appropriate LTX variant based on task and hints.

    Args:
        task_type: "t2v", "i2v", "v2v", "txt2vid", "img2vid", "vid2vid"
        hints: Optional hints dict (e.g., {"speed": True, "distilled": True})
        profile: Optional profile with preferences
        available_vram_gb: Available VRAM for selection logic

    Returns:
        BranchSelection with selected variant and explain trace
    """
    hints = hints or {}
    profile = profile or {}
    explain_trace = []
    rejected = []

    # Normalize task type
    task_lower = task_type.lower()
    explain_trace.append(
        {
            "step": "task_normalization",
            "input": task_type,
            "normalized": task_lower,
        }
    )

    # Determine base task category
    if task_lower in ["i2v", "img2vid", "image2video", "image-to-video"]:
        base_task = "i2v"
        explain_trace.append({"step": "base_task", "result": "i2v", "reason": "image input detected"})
    elif task_lower in ["v2v", "vid2vid", "video2video", "video-to-video"]:
        base_task = "v2v"
        explain_trace.append({"step": "base_task", "result": "v2v", "reason": "video guide detected"})
    elif task_lower in ["t2v", "txt2vid", "text2video", "text-to-video"]:
        base_task = "t2v"
        explain_trace.append({"step": "base_task", "result": "t2v", "reason": "text-only generation"})
    else:
        base_task = "t2v"  # Default
        explain_trace.append({"step": "base_task", "result": "t2v", "reason": "default fallback"})

    # Score variants based on hints
    variant_scores = {}

    if base_task == "i2v":
        # I2V variant selection
        if hints.get("speed") or hints.get("distilled") or hints.get("fast"):
            variant_scores[LTXVariant.I2V_DISTILLED] = 100
            explain_trace.append(
                {
                    "step": "variant_scoring",
                    "variant": "i2v_distilled",
                    "score": 100,
                    "reason": "speed/distilled hint matched",
                }
            )
            rejected.append(
                {
                    "variant": "i2v_standard",
                    "reason": "rejected: speed/distilled hint prioritizes fast variant",
                }
            )
        else:
            variant_scores[LTXVariant.I2V_STANDARD] = 90
            explain_trace.append(
                {
                    "step": "variant_scoring",
                    "variant": "i2v_standard",
                    "score": 90,
                    "reason": "default quality I2V",
                }
            )
            if not (hints.get("speed") or hints.get("distilled")):
                rejected.append(
                    {
                        "variant": "i2v_distilled",
                        "reason": "rejected: no speed/distilled hint, quality mode preferred",
                    }
                )

    elif base_task == "v2v":
        # V2V is the only option for video-to-video
        variant_scores[LTXVariant.V2V] = 100
        explain_trace.append(
            {
                "step": "variant_scoring",
                "variant": "v2v",
                "score": 100,
                "reason": "video-to-video task requires LTXVAddGuide",
            }
        )

    else:  # T2V
        # T2V variant selection
        if hints.get("enhanced") or hints.get("stg") or hints.get("quality"):
            variant_scores[LTXVariant.T2V_ENHANCED] = 100
            explain_trace.append(
                {
                    "step": "variant_scoring",
                    "variant": "txt2vid_enhanced",
                    "score": 100,
                    "reason": "enhanced/stg/quality hint matched - LTXVApplySTG",
                }
            )
            rejected.extend(
                [
                    {"variant": "txt2vid_distilled", "reason": "rejected: enhanced mode requires full pipeline"},
                    {"variant": "txt2vid", "reason": "rejected: enhanced mode requested"},
                ]
            )
        elif hints.get("speed") or hints.get("distilled") or hints.get("fast"):
            variant_scores[LTXVariant.T2V_DISTILLED] = 100
            explain_trace.append(
                {
                    "step": "variant_scoring",
                    "variant": "txt2vid_distilled",
                    "score": 100,
                    "reason": "speed/distilled hint - SamplerCustom + Gemma",
                }
            )
            rejected.extend(
                [
                    {"variant": "txt2vid_enhanced", "reason": "rejected: speed mode, enhanced too slow"},
                    {"variant": "txt2vid", "reason": "rejected: distilled faster with good quality"},
                ]
            )
        else:
            variant_scores[LTXVariant.T2V_STANDARD] = 80
            explain_trace.append(
                {
                    "step": "variant_scoring",
                    "variant": "txt2vid",
                    "score": 80,
                    "reason": "default T2V with AV pipeline",
                }
            )
            if not hints.get("speed"):
                rejected.append(
                    {
                        "variant": "txt2vid_distilled",
                        "reason": "rejected: default quality mode, distilled available on request",
                    }
                )

    # VRAM-based adjustments
    if available_vram_gb < 16:
        explain_trace.append(
            {
                "step": "vram_constraint",
                "available_vram": available_vram_gb,
                "action": "prefer_distilled_if_available",
            }
        )
        # Boost distilled variants if VRAM constrained
        if LTXVariant.T2V_DISTILLED in variant_scores:
            variant_scores[LTXVariant.T2V_DISTILLED] += 20
        if LTXVariant.I2V_DISTILLED in variant_scores:
            variant_scores[LTXVariant.I2V_DISTILLED] += 20

    # Select highest scoring variant
    selected_variant = max(variant_scores.items(), key=lambda x: x[1])[0]
    explain_trace.append(
        {
            "step": "final_selection",
            "selected": selected_variant.value,
            "all_scores": {k.value: v for k, v in variant_scores.items()},
        }
    )

    # Determine sampler type
    if selected_variant in [LTXVariant.T2V_DISTILLED, LTXVariant.I2V_DISTILLED]:
        sampler_type = LTXSampler.SAMPLER_CUSTOM
        explain_trace.append(
            {
                "step": "sampler_selection",
                "selected": "SamplerCustom",
                "reason": "distilled variants use simplified sampler",
            }
        )
    else:
        sampler_type = LTXSampler.SAMPLER_CUSTOM_ADVANCED
        explain_trace.append(
            {
                "step": "sampler_selection",
                "selected": "SamplerCustomAdvanced",
                "reason": "full pipeline requires advanced sampler with CFGGuider",
            }
        )

    # Get canonical nodes for selected variant
    canonical_nodes = LTX_CANONICAL_NODES.get(selected_variant, {})

    return BranchSelection(
        selected_variant=selected_variant,
        selected_skeleton=selected_variant.value,
        canonical_nodes=canonical_nodes,
        sampler_type=sampler_type,
        explain_trace=explain_trace,
        rejected_alternatives=rejected,
        hints_matched=hints,
    )


def get_ltx_workflow_with_trace(
    task_type: str,
    prompt: str,
    image_path: Optional[str] = None,
    hints: Optional[Dict[str, Any]] = None,
    available_vram_gb: float = 24.0,
) -> Dict[str, Any]:
    """
    Get complete LTX workflow with full explain trace.

    Args:
        task_type: Task type (t2v, i2v, v2v, etc.)
        prompt: Generation prompt
        image_path: Optional image path for I2V/V2V
        hints: Optional selection hints
        available_vram_gb: Available VRAM

    Returns:
        Dict with workflow, selection details, and explain trace
    """
    # Get branch selection
    selection = select_ltx_variant(task_type, hints, available_vram_gb=available_vram_gb)

    # Build result with full trace
    result = {
        "workflow_type": selection.selected_variant.value,
        "sampler": selection.sampler_type.value,
        "canonical_nodes": selection.canonical_nodes,
        "selection_trace": selection.explain_trace,
        "rejected_alternatives": selection.rejected_alternatives,
        "hints_applied": selection.hints_matched,
    }

    # Add task-specific parameters
    if selection.selected_variant in [LTXVariant.I2V_STANDARD, LTXVariant.I2V_DISTILLED]:
        result["requires_image"] = True
        if not image_path:
            result["warning"] = "I2V variant selected but no image_path provided"

    if selection.selected_variant == LTXVariant.V2V:
        result["requires_guide"] = True
        if not image_path:
            result["warning"] = "V2V variant selected but no guide image/video provided"

    return result


def list_available_ltx_variants() -> List[Dict[str, Any]]:
    """List all available LTX variants with their characteristics."""
    return [
        {
            "variant": v.value,
            "canonical_nodes": LTX_CANONICAL_NODES.get(v, {}),
            "use_cases": _get_use_cases(v),
        }
        for v in LTXVariant
    ]


def _get_use_cases(variant: LTXVariant) -> List[str]:
    """Get typical use cases for a variant."""
    use_cases = {
        LTXVariant.T2V_STANDARD: ["text-to-video", "audio generation", "general purpose"],
        LTXVariant.T2V_DISTILLED: ["fast generation", "iterations", "lower VRAM"],
        LTXVariant.T2V_ENHANCED: ["maximum quality", "temporal consistency", "STG"],
        LTXVariant.I2V_STANDARD: ["image-to-video", "animate image", "audio sync"],
        LTXVariant.I2V_DISTILLED: ["fast I2V", "quick iterations"],
        LTXVariant.V2V: ["video-to-video", "style transfer", "guided generation"],
    }
    return use_cases.get(variant, [])


def get_ltx_model_constraints(variant: LTXVariant) -> Dict[str, Any]:
    """Get model constraints for specific LTX variant."""
    base_constraints = {
        "resolution_divisible_by": 32,
        "frame_formula": "8n+1",
        "valid_frame_counts": [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121],
        "default_frames": 97,
        "cfg_range": [1.0, 7.0],
        "default_cfg": 3.0,
    }

    variant_specific = {
        LTXVariant.T2V_DISTILLED: {
            "model_file": "ltx-2-19b-distilled-fp8.safetensors",
            "steps": 10,
            "cfg": 2.5,
        },
        LTXVariant.T2V_ENHANCED: {
            "model_file": "ltx-2-19b-dev-fp8.safetensors",
            "stg_blocks": "14, 19",
            "steps": 30,
        },
        LTXVariant.I2V_DISTILLED: {
            "model_file": "ltx-2-19b-distilled-fp8.safetensors",
            "steps": 10,
            "strength": 40,
        },
    }

    constraints = base_constraints.copy()
    if variant in variant_specific:
        constraints.update(variant_specific[variant])

    return constraints


# Compile report generator
def generate_compile_report(selection: BranchSelection) -> str:
    """Generate human-readable compile report with branch rationale."""
    lines = [
        "=" * 60,
        "LTX Workflow Compile Report",
        "=" * 60,
        "",
        f"Selected Variant: {selection.selected_variant.value}",
        f"Skeleton: {selection.selected_skeleton}",
        f"Sampler: {selection.sampler_type.value}",
        "",
        "Canonical Node Chain:",
        "-" * 40,
    ]

    for node_name, class_type in selection.canonical_nodes.items():
        lines.append(f"  {node_name:20} -> {class_type}")

    lines.extend(
        [
            "",
            "Selection Rationale (Explain Trace):",
            "-" * 40,
        ]
    )

    for entry in selection.explain_trace:
        step = entry.get("step", "unknown")
        if "result" in entry:
            lines.append(f"  [{step}] {entry['result']} - {entry.get('reason', '')}")
        elif "selected" in entry:
            lines.append(f"  [{step}] Selected: {entry['selected']}")
            if "reason" in entry:
                lines.append(f"           Reason: {entry['reason']}")
        else:
            lines.append(f"  [{step}] {json.dumps(entry, indent=2)}")

    if selection.rejected_alternatives:
        lines.extend(
            [
                "",
                "Rejected Alternatives:",
                "-" * 40,
            ]
        )
        for alt in selection.rejected_alternatives:
            lines.append(f"  {alt['variant']}: {alt['reason']}")

    lines.extend(
        [
            "",
            "=" * 60,
        ]
    )

    return "\n".join(lines)


# Advanced branch support for ModifyLTXModel and inversion/interpolation
def get_advanced_ltx_branches() -> Dict[str, Any]:
    """
    Get advanced LTX branch configurations.

    Includes:
    - ModifyLTXModel for model patching
    - Inversion compatible branches
    - Interpolation branches
    """
    return {
        "modify_ltx_model": {
            "node": "ModifyLTXModel",
            "use_cases": ["model patching", "custom schedulers", "experimental"],
            "compatible_variants": [LTXVariant.T2V_STANDARD, LTXVariant.I2V_STANDARD],
            "notes": "Insert between CheckpointLoaderSimple and sampler",
        },
        "inversion_compatible": {
            "node": "LTXInversionSampler",
            "use_cases": ["video inversion", "style transfer", "editing"],
            "compatible_variants": [LTXVariant.T2V_STANDARD],
            "requires": ["inversion_checkpoint"],
        },
        "interpolation": {
            "node": "LTXVInterpolation",
            "use_cases": ["frame interpolation", "smooth transitions"],
            "compatible_variants": [LTXVariant.T2V_STANDARD, LTXVariant.I2V_STANDARD],
            "notes": "Add after VAEDecodeTiled for frame smoothing",
        },
        "audio_reactive": {
            "node": "LTXVAudioReactive",
            "use_cases": ["music visualization", "beat sync"],
            "compatible_variants": [LTXVariant.T2V_STANDARD, LTXVariant.T2V_ENHANCED],
            "requires": ["audio_input"],
        },
    }
