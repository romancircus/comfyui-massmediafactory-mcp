"""
Model Registry - Single Source of Truth for Model Definitions

This module centralizes all model constraints, defaults, and aliases.
All other modules should import from here rather than defining their own model data.

Usage:
    from .model_registry import (
        get_model_constraints,
        get_model_defaults,
        get_canonical_model_key,
        list_supported_models,
        MODEL_CONSTRAINTS,
        MODEL_DEFAULTS,
        MODEL_ALIASES,
    )
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


# =============================================================================
# Model Type Definitions
# =============================================================================


@dataclass
class CFGSpec:
    """CFG/guidance specification for a model."""

    min: float
    max: float
    default: float
    via: Optional[str] = None  # e.g., "FluxGuidance" for FLUX models
    note: str = ""


@dataclass
class ResolutionSpec:
    """Resolution specification for a model."""

    divisible_by: int
    native: List[int]  # [width, height]
    max: Optional[List[int]] = None  # [max_width, max_height]
    min: int = 256
    note: str = ""


@dataclass
class FrameSpec:
    """Frame count specification for video models."""

    default: int
    max: Optional[int] = None
    formula: Optional[str] = None  # e.g., "8n+1" for LTX
    valid_examples: Optional[List[int]] = None
    note: str = ""


@dataclass
class StepsSpec:
    """Sampling steps specification."""

    default: int
    min: int
    max: int


@dataclass
class SchedulerSpec:
    """Scheduler parameters for a model."""

    default: str = "simple"
    max_shift: Optional[float] = None
    base_shift: Optional[float] = None
    stretch: Optional[bool] = None
    note: str = ""


@dataclass
class ModelConstraints:
    """Complete constraint specification for a model."""

    display_name: str
    model_type: str  # "image", "video", "edit"
    cfg: CFGSpec
    resolution: ResolutionSpec
    steps: StepsSpec
    required_nodes: Dict[str, Any] = field(default_factory=dict)
    forbidden_nodes: Dict[str, str] = field(default_factory=dict)
    frames: Optional[FrameSpec] = None
    scheduler: Optional[SchedulerSpec] = None
    shift: Optional[Dict[str, Any]] = None
    denoise: Optional[Dict[str, Any]] = None
    sampler_params: Optional[Dict[str, Any]] = None
    clip_models: Optional[Dict[str, str]] = None
    strengths: Optional[List[str]] = None
    workflow_notes: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backwards compatibility."""
        result = {
            "display_name": self.display_name,
            "type": self.model_type,
            "cfg": {
                "min": self.cfg.min,
                "max": self.cfg.max,
                "default": self.cfg.default,
            },
            "resolution": {
                "divisible_by": self.resolution.divisible_by,
                "native": self.resolution.native,
                "min": self.resolution.min,
            },
            "steps": {
                "default": self.steps.default,
                "min": self.steps.min,
                "max": self.steps.max,
            },
        }

        if self.cfg.via:
            result["cfg"]["via"] = self.cfg.via
        if self.cfg.note:
            result["cfg"]["note"] = self.cfg.note

        if self.resolution.max:
            result["resolution"]["max"] = self.resolution.max
        if self.resolution.note:
            result["resolution"]["note"] = self.resolution.note

        if self.frames:
            result["frames"] = {
                "default": self.frames.default,
            }
            if self.frames.max:
                result["frames"]["max"] = self.frames.max
            if self.frames.formula:
                result["frames"]["formula"] = self.frames.formula
            if self.frames.valid_examples:
                result["frames"]["valid_examples"] = self.frames.valid_examples
            if self.frames.note:
                result["frames"]["note"] = self.frames.note

        if self.required_nodes:
            result["required_nodes"] = self.required_nodes
        if self.forbidden_nodes:
            result["forbidden_nodes"] = self.forbidden_nodes
        if self.scheduler:
            result["scheduler_params"] = {
                "scheduler": self.scheduler.default,
            }
            if self.scheduler.max_shift is not None:
                result["scheduler_params"]["max_shift"] = self.scheduler.max_shift
            if self.scheduler.base_shift is not None:
                result["scheduler_params"]["base_shift"] = self.scheduler.base_shift
            if self.scheduler.stretch is not None:
                result["scheduler_params"]["stretch"] = self.scheduler.stretch
        if self.shift:
            result["shift"] = self.shift
        if self.denoise:
            result["denoise"] = self.denoise
        if self.sampler_params:
            result["sampler_params"] = self.sampler_params
        if self.clip_models:
            result["clip_models"] = self.clip_models
        if self.strengths:
            result["strengths"] = self.strengths
        if self.workflow_notes:
            result["workflow_notes"] = self.workflow_notes

        return result


# =============================================================================
# Model Constraints Registry
# =============================================================================

_MODEL_REGISTRY: Dict[str, ModelConstraints] = {
    "ltx2": ModelConstraints(
        display_name="LTX-Video 2.0",
        model_type="video",
        cfg=CFGSpec(
            min=2.5, max=4.0, default=3.0, note="LTX-2 is optimized for low CFG. Higher values cause artifacts."
        ),
        resolution=ResolutionSpec(
            divisible_by=8, native=[768, 512], max=[1920, 1088], note="Width and height must be divisible by 8"
        ),
        frames=FrameSpec(
            default=97,
            formula="8n+1",
            valid_examples=[9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121],
            note="Frame count must be 8n+1 format",
        ),
        steps=StepsSpec(default=30, min=25, max=35),
        scheduler=SchedulerSpec(default="euler", max_shift=2.05, base_shift=0.95, stretch=True),
        required_nodes={
            "loader": "LTXVLoader",
            "sampler": "SamplerCustom",
            "scheduler": "LTXVScheduler",
            "conditioning_wrapper": "LTXVConditioning",
            "latent": "EmptyLTXVLatentVideo",
            "output": "VHS_VideoCombine",
        },
        forbidden_nodes={
            "KSampler": "Use SamplerCustom with LTXVScheduler instead",
            "CheckpointLoaderSimple": "Use LTXVLoader for LTX-2 models",
            "EmptyLatentImage": "Use EmptyLTXVLatentVideo for video",
            "SaveImage": "Use VHS_VideoCombine for video output",
        },
    ),
    "flux2": ModelConstraints(
        display_name="FLUX.2",
        model_type="image",
        cfg=CFGSpec(
            min=2.5,
            max=5.0,
            default=3.5,
            via="FluxGuidance",
            note="FLUX uses FluxGuidance node instead of cfg parameter",
        ),
        resolution=ResolutionSpec(
            divisible_by=16, native=[1024, 1024], max=[2048, 2048], note="Width and height must be divisible by 16"
        ),
        steps=StepsSpec(default=20, min=15, max=50),
        scheduler=SchedulerSpec(default="simple", note="Use 'simple' scheduler"),
        required_nodes={
            "loader": ["UNETLoader", "DualCLIPLoader", "VAELoader"],
            "sampler": "SamplerCustomAdvanced",
            "scheduler": "BasicScheduler",
            "conditioning_wrapper": "FluxGuidance",
            "guider": "BasicGuider",
            "noise": "RandomNoise",
            "latent": "EmptySD3LatentImage",
            "output": "SaveImage",
        },
        forbidden_nodes={
            "KSampler": "Use SamplerCustomAdvanced with BasicGuider",
            "CheckpointLoaderSimple": "Use UNETLoader + DualCLIPLoader + VAELoader",
            "EmptyLatentImage": "Use EmptySD3LatentImage for FLUX",
        },
        clip_models={"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp16.safetensors", "type": "flux"},
    ),
    "wan26": ModelConstraints(
        display_name="Wan 2.1/2.2 I2V (WanVideoWrapper)",
        model_type="video",
        cfg=CFGSpec(min=4.0, max=7.0, default=5.0, note="Wan uses standard CFG in WanVideoSampler"),
        resolution=ResolutionSpec(
            divisible_by=8, native=[832, 480], note="Use 480p for faster generation, 720p for quality"
        ),
        frames=FrameSpec(default=81, max=121, note="81 frames is ~5s at 16fps"),
        steps=StepsSpec(default=30, min=20, max=50),
        sampler_params={"shift": 5.0, "scheduler": "unipc"},
        required_nodes={
            "loader": "WanVideoModelLoader",
            "vae": "WanVideoVAELoader",
            "text_encoder": "LoadWanVideoT5TextEncoder",
            "clip_vision": "LoadWanVideoClipTextEncoder",
            "text_encode": "WanVideoTextEncode",
            "clip_encode": "WanVideoClipVisionEncode",
            "image_encode": "WanVideoImageToVideoEncode",
            "sampler": "WanVideoSampler",
            "decoder": "WanVideoDecode",
            "output": "SaveVideo",
        },
        forbidden_nodes={
            "CheckpointLoaderSimple": "Use WanVideoModelLoader",
            "KSampler": "Use WanVideoSampler",
            "DownloadAndLoadWanModel": "Non-existent node, use WanVideoModelLoader",
            "VHS_VideoCombine": "Not installed, use CreateVideo + SaveVideo",
        },
    ),
    "qwen": ModelConstraints(
        display_name="Qwen Image",
        model_type="image",
        cfg=CFGSpec(min=3.0, max=5.0, default=3.5, note="Qwen works best with low CFG (3.0-4.0)"),
        resolution=ResolutionSpec(
            divisible_by=8, native=[1296, 1296], note="1296x1296 is native, good for text rendering"
        ),
        steps=StepsSpec(default=50, min=35, max=60),
        shift={
            "min": 7.0,
            "max": 13.0,
            "default": 7.0,
            "note": "CRITICAL: shift=3.1 causes blurry output. Use 7.0 for sharp, 12-13 for maximum sharpness",
        },
        scheduler=SchedulerSpec(default="simple", note="Use 'simple' scheduler, not 'normal'"),
        required_nodes={
            "loader": "UNETLoader",
            "clip": "CLIPLoader",
            "vae": "VAELoader",
            "shift_node": "ModelSamplingAuraFlow",
            "sampler": "KSampler",
            "latent": "EmptySD3LatentImage",
            "output": "SaveImage",
        },
        strengths=[
            "Text rendering",
            "Posters and logos",
            "Complex layouts",
            "UI design mockups",
            "Photorealistic portraits",
        ],
    ),
    "sdxl": ModelConstraints(
        display_name="SDXL",
        model_type="image",
        cfg=CFGSpec(min=5.0, max=10.0, default=7.0, note="SDXL works best with CFG 6-8"),
        resolution=ResolutionSpec(divisible_by=8, native=[1024, 1024], note="Use SDXL-optimized aspect ratios"),
        steps=StepsSpec(default=25, min=15, max=50),
        required_nodes={
            "loader": "CheckpointLoaderSimple",
            "sampler": "KSampler",
            "latent": "EmptyLatentImage",
            "output": "SaveImage",
        },
    ),
    "hunyuan15": ModelConstraints(
        display_name="HunyuanVideo 1.5",
        model_type="video",
        cfg=CFGSpec(min=4.0, max=8.0, default=6.0),
        resolution=ResolutionSpec(divisible_by=16, native=[1280, 720], note="720p native, supports up to 1080p"),
        frames=FrameSpec(default=81, max=129, note="129 frames is ~5 seconds at 24fps"),
        steps=StepsSpec(default=30, min=20, max=50),
        required_nodes={
            "loader": "HunyuanVideoModelLoader",
            "sampler": "HunyuanVideoSampler",
            "output": "VHS_VideoCombine",
        },
    ),
    "qwen_edit": ModelConstraints(
        display_name="Qwen Image Edit 2511",
        model_type="edit",
        cfg=CFGSpec(
            min=1.5,
            max=3.0,
            default=2.0,
            note="CFG is PRIMARY color control. >4 causes oversaturation/color distortion. Keep low (2.0-2.5).",
        ),
        resolution=ResolutionSpec(
            divisible_by=8, native=[720, 1280], note="Supports arbitrary resolutions divisible by 8"
        ),
        steps=StepsSpec(default=20, min=15, max=30),
        denoise={
            "default": 1.0,
            "note": "MUST be 1.0 for background replacement. Lower values preserve original latent including background.",
        },
        required_nodes={
            "loader": ["UNETLoader", "CLIPLoader", "VAELoader"],
            "text_encoder": "TextEncodeQwenImageEditPlus",
            "latent": "EmptyQwenImageLayeredLatentImage",
            "sampler": "KSampler",
            "output": "SaveImage",
        },
        forbidden_nodes={
            "VAEEncode": "Do NOT use VAEEncode for background replacement. Use EmptyQwenImageLayeredLatentImage instead.",
            "TextEncodeQwenImageEdit": "Use TextEncodeQwenImageEditPlus for better instruction following",
        },
        workflow_notes={
            "background_replacement": "Pass original image to TextEncodeQwenImageEditPlus image1 input. Model regenerates while matching reference.",
            "layers": "Use layers=0 in EmptyQwenImageLayeredLatentImage for better character preservation.",
        },
    ),
    "z_turbo": ModelConstraints(
        display_name="Z-Image-Turbo",
        model_type="image",
        cfg=CFGSpec(
            min=2.0, max=5.0, default=3.0, note="Z-Turbo works best with low CFG. Higher values can cause artifacts."
        ),
        resolution=ResolutionSpec(
            divisible_by=8, native=[1024, 1024], max=[2048, 2048], note="Width and height must be divisible by 8"
        ),
        steps=StepsSpec(default=4, min=1, max=10),
        scheduler=SchedulerSpec(default="simple", note="Fast scheduler for turbo mode"),
        required_nodes={
            "loader": "CheckpointLoaderSimple",
            "sampler": "KSampler",
            "latent": "EmptyLatentImage",
            "output": "SaveImage",
        },
        workflow_notes={
            "speed": "4-step generation for ultra-fast inference",
            "quality": "Slightly lower quality than 20-step models but 5x faster",
        },
    ),
    "cogvideox_5b": ModelConstraints(
        display_name="CogVideoX-5B",
        model_type="video",
        cfg=CFGSpec(min=4.0, max=8.0, default=6.0, note="CogVideoX works well with standard CFG 5-7"),
        resolution=ResolutionSpec(
            divisible_by=16, native=[720, 480], max=[1280, 720], note="Width and height must be divisible by 16"
        ),
        frames=FrameSpec(
            default=49, max=81, valid_examples=[17, 33, 49, 65, 81], note="49 frames is ~2 seconds at 24fps"
        ),
        steps=StepsSpec(default=50, min=30, max=100),
        required_nodes={
            "loader": "CogVideoLoader",
            "sampler": "CogVideoSampler",
            "latent": "EmptyCogVideoLatent",
            "output": "VHS_VideoCombine",
        },
        workflow_notes={
            "quality": "High-quality video generation from Tsinghua",
            "speed": "Requires ~12GB VRAM for 5B model",
        },
    ),
}


# =============================================================================
# Model Aliases - Maps shorthand names to canonical keys
# =============================================================================

MODEL_ALIASES: Dict[str, str] = {
    # LTX aliases
    "ltx": "ltx2",
    "ltx-2": "ltx2",
    "ltxv": "ltx2",
    "ltx_video": "ltx2",
    # FLUX aliases
    "flux": "flux2",
    "flux.2": "flux2",
    "flux-2": "flux2",
    "flux.2-dev": "flux2",
    # Wan aliases
    "wan": "wan26",
    "wan2.6": "wan26",
    "wan-2.6": "wan26",
    # HunyuanVideo aliases
    "hunyuan": "hunyuan15",
    "hunyuan1.5": "hunyuan15",
    "hunyuan-1.5": "hunyuan15",
    "hunyuanvideo": "hunyuan15",
    # Qwen aliases
    "qwen_image": "qwen",
    "qwen-image": "qwen",
    "qwen2512": "qwen",
    # Qwen Edit aliases
    "qwen-edit": "qwen_edit",
    "qwenedit": "qwen_edit",
    # Z-Image-Turbo aliases
    "z-turbo": "z_turbo",
    "zturbo": "z_turbo",
    "z_image_turbo": "z_turbo",
    # CogVideoX aliases
    "cogvideo": "cogvideox_5b",
    "cogvideo-5b": "cogvideox_5b",
    "cogvideox": "cogvideox_5b",
}


# =============================================================================
# Workflow Type Aliases - Maps shorthand types to canonical types
# =============================================================================

WORKFLOW_TYPE_ALIASES: Dict[str, str] = {
    # Text-to-video
    "t2v": "txt2vid",
    "text-to-video": "txt2vid",
    "text2video": "txt2vid",
    # Image-to-video
    "i2v": "img2vid",
    "image-to-video": "img2vid",
    "image2video": "img2vid",
    # Text-to-image
    "t2i": "txt2img",
    "text-to-image": "txt2img",
    "text2image": "txt2img",
    # Edit
    "image-edit": "edit",
    "img-edit": "edit",
}


# =============================================================================
# Model-to-Skeleton Mapping - Maps (model, type) to skeleton keys
# =============================================================================

MODEL_SKELETON_MAP: Dict[Tuple[str, str], Tuple[str, str]] = {
    # LTX-2 Text-to-Video
    ("ltx", "t2v"): ("ltx2", "txt2vid"),
    ("ltx", "txt2vid"): ("ltx2", "txt2vid"),
    ("ltx", "text-to-video"): ("ltx2", "txt2vid"),
    ("ltx2", "t2v"): ("ltx2", "txt2vid"),
    ("ltx2", "txt2vid"): ("ltx2", "txt2vid"),
    # LTX-2 Image-to-Video
    ("ltx", "i2v"): ("ltx2", "img2vid"),
    ("ltx", "img2vid"): ("ltx2", "img2vid"),
    ("ltx", "image-to-video"): ("ltx2", "img2vid"),
    ("ltx2", "i2v"): ("ltx2", "img2vid"),
    ("ltx2", "img2vid"): ("ltx2", "img2vid"),
    # FLUX.2 Text-to-Image
    ("flux", "t2i"): ("flux2", "txt2img"),
    ("flux", "txt2img"): ("flux2", "txt2img"),
    ("flux", "text-to-image"): ("flux2", "txt2img"),
    ("flux2", "t2i"): ("flux2", "txt2img"),
    ("flux2", "txt2img"): ("flux2", "txt2img"),
    # Wan 2.6 Text-to-Video
    ("wan", "t2v"): ("wan26", "txt2vid"),
    ("wan", "txt2vid"): ("wan26", "txt2vid"),
    ("wan26", "t2v"): ("wan26", "txt2vid"),
    ("wan26", "txt2vid"): ("wan26", "txt2vid"),
    # Wan 2.6 Image-to-Video
    ("wan", "i2v"): ("wan26", "img2vid"),
    ("wan", "img2vid"): ("wan26", "img2vid"),
    ("wan26", "i2v"): ("wan26", "img2vid"),
    ("wan26", "img2vid"): ("wan26", "img2vid"),
    # Qwen Text-to-Image
    ("qwen", "t2i"): ("qwen", "txt2img"),
    ("qwen", "txt2img"): ("qwen", "txt2img"),
    # SDXL Text-to-Image
    ("sdxl", "t2i"): ("sdxl", "txt2img"),
    ("sdxl", "txt2img"): ("sdxl", "txt2img"),
    # HunyuanVideo 1.5 Text-to-Video
    ("hunyuan", "t2v"): ("hunyuan15", "txt2vid"),
    ("hunyuan", "txt2vid"): ("hunyuan15", "txt2vid"),
    ("hunyuan15", "t2v"): ("hunyuan15", "txt2vid"),
    ("hunyuan15", "txt2vid"): ("hunyuan15", "txt2vid"),
    # HunyuanVideo 1.5 Image-to-Video
    ("hunyuan", "i2v"): ("hunyuan15", "img2vid"),
    ("hunyuan", "img2vid"): ("hunyuan15", "img2vid"),
    ("hunyuan15", "i2v"): ("hunyuan15", "img2vid"),
    ("hunyuan15", "img2vid"): ("hunyuan15", "img2vid"),
    # Z-Image-Turbo Text-to-Image
    ("z_turbo", "t2i"): ("z_turbo", "txt2img"),
    ("z_turbo", "txt2img"): ("z_turbo", "txt2img"),
    ("z-turbo", "t2i"): ("z_turbo", "txt2img"),
    ("zturbo", "t2i"): ("z_turbo", "txt2img"),
    # CogVideoX-5B Text-to-Video
    ("cogvideox_5b", "t2v"): ("cogvideox_5b", "txt2vid"),
    ("cogvideox_5b", "txt2vid"): ("cogvideox_5b", "txt2vid"),
    ("cogvideo", "t2v"): ("cogvideox_5b", "txt2vid"),
    ("cogvideo-5b", "t2v"): ("cogvideox_5b", "txt2vid"),
    ("cogvideox", "t2v"): ("cogvideox_5b", "txt2vid"),
}


# =============================================================================
# Default Generation Parameters by Model
# =============================================================================

MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "ltx2": {
        "width": 768,
        "height": 512,
        "frames": 97,
        "steps": 30,
        "cfg": 3.0,
        "fps": 24,
    },
    "flux2": {
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "guidance": 3.5,
    },
    "wan26": {
        "width": 832,
        "height": 480,
        "frames": 81,
        "steps": 30,
        "cfg": 5.0,
        "fps": 24,
    },
    "qwen": {
        "width": 1296,
        "height": 1296,
        "steps": 50,
        "cfg": 3.5,
        "shift": 7.0,
    },
    "sdxl": {
        "width": 1024,
        "height": 1024,
        "steps": 25,
        "cfg": 7.0,
    },
    "hunyuan15": {
        "width": 1280,
        "height": 720,
        "frames": 81,
        "steps": 30,
        "cfg": 6.0,
        "fps": 24,
    },
    "qwen_edit": {
        "width": 720,
        "height": 1280,
        "steps": 20,
        "cfg": 2.0,
        "denoise": 1.0,
    },
    "z_turbo": {
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "cfg": 3.0,
    },
    "cogvideox_5b": {
        "width": 720,
        "height": 480,
        "frames": 49,
        "steps": 50,
        "cfg": 6.0,
        "fps": 24,
    },
}

# Add aliases to defaults
for alias, canonical in MODEL_ALIASES.items():
    if canonical in MODEL_DEFAULTS and alias not in MODEL_DEFAULTS:
        MODEL_DEFAULTS[alias] = MODEL_DEFAULTS[canonical]


# =============================================================================
# Resolution Specs for Validation (backwards compatibility)
# =============================================================================

MODEL_RESOLUTION_SPECS: Dict[str, Dict[str, Any]] = {}
for model_key, constraints in _MODEL_REGISTRY.items():
    MODEL_RESOLUTION_SPECS[model_key] = {
        "native": constraints.resolution.native[0] if constraints.resolution.native else 1024,
        "divisible_by": constraints.resolution.divisible_by,
        "min": constraints.resolution.min,
        "max": constraints.resolution.max[0] if constraints.resolution.max else 2048,
    }


# =============================================================================
# Public API
# =============================================================================


def resolve_model_name(model: str) -> str:
    """
    Resolve a model name or alias to its canonical name.

    Args:
        model: Model name or alias (e.g., "ltx", "flux", "wan26")

    Returns:
        Canonical model name (e.g., "ltx2", "flux2", "wan26")
    """
    model_lower = model.lower()
    return MODEL_ALIASES.get(model_lower, model_lower)


def resolve_workflow_type(workflow_type: str) -> str:
    """
    Resolve a workflow type or alias to its canonical type.

    Args:
        workflow_type: Workflow type or alias (e.g., "t2v", "i2v", "t2i")

    Returns:
        Canonical workflow type (e.g., "txt2vid", "img2vid", "txt2img")
    """
    type_lower = workflow_type.lower()
    return WORKFLOW_TYPE_ALIASES.get(type_lower, type_lower)


def get_canonical_model_key(model: str, workflow_type: str) -> Optional[Tuple[str, str]]:
    """
    Get the canonical (model, type) key for skeleton lookup.

    Args:
        model: Model name or alias
        workflow_type: Workflow type or alias

    Returns:
        Tuple of (canonical_model, canonical_type) or None if not found
    """
    model_lower = model.lower()
    type_lower = workflow_type.lower()

    # Try direct lookup first
    key = (model_lower, type_lower)
    if key in MODEL_SKELETON_MAP:
        return MODEL_SKELETON_MAP[key]

    # Try with resolved model name
    resolved_model = resolve_model_name(model_lower)
    key = (resolved_model, type_lower)
    if key in MODEL_SKELETON_MAP:
        return MODEL_SKELETON_MAP[key]

    # Try with resolved workflow type
    resolved_type = resolve_workflow_type(type_lower)
    key = (model_lower, resolved_type)
    if key in MODEL_SKELETON_MAP:
        return MODEL_SKELETON_MAP[key]

    # Try with both resolved
    key = (resolved_model, resolved_type)
    if key in MODEL_SKELETON_MAP:
        return MODEL_SKELETON_MAP[key]

    return None


def get_model_constraints(model: str) -> Dict[str, Any]:
    """
    Get constraints for a model.

    Args:
        model: Model name or alias

    Returns:
        Constraints dict or error dict if model not found
    """
    canonical = resolve_model_name(model)

    if canonical not in _MODEL_REGISTRY:
        return {"error": f"No constraints for model '{model}'", "available": list(_MODEL_REGISTRY.keys())}

    return _MODEL_REGISTRY[canonical].to_dict()


def get_model_constraints_object(model: str) -> Optional[ModelConstraints]:
    """
    Get constraints object for a model (for internal use).

    Args:
        model: Model name or alias

    Returns:
        ModelConstraints object or None if not found
    """
    canonical = resolve_model_name(model)
    return _MODEL_REGISTRY.get(canonical)


def get_model_defaults(model: str) -> Dict[str, Any]:
    """
    Get default generation parameters for a model.

    Args:
        model: Model name or alias

    Returns:
        Dict of default parameters or empty dict if model not found
    """
    canonical = resolve_model_name(model)
    return MODEL_DEFAULTS.get(canonical, {})


def list_supported_models() -> List[str]:
    """
    Get list of all supported canonical model names.

    Returns:
        List of model names
    """
    return list(_MODEL_REGISTRY.keys())


def list_model_aliases() -> Dict[str, str]:
    """
    Get mapping of aliases to canonical names.

    Returns:
        Dict mapping alias -> canonical name
    """
    return MODEL_ALIASES.copy()


def is_video_model(model: str) -> bool:
    """
    Check if a model is a video generation model.

    Args:
        model: Model name or alias

    Returns:
        True if video model, False otherwise
    """
    canonical = resolve_model_name(model)
    constraints = _MODEL_REGISTRY.get(canonical)
    return constraints is not None and constraints.model_type == "video"


def get_resolution_spec(model: str) -> Optional[Dict[str, Any]]:
    """
    Get resolution specification for a model.

    Args:
        model: Model name or alias

    Returns:
        Resolution spec dict or None if model not found
    """
    canonical = resolve_model_name(model)
    return MODEL_RESOLUTION_SPECS.get(canonical)


def validate_model_exists(model: str) -> Tuple[bool, str]:
    """
    Check if a model exists in the registry.

    Args:
        model: Model name or alias

    Returns:
        Tuple of (exists, canonical_name_or_error_message)
    """
    canonical = resolve_model_name(model)
    if canonical in _MODEL_REGISTRY:
        return True, canonical
    return False, f"Unknown model '{model}'. Available: {list(_MODEL_REGISTRY.keys())}"


# =============================================================================
# Backwards Compatibility - Export as dict for existing code
# =============================================================================

MODEL_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    key: constraints.to_dict() for key, constraints in _MODEL_REGISTRY.items()
}
