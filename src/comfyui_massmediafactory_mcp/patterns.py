"""
Workflow Patterns Module

Provides exact working workflow patterns that prevent Claude from drifting
when constructing ComfyUI workflows.

Key tools:
- get_workflow_skeleton(): Returns exact working workflow JSON
- get_model_constraints(): Returns hard constraints per model
- get_node_chain(): Returns ordered nodes with connections
- validate_against_pattern(): Detects drift from working patterns

NOTE: Model constraints are now centralized in model_registry.py.
This module imports from there for backwards compatibility.
"""

import copy
import time
from typing import Dict, Any, List, Optional

# Import model constraints from centralized registry
from .model_registry import (
    MODEL_CONSTRAINTS,
    get_model_constraints as _get_registry_constraints,
)

# =============================================================================
# Skeleton Caching System
# =============================================================================

_SKELETON_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes
_CACHE_HITS = 0
_CACHE_MISSES = 0


def _get_cache_key(model: str, task: str) -> str:
    """Generate cache key for model/task combination."""
    return f"{model.lower()}:{task.lower()}"


def _get_skeleton_from_cache(model: str, task: str) -> Optional[Dict[str, Any]]:
    """
    Get skeleton from cache if valid.

    Returns cached skeleton if it exists and hasn't expired.
    Updates cache metrics for monitoring.
    """
    global _CACHE_HITS, _CACHE_MISSES

    cache_key = _get_cache_key(model, task)
    cached = _SKELETON_CACHE.get(cache_key)

    if cached is None:
        _CACHE_MISSES += 1
        return None

    # Check TTL
    if time.time() - cached["loaded_at"] > _CACHE_TTL_SECONDS:
        # Expired - remove from cache
        del _SKELETON_CACHE[cache_key]
        _CACHE_MISSES += 1
        return None

    _CACHE_HITS += 1
    return copy.deepcopy(cached["data"])


def _store_skeleton_in_cache(model: str, task: str, data: Dict[str, Any]) -> None:
    """Store skeleton in cache with timestamp."""
    cache_key = _get_cache_key(model, task)
    _SKELETON_CACHE[cache_key] = {
        "data": data,
        "loaded_at": time.time(),
    }


def get_cache_stats() -> Dict[str, Any]:
    """
    Get skeleton cache statistics.

    Returns:
        {
            "hits": int,
            "misses": int,
            "hit_rate": float,
            "entries": int,
            "ttl_seconds": int,
        }
    """
    total = _CACHE_HITS + _CACHE_MISSES
    hit_rate = _CACHE_HITS / total if total > 0 else 0.0

    return {
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "hit_rate": round(hit_rate, 4),
        "entries": len(_SKELETON_CACHE),
        "ttl_seconds": _CACHE_TTL_SECONDS,
    }


def clear_skeleton_cache() -> None:
    """Clear all cached skeletons. Useful for testing or memory management."""
    global _CACHE_HITS, _CACHE_MISSES
    _SKELETON_CACHE.clear()
    _CACHE_HITS = 0
    _CACHE_MISSES = 0


# =============================================================================
# Workflow Skeletons
# =============================================================================

WORKFLOW_SKELETONS = {
    ("ltx2", "txt2vid"): {
        "_meta": {
            "description": "LTX-2 text-to-video AV pipeline (full quality)",
            "model": "LTX-2 19B Dev",
            "type": "txt2vid",
            "parameters": ["PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "FRAMES", "FPS", "STEPS", "CFG", "PREFIX"],
            "defaults": {
                "WIDTH": 768,
                "HEIGHT": 512,
                "SEED": 42,
                "FRAMES": 97,
                "FPS": 24,
                "STEPS": 30,
                "CFG": 3.0,
                "PREFIX": "ltx2_output",
                "NEGATIVE": "worst quality, inconsistent motion, blurry, jittery, distorted",
            },
        },
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load LTX-2 Model"},
            "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"},
        },
        "2": {
            "class_type": "LTXAVTextEncoderLoader",
            "_meta": {"title": "Load Gemma-3 Text Encoder"},
            "inputs": {
                "text_encoder": "gemma_3_12B_it.safetensors",
                "ckpt_name": "ltx-2-19b-dev-fp8.safetensors",
                "device": "default",
            },
        },
        "3": {
            "class_type": "LTXVAudioVAELoader",
            "_meta": {"title": "Load Audio VAE"},
            "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Positive Prompt"},
            "inputs": {"text": "{{PROMPT}}", "clip": ["2", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Negative Prompt"},
            "inputs": {"text": "{{NEGATIVE}}", "clip": ["2", 0]},
        },
        "6": {
            "class_type": "LTXVConditioning",
            "_meta": {"title": "Apply Frame Rate Conditioning"},
            "inputs": {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": "{{FPS}}"},
        },
        "7": {
            "class_type": "EmptyLTXVLatentVideo",
            "_meta": {"title": "Create Empty Video Latent"},
            "inputs": {"width": "{{WIDTH}}", "height": "{{HEIGHT}}", "length": "{{FRAMES}}", "batch_size": 1},
        },
        "8": {
            "class_type": "LTXVEmptyLatentAudio",
            "_meta": {"title": "Create Empty Audio Latent"},
            "inputs": {"audio_vae": ["3", 0], "frames_number": "{{FRAMES}}", "frame_rate": "{{FPS}}", "batch_size": 1},
        },
        "9": {
            "class_type": "LTXVConcatAVLatent",
            "_meta": {"title": "Concat Audio+Video Latents"},
            "inputs": {"video_latent": ["7", 0], "audio_latent": ["8", 0]},
        },
        "10": {
            "class_type": "LTXVScheduler",
            "_meta": {"title": "LTX Scheduler"},
            "inputs": {
                "steps": "{{STEPS}}",
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["9", 0],
            },
        },
        "11": {
            "class_type": "RandomNoise",
            "_meta": {"title": "Random Noise"},
            "inputs": {"noise_seed": "{{SEED}}"},
        },
        "12": {
            "class_type": "KSamplerSelect",
            "_meta": {"title": "Select Sampler"},
            "inputs": {"sampler_name": "euler"},
        },
        "13": {
            "class_type": "CFGGuider",
            "_meta": {"title": "CFG Guider"},
            "inputs": {"model": ["1", 0], "positive": ["6", 0], "negative": ["6", 1], "cfg": "{{CFG}}"},
        },
        "14": {
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": "Sample Video"},
            "inputs": {
                "noise": ["11", 0],
                "guider": ["13", 0],
                "sampler": ["12", 0],
                "sigmas": ["10", 0],
                "latent_image": ["9", 0],
            },
        },
        "15": {
            "class_type": "LTXVSeparateAVLatent",
            "_meta": {"title": "Separate Audio+Video"},
            "inputs": {"av_latent": ["14", 0]},
        },
        "16": {
            "class_type": "VAEDecodeTiled",
            "_meta": {"title": "Decode Video Latent"},
            "inputs": {
                "samples": ["15", 0],
                "vae": ["1", 2],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
        },
        "17": {
            "class_type": "LTXVAudioVAEDecode",
            "_meta": {"title": "Decode Audio Latent"},
            "inputs": {"samples": ["15", 1], "audio_vae": ["3", 0]},
        },
        "18": {
            "class_type": "CreateVideo",
            "_meta": {"title": "Create Video"},
            "inputs": {"images": ["16", 0], "audio": ["17", 0], "fps": "{{FPS}}"},
        },
        "19": {
            "class_type": "SaveVideo",
            "_meta": {"title": "Save Video"},
            "inputs": {"video": ["18", 0], "filename_prefix": "{{PREFIX}}", "format": "mp4", "codec": "auto"},
        },
    },
    ("ltx2", "txt2vid_distilled"): {
        "_meta": {
            "description": "LTX-2 text-to-video (distilled, 3-4x faster)",
            "model": "LTX-2 19B Distilled",
            "type": "txt2vid",
            "parameters": ["PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "FRAMES", "FPS", "STEPS"],
            "defaults": {
                "WIDTH": 768,
                "HEIGHT": 512,
                "SEED": 42,
                "FRAMES": 97,
                "FPS": 24,
                "STEPS": 10,
                "NEGATIVE": "worst quality, inconsistent motion, blurry, jittery, distorted",
            },
        },
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load LTX-2 Distilled Model"},
            "inputs": {"ckpt_name": "ltx-2-19b-distilled-fp8.safetensors"},
        },
        "2": {
            "class_type": "LTXVGemmaCLIPModelLoader",
            "_meta": {"title": "Load Gemma-3 Text Encoder"},
            "inputs": {
                "gemma_path": "gemma_3_12B_it_fp8_e4m3fn.safetensors",
                "ltxv_path": "ltx-2-19b-distilled-fp8.safetensors",
                "max_length": 1024,
            },
        },
        "3": {
            "class_type": "LTXVGemmaEnhancePrompt",
            "_meta": {"title": "Enhance Prompt with AI"},
            "inputs": {
                "clip": ["2", 0],
                "prompt": "{{PROMPT}}",
                "max_tokens": 512,
                "bypass_i2v": False,
                "seed": "{{SEED}}",
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Enhanced Prompt"},
            "inputs": {"clip": ["2", 0], "text": ["3", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Negative Prompt"},
            "inputs": {"clip": ["2", 0], "text": "{{NEGATIVE}}"},
        },
        "6": {
            "class_type": "LTXVConditioning",
            "_meta": {"title": "Apply Frame Rate Conditioning"},
            "inputs": {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": "{{FPS}}"},
        },
        "7": {
            "class_type": "EmptyLTXVLatentVideo",
            "_meta": {"title": "Create Empty Latent"},
            "inputs": {"width": "{{WIDTH}}", "height": "{{HEIGHT}}", "length": "{{FRAMES}}", "batch_size": 1},
        },
        "8": {
            "class_type": "LTXVScheduler",
            "_meta": {"title": "LTX Scheduler (Distilled)"},
            "inputs": {"steps": "{{STEPS}}", "max_shift": 2.05, "base_shift": 0.95, "stretch": True},
        },
        "9": {
            "class_type": "KSamplerSelect",
            "_meta": {"title": "Select Sampler"},
            "inputs": {"sampler_name": "euler"},
        },
        "10": {
            "class_type": "SamplerCustom",
            "_meta": {"title": "Sample Video"},
            "inputs": {
                "model": ["1", 0],
                "add_noise": True,
                "noise_seed": "{{SEED}}",
                "cfg": 2.5,
                "positive": ["6", 0],
                "negative": ["6", 1],
                "sampler": ["9", 0],
                "sigmas": ["8", 0],
                "latent_image": ["7", 0],
            },
        },
        "11": {
            "class_type": "VAEDecode",
            "_meta": {"title": "Decode Latent to Video"},
            "inputs": {"samples": ["10", 0], "vae": ["1", 2]},
        },
        "12": {
            "class_type": "CreateVideo",
            "_meta": {"title": "Create Video"},
            "inputs": {"images": ["11", 0], "fps": "{{FPS}}"},
        },
        "13": {
            "class_type": "SaveVideo",
            "_meta": {"title": "Save Video"},
            "inputs": {"video": ["12", 0], "filename_prefix": "ltx2_distilled", "format": "mp4", "codec": "auto"},
        },
    },
    ("ltx2", "img2vid"): {
        "_meta": {
            "description": "LTX-2 image-to-video AV pipeline",
            "model": "LTX-2 19B Dev",
            "type": "img2vid",
            "parameters": [
                "IMAGE_PATH",
                "PROMPT",
                "NEGATIVE",
                "SEED",
                "WIDTH",
                "HEIGHT",
                "FRAMES",
                "FPS",
                "STEPS",
                "CFG",
                "STRENGTH",
                "CRF",
                "BLUR_RADIUS",
                "PREFIX",
            ],
            "defaults": {
                "WIDTH": 768,
                "HEIGHT": 512,
                "SEED": 42,
                "FRAMES": 97,
                "FPS": 24,
                "STEPS": 30,
                "CFG": 3.0,
                "STRENGTH": 0.85,
                "CRF": 38,
                "BLUR_RADIUS": 1,
                "PREFIX": "ltx2_i2v",
                "NEGATIVE": "worst quality, inconsistent motion, blurry, jittery, distorted",
            },
        },
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load LTX-2 Model"},
            "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"},
        },
        "2": {
            "class_type": "LTXAVTextEncoderLoader",
            "_meta": {"title": "Load Gemma-3 Text Encoder"},
            "inputs": {
                "text_encoder": "gemma_3_12B_it.safetensors",
                "ckpt_name": "ltx-2-19b-dev-fp8.safetensors",
                "device": "default",
            },
        },
        "3": {
            "class_type": "LTXVAudioVAELoader",
            "_meta": {"title": "Load Audio VAE"},
            "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Positive Prompt"},
            "inputs": {"text": "{{PROMPT}}", "clip": ["2", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Negative Prompt"},
            "inputs": {"text": "{{NEGATIVE}}", "clip": ["2", 0]},
        },
        "6": {
            "class_type": "LTXVConditioning",
            "_meta": {"title": "Apply Frame Rate Conditioning"},
            "inputs": {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": "{{FPS}}"},
        },
        "20": {
            "class_type": "LoadImage",
            "_meta": {"title": "Load Input Image"},
            "inputs": {"image": "{{IMAGE_PATH}}"},
        },
        "21": {
            "class_type": "LTXVImgToVideoAdvanced",
            "_meta": {"title": "Create I2V Latent"},
            "inputs": {
                "positive": ["6", 0],
                "negative": ["6", 1],
                "vae": ["1", 2],
                "image": ["20", 0],
                "width": "{{WIDTH}}",
                "height": "{{HEIGHT}}",
                "length": "{{FRAMES}}",
                "batch_size": 1,
                "crf": "{{CRF}}",
                "blur_radius": "{{BLUR_RADIUS}}",
                "interpolation": "lanczos",
                "crop": "disabled",
                "strength": "{{STRENGTH}}",
            },
        },
        "8": {
            "class_type": "LTXVEmptyLatentAudio",
            "_meta": {"title": "Create Empty Audio Latent"},
            "inputs": {"audio_vae": ["3", 0], "frames_number": "{{FRAMES}}", "frame_rate": "{{FPS}}", "batch_size": 1},
        },
        "9": {
            "class_type": "LTXVConcatAVLatent",
            "_meta": {"title": "Concat Audio+Video Latents"},
            "inputs": {"video_latent": ["21", 2], "audio_latent": ["8", 0]},
        },
        "10": {
            "class_type": "LTXVScheduler",
            "_meta": {"title": "LTX Scheduler"},
            "inputs": {
                "steps": "{{STEPS}}",
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["9", 0],
            },
        },
        "11": {
            "class_type": "RandomNoise",
            "_meta": {"title": "Random Noise"},
            "inputs": {"noise_seed": "{{SEED}}"},
        },
        "12": {
            "class_type": "KSamplerSelect",
            "_meta": {"title": "Select Sampler"},
            "inputs": {"sampler_name": "euler"},
        },
        "13": {
            "class_type": "CFGGuider",
            "_meta": {"title": "CFG Guider"},
            "inputs": {"model": ["1", 0], "positive": ["21", 0], "negative": ["21", 1], "cfg": "{{CFG}}"},
        },
        "14": {
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": "Sample Video"},
            "inputs": {
                "noise": ["11", 0],
                "guider": ["13", 0],
                "sampler": ["12", 0],
                "sigmas": ["10", 0],
                "latent_image": ["9", 0],
            },
        },
        "15": {
            "class_type": "LTXVSeparateAVLatent",
            "_meta": {"title": "Separate Audio+Video"},
            "inputs": {"av_latent": ["14", 0]},
        },
        "16": {
            "class_type": "VAEDecodeTiled",
            "_meta": {"title": "Decode Video Latent"},
            "inputs": {
                "samples": ["15", 0],
                "vae": ["1", 2],
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
        },
        "17": {
            "class_type": "LTXVAudioVAEDecode",
            "_meta": {"title": "Decode Audio Latent"},
            "inputs": {"samples": ["15", 1], "audio_vae": ["3", 0]},
        },
        "18": {
            "class_type": "CreateVideo",
            "_meta": {"title": "Create Video"},
            "inputs": {"images": ["16", 0], "audio": ["17", 0], "fps": "{{FPS}}"},
        },
        "19": {
            "class_type": "SaveVideo",
            "_meta": {"title": "Save Video"},
            "inputs": {"video": ["18", 0], "filename_prefix": "{{PREFIX}}", "format": "mp4", "codec": "auto"},
        },
    },
    ("flux2", "txt2img"): {
        "_meta": {
            "description": "FLUX.2-dev text-to-image",
            "model": "FLUX.2-dev",
            "type": "txt2img",
            "parameters": ["PROMPT", "SEED", "WIDTH", "HEIGHT", "STEPS", "GUIDANCE"],
            "defaults": {"WIDTH": 1024, "HEIGHT": 1024, "SEED": 42, "STEPS": 20, "GUIDANCE": 3.5},
        },
        "1": {
            "class_type": "UNETLoader",
            "_meta": {"title": "Load FLUX UNET"},
            "inputs": {"unet_name": "flux1-dev-fp8.safetensors", "weight_dtype": "fp8_e4m3fn"},
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "_meta": {"title": "Load Dual CLIP"},
            "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp16.safetensors", "type": "flux"},
        },
        "3": {"class_type": "VAELoader", "_meta": {"title": "Load VAE"}, "inputs": {"vae_name": "ae.safetensors"}},
        "4": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Prompt"},
            "inputs": {"clip": ["2", 0], "text": "{{PROMPT}}"},
        },
        "5": {
            "class_type": "FluxGuidance",
            "_meta": {"title": "Apply Guidance"},
            "inputs": {"conditioning": ["4", 0], "guidance": "{{GUIDANCE}}"},
        },
        "6": {
            "class_type": "EmptySD3LatentImage",
            "_meta": {"title": "Create Empty Latent"},
            "inputs": {"width": "{{WIDTH}}", "height": "{{HEIGHT}}", "batch_size": 1},
        },
        "7": {
            "class_type": "KSamplerSelect",
            "_meta": {"title": "Select Sampler"},
            "inputs": {"sampler_name": "euler"},
        },
        "8": {
            "class_type": "BasicScheduler",
            "_meta": {"title": "Basic Scheduler"},
            "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": "{{STEPS}}", "denoise": 1.0},
        },
        "9": {"class_type": "RandomNoise", "_meta": {"title": "Generate Noise"}, "inputs": {"noise_seed": "{{SEED}}"}},
        "10": {
            "class_type": "BasicGuider",
            "_meta": {"title": "Basic Guider"},
            "inputs": {"model": ["1", 0], "conditioning": ["5", 0]},
        },
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": "Advanced Sampler"},
            "inputs": {
                "noise": ["9", 0],
                "guider": ["10", 0],
                "sampler": ["7", 0],
                "sigmas": ["8", 0],
                "latent_image": ["6", 0],
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "_meta": {"title": "Decode Latent"},
            "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
        },
        "13": {
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
            "inputs": {"images": ["12", 0], "filename_prefix": "flux2_output"},
        },
    },
    ("wan26", "img2vid"): {
        "_meta": {
            "description": "Wan 2.6 image-to-video",
            "model": "Wan 2.6 14B",
            "type": "img2vid",
            "parameters": ["IMAGE_PATH", "PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "FRAMES"],
            "defaults": {
                "WIDTH": 832,
                "HEIGHT": 480,
                "SEED": 42,
                "FRAMES": 81,
                "NEGATIVE": "worst quality, blurry, distorted",
            },
        },
        "1": {
            "class_type": "WanVideoModelLoader",
            "_meta": {"title": "Load Wan Model"},
            "inputs": {
                "model": "wan_i2v_fp8/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
                "base_precision": "bf16",
                "quantization": "fp8_e4m3fn_scaled",
                "load_device": "offload_device",
            },
        },
        "2": {
            "class_type": "WanVideoVAELoader",
            "_meta": {"title": "Load VAE"},
            "inputs": {"model_name": "wan_2.1_vae.safetensors", "precision": "bf16"},
        },
        "3": {
            "class_type": "LoadWanVideoT5TextEncoder",
            "_meta": {"title": "Load T5 Text Encoder"},
            "inputs": {"model_name": "umt5_xxl_fp16.safetensors", "precision": "bf16", "load_device": "offload_device"},
        },
        "4": {
            "class_type": "LoadWanVideoClipTextEncoder",
            "_meta": {"title": "Load CLIP Vision"},
            "inputs": {
                "model_name": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "precision": "fp16",
                "load_device": "offload_device",
            },
        },
        "5": {"class_type": "LoadImage", "_meta": {"title": "Load Input Image"}, "inputs": {"image": "{{IMAGE_PATH}}"}},
        "6": {
            "class_type": "WanVideoTextEncode",
            "_meta": {"title": "Encode Text"},
            "inputs": {
                "positive_prompt": "{{PROMPT}}",
                "negative_prompt": "{{NEGATIVE}}",
                "t5": ["3", 0],
                "force_offload": True,
            },
        },
        "7": {
            "class_type": "WanVideoClipVisionEncode",
            "_meta": {"title": "Encode CLIP Vision"},
            "inputs": {
                "clip_vision": ["4", 0],
                "image_1": ["5", 0],
                "strength_1": 1.0,
                "strength_2": 1.0,
                "crop": "center",
                "combine_embeds": "average",
                "force_offload": True,
            },
        },
        "8": {
            "class_type": "WanVideoImageToVideoEncode",
            "_meta": {"title": "Encode Image for I2V"},
            "inputs": {
                "width": "{{WIDTH}}",
                "height": "{{HEIGHT}}",
                "num_frames": "{{FRAMES}}",
                "noise_aug_strength": 0.0,
                "start_latent_strength": 1.0,
                "end_latent_strength": 1.0,
                "force_offload": True,
                "vae": ["2", 0],
                "clip_embeds": ["7", 0],
                "start_image": ["5", 0],
            },
        },
        "9": {
            "class_type": "WanVideoSampler",
            "_meta": {"title": "Sample Video"},
            "inputs": {
                "model": ["1", 0],
                "image_embeds": ["8", 0],
                "text_embeds": ["6", 0],
                "steps": 30,
                "cfg": 5.0,
                "shift": 5.0,
                "seed": "{{SEED}}",
                "force_offload": True,
                "scheduler": "unipc",
                "riflex_freq_index": 0,
            },
        },
        "10": {
            "class_type": "WanVideoDecode",
            "_meta": {"title": "Decode Video"},
            "inputs": {
                "vae": ["2", 0],
                "samples": ["9", 0],
                "enable_vae_tiling": False,
                "tile_x": 272,
                "tile_y": 272,
                "tile_stride_x": 144,
                "tile_stride_y": 128,
            },
        },
        "11": {
            "class_type": "CreateVideo",
            "_meta": {"title": "Create Video"},
            "inputs": {"images": ["10", 0], "fps": 16.0},
        },
        "12": {
            "class_type": "SaveVideo",
            "_meta": {"title": "Save Video"},
            "inputs": {"video": ["11", 0], "filename_prefix": "wan26_i2v", "format": "mp4", "codec": "h264"},
        },
    },
    ("wan26", "txt2vid"): {
        "_meta": {
            "description": "Wan 2.6 text-to-video using WanVideoWrapper nodes",
            "model": "Wan 2.6 14B",
            "type": "txt2vid",
            "parameters": ["PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "FRAMES", "STEPS", "CFG", "SHIFT", "FPS"],
            "defaults": {
                "WIDTH": 832,
                "HEIGHT": 480,
                "SEED": 42,
                "FRAMES": 81,
                "STEPS": 30,
                "CFG": 5.0,
                "SHIFT": 5.0,
                "FPS": 16,
                "NEGATIVE": "worst quality, blurry, distorted",
            },
        },
        "1": {
            "class_type": "WanVideoModelLoader",
            "_meta": {"title": "Load Wan T2V Model"},
            "inputs": {
                "model": "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
                "base_precision": "bf16",
                "quantization": "disabled",
                "load_device": "offload_device",
            },
        },
        "2": {
            "class_type": "WanVideoVAELoader",
            "_meta": {"title": "Load Wan VAE"},
            "inputs": {"model_name": "wan_2.1_vae.safetensors", "precision": "bf16"},
        },
        "3": {
            "class_type": "LoadWanVideoT5TextEncoder",
            "_meta": {"title": "Load T5 Text Encoder"},
            "inputs": {"model_name": "umt5_xxl_fp16.safetensors", "precision": "bf16", "load_device": "offload_device"},
        },
        "4": {
            "class_type": "WanVideoTextEncode",
            "_meta": {"title": "Encode Text"},
            "inputs": {
                "positive_prompt": "{{PROMPT}}",
                "negative_prompt": "{{NEGATIVE}}",
                "t5": ["3", 0],
                "force_offload": True,
            },
        },
        "5": {
            "class_type": "WanVideoEmptyEmbeds",
            "_meta": {"title": "Empty Image Embeds (T2V)"},
            "inputs": {"width": "{{WIDTH}}", "height": "{{HEIGHT}}", "num_frames": "{{FRAMES}}"},
        },
        "6": {
            "class_type": "WanVideoSampler",
            "_meta": {"title": "Sample Video"},
            "inputs": {
                "model": ["1", 0],
                "image_embeds": ["5", 0],
                "text_embeds": ["4", 0],
                "steps": "{{STEPS}}",
                "cfg": "{{CFG}}",
                "shift": "{{SHIFT}}",
                "seed": "{{SEED}}",
                "force_offload": True,
                "scheduler": "unipc",
                "riflex_freq_index": 0,
            },
        },
        "7": {
            "class_type": "WanVideoDecode",
            "_meta": {"title": "Decode Video"},
            "inputs": {
                "vae": ["2", 0],
                "samples": ["6", 0],
                "enable_vae_tiling": False,
                "tile_x": 272,
                "tile_y": 272,
                "tile_stride_x": 144,
                "tile_stride_y": 128,
            },
        },
        "8": {
            "class_type": "CreateVideo",
            "_meta": {"title": "Create Video"},
            "inputs": {"images": ["7", 0], "fps": "{{FPS}}"},
        },
        "9": {
            "class_type": "SaveVideo",
            "_meta": {"title": "Save Video"},
            "inputs": {"video": ["8", 0], "filename_prefix": "wan26_t2v", "format": "mp4", "codec": "h264"},
        },
    },
    ("qwen", "txt2img"): {
        "_meta": {
            "description": "Qwen text-to-image (best for text rendering and photorealistic portraits)",
            "model": "Qwen Image 2512",
            "type": "txt2img",
            "parameters": ["PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "STEPS", "CFG", "SHIFT"],
            "defaults": {
                "WIDTH": 1296,
                "HEIGHT": 1296,
                "SEED": 42,
                "STEPS": 50,
                "CFG": 3.5,
                "SHIFT": 7.0,
                "NEGATIVE": "worst quality, blurry, distorted",
            },
        },
        "1": {
            "class_type": "UNETLoader",
            "_meta": {"title": "Load Qwen UNET"},
            "inputs": {"unet_name": "qwen_image_2512_fp8_e4m3fn.safetensors", "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load Qwen CLIP"},
            "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"},
        },
        "3": {
            "class_type": "VAELoader",
            "_meta": {"title": "Load Qwen VAE"},
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Positive Prompt"},
            "inputs": {"clip": ["2", 0], "text": "{{PROMPT}}"},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Negative Prompt"},
            "inputs": {"clip": ["2", 0], "text": "{{NEGATIVE}}"},
        },
        "6": {
            "class_type": "ModelSamplingAuraFlow",
            "_meta": {"title": "Apply Shift (CRITICAL: 7.0 for sharp output)"},
            "inputs": {"model": ["1", 0], "shift": "{{SHIFT}}"},
        },
        "7": {
            "class_type": "EmptySD3LatentImage",
            "_meta": {"title": "Create Empty Latent"},
            "inputs": {"width": "{{WIDTH}}", "height": "{{HEIGHT}}", "batch_size": 1},
        },
        "8": {
            "class_type": "KSampler",
            "_meta": {"title": "Sample Image"},
            "inputs": {
                "model": ["6", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["7", 0],
                "seed": "{{SEED}}",
                "steps": "{{STEPS}}",
                "cfg": "{{CFG}}",
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "_meta": {"title": "Decode Latent"},
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
        },
        "10": {
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
            "inputs": {"images": ["9", 0], "filename_prefix": "qwen_output"},
        },
    },
    ("sdxl", "txt2img"): {
        "_meta": {
            "description": "SDXL text-to-image",
            "model": "SDXL Base",
            "type": "txt2img",
            "parameters": ["PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "STEPS", "CFG"],
            "defaults": {
                "WIDTH": 1024,
                "HEIGHT": 1024,
                "SEED": 42,
                "STEPS": 25,
                "CFG": 7.0,
                "NEGATIVE": "worst quality, low quality, blurry",
            },
        },
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load SDXL Model"},
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Positive Prompt"},
            "inputs": {"clip": ["1", 1], "text": "{{PROMPT}}"},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Negative Prompt"},
            "inputs": {"clip": ["1", 1], "text": "{{NEGATIVE}}"},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Create Empty Latent"},
            "inputs": {"width": "{{WIDTH}}", "height": "{{HEIGHT}}", "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "_meta": {"title": "Sample Image"},
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": "{{SEED}}",
                "steps": "{{STEPS}}",
                "cfg": "{{CFG}}",
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "_meta": {"title": "Decode Latent"},
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
            "inputs": {"images": ["6", 0], "filename_prefix": "sdxl_output"},
        },
    },
    ("hunyuan15", "txt2vid"): {
        "_meta": {
            "description": "HunyuanVideo 1.5 text-to-video",
            "model": "HunyuanVideo 1.5",
            "type": "txt2vid",
            "parameters": ["PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "FRAMES", "STEPS", "CFG"],
            "defaults": {
                "WIDTH": 1280,
                "HEIGHT": 720,
                "SEED": 42,
                "FRAMES": 81,
                "STEPS": 30,
                "CFG": 6.0,
                "NEGATIVE": "worst quality, blurry, distorted, low resolution",
            },
        },
        "1": {
            "class_type": "HunyuanVideoModelLoader",
            "_meta": {"title": "Load HunyuanVideo Model"},
            "inputs": {"model_name": "hunyuanvideo_t2v_720p_bf16.safetensors", "precision": "bf16"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP Text Encoder"},
            "inputs": {"clip_name": "clip_l.safetensors", "type": "hunyuan_video"},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Positive Prompt"},
            "inputs": {"clip": ["2", 0], "text": "{{PROMPT}}"},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Negative Prompt"},
            "inputs": {"clip": ["2", 0], "text": "{{NEGATIVE}}"},
        },
        "5": {
            "class_type": "EmptyHunyuanLatentVideo",
            "_meta": {"title": "Create Empty Video Latent"},
            "inputs": {"width": "{{WIDTH}}", "height": "{{HEIGHT}}", "length": "{{FRAMES}}", "batch_size": 1},
        },
        "6": {
            "class_type": "HunyuanVideoSampler",
            "_meta": {"title": "Sample Video"},
            "inputs": {
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent": ["5", 0],
                "seed": "{{SEED}}",
                "steps": "{{STEPS}}",
                "cfg": "{{CFG}}",
                "sampler": "euler",
                "scheduler": "normal",
            },
        },
        "7": {
            "class_type": "HunyuanVideoVAEDecode",
            "_meta": {"title": "Decode Latent to Video"},
            "inputs": {"samples": ["6", 0], "vae": ["1", 1]},
        },
        "8": {
            "class_type": "VHS_VideoCombine",
            "_meta": {"title": "Save Video"},
            "inputs": {
                "images": ["7", 0],
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": "hunyuan_output",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
    },
    ("hunyuan15", "img2vid"): {
        "_meta": {
            "description": "HunyuanVideo 1.5 image-to-video",
            "model": "HunyuanVideo 1.5",
            "type": "img2vid",
            "parameters": ["IMAGE_PATH", "PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "FRAMES", "STEPS", "CFG"],
            "defaults": {
                "WIDTH": 1280,
                "HEIGHT": 720,
                "SEED": 42,
                "FRAMES": 81,
                "STEPS": 30,
                "CFG": 6.0,
                "NEGATIVE": "worst quality, blurry, distorted, low resolution",
            },
        },
        "1": {
            "class_type": "HunyuanVideoModelLoader",
            "_meta": {"title": "Load HunyuanVideo Model"},
            "inputs": {"model_name": "hunyuanvideo_t2v_720p_bf16.safetensors", "precision": "bf16"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP Text Encoder"},
            "inputs": {"clip_name": "clip_l.safetensors", "type": "hunyuan_video"},
        },
        "3": {"class_type": "LoadImage", "_meta": {"title": "Load Input Image"}, "inputs": {"image": "{{IMAGE_PATH}}"}},
        "4": {
            "class_type": "HunyuanVideoImageEncode",
            "_meta": {"title": "Encode Reference Image"},
            "inputs": {"image": ["3", 0], "vae": ["1", 1]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Positive Prompt"},
            "inputs": {"clip": ["2", 0], "text": "{{PROMPT}}"},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Negative Prompt"},
            "inputs": {"clip": ["2", 0], "text": "{{NEGATIVE}}"},
        },
        "7": {
            "class_type": "EmptyHunyuanLatentVideo",
            "_meta": {"title": "Create Empty Video Latent"},
            "inputs": {"width": "{{WIDTH}}", "height": "{{HEIGHT}}", "length": "{{FRAMES}}", "batch_size": 1},
        },
        "8": {
            "class_type": "HunyuanVideoSampler",
            "_meta": {"title": "Sample Video"},
            "inputs": {
                "model": ["1", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent": ["7", 0],
                "image_embeds": ["4", 0],
                "seed": "{{SEED}}",
                "steps": "{{STEPS}}",
                "cfg": "{{CFG}}",
                "sampler": "euler",
                "scheduler": "normal",
            },
        },
        "9": {
            "class_type": "HunyuanVideoVAEDecode",
            "_meta": {"title": "Decode Latent to Video"},
            "inputs": {"samples": ["8", 0], "vae": ["1", 1]},
        },
        "10": {
            "class_type": "VHS_VideoCombine",
            "_meta": {"title": "Save Video"},
            "inputs": {
                "images": ["9", 0],
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": "hunyuan_i2v",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
    },
    ("z_turbo", "txt2img"): {
        "_meta": {
            "description": "Z-Image-Turbo fast text-to-image (4-step inference)",
            "model": "Z-Image-Turbo",
            "type": "txt2img",
            "parameters": ["PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "STEPS", "CFG"],
            "defaults": {
                "WIDTH": 1024,
                "HEIGHT": 1024,
                "SEED": 42,
                "STEPS": 4,
                "CFG": 3.0,
                "NEGATIVE": "worst quality, blurry, distorted",
            },
        },
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Z-Turbo Model"},
            "inputs": {"ckpt_name": "z_image_turbo.safetensors"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Positive Prompt"},
            "inputs": {"clip": ["1", 1], "text": "{{PROMPT}}"},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Negative Prompt"},
            "inputs": {"clip": ["1", 1], "text": "{{NEGATIVE}}"},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Create Empty Latent"},
            "inputs": {"width": "{{WIDTH}}", "height": "{{HEIGHT}}", "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "_meta": {"title": "Sample Image"},
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": "{{SEED}}",
                "steps": "{{STEPS}}",
                "cfg": "{{CFG}}",
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "_meta": {"title": "Decode Latent"},
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
            "inputs": {"images": ["6", 0], "filename_prefix": "zturbo_output"},
        },
    },
    ("cogvideox_5b", "txt2vid"): {
        "_meta": {
            "description": "CogVideoX-5B text-to-video from Tsinghua",
            "model": "CogVideoX-5B",
            "type": "txt2vid",
            "parameters": ["PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "FRAMES", "STEPS", "CFG"],
            "defaults": {
                "WIDTH": 720,
                "HEIGHT": 480,
                "SEED": 42,
                "FRAMES": 49,
                "STEPS": 50,
                "CFG": 6.0,
                "NEGATIVE": "worst quality, blurry, distorted, low resolution",
            },
        },
        "1": {
            "class_type": "CogVideoLoader",
            "_meta": {"title": "Load CogVideoX Model"},
            "inputs": {"model_name": "cogvideox_5b_bf16.safetensors", "precision": "bf16"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP Text Encoder"},
            "inputs": {"clip_name": "clip_l.safetensors", "type": "cogvideo"},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Positive Prompt"},
            "inputs": {"clip": ["2", 0], "text": "{{PROMPT}}"},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Encode Negative Prompt"},
            "inputs": {"clip": ["2", 0], "text": "{{NEGATIVE}}"},
        },
        "5": {
            "class_type": "EmptyCogVideoLatent",
            "_meta": {"title": "Create Empty Video Latent"},
            "inputs": {"width": "{{WIDTH}}", "height": "{{HEIGHT}}", "length": "{{FRAMES}}", "batch_size": 1},
        },
        "6": {
            "class_type": "CogVideoSampler",
            "_meta": {"title": "Sample Video"},
            "inputs": {
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent": ["5", 0],
                "seed": "{{SEED}}",
                "steps": "{{STEPS}}",
                "cfg": "{{CFG}}",
                "sampler": "euler",
                "scheduler": "normal",
            },
        },
        "7": {
            "class_type": "CogVideoVAEDecode",
            "_meta": {"title": "Decode Latent to Video"},
            "inputs": {"samples": ["6", 0], "vae": ["1", 1]},
        },
        "8": {
            "class_type": "VHS_VideoCombine",
            "_meta": {"title": "Save Video"},
            "inputs": {
                "images": ["7", 0],
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": "cogvideo_output",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
    },
}


# =============================================================================
# Node Chains (ordered nodes with connection info)
# =============================================================================

NODE_CHAINS = {
    ("ltx2", "txt2vid"): [
        {"id": "1", "class_type": "CheckpointLoaderSimple", "outputs": {"MODEL": 0, "CLIP": 1, "VAE": 2}},
        {"id": "2", "class_type": "LTXAVTextEncoderLoader", "outputs": {"CLIP": 0}},
        {"id": "3", "class_type": "LTXVAudioVAELoader", "outputs": {"VAE": 0}},
        {"id": "4", "class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0]}, "outputs": {"CONDITIONING": 0}},
        {"id": "5", "class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0]}, "outputs": {"CONDITIONING": 0}},
        {
            "id": "6",
            "class_type": "LTXVConditioning",
            "inputs": {"positive": ["4", 0], "negative": ["5", 0]},
            "outputs": {"POSITIVE": 0, "NEGATIVE": 1},
        },
        {"id": "7", "class_type": "EmptyLTXVLatentVideo", "outputs": {"LATENT": 0}},
        {"id": "8", "class_type": "LTXVEmptyLatentAudio", "inputs": {"audio_vae": ["3", 0]}, "outputs": {"LATENT": 0}},
        {
            "id": "9",
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video_latent": ["7", 0], "audio_latent": ["8", 0]},
            "outputs": {"LATENT": 0},
        },
        {"id": "10", "class_type": "LTXVScheduler", "inputs": {"latent": ["9", 0]}, "outputs": {"SIGMAS": 0}},
        {"id": "11", "class_type": "RandomNoise", "outputs": {"NOISE": 0}},
        {"id": "12", "class_type": "KSamplerSelect", "outputs": {"SAMPLER": 0}},
        {
            "id": "13",
            "class_type": "CFGGuider",
            "inputs": {"model": ["1", 0], "positive": ["6", 0], "negative": ["6", 1]},
            "outputs": {"GUIDER": 0},
        },
        {
            "id": "14",
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["11", 0],
                "guider": ["13", 0],
                "sampler": ["12", 0],
                "sigmas": ["10", 0],
                "latent_image": ["9", 0],
            },
            "outputs": {"LATENT": 0},
        },
        {
            "id": "15",
            "class_type": "LTXVSeparateAVLatent",
            "inputs": {"av_latent": ["14", 0]},
            "outputs": {"video_latent": 0, "audio_latent": 1},
        },
        {
            "id": "16",
            "class_type": "VAEDecodeTiled",
            "inputs": {"samples": ["15", 0], "vae": ["1", 2]},
            "outputs": {"IMAGE": 0},
        },
        {
            "id": "17",
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {"samples": ["15", 1], "audio_vae": ["3", 0]},
            "outputs": {"AUDIO": 0},
        },
        {
            "id": "18",
            "class_type": "CreateVideo",
            "inputs": {"images": ["16", 0], "audio": ["17", 0]},
            "outputs": {"VIDEO": 0},
        },
        {"id": "19", "class_type": "SaveVideo", "inputs": {"video": ["18", 0]}, "outputs": {}},
    ],
    ("flux2", "txt2img"): [
        {"id": "1", "class_type": "UNETLoader", "outputs": {"MODEL": 0}},
        {"id": "2", "class_type": "DualCLIPLoader", "outputs": {"CLIP": 0}},
        {"id": "3", "class_type": "VAELoader", "outputs": {"VAE": 0}},
        {"id": "4", "class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0]}, "outputs": {"CONDITIONING": 0}},
        {"id": "5", "class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0]}, "outputs": {"CONDITIONING": 0}},
        {"id": "6", "class_type": "EmptySD3LatentImage", "outputs": {"LATENT": 0}},
        {"id": "7", "class_type": "KSamplerSelect", "outputs": {"SAMPLER": 0}},
        {"id": "8", "class_type": "BasicScheduler", "inputs": {"model": ["1", 0]}, "outputs": {"SIGMAS": 0}},
        {"id": "9", "class_type": "RandomNoise", "outputs": {"NOISE": 0}},
        {
            "id": "10",
            "class_type": "BasicGuider",
            "inputs": {"model": ["1", 0], "conditioning": ["5", 0]},
            "outputs": {"GUIDER": 0},
        },
        {
            "id": "11",
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["10", 0],
                "sampler": ["7", 0],
                "sigmas": ["8", 0],
                "latent_image": ["6", 0],
            },
            "outputs": {"LATENT": 0},
        },
        {
            "id": "12",
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
            "outputs": {"IMAGE": 0},
        },
        {"id": "13", "class_type": "SaveImage", "inputs": {"images": ["12", 0]}, "outputs": {}},
    ],
    ("wan26", "img2vid"): [
        {"id": "1", "class_type": "WanVideoModelLoader", "outputs": {"WANVIDEOMODEL": 0}},
        {"id": "2", "class_type": "WanVideoVAELoader", "outputs": {"WANVAE": 0}},
        {"id": "3", "class_type": "LoadWanVideoT5TextEncoder", "outputs": {"WANTEXTENCODER": 0}},
        {"id": "4", "class_type": "LoadWanVideoClipTextEncoder", "outputs": {"CLIP_VISION": 0}},
        {"id": "5", "class_type": "LoadImage", "outputs": {"IMAGE": 0, "MASK": 1}},
        {
            "id": "6",
            "class_type": "WanVideoTextEncode",
            "inputs": {"t5": ["3", 0]},
            "outputs": {"WANVIDEOTEXTEMBEDS": 0},
        },
        {
            "id": "7",
            "class_type": "WanVideoClipVisionEncode",
            "inputs": {"clip_vision": ["4", 0], "image_1": ["5", 0]},
            "outputs": {"WANVIDIMAGE_CLIPEMBEDS": 0},
        },
        {
            "id": "8",
            "class_type": "WanVideoImageToVideoEncode",
            "inputs": {"vae": ["2", 0], "clip_embeds": ["7", 0], "start_image": ["5", 0]},
            "outputs": {"WANVIDIMAGE_EMBEDS": 0},
        },
        {
            "id": "9",
            "class_type": "WanVideoSampler",
            "inputs": {"model": ["1", 0], "image_embeds": ["8", 0], "text_embeds": ["6", 0]},
            "outputs": {"LATENT": 0},
        },
        {
            "id": "10",
            "class_type": "WanVideoDecode",
            "inputs": {"vae": ["2", 0], "samples": ["9", 0]},
            "outputs": {"IMAGE": 0},
        },
        {"id": "11", "class_type": "CreateVideo", "inputs": {"images": ["10", 0]}, "outputs": {"VIDEO": 0}},
        {"id": "12", "class_type": "SaveVideo", "inputs": {"video": ["11", 0]}, "outputs": {}},
    ],
    ("wan26", "txt2vid"): [
        {"id": "1", "class_type": "WanVideoModelLoader", "outputs": {"WANVIDEOMODEL": 0}},
        {"id": "2", "class_type": "WanVideoVAELoader", "outputs": {"WANVAE": 0}},
        {"id": "3", "class_type": "LoadWanVideoT5TextEncoder", "outputs": {"WANTEXTENCODER": 0}},
        {
            "id": "4",
            "class_type": "WanVideoTextEncode",
            "inputs": {"t5": ["3", 0]},
            "outputs": {"WANVIDEOTEXTEMBEDS": 0},
        },
        {"id": "5", "class_type": "WanVideoEmptyEmbeds", "outputs": {"WANVIDIMAGE_EMBEDS": 0}},
        {
            "id": "6",
            "class_type": "WanVideoSampler",
            "inputs": {"model": ["1", 0], "image_embeds": ["5", 0], "text_embeds": ["4", 0]},
            "outputs": {"LATENT": 0},
        },
        {
            "id": "7",
            "class_type": "WanVideoDecode",
            "inputs": {"vae": ["2", 0], "samples": ["6", 0]},
            "outputs": {"IMAGE": 0},
        },
        {"id": "8", "class_type": "CreateVideo", "inputs": {"images": ["7", 0]}, "outputs": {"VIDEO": 0}},
        {"id": "9", "class_type": "SaveVideo", "inputs": {"video": ["8", 0]}, "outputs": {}},
    ],
    ("sdxl", "txt2img"): [
        {"id": "1", "class_type": "CheckpointLoaderSimple", "outputs": {"MODEL": 0, "CLIP": 1, "VAE": 2}},
        {"id": "2", "class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}, "outputs": {"CONDITIONING": 0}},
        {"id": "3", "class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}, "outputs": {"CONDITIONING": 0}},
        {"id": "4", "class_type": "EmptyLatentImage", "outputs": {"LATENT": 0}},
        {
            "id": "5",
            "class_type": "KSampler",
            "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]},
            "outputs": {"LATENT": 0},
        },
        {
            "id": "6",
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
            "outputs": {"IMAGE": 0},
        },
        {"id": "7", "class_type": "SaveImage", "inputs": {"images": ["6", 0]}, "outputs": {}},
    ],
    ("qwen", "txt2img"): [
        {"id": "1", "class_type": "UNETLoader", "outputs": {"MODEL": 0}},
        {"id": "2", "class_type": "CLIPLoader", "outputs": {"CLIP": 0}},
        {"id": "3", "class_type": "VAELoader", "outputs": {"VAE": 0}},
        {"id": "4", "class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0]}, "outputs": {"CONDITIONING": 0}},
        {"id": "5", "class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0]}, "outputs": {"CONDITIONING": 0}},
        {
            "id": "6",
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0]},
            "outputs": {"MODEL": 0},
            "note": "CRITICAL: shift=7.0 for sharp output",
        },
        {"id": "7", "class_type": "EmptySD3LatentImage", "outputs": {"LATENT": 0}},
        {
            "id": "8",
            "class_type": "KSampler",
            "inputs": {"model": ["6", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["7", 0]},
            "outputs": {"LATENT": 0},
        },
        {
            "id": "9",
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
            "outputs": {"IMAGE": 0},
        },
        {"id": "10", "class_type": "SaveImage", "inputs": {"images": ["9", 0]}, "outputs": {}},
    ],
    ("hunyuan15", "txt2vid"): [
        {"id": "1", "class_type": "HunyuanVideoModelLoader", "outputs": {"MODEL": 0, "VAE": 1}},
        {"id": "2", "class_type": "CLIPLoader", "outputs": {"CLIP": 0}},
        {"id": "3", "class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0]}, "outputs": {"CONDITIONING": 0}},
        {"id": "4", "class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0]}, "outputs": {"CONDITIONING": 0}},
        {"id": "5", "class_type": "EmptyHunyuanLatentVideo", "outputs": {"LATENT": 0}},
        {
            "id": "6",
            "class_type": "HunyuanVideoSampler",
            "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent": ["5", 0]},
            "outputs": {"LATENT": 0},
        },
        {
            "id": "7",
            "class_type": "HunyuanVideoVAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["1", 1]},
            "outputs": {"IMAGE": 0},
        },
        {"id": "8", "class_type": "VHS_VideoCombine", "inputs": {"images": ["7", 0]}, "outputs": {"FILENAMES": 0}},
    ],
    ("hunyuan15", "img2vid"): [
        {"id": "1", "class_type": "HunyuanVideoModelLoader", "outputs": {"MODEL": 0, "VAE": 1}},
        {"id": "2", "class_type": "CLIPLoader", "outputs": {"CLIP": 0}},
        {"id": "3", "class_type": "LoadImage", "outputs": {"IMAGE": 0, "MASK": 1}},
        {
            "id": "4",
            "class_type": "HunyuanVideoImageEncode",
            "inputs": {"image": ["3", 0], "vae": ["1", 1]},
            "outputs": {"IMAGE_EMBEDS": 0},
        },
        {"id": "5", "class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0]}, "outputs": {"CONDITIONING": 0}},
        {"id": "6", "class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0]}, "outputs": {"CONDITIONING": 0}},
        {"id": "7", "class_type": "EmptyHunyuanLatentVideo", "outputs": {"LATENT": 0}},
        {
            "id": "8",
            "class_type": "HunyuanVideoSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent": ["7", 0],
                "image_embeds": ["4", 0],
            },
            "outputs": {"LATENT": 0},
        },
        {
            "id": "9",
            "class_type": "HunyuanVideoVAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["1", 1]},
            "outputs": {"IMAGE": 0},
        },
        {"id": "10", "class_type": "VHS_VideoCombine", "inputs": {"images": ["9", 0]}, "outputs": {"FILENAMES": 0}},
    ],
}


# =============================================================================
# API Functions
# =============================================================================


def get_workflow_skeleton(model: str, task: str) -> Dict[str, Any]:
    """
    Get exact working workflow JSON for model+task.

    Uses in-memory caching to reduce overhead (~10ms speedup per call).
    Cache TTL: 5 minutes. Cache stats available via get_cache_stats().

    Args:
        model: Model identifier (ltx2, flux2, wan26, qwen, sdxl)
        task: Task type (txt2vid, img2vid, txt2img, txt2vid_distilled)

    Returns:
        Complete workflow JSON with {{PLACEHOLDER}} parameters
    """
    # Try cache first
    cached = _get_skeleton_from_cache(model, task)
    if cached is not None:
        return cached

    # Load from registry
    key = (model.lower(), task.lower())

    if key not in WORKFLOW_SKELETONS:
        # Try to find partial matches
        available = [f"{m}/{t}" for m, t in WORKFLOW_SKELETONS.keys()]
        return {"error": f"No skeleton for {model}/{task}", "available": available}

    # Deep copy and cache
    result = copy.deepcopy(WORKFLOW_SKELETONS[key])
    _store_skeleton_in_cache(model, task, result)
    return result


def get_model_constraints(model: str) -> Dict[str, Any]:
    """
    Get hard constraints for a model.

    Args:
        model: Model identifier (ltx2, flux2, wan26, qwen, sdxl, hunyuan15)

    Returns:
        Constraints dict with cfg, resolution, frames, required_nodes, forbidden_nodes

    NOTE: Delegates to model_registry.get_model_constraints() for centralized data.
    """
    return _get_registry_constraints(model)


def get_node_chain(model: str, task: str) -> List[Dict[str, Any]]:
    """
    Get ordered list of required nodes with connections.

    Args:
        model: Model identifier
        task: Task type

    Returns:
        List of nodes in order with input/output slot information
    """
    key = (model.lower(), task.lower())

    if key not in NODE_CHAINS:
        available = [f"{m}/{t}" for m, t in NODE_CHAINS.keys()]
        return {"error": f"No node chain for {model}/{task}", "available": available}

    return copy.deepcopy(NODE_CHAINS[key])


def validate_against_pattern(workflow: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Validate a workflow against known working patterns.

    Args:
        workflow: Workflow JSON to validate
        model: Model identifier for constraint lookup

    Returns:
        {
            "valid": bool,
            "errors": list of error messages,
            "warnings": list of warnings,
            "suggestions": list of fixes
        }
    """
    errors = []
    warnings = []
    suggestions = []

    model_lower = model.lower()
    constraints = MODEL_CONSTRAINTS.get(model_lower, {})

    if not constraints:
        return {
            "valid": True,
            "errors": [],
            "warnings": [f"No constraints defined for model '{model}', skipping validation"],
            "suggestions": [],
        }

    # Check for forbidden nodes
    forbidden = constraints.get("forbidden_nodes", {})
    for node_id, node in workflow.items():
        if node_id.startswith("_"):
            continue

        class_type = node.get("class_type", "")
        if class_type in forbidden:
            errors.append(f"Node {node_id} uses forbidden '{class_type}': {forbidden[class_type]}")

    # Check for required nodes
    required = constraints.get("required_nodes", {})
    workflow_types = {n.get("class_type") for n in workflow.values() if isinstance(n, dict) and "class_type" in n}

    for role, required_type in required.items():
        if isinstance(required_type, list):
            # All must be present (e.g., FLUX loaders)
            for rt in required_type:
                if rt not in workflow_types:
                    errors.append(f"Missing required {role} node: {rt}")
        else:
            if required_type not in workflow_types:
                errors.append(f"Missing required {role} node: {required_type}")

    # Check CFG values
    cfg_constraints = constraints.get("cfg", {})
    if cfg_constraints:
        for node_id, node in workflow.items():
            if node_id.startswith("_"):
                continue

            inputs = node.get("inputs", {})
            if "cfg" in inputs:
                cfg_val = inputs["cfg"]
                if isinstance(cfg_val, (int, float)):
                    if cfg_constraints.get("min") and cfg_val < cfg_constraints["min"]:
                        errors.append(f"CFG {cfg_val} is below minimum {cfg_constraints['min']} for {model}")
                    if cfg_constraints.get("max") and cfg_val > cfg_constraints["max"]:
                        errors.append(f"CFG {cfg_val} exceeds maximum {cfg_constraints['max']} for {model}")

            # Check for guidance in FluxGuidance
            if node.get("class_type") == "FluxGuidance" and "guidance" in inputs:
                guidance_val = inputs["guidance"]
                if isinstance(guidance_val, (int, float)):
                    if cfg_constraints.get("min") and guidance_val < cfg_constraints["min"]:
                        warnings.append(
                            f"Guidance {guidance_val} is low for FLUX (typical: {cfg_constraints.get('default', 3.5)})"
                        )

    # Check resolution constraints
    res_constraints = constraints.get("resolution", {})
    if res_constraints:
        divisible_by = res_constraints.get("divisible_by", 8)
        for node_id, node in workflow.items():
            if node_id.startswith("_"):
                continue

            inputs = node.get("inputs", {})
            for dim in ["width", "height"]:
                if dim in inputs:
                    val = inputs[dim]
                    if isinstance(val, int) and val % divisible_by != 0:
                        errors.append(f"{dim.capitalize()} {val} not divisible by {divisible_by} for {model}")

    # Check frame count for LTX
    if model_lower == "ltx2":
        frame_constraints = constraints.get("frames", {})
        for node_id, node in workflow.items():
            if node_id.startswith("_"):
                continue

            inputs = node.get("inputs", {})
            if "length" in inputs:
                frames = inputs["length"]
                if isinstance(frames, int) and (frames - 1) % 8 != 0:
                    valid_examples = frame_constraints.get("valid_examples", [])
                    errors.append(f"Frame count {frames} doesn't follow 8n+1 rule. Valid: {valid_examples}")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings, "suggestions": suggestions}


def list_available_patterns() -> Dict[str, Any]:
    """
    List all available workflow patterns.

    Returns:
        {
            "skeletons": [{"model": "ltx2", "task": "txt2vid", "description": "..."}],
            "models": ["ltx2", "flux2", ...],
            "total": count
        }
    """
    skeletons = []
    for (model, task), workflow in WORKFLOW_SKELETONS.items():
        meta = workflow.get("_meta", {})
        skeletons.append(
            {
                "model": model,
                "task": task,
                "description": meta.get("description", ""),
                "type": meta.get("type", ""),
                "parameters": meta.get("parameters", []),
            }
        )

    return {"skeletons": skeletons, "models": list(MODEL_CONSTRAINTS.keys()), "total": len(skeletons)}
