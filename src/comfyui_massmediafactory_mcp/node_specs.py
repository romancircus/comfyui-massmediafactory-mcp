"""
Node Specifications for LLM Reference

Provides exact input/output specifications for ComfyUI nodes
to prevent LLM hallucination of node names or wrong input types.
"""

# Node specifications extracted from 02_NODE_LIBRARY.md
NODE_SPECS = {
    # Core Loading Nodes
    "CheckpointLoaderSimple": {
        "class": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "STRING (filename)"
        },
        "outputs": ["MODEL", "CLIP", "VAE"],
        "category": "loaders"
    },
    "UNETLoader": {
        "class": "UNETLoader",
        "inputs": {
            "unet_name": "STRING (filename)",
            "weight_dtype": "STRING (default|fp8_e4m3fn|fp8_e5m2|fp16)"
        },
        "outputs": ["MODEL"],
        "category": "loaders"
    },
    "CLIPLoader": {
        "class": "CLIPLoader",
        "inputs": {
            "clip_name": "STRING (filename)",
            "type": "STRING (sd1|sd2|sdxl|flux|qwen_image)",
            "device": "STRING (default|cpu)"
        },
        "outputs": ["CLIP"],
        "category": "loaders"
    },
    "DualCLIPLoader": {
        "class": "DualCLIPLoader",
        "inputs": {
            "clip_name1": "STRING (CLIP L)",
            "clip_name2": "STRING (T5 XXL)",
            "type": "STRING (flux|sd3)"
        },
        "outputs": ["CLIP"],
        "category": "loaders"
    },
    "VAELoader": {
        "class": "VAELoader",
        "inputs": {
            "vae_name": "STRING (filename)"
        },
        "outputs": ["VAE"],
        "category": "loaders"
    },
    "LoraLoader": {
        "class": "LoraLoader",
        "inputs": {
            "model": "MODEL",
            "clip": "CLIP",
            "lora_name": "STRING (filename)",
            "strength_model": "FLOAT (0.0-2.0, default: 1.0)",
            "strength_clip": "FLOAT (0.0-2.0, default: 1.0)"
        },
        "outputs": ["MODEL", "CLIP"],
        "category": "loaders"
    },
    "LoraLoaderModelOnly": {
        "class": "LoraLoaderModelOnly",
        "inputs": {
            "model": "MODEL",
            "lora_name": "STRING (filename)",
            "strength_model": "FLOAT (0.0-2.0, default: 1.0)"
        },
        "outputs": ["MODEL"],
        "category": "loaders"
    },

    # Text Encoding Nodes
    "CLIPTextEncode": {
        "class": "CLIPTextEncode",
        "inputs": {
            "clip": "CLIP",
            "text": "STRING (prompt)"
        },
        "outputs": ["CONDITIONING"],
        "category": "conditioning"
    },
    "FluxGuidance": {
        "class": "FluxGuidance",
        "inputs": {
            "conditioning": "CONDITIONING",
            "guidance": "FLOAT (0.0-100.0, default: 3.5)"
        },
        "outputs": ["CONDITIONING"],
        "category": "conditioning",
        "notes": "FLUX-specific. Replaces CFG parameter."
    },

    # LTX-Video Specific Nodes
    "LTXVConditioning": {
        "class": "LTXVConditioning",
        "inputs": {
            "positive": "CONDITIONING",
            "negative": "CONDITIONING",
            "frame_rate": "FLOAT (default: 24)"
        },
        "outputs": ["CONDITIONING (positive)", "CONDITIONING (negative)"],
        "category": "ltx",
        "notes": "Required wrapper for LTX conditioning."
    },
    "LTXVScheduler": {
        "class": "LTXVScheduler",
        "inputs": {
            "steps": "INT (1-100, default: 30)",
            "max_shift": "FLOAT (default: 2.05)",
            "base_shift": "FLOAT (default: 0.95)",
            "stretch": "BOOLEAN (default: true)",
            "terminal": "FLOAT (default: 0.1)"
        },
        "outputs": ["SIGMAS"],
        "category": "ltx"
    },
    "EmptyLTXVLatentVideo": {
        "class": "EmptyLTXVLatentVideo",
        "inputs": {
            "width": "INT (divisible by 8, default: 768)",
            "height": "INT (divisible by 8, default: 512)",
            "length": "INT (must be 8n+1: 9,17,25,...97,...121)",
            "batch_size": "INT (default: 1)"
        },
        "outputs": ["LATENT"],
        "category": "ltx",
        "notes": "Frame count MUST be 8n+1 (e.g., 97, 121)."
    },
    "LTXVImgToVideo": {
        "class": "LTXVImgToVideo",
        "inputs": {
            "image": "IMAGE",
            "vae": "VAE",
            "width": "INT",
            "height": "INT",
            "length": "INT (8n+1)"
        },
        "outputs": ["LATENT"],
        "category": "ltx"
    },
    "LTXVPreprocess": {
        "class": "LTXVPreprocess",
        "inputs": {
            "image": "IMAGE",
            "strength": "INT (0-100, default: 40)"
        },
        "outputs": ["IMAGE"],
        "category": "ltx"
    },
    "LTXVGemmaCLIPModelLoader": {
        "class": "LTXVGemmaCLIPModelLoader",
        "inputs": {
            "model_name": "STRING (gemma_3_12B_it_fp8)"
        },
        "outputs": ["GEMMA_MODEL"],
        "category": "ltx"
    },
    "LTXVGemmaEnhancePrompt": {
        "class": "LTXVGemmaEnhancePrompt",
        "inputs": {
            "gemma_model": "GEMMA_MODEL",
            "prompt": "STRING"
        },
        "outputs": ["STRING (enhanced prompt)"],
        "category": "ltx"
    },

    # Wan 2.1 Specific Nodes
    "WanVideoModelLoader": {
        "class": "WanVideoModelLoader",
        "inputs": {
            "model_name": "STRING (wan_2.1_*.safetensors)"
        },
        "outputs": ["MODEL", "VAE"],
        "category": "wan"
    },
    "EmptyWanLatentVideo": {
        "class": "EmptyWanLatentVideo",
        "inputs": {
            "width": "INT (480p: 832, 720p: 1280)",
            "height": "INT (480p: 480, 720p: 720)",
            "frames": "INT (default: 81)",
            "batch_size": "INT (default: 1)"
        },
        "outputs": ["LATENT"],
        "category": "wan"
    },
    "WanImageEncode": {
        "class": "WanImageEncode",
        "inputs": {
            "image": "IMAGE",
            "vae": "VAE"
        },
        "outputs": ["LATENT"],
        "category": "wan"
    },
    "WanVAEDecode": {
        "class": "WanVAEDecode",
        "inputs": {
            "samples": "LATENT",
            "vae": "VAE"
        },
        "outputs": ["IMAGE"],
        "category": "wan"
    },

    # FLUX Specific Nodes
    "ModelSamplingFlux": {
        "class": "ModelSamplingFlux",
        "inputs": {
            "model": "MODEL",
            "width": "INT",
            "height": "INT"
        },
        "outputs": ["MODEL"],
        "category": "flux"
    },

    # Qwen Specific Nodes
    "ModelSamplingAuraFlow": {
        "class": "ModelSamplingAuraFlow",
        "inputs": {
            "model": "MODEL",
            "shift": "FLOAT (default: 3.1)"
        },
        "outputs": ["MODEL"],
        "category": "qwen"
    },

    # Latent Generation Nodes
    "EmptyLatentImage": {
        "class": "EmptyLatentImage",
        "inputs": {
            "width": "INT (divisible by 8)",
            "height": "INT (divisible by 8)",
            "batch_size": "INT (default: 1)"
        },
        "outputs": ["LATENT"],
        "category": "latent"
    },
    "EmptySD3LatentImage": {
        "class": "EmptySD3LatentImage",
        "inputs": {
            "width": "INT (divisible by 16 for FLUX)",
            "height": "INT (divisible by 16 for FLUX)",
            "batch_size": "INT (default: 1)"
        },
        "outputs": ["LATENT"],
        "category": "latent",
        "notes": "Use for FLUX (divisible by 16) and SD3."
    },

    # Sampler Nodes
    "KSampler": {
        "class": "KSampler",
        "inputs": {
            "model": "MODEL",
            "positive": "CONDITIONING",
            "negative": "CONDITIONING",
            "latent_image": "LATENT",
            "seed": "INT",
            "steps": "INT (1-100)",
            "cfg": "FLOAT (1.0-30.0)",
            "sampler_name": "STRING (euler|euler_ancestral|dpmpp_2m|...)",
            "scheduler": "STRING (normal|karras|exponential|simple|...)",
            "denoise": "FLOAT (0.0-1.0, default: 1.0)"
        },
        "outputs": ["LATENT"],
        "category": "sampling",
        "notes": "Do NOT use for video models. Use SamplerCustom instead."
    },
    "KSamplerSelect": {
        "class": "KSamplerSelect",
        "inputs": {
            "sampler_name": "STRING (euler|dpmpp_2m|res_multistep|...)"
        },
        "outputs": ["SAMPLER"],
        "category": "sampling"
    },
    "SamplerCustom": {
        "class": "SamplerCustom",
        "inputs": {
            "model": "MODEL",
            "positive": "CONDITIONING",
            "negative": "CONDITIONING",
            "sampler": "SAMPLER",
            "sigmas": "SIGMAS",
            "latent_image": "LATENT",
            "add_noise": "BOOLEAN (default: true)",
            "noise_seed": "INT"
        },
        "outputs": ["LATENT", "LATENT (denoised)"],
        "category": "sampling",
        "notes": "REQUIRED for video models (LTX, Wan)."
    },
    "BasicScheduler": {
        "class": "BasicScheduler",
        "inputs": {
            "model": "MODEL",
            "scheduler": "STRING (normal|karras|exponential|simple|sgm_uniform)",
            "steps": "INT (1-100)",
            "denoise": "FLOAT (0.0-1.0, default: 1.0)"
        },
        "outputs": ["SIGMAS"],
        "category": "sampling"
    },

    # Decode/Encode Nodes
    "VAEDecode": {
        "class": "VAEDecode",
        "inputs": {
            "samples": "LATENT",
            "vae": "VAE"
        },
        "outputs": ["IMAGE"],
        "category": "vae"
    },
    "VAEEncode": {
        "class": "VAEEncode",
        "inputs": {
            "pixels": "IMAGE",
            "vae": "VAE"
        },
        "outputs": ["LATENT"],
        "category": "vae"
    },

    # Output Nodes
    "SaveImage": {
        "class": "SaveImage",
        "inputs": {
            "images": "IMAGE",
            "filename_prefix": "STRING (default: ComfyUI)"
        },
        "outputs": [],
        "category": "output"
    },
    "SaveAnimatedWEBP": {
        "class": "SaveAnimatedWEBP",
        "inputs": {
            "images": "IMAGE",
            "filename_prefix": "STRING",
            "fps": "INT (default: 24)",
            "lossless": "BOOLEAN (default: false)",
            "quality": "INT (0-100, default: 80)"
        },
        "outputs": [],
        "category": "output"
    },
    "VHS_VideoCombine": {
        "class": "VHS_VideoCombine",
        "inputs": {
            "images": "IMAGE",
            "frame_rate": "FLOAT (default: 24)",
            "format": "STRING (webm_video|mp4_video|...)"
        },
        "outputs": ["VHS_FILENAMES"],
        "category": "output"
    },

    # Image Input Nodes
    "LoadImage": {
        "class": "LoadImage",
        "inputs": {
            "image": "STRING (filename)"
        },
        "outputs": ["IMAGE", "MASK"],
        "category": "input"
    },
    "ImageScale": {
        "class": "ImageScale",
        "inputs": {
            "image": "IMAGE",
            "upscale_method": "STRING (nearest|bilinear|bicubic|lanczos)",
            "width": "INT",
            "height": "INT",
            "crop": "STRING (disabled|center)"
        },
        "outputs": ["IMAGE"],
        "category": "image"
    },
}


def get_node_spec(node_class_name: str) -> dict:
    """
    Get the specification for a node by class name.

    Args:
        node_class_name: The node class name (e.g., "LTXVConditioning")

    Returns:
        Node specification with inputs, outputs, and notes.
    """
    spec = NODE_SPECS.get(node_class_name)
    if spec:
        return spec
    return {"error": f"Unknown node: {node_class_name}", "available": list(NODE_SPECS.keys())}


def list_node_specs(category: str = None) -> list:
    """
    List all node specifications, optionally filtered by category.

    Args:
        category: Optional category filter (loaders, conditioning, ltx, wan, flux, qwen, etc.)

    Returns:
        List of node specifications.
    """
    if category:
        return [
            {"class": k, **v}
            for k, v in NODE_SPECS.items()
            if v.get("category") == category
        ]
    return [{"class": k, **v} for k, v in NODE_SPECS.items()]


def get_node_categories() -> list:
    """Get all available node categories."""
    return list(set(v.get("category", "uncategorized") for v in NODE_SPECS.values()))
