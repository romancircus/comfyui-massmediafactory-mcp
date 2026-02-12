"""
Dynamic Node Output Type Registry

Replaces the hardcoded NODE_OUTPUT_TYPES in topology_validator.py with
dynamic discovery from ComfyUI's /object_info endpoint.

Queries ComfyUI once, caches the result, and provides a fallback to
hardcoded entries if ComfyUI is unreachable.

Usage:
    from .node_registry import get_node_output_types, get_type_compatibility

    output_types = get_node_output_types()  # dict: class_type -> [output_types]
    compat = get_type_compatibility()       # dict: output_type -> [compatible_input_names]
"""

import logging
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger("comfyui-mcp")

# Cache TTL in seconds (5 minutes — nodes don't change during a session)
_CACHE_TTL = 300

# Thread-safe cache state
_cache_lock = threading.Lock()
_cached_output_types: Optional[Dict[str, List[str]]] = None
_cached_type_compat: Optional[Dict[str, List[str]]] = None
_cache_timestamp: float = 0


# =============================================================================
# Hardcoded Fallback — used when ComfyUI is unreachable
# =============================================================================

_FALLBACK_OUTPUT_TYPES: Dict[str, List[str]] = {
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


# =============================================================================
# Dynamic Discovery
# =============================================================================


def _fetch_all_node_output_types() -> Optional[Dict[str, List[str]]]:
    """
    Query ComfyUI /object_info and extract output types for every registered node.

    Returns:
        Dict mapping class_type -> list of output type strings, or None on failure.
    """
    try:
        from .client import get_client

        client = get_client()
        object_info = client.get_object_info()

        if "error" in object_info:
            logger.warning("node_registry: /object_info returned error: %s", object_info["error"])
            return None

        registry = {}
        for class_type, node_info in object_info.items():
            if not isinstance(node_info, dict):
                continue
            output_types = node_info.get("output", [])
            # output is a list of type strings like ["MODEL", "CLIP", "VAE"]
            if isinstance(output_types, list):
                registry[class_type] = list(output_types)

        logger.info("node_registry: discovered %d node types from ComfyUI", len(registry))
        return registry

    except Exception as e:
        logger.warning("node_registry: failed to fetch from ComfyUI: %s", e)
        return None


def _build_type_compatibility(registry: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Build type compatibility map from the full node registry.

    Scans all node input definitions to learn which input names accept which types.
    Falls back to the hardcoded map if ComfyUI is unreachable.

    Returns:
        Dict mapping output_type -> list of compatible input names.
    """
    try:
        from .client import get_client

        client = get_client()
        object_info = client.get_object_info()

        if "error" in object_info:
            return _FALLBACK_TYPE_COMPATIBILITY.copy()

        # Collect: for each type, which input field names accept it
        type_to_inputs: Dict[str, set] = {}

        for _class_type, node_info in object_info.items():
            if not isinstance(node_info, dict):
                continue
            inputs = node_info.get("input", {})
            for category in ("required", "optional"):
                cat_inputs = inputs.get(category, {})
                if not isinstance(cat_inputs, dict):
                    continue
                for input_name, input_spec in cat_inputs.items():
                    if isinstance(input_spec, list) and len(input_spec) > 0:
                        input_type = input_spec[0]
                        if isinstance(input_type, str) and input_type.isupper():
                            if input_type not in type_to_inputs:
                                type_to_inputs[input_type] = set()
                            type_to_inputs[input_type].add(input_name.lower())

        # Convert to sorted lists
        result = {t: sorted(names) for t, names in type_to_inputs.items()}
        logger.info("node_registry: built type compatibility for %d types", len(result))
        return result

    except Exception:
        return _FALLBACK_TYPE_COMPATIBILITY.copy()


# Hardcoded type compatibility fallback
_FALLBACK_TYPE_COMPATIBILITY: Dict[str, List[str]] = {
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


# =============================================================================
# Public API (cached)
# =============================================================================


def get_node_output_types(force_refresh: bool = False) -> Dict[str, List[str]]:
    """
    Get the node output type registry.

    First call queries ComfyUI and caches the result. Subsequent calls
    return the cache until TTL expires. Falls back to hardcoded entries
    if ComfyUI is unreachable.

    Args:
        force_refresh: Bypass cache and re-query ComfyUI.

    Returns:
        Dict mapping node class_type -> list of output type strings.
    """
    global _cached_output_types, _cache_timestamp

    with _cache_lock:
        now = time.time()
        if not force_refresh and _cached_output_types is not None and (now - _cache_timestamp) < _CACHE_TTL:
            return _cached_output_types

    # Fetch outside the lock (network I/O can be slow)
    fetched = _fetch_all_node_output_types()

    with _cache_lock:
        if fetched is not None:
            _cached_output_types = fetched
            _cache_timestamp = time.time()
            return _cached_output_types

        # Fallback: return previous cache if available
        if _cached_output_types is not None:
            return _cached_output_types

        logger.info("node_registry: using fallback hardcoded registry (%d entries)", len(_FALLBACK_OUTPUT_TYPES))
        _cached_output_types = _FALLBACK_OUTPUT_TYPES.copy()
        _cache_timestamp = time.time()
        return _cached_output_types


def get_type_compatibility(force_refresh: bool = False) -> Dict[str, List[str]]:
    """
    Get the type compatibility map.

    Maps output type names to compatible input field names.

    Args:
        force_refresh: Bypass cache and re-query ComfyUI.

    Returns:
        Dict mapping output_type -> list of compatible input names.
    """
    global _cached_type_compat

    with _cache_lock:
        if not force_refresh and _cached_type_compat is not None:
            return _cached_type_compat

    # Ensure output types are loaded first (has its own locking)
    registry = get_node_output_types(force_refresh=force_refresh)
    compat = _build_type_compatibility(registry)

    with _cache_lock:
        _cached_type_compat = compat
        return _cached_type_compat


def invalidate_cache():
    """Clear the cached registry, forcing a re-fetch on next access."""
    global _cached_output_types, _cached_type_compat, _cache_timestamp
    with _cache_lock:
        _cached_output_types = None
        _cached_type_compat = None
        _cache_timestamp = 0


def get_registry_stats() -> dict:
    """
    Get statistics about the current node registry.

    Returns:
        Dict with node count, source (dynamic/fallback), cache age, etc.
    """
    with _cache_lock:
        if _cached_output_types is None:
            return {"loaded": False, "source": "none", "node_count": 0}

        is_fallback = _cached_output_types is _FALLBACK_OUTPUT_TYPES or len(_cached_output_types) <= len(
            _FALLBACK_OUTPUT_TYPES
        )
        cache_age = time.time() - _cache_timestamp if _cache_timestamp > 0 else -1

        return {
            "loaded": True,
            "source": "fallback" if is_fallback else "dynamic",
            "node_count": len(_cached_output_types),
            "fallback_count": len(_FALLBACK_OUTPUT_TYPES),
            "cache_age_seconds": round(cache_age, 1),
            "cache_ttl_seconds": _CACHE_TTL,
        }
