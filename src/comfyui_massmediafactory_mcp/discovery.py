"""
Discovery Tools

Tools for discovering available models, nodes, and capabilities in ComfyUI.
"""

from typing import Optional
from .client import get_client


def list_checkpoints() -> dict:
    """
    List all available checkpoint models in ComfyUI.
    Returns model filenames that can be used with CheckpointLoaderSimple or UNETLoader.
    """
    client = get_client()
    result = client.get_object_info("CheckpointLoaderSimple")

    if "error" in result:
        return result

    try:
        models = result["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        return {"checkpoints": models, "count": len(models)}
    except (KeyError, IndexError):
        return {"checkpoints": [], "count": 0, "note": "Could not parse checkpoint list"}


def list_unets() -> dict:
    """
    List all available UNET models (for Flux, etc.).
    """
    client = get_client()
    result = client.get_object_info("UNETLoader")

    if "error" in result:
        return result

    try:
        models = result["UNETLoader"]["input"]["required"]["unet_name"][0]
        return {"unets": models, "count": len(models)}
    except (KeyError, IndexError):
        return {"unets": [], "count": 0, "note": "Could not parse UNET list"}


def list_loras() -> dict:
    """
    List all available LoRA models in ComfyUI.
    """
    client = get_client()
    result = client.get_object_info("LoraLoader")

    if "error" in result:
        return result

    try:
        loras = result["LoraLoader"]["input"]["required"]["lora_name"][0]
        return {"loras": loras, "count": len(loras)}
    except (KeyError, IndexError):
        return {"loras": [], "count": 0, "note": "Could not parse LoRA list"}


def list_vaes() -> dict:
    """
    List all available VAE models in ComfyUI.
    """
    client = get_client()
    result = client.get_object_info("VAELoader")

    if "error" in result:
        return result

    try:
        vaes = result["VAELoader"]["input"]["required"]["vae_name"][0]
        return {"vaes": vaes, "count": len(vaes)}
    except (KeyError, IndexError):
        return {"vaes": [], "count": 0, "note": "Could not parse VAE list"}


def list_clip_models() -> dict:
    """
    List all available CLIP models.
    """
    client = get_client()
    result = client.get_object_info("CLIPLoader")

    if "error" in result:
        return result

    try:
        clips = result["CLIPLoader"]["input"]["required"]["clip_name"][0]
        return {"clips": clips, "count": len(clips)}
    except (KeyError, IndexError):
        return {"clips": [], "count": 0, "note": "Could not parse CLIP list"}


def list_controlnets() -> dict:
    """
    List all available ControlNet models.
    """
    client = get_client()
    result = client.get_object_info("ControlNetLoader")

    if "error" in result:
        return result

    try:
        controlnets = result["ControlNetLoader"]["input"]["required"]["control_net_name"][0]
        return {"controlnets": controlnets, "count": len(controlnets)}
    except (KeyError, IndexError):
        return {"controlnets": [], "count": 0, "note": "Could not parse ControlNet list"}


def get_node_info(node_type: str) -> dict:
    """
    Get detailed information about a specific ComfyUI node type.

    Args:
        node_type: The node class name (e.g., "KSampler", "CLIPTextEncode")

    Returns:
        Node schema including inputs, outputs, and their types.
    """
    client = get_client()
    result = client.get_object_info(node_type)

    if "error" in result:
        return result

    if node_type not in result:
        return {"error": f"Node type '{node_type}' not found"}

    node = result[node_type]
    return {
        "name": node_type,
        "category": node.get("category", "unknown"),
        "description": node.get("description", ""),
        "inputs": node.get("input", {}),
        "outputs": node.get("output", []),
        "output_names": node.get("output_name", []),
    }


def search_nodes(query: str, limit: int = 50) -> dict:
    """
    Search for ComfyUI nodes by name or category.

    Args:
        query: Search term (e.g., "sampler", "image", "video", "flux")
        limit: Maximum results to return

    Returns:
        List of matching node types.
    """
    client = get_client()
    result = client.get_object_info()

    if "error" in result:
        return result

    query_lower = query.lower()
    matches = []

    for node_name, node_info in result.items():
        score = 0

        # Exact name match scores highest
        if query_lower == node_name.lower():
            score = 100
        # Name contains query
        elif query_lower in node_name.lower():
            score = 50
        # Category contains query
        elif query_lower in node_info.get("category", "").lower():
            score = 25
        # Description contains query
        elif query_lower in node_info.get("description", "").lower():
            score = 10

        if score > 0:
            matches.append({
                "name": node_name,
                "category": node_info.get("category", "unknown"),
                "score": score,
            })

    # Sort by score descending
    matches.sort(key=lambda x: x["score"], reverse=True)

    return {
        "matches": matches[:limit],
        "total": len(matches),
        "query": query,
    }


def get_all_models() -> dict:
    """
    Get a summary of all available models across all types.
    """
    return {
        "checkpoints": list_checkpoints(),
        "unets": list_unets(),
        "loras": list_loras(),
        "vaes": list_vaes(),
        "clips": list_clip_models(),
        "controlnets": list_controlnets(),
    }
