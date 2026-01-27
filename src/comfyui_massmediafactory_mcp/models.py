"""
Model Management Module

Provides tools for searching, downloading, and managing ComfyUI models.
Supports Civitai and HuggingFace as model sources.
"""

import os
import re
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional


# Default ComfyUI paths
def get_comfyui_base() -> str:
    """Get ComfyUI base directory from environment or default."""
    return os.environ.get("COMFYUI_PATH", os.path.expanduser("~/ComfyUI"))


# Model type to directory mapping
MODEL_DIRECTORIES = {
    "checkpoint": "models/checkpoints",
    "unet": "models/unet",
    "lora": "models/loras",
    "vae": "models/vae",
    "controlnet": "models/controlnet",
    "clip": "models/clip",
    "upscaler": "models/upscale_models",
    "embedding": "models/embeddings",
}


def search_civitai(
    query: str,
    model_type: str = None,
    nsfw: bool = False,
    limit: int = 10,
) -> dict:
    """
    Search Civitai for models by query.

    Args:
        query: Search term (e.g., "anime style", "flux lora", "cinematic")
        model_type: Filter by type: "Checkpoint", "LORA", "TextualInversion",
                   "Hypernetwork", "AestheticGradient", "Controlnet", "Upscaler"
        nsfw: Include NSFW models (default False)
        limit: Maximum results to return (default 10, max 100)

    Returns:
        {
            "models": [
                {
                    "name": "Model Name",
                    "type": "LORA",
                    "download_url": "https://...",
                    "filename": "model.safetensors",
                    "trigger_words": ["trigger1", "trigger2"],
                    "base_model": "SD 1.5",
                    "rating": 4.8,
                    "downloads": 12345
                }
            ],
            "total": 50
        }
    """
    base_url = "https://civitai.com/api/v1/models"

    params = {
        "query": query,
        "limit": min(limit, 100),
        "nsfw": str(nsfw).lower(),
        "sort": "Highest Rated",
    }

    if model_type:
        # Map common names to Civitai types
        type_mapping = {
            "checkpoint": "Checkpoint",
            "lora": "LORA",
            "embedding": "TextualInversion",
            "controlnet": "Controlnet",
            "upscaler": "Upscaler",
        }
        params["types"] = type_mapping.get(model_type.lower(), model_type)

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e), "models": [], "total": 0}

    models = []
    for item in data.get("items", []):
        # Get the latest/best version
        versions = item.get("modelVersions", [])
        if not versions:
            continue

        version = versions[0]  # Latest version
        files = version.get("files", [])

        # Find the primary file (safetensors preferred)
        primary_file = None
        for f in files:
            if f.get("primary"):
                primary_file = f
                break
            if f.get("name", "").endswith(".safetensors"):
                primary_file = f
                break
        if not primary_file and files:
            primary_file = files[0]

        if not primary_file:
            continue

        models.append({
            "name": item.get("name"),
            "type": item.get("type"),
            "download_url": primary_file.get("downloadUrl"),
            "filename": primary_file.get("name"),
            "size_mb": round(primary_file.get("sizeKB", 0) / 1024, 1),
            "trigger_words": version.get("trainedWords", []),
            "base_model": version.get("baseModel"),
            "rating": item.get("stats", {}).get("rating"),
            "downloads": item.get("stats", {}).get("downloadCount"),
            "civitai_url": f"https://civitai.com/models/{item.get('id')}",
        })

    return {
        "models": models,
        "total": data.get("metadata", {}).get("totalItems", len(models)),
        "query": query,
    }


def download_model(
    url: str,
    model_type: str,
    filename: str = None,
    overwrite: bool = False,
) -> dict:
    """
    Download a model to the appropriate ComfyUI directory.

    Args:
        url: Download URL (Civitai or HuggingFace direct link)
        model_type: Type of model: "checkpoint", "unet", "lora", "vae",
                   "controlnet", "clip", "upscaler", "embedding"
        filename: Target filename (auto-detected from URL if not provided)
        overwrite: Whether to overwrite existing files (default False)

    Returns:
        {
            "success": True,
            "path": "/path/to/model.safetensors",
            "size_mb": 2048.5
        }

    Example:
        # Download a LoRA from Civitai
        result = download_model(
            url="https://civitai.com/api/download/models/123456",
            model_type="lora"
        )
    """
    comfyui_base = get_comfyui_base()

    # Get target directory
    model_type_lower = model_type.lower()
    if model_type_lower not in MODEL_DIRECTORIES:
        return {
            "success": False,
            "error": f"Unknown model type: {model_type}. Valid types: {list(MODEL_DIRECTORIES.keys())}",
        }

    target_dir = Path(comfyui_base) / MODEL_DIRECTORIES[model_type_lower]

    # Create directory if needed
    target_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename
    if not filename:
        # Extract from URL or Content-Disposition
        if "civitai.com" in url:
            # Civitai URLs: need to follow redirect to get filename
            filename = _extract_civitai_filename(url)
        elif "huggingface.co" in url:
            # HuggingFace URLs typically have filename in path
            filename = url.split("/")[-1].split("?")[0]
        else:
            filename = url.split("/")[-1].split("?")[0]

        if not filename or filename == "download":
            filename = f"model_{model_type_lower}.safetensors"

    target_path = target_dir / filename

    # Check if exists
    if target_path.exists() and not overwrite:
        return {
            "success": False,
            "error": f"File already exists: {target_path}. Set overwrite=True to replace.",
            "path": str(target_path),
        }

    # Download with progress
    try:
        # Handle Civitai API token if needed
        headers = {}
        if "civitai.com" in url:
            token = os.environ.get("CIVITAI_API_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=300) as response:
            # Get content-disposition filename if available
            cd = response.headers.get("Content-Disposition")
            if cd and "filename=" in cd:
                match = re.search(r'filename="?([^";\n]+)"?', cd)
                if match:
                    filename = match.group(1)
                    target_path = target_dir / filename

            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192 * 16  # 128KB chunks

            with open(target_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

        size_mb = round(target_path.stat().st_size / (1024 * 1024), 1)

        return {
            "success": True,
            "path": str(target_path),
            "filename": filename,
            "size_mb": size_mb,
            "model_type": model_type_lower,
        }

    except Exception as e:
        # Clean up partial download
        if target_path.exists():
            target_path.unlink()
        return {
            "success": False,
            "error": str(e),
        }


def _extract_civitai_filename(url: str) -> str:
    """Extract filename from Civitai download URL."""
    try:
        # Try HEAD request to get Content-Disposition
        req = urllib.request.Request(url, method="HEAD")
        token = os.environ.get("CIVITAI_API_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")

        with urllib.request.urlopen(req, timeout=10) as response:
            cd = response.headers.get("Content-Disposition")
            if cd and "filename=" in cd:
                match = re.search(r'filename="?([^";\n]+)"?', cd)
                if match:
                    return match.group(1)
    except Exception:
        pass

    # Fallback: extract model ID from URL
    match = re.search(r"/models/(\d+)", url)
    if match:
        return f"civitai_{match.group(1)}.safetensors"

    return None


def get_model_info(model_path: str) -> dict:
    """
    Get metadata about an installed model.

    Args:
        model_path: Path to the model file (can be relative to ComfyUI models dir)

    Returns:
        {
            "exists": True,
            "filename": "model.safetensors",
            "size_mb": 2048.5,
            "type": "safetensors",
            "full_path": "/path/to/model.safetensors"
        }
    """
    comfyui_base = get_comfyui_base()

    # Handle relative paths
    path = Path(model_path)
    if not path.is_absolute():
        # Search in model directories
        for subdir in MODEL_DIRECTORIES.values():
            full_path = Path(comfyui_base) / subdir / model_path
            if full_path.exists():
                path = full_path
                break

    if not path.exists():
        return {
            "exists": False,
            "path": str(model_path),
            "error": "Model not found",
        }

    return {
        "exists": True,
        "filename": path.name,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
        "type": path.suffix.lstrip("."),
        "full_path": str(path),
    }


def list_installed_models(model_type: str = None) -> dict:
    """
    List all installed models, optionally filtered by type.

    Args:
        model_type: Filter by type (e.g., "lora", "checkpoint")

    Returns:
        {
            "models": [
                {"name": "model.safetensors", "type": "lora", "size_mb": 123.4}
            ],
            "total": 50
        }
    """
    comfyui_base = get_comfyui_base()
    models = []

    dirs_to_check = MODEL_DIRECTORIES
    if model_type:
        model_type_lower = model_type.lower()
        if model_type_lower in MODEL_DIRECTORIES:
            dirs_to_check = {model_type_lower: MODEL_DIRECTORIES[model_type_lower]}

    for mtype, subdir in dirs_to_check.items():
        model_dir = Path(comfyui_base) / subdir
        if not model_dir.exists():
            continue

        for f in model_dir.glob("*"):
            if f.is_file() and f.suffix in (".safetensors", ".ckpt", ".pt", ".pth", ".bin"):
                models.append({
                    "name": f.name,
                    "type": mtype,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                })

    return {
        "models": sorted(models, key=lambda x: x["name"]),
        "total": len(models),
    }
