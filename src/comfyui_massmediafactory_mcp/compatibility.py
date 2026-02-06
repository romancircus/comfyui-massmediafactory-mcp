"""
Model Compatibility Matrix for MassMediaFactory MCP

Cross-references installed models, VRAM availability, and supported workflow types.
"""


def get_compatibility_matrix() -> dict:
    """
    Get model compatibility matrix showing which models work with which workflow types.

    Cross-references:
    - Model constraints from model_registry
    - Installed models from discovery
    - Available VRAM from system stats

    Returns:
        {
            "models": [
                {
                    "name": "flux2",
                    "workflow_types": ["t2i"],
                    "installed": True,
                    "fits_vram": True,
                    "vram_required_gb": 12.0,
                    "constraints": {...}
                }
            ],
            "vram_available_gb": 32.0,
            "recommendations": [...]
        }
    """
    from .model_registry import MODEL_CONSTRAINTS, get_model_defaults
    from . import discovery

    # Get available VRAM
    vram_available_gb = None
    try:
        from .client import get_client

        client = get_client()
        stats = client.get_system_stats()
        if "error" not in stats:
            devices = stats.get("devices", [])
            if devices:
                vram_available_gb = round(devices[0].get("vram_free", 0) / (1024**3), 2)
    except Exception:
        pass

    # Get installed models
    installed_checkpoints = set()
    installed_unets = set()
    try:
        checkpoints = discovery.list_checkpoints()
        if isinstance(checkpoints, dict) and "checkpoints" in checkpoints:
            installed_checkpoints = {c.lower() for c in checkpoints["checkpoints"]}
        unets = discovery.list_unets()
        if isinstance(unets, dict) and "unet" in unets:
            installed_unets = {u.lower() for u in unets["unet"]}
    except Exception:
        pass

    all_installed = installed_checkpoints | installed_unets

    # VRAM requirements (approximate, in GB)
    VRAM_REQUIREMENTS = {
        "flux2": 12.0,
        "ltx2": 8.0,
        "wan26": 12.0,
        "qwen": 10.0,
        "qwen_edit": 10.0,
        "sdxl": 6.0,
        "hunyuan15": 14.0,
    }

    # Model name keywords for matching installed files
    MODEL_KEYWORDS = {
        "flux2": ["flux"],
        "ltx2": ["ltx"],
        "wan26": ["wan"],
        "qwen": ["qwen"],
        "qwen_edit": ["qwen"],
        "sdxl": ["sdxl", "sd_xl"],
        "hunyuan15": ["hunyuan"],
    }

    # Workflow type mappings
    WORKFLOW_TYPES = {
        "flux2": ["t2i"],
        "ltx2": ["t2v", "i2v"],
        "wan26": ["t2v", "i2v"],
        "qwen": ["t2i"],
        "qwen_edit": ["edit"],
        "sdxl": ["t2i"],
        "hunyuan15": ["t2v", "i2v"],
    }

    models = []
    recommendations = []

    for model_name, constraints in MODEL_CONSTRAINTS.items():
        keywords = MODEL_KEYWORDS.get(model_name, [model_name])
        installed = any(any(kw in fname for kw in keywords) for fname in all_installed)

        vram_required = VRAM_REQUIREMENTS.get(model_name)
        fits_vram = None
        if vram_available_gb is not None and vram_required is not None:
            fits_vram = vram_available_gb >= vram_required

        workflow_types = WORKFLOW_TYPES.get(model_name, [])
        defaults = get_model_defaults(model_name)

        model_info = {
            "name": model_name,
            "workflow_types": workflow_types,
            "installed": installed,
            "fits_vram": fits_vram,
            "vram_required_gb": vram_required,
            "default_resolution": f"{defaults.get('width', '?')}x{defaults.get('height', '?')}",
            "default_steps": defaults.get("steps"),
            "cfg_range": [
                constraints.get("cfg", {}).get("min"),
                constraints.get("cfg", {}).get("max"),
            ],
        }
        models.append(model_info)

        # Generate recommendations
        if installed and fits_vram is True:
            for wt in workflow_types:
                recommendations.append(
                    {
                        "model": model_name,
                        "workflow_type": wt,
                        "status": "ready",
                        "message": f"{model_name} is installed and fits in VRAM for {wt}",
                    }
                )
        elif installed and fits_vram is False:
            recommendations.append(
                {
                    "model": model_name,
                    "status": "vram_limited",
                    "message": f"{model_name} installed but needs {vram_required}GB, only {vram_available_gb}GB free",
                }
            )
        elif not installed:
            recommendations.append(
                {
                    "model": model_name,
                    "status": "not_installed",
                    "message": f"{model_name} not found in installed models",
                }
            )

    return {
        "models": models,
        "vram_available_gb": vram_available_gb,
        "recommendations": recommendations,
        "total_models": len(models),
        "installed_count": sum(1 for m in models if m["installed"]),
    }
