"""
Workflow Templates

Pre-built workflow templates for SOTA models with {{PLACEHOLDER}} syntax.
Includes validation to ensure templates are well-formed.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..param_inject import inject_placeholders
from .. import discovery

# Import error formatters with fallback for module loading
try:
    from core.errors import format_template_metadata

    _HAS_CORE_ERRORS = True
except (ImportError, ModuleNotFoundError):
    _HAS_CORE_ERRORS = False
    format_template_metadata = None


TEMPLATES_DIR = Path(__file__).parent

# Cached installed models (populated lazily per list_templates call)
_installed_models_cache: Optional[List[str]] = None


# Keywords to match template model names against installed model filenames
MODEL_FILENAME_KEYWORDS = {
    "flux2": ["flux"],
    "ltx2": ["ltx"],
    "wan26": ["wan"],
    "wan22": ["wan"],
    "qwen": ["qwen"],
    "qwen_edit": ["qwen"],
    "hunyuan15": ["hunyuan"],
    "sdxl": ["sdxl", "sd_xl"],
    "telestyle": ["telestyle", "qwen"],
    "z_turbo": ["z-image-turbo", "z_turbo", "zturbo"],
    "audio": [],  # Audio templates don't need model files
    "utility": [],  # Utility templates don't need model files
}


def _get_installed_models() -> List[str]:
    """Get all installed model filenames from ComfyUI. Cached per call."""
    global _installed_models_cache
    if _installed_models_cache is not None:
        return _installed_models_cache

    all_models = []
    try:
        checkpoints = discovery.list_checkpoints()
        if "checkpoints" in checkpoints:
            all_models.extend(checkpoints["checkpoints"])
    except Exception:
        pass

    try:
        unets = discovery.list_unets()
        if "unets" in unets:
            all_models.extend(unets["unets"])
    except Exception:
        pass

    # Lowercase for matching
    _installed_models_cache = [m.lower() for m in all_models]
    return _installed_models_cache


def _is_model_installed(model_type_normalized: str) -> bool:
    """Check if a model type has any matching installed models."""
    if not model_type_normalized:
        return True  # Unknown model type - don't filter

    # Audio and utility templates don't require model files
    if model_type_normalized in ("audio", "utility"):
        return True

    keywords = MODEL_FILENAME_KEYWORDS.get(model_type_normalized, [])
    if not keywords:
        return True  # No keywords defined - don't filter

    installed = _get_installed_models()
    if not installed:
        return True  # ComfyUI unreachable - don't filter

    return any(any(kw in model_file for kw in keywords) for model_file in installed)


# Model type mapping for filtering
MODEL_TYPE_MAP = {
    "flux2": ["flux2", "flux.2", "flux2-dev", "flux", "fl.2"],
    "ltx2": ["ltx2", "ltx-2", "ltxvideo", "ltx"],
    "wan26": ["wan26", "wan 2.6", "wan2.6", "wan 2.1", "wan2.1"],
    "wan22": ["wan22", "wan 2.2", "wan2.2", "wan 2.2 s2v", "wan"],
    "qwen": ["qwen", "qwen-image"],
    "qwen_edit": ["qwen_edit", "qwen-edit", "qwen_image_edit"],
    "hunyuan15": ["hunyuanvideo 1.5", "hunyuan15", "hunyuan"],
    "sdxl": ["sdxl", "stable diffusion xl"],
    "telestyle": ["telestyle", "tele_style"],
    "z_turbo": ["z-image-turbo", "z_turbo", "z-turbo", "zturbo", "z_image_turbo"],
    "audio": ["audio", "f5_tts", "chatterbox", "qwen3_tts"],
    "utility": ["utility", "video"],
}


def get_model_type(model_name: str) -> Optional[str]:
    """Normalize model name to canonical type."""
    if not model_name:
        return None
    model_lower = model_name.lower()

    # Check for exact matches first
    # Order matters: more specific variants before general ones
    # - qwen_edit before qwen (both contain "qwen")
    # - wan26 before wan22 (wan22 has bare "wan" as fallback for unversioned names)
    for canonical in [
        "qwen_edit",
        "flux2",
        "ltx2",
        "wan26",
        "wan22",
        "telestyle",
        "qwen",
        "hunyuan15",
        "sdxl",
        "z_turbo",
        "audio",
        "utility",
    ]:
        if canonical not in MODEL_TYPE_MAP:
            continue
        for var in MODEL_TYPE_MAP[canonical]:
            if var.lower() in model_lower:
                return canonical

    return model_lower if model_lower else None


def validate_template(template: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Validate template structure.

    Args:
        template: Template dict to validate

    Returns:
        Tuple of (errors, warnings)
    """
    errors = []
    warnings = []

    # Check _meta section exists
    if "_meta" not in template:
        errors.append("Missing _meta section")
        return errors, warnings

    meta = template["_meta"]

    # Check required _meta fields
    required_meta = ["description", "model", "type", "parameters", "defaults"]
    for field in required_meta:
        if field not in meta:
            if field == "defaults":
                warnings.append(f"Missing _meta.{field} (optional but recommended)")
            else:
                errors.append(f"Missing _meta.{field}")

    # Check versioning fields (recommended but not required)
    if "version" not in meta:
        warnings.append("Missing _meta.version (run sync_templates.py to stamp)")
    if "hub_hash" not in meta:
        warnings.append("Missing _meta.hub_hash (run sync_templates.py to stamp)")

    # Check that parameters is a list
    if "parameters" in meta and not isinstance(meta["parameters"], list):
        errors.append("_meta.parameters must be a list")

    # Check that defaults is a dict
    if "defaults" in meta and not isinstance(meta["defaults"], dict):
        errors.append("_meta.defaults must be a dict")

    # Find all placeholders in workflow
    workflow_str = json.dumps(template)
    placeholders = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", workflow_str))

    # Check placeholders are declared in parameters
    declared_params = set(meta.get("parameters", []))
    for placeholder in placeholders:
        if placeholder not in declared_params:
            warnings.append(f"Undeclared placeholder: {{{{{placeholder}}}}}")

    # Check declared parameters have placeholders
    for param in declared_params:
        if param not in placeholders:
            warnings.append(f"Declared parameter '{param}' not used in workflow")

    # Check node structure
    node_count = 0
    for key, value in template.items():
        if key.startswith("_"):
            continue

        node_count += 1

        if not isinstance(value, dict):
            errors.append(f"Node '{key}' must be a dict")
            continue

        if "class_type" not in value:
            errors.append(f"Node '{key}' missing class_type")

        if "inputs" not in value:
            warnings.append(f"Node '{key}' has no inputs (may be intentional)")

        # Check connections format
        inputs = value.get("inputs", {})
        for input_name, input_value in inputs.items():
            if isinstance(input_value, list):
                if len(input_value) != 2:
                    errors.append(
                        f"Node '{key}' input '{input_name}' has invalid connection format (expected [node_id, slot])"
                    )
                elif not isinstance(input_value[1], int):
                    errors.append(f"Node '{key}' input '{input_name}' slot must be integer")

    if node_count == 0:
        errors.append("Template has no nodes")

    return errors, warnings


def load_template(name: str, validate: bool = True) -> Dict[str, Any]:
    """
    Load a template by name with optional validation.

    Args:
        name: Template name (without .json extension)
        validate: Whether to validate the template (default True)

    Returns:
        Template dict, or {"error": ...} if not found or invalid
    """
    template_file = TEMPLATES_DIR / f"{name}.json"
    if not template_file.exists():
        # Use centralized error format
        if _HAS_CORE_ERRORS and format_template_metadata is not None:
            return format_template_metadata(template_name=name, missing_fields=["Template file doesn't exist"])
        return {"error": f"Template '{name}' not found"}

    try:
        template = json.loads(template_file.read_text())
    except json.JSONDecodeError as e:
        return {"error": f"Template '{name}' has invalid JSON: {e}"}

    if validate:
        errors, warnings = validate_template(template)
        if errors:
            # Use centralized error formatter if available
            if _HAS_CORE_ERRORS and format_template_metadata is not None:
                # Ensure errors is a list for type safety
                errors_list = list(errors) if errors else []

                # Check if missing _meta fields
                if "_meta" not in template:
                    return format_template_metadata(
                        template_name=name,
                        missing_fields=["_meta section"],
                        errors=errors_list,
                    )
                # Check for required _meta fields
                meta = template.get("_meta", {})
                missing = [f for f in ["description", "model", "type", "parameters"] if f not in meta]
                if missing:
                    return format_template_metadata(template_name=name, missing_fields=missing, errors=errors_list)
                # Generic validation error
                return format_template_metadata(template_name=name, errors=errors_list)
            # Fallback with old format
            return {
                "error": f"Template '{name}' has validation errors",
                "validation_errors": errors,
                "validation_warnings": warnings,
            }
        # Attach warnings to template metadata if any
        if warnings:
            template["_validation_warnings"] = warnings

    return template


def list_templates(
    validate: bool = False,
    only_installed: bool = False,
    model_type: str = None,
    tags: List[str] = None,
) -> Dict[str, Any]:
    """
    List all available templates.

    Args:
        validate: Whether to validate each template (default False for speed)
        only_installed: Filter to only templates with installed models
        model_type: Filter by model type (flux2, ltx2, wan26, etc.)
        tags: Filter templates with specific tags (AND matching)

    Returns:
        Dict with templates list and count
    """
    if tags is None:
        tags = []

    # Reset installed model cache for fresh results
    global _installed_models_cache
    if only_installed:
        _installed_models_cache = None

    templates = []
    validation_issues = []

    for f in TEMPLATES_DIR.glob("*.json"):
        if f.name.startswith("."):
            continue
        try:
            data = json.loads(f.read_text())
            meta = data.get("_meta", {})

            # Extract model from _meta.model and normalize
            model = meta.get("model", "")
            model_type_normalized = get_model_type(model)

            # Template filename can override model_type resolution for ambiguous cases
            # (e.g., telestyle_image uses same model as qwen_edit but is a different type)
            if f.stem.startswith("telestyle"):
                model_type_normalized = "telestyle"

            # Apply model_type filter
            if model_type and model_type_normalized != model_type.lower():
                continue

            # Apply tags filter (AND matching - all tags must be present)
            if tags:
                template_tags = meta.get("tags", [])
                if not all(tag in template_tags for tag in tags):
                    continue

            # Apply only_installed filter (check if model is available)
            if only_installed:
                if not _is_model_installed(model_type_normalized):
                    continue

            template_info = {
                "name": f.stem,
                "description": meta.get("description", ""),
                "model": model,
                "type": meta.get("type", ""),
                "parameters": meta.get("parameters", []),
                "vram_min": meta.get("vram_min"),
                "tags": meta.get("tags", []),
            }

            if validate:
                errors, warnings = validate_template(data)
                if errors or warnings:
                    template_info["validation"] = {
                        "valid": len(errors) == 0,
                        "errors": errors,
                        "warnings": warnings,
                    }
                    if errors:
                        validation_issues.append(f.stem)

            templates.append(template_info)

        except json.JSONDecodeError:
            validation_issues.append(f.stem)

    result = {"templates": templates, "count": len(templates)}
    if validation_issues:
        result["templates_with_issues"] = validation_issues

    return result


def inject_parameters(template: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inject parameters into a template.

    Replaces {{PARAM_NAME}} placeholders with actual values.
    Template defaults from _meta.defaults are used as fallbacks for any
    parameters not explicitly provided.
    Strips _meta and other metadata keys before injection.

    Args:
        template: Template dict with placeholders
        params: Dict of parameter values

    Returns:
        Workflow dict with parameters injected (metadata stripped)
    """
    # Merge template defaults with user params (user params take precedence)
    meta = template.get("_meta", {})
    defaults = meta.get("defaults", {})
    merged_params = {**defaults, **params}
    return inject_placeholders(template, merged_params, strip_meta=True)


def get_template_parameters(name: str) -> Dict[str, Any]:
    """
    Get parameters and defaults for a template.

    Args:
        name: Template name

    Returns:
        Dict with parameters list and defaults
    """
    template = load_template(name, validate=False)
    if "error" in template:
        return template

    meta = template.get("_meta", {})
    return {
        "name": name,
        "parameters": meta.get("parameters", []),
        "defaults": meta.get("defaults", {}),
        "description": meta.get("description", ""),
        "type": meta.get("type", ""),
    }


def validate_all_templates() -> Dict[str, Any]:
    """
    Validate all templates in the directory.

    Returns:
        Dict with validation results for each template
    """
    results = {}

    for f in TEMPLATES_DIR.glob("*.json"):
        if f.name.startswith("."):
            continue
        try:
            data = json.loads(f.read_text())
            errors, warnings = validate_template(data)
            results[f.stem] = {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
            }
        except json.JSONDecodeError as e:
            results[f.stem] = {
                "valid": False,
                "errors": [f"Invalid JSON: {e}"],
                "warnings": [],
            }

    valid_count = sum(1 for r in results.values() if r["valid"])
    return {
        "results": results,
        "total": len(results),
        "valid": valid_count,
        "invalid": len(results) - valid_count,
    }
