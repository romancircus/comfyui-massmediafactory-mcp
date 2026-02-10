"""ComfyUI MassMediaFactory MCP Server - Main entry point."""

import json
import os
from pathlib import Path
from typing import List
from mcp.server.fastmcp import FastMCP

from . import discovery
from . import execution
from . import validation
from . import templates
from . import patterns
from . import publish
from . import reference_docs
from . import topology_validator
from . import workflow_generator
from . import prompt_enhance
from .mcp_utils import (
    mcp_error,
    mcp_tool_wrapper,
    paginate,
)

# Initialize MCP server
mcp = FastMCP(
    "comfyui-massmediafactory",
    instructions="ComfyUI workflow orchestration for image and video generation",
)


def _to_mcp_response(result: dict) -> dict:
    """Convert result to MCP format with isError flag."""
    if isinstance(result, dict) and "error" in result and "isError" not in result:
        return {
            **result,
            "isError": True,
            "code": result.get("code", "TOOL_ERROR"),
        }
    return result


def _validate_path(path: str, allowed_base: str) -> tuple[bool, str]:
    """Validate path is within allowed directory. Prevents path traversal."""
    try:
        resolved = Path(path).resolve()
        allowed = Path(allowed_base).resolve()
        # Use is_relative_to() to prevent prefix attacks like /home/user_evil matching /home/user
        if not resolved.is_relative_to(allowed):
            return False, f"Path '{path}' is outside allowed directory '{allowed_base}'"
        return True, str(resolved)
    except Exception as e:
        return False, f"Invalid path: {e}"


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL is from allowed domains. Prevents arbitrary downloads."""
    from urllib.parse import urlparse

    allowed_domains = [
        "civitai.com",
        "huggingface.co",
        "github.com",
        "raw.githubusercontent.com",
    ]

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False, f"URL '{url}' has no hostname"

        # Check exact match or subdomain match (e.g., www.civitai.com, api.huggingface.co)
        hostname_lower = hostname.lower()
        for domain in allowed_domains:
            if hostname_lower == domain or hostname_lower.endswith(f".{domain}"):
                return True, ""

        return (
            False,
            f"URL '{url}' is not from allowed domains: {', '.join(allowed_domains)}",
        )
    except Exception as e:
        return False, f"Invalid URL: {e}"


def _escape_user_content(text: str) -> str:
    """Escape user content for safe inclusion in prompts. Prevents injection."""
    cleaned = "".join(char for char in text if char.isprintable() or char in "\n\t")
    return cleaned[:10000]


# =============================================================================
# Discovery Tools (3)
# =============================================================================


@mcp.tool()
@mcp_tool_wrapper
def list_models(model_type: str = "all") -> dict:
    """List models by type. type: checkpoint|unet|lora|vae|controlnet|all"""
    if model_type == "all":
        return _to_mcp_response(discovery.get_all_models())
    type_map = {
        "checkpoint": discovery.list_checkpoints,
        "unet": discovery.list_unets,
        "lora": discovery.list_loras,
        "vae": discovery.list_vaes,
        "controlnet": discovery.list_controlnets,
    }
    if model_type not in type_map:
        return mcp_error(f"Unknown model type: {model_type}. Use: checkpoint|unet|lora|vae|controlnet|all")
    return _to_mcp_response(type_map[model_type]())


@mcp.tool()
@mcp_tool_wrapper
def get_node_info(node_type: str) -> dict:
    """Get ComfyUI node schema by class name."""
    return _to_mcp_response(discovery.get_node_info(node_type))


@mcp.tool()
@mcp_tool_wrapper
def search_nodes(query: str) -> dict:
    """Search ComfyUI nodes by name/category."""
    return discovery.search_nodes(query)


# =============================================================================
# System Tools (3)
# =============================================================================


@mcp.tool()
@mcp_tool_wrapper
def get_system_stats() -> dict:
    """Get GPU VRAM and system stats."""
    return execution.get_system_stats()


@mcp.tool()
@mcp_tool_wrapper
def free_memory(unload_models: bool = False) -> dict:
    """Free GPU memory. unload_models=True to clear all."""
    return execution.free_memory(unload_models)


@mcp.tool()
@mcp_tool_wrapper
def interrupt() -> dict:
    """Stop currently running workflow."""
    return execution.interrupt_execution()


# =============================================================================
# I/O Tools (2)
# =============================================================================


@mcp.tool()
@mcp_tool_wrapper
def upload_image(
    image_path: str,
    filename: str = None,
    subfolder: str = "",
    overwrite: bool = True,
) -> dict:
    """Upload image for ControlNet/I2V workflows."""
    home_dir = os.path.expanduser("~")
    valid, resolved_path = _validate_path(image_path, home_dir)
    if not valid:
        return mcp_error(resolved_path, "VALIDATION_ERROR")
    return execution.upload_image(resolved_path, filename, subfolder, overwrite)


@mcp.tool()
@mcp_tool_wrapper
def download_output(asset_id: str, output_path: str) -> dict:
    """Download asset to local file."""
    home_dir = os.path.expanduser("~")
    valid, resolved_path = _validate_path(output_path, home_dir)
    if not valid:
        return mcp_error(resolved_path, "VALIDATION_ERROR")
    return execution.download_output(asset_id, resolved_path)


# =============================================================================
# Publishing Tools (3)
# =============================================================================


@mcp.tool()
@mcp_tool_wrapper
def publish_asset(
    asset_id: str,
    target_filename: str = None,
    manifest_key: str = None,
    publish_dir: str = None,
    web_optimize: bool = True,
) -> dict:
    """Publish asset to web dir. Use target_filename or manifest_key."""
    return publish.publish_asset(
        asset_id=asset_id,
        target_filename=target_filename,
        manifest_key=manifest_key,
        publish_dir=publish_dir,
        web_optimize=web_optimize,
    )


@mcp.tool()
@mcp_tool_wrapper
def get_publish_info() -> dict:
    """Get current publish directory config."""
    return publish.get_publish_info()


@mcp.tool()
@mcp_tool_wrapper
def set_publish_dir(publish_dir: str) -> dict:
    """Set publish directory path."""
    return publish.set_publish_dir(publish_dir)


# =============================================================================
# Validation Tool (1)
# =============================================================================


@mcp.tool()
@mcp_tool_wrapper
def validate_workflow(
    workflow: dict,
    model: str = None,
    auto_fix: bool = False,
    check_pattern: bool = False,
) -> dict:
    """Validate workflow. auto_fix=True to correct params. check_pattern=True for drift detection."""
    import json

    errors = []
    warnings = []
    corrections = []
    result_workflow = workflow

    # 1. Basic validation (node types, connections)
    basic_result = validation.validate_workflow(workflow)
    if "error" in basic_result:
        return _to_mcp_response(basic_result)
    if basic_result.get("errors"):
        errors.extend(basic_result["errors"])
    if basic_result.get("warnings"):
        warnings.extend(basic_result["warnings"])

    # 2. Topology validation (resolution, frames, CFG, model constraints)
    workflow_json = json.dumps(workflow)
    topo_result = topology_validator.validate_topology(workflow_json, model)
    if topo_result.get("errors"):
        errors.extend(topo_result["errors"])
    if topo_result.get("warnings"):
        warnings.extend(topo_result["warnings"])
    detected_model = topo_result.get("model_detected", model)

    # 3. Pattern validation (drift detection)
    if check_pattern and detected_model:
        pattern_result = patterns.validate_against_pattern(workflow, detected_model)
        if pattern_result.get("errors"):
            errors.extend(pattern_result["errors"])
        if pattern_result.get("warnings"):
            warnings.extend(pattern_result["warnings"])

    # 4. Auto-fix if requested
    if auto_fix and errors:
        fix_result = topology_validator.auto_correct_parameters(workflow, detected_model)
        if fix_result.get("corrections"):
            corrections = fix_result["corrections"]
            result_workflow = fix_result.get("workflow", workflow)
            # Re-validate after fixing
            errors = []  # Clear errors, re-check
            recheck = topology_validator.validate_topology(json.dumps(result_workflow), detected_model)
            if recheck.get("errors"):
                errors = recheck["errors"]

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "corrections": corrections if auto_fix else [],
        "workflow": result_workflow if auto_fix else None,
        "model_detected": detected_model,
    }


# Internal helper - not exposed as MCP tool (too specialized)
def check_connection(source_type: str, source_slot: int, target_type: str, target_input: str) -> dict:
    """Check if node connection is type-compatible."""
    return validation.check_node_compatibility(source_type, source_slot, target_type, target_input)


# =============================================================================
# Template Tools (2)
# =============================================================================


@mcp.tool()
@mcp_tool_wrapper
def list_workflow_templates(
    limit: int = 50,
    cursor: str = None,
    only_installed: bool = False,
    model_type: str = None,
    tags: List[str] = None,
) -> dict:
    """List available workflow templates. Paginated. Supports filtering."""
    if tags is None:
        tags = []
    result = templates.list_templates(validate=False, only_installed=only_installed, model_type=model_type, tags=tags)
    if "error" in result:
        return _to_mcp_response(result)
    template_list = result.get("templates", [])
    paginated = paginate(template_list, cursor=cursor, limit=limit)
    return {
        "templates": paginated["items"],
        "nextCursor": paginated["nextCursor"],
        "total": paginated["total"],
    }


@mcp.tool()
@mcp_tool_wrapper
def get_template(name: str) -> dict:
    """Get workflow template by name."""
    return _to_mcp_response(templates.load_template(name))


# =============================================================================
# Workflow Pattern Tools (3)
# =============================================================================


@mcp.tool()
@mcp_tool_wrapper
def get_workflow_skeleton(model: str, task: str) -> dict:
    """Get tested workflow structure for model+task."""
    return patterns.get_workflow_skeleton(model, task)


@mcp.tool()
@mcp_tool_wrapper
def get_model_constraints(model: str) -> dict:
    """Get model constraints (CFG, resolution, frames, required nodes)."""
    return patterns.get_model_constraints(model)


@mcp.tool()
@mcp_tool_wrapper
def get_node_chain(model: str, task: str) -> dict:
    """Get ordered nodes with exact connection slots."""
    result = patterns.get_node_chain(model, task)
    if isinstance(result, dict) and "error" in result:
        return result
    return {"nodes": result, "model": model, "task": task, "count": len(result)}


# =============================================================================
# Prompt Tool (1)
# =============================================================================


@mcp.tool()
@mcp_tool_wrapper
def enhance_prompt(
    prompt: str,
    model: str = "flux",
    style: str = None,
    use_llm: bool = True,
    llm_model: str = None,
) -> dict:
    """Enhance a generation prompt with model-specific quality tokens and optional LLM rewriting.

    Uses local Ollama LLM to intelligently rewrite prompts for better generation results.
    Falls back to token injection if LLM unavailable.

    Args:
        prompt: The original prompt to enhance.
        model: Target generation model (flux, sdxl, qwen, wan, ltx).
        style: Optional style preset (cinematic, anime, photorealistic).
        use_llm: Use Ollama LLM for intelligent rewriting (default True).
        llm_model: Ollama model to use (default: qwen3:8b).
    """
    return _to_mcp_response(
        prompt_enhance.enhance_prompt(
            prompt=prompt,
            model=model,
            style=style,
            use_llm=use_llm,
            llm_model=llm_model,
        )
    )


# =============================================================================
# MCP Resources (13)
# =============================================================================


@mcp.resource(
    "comfyui://docs/patterns/ltx",
    name="LTX Video Pattern",
    description="LLM reference pattern for LTX-Video workflow generation",
    mime_type="text/markdown",
)
def resource_pattern_ltx() -> str:
    """LTX Video pattern documentation."""
    result = reference_docs.get_model_pattern("ltx")
    return result.get("pattern", json.dumps(result))


@mcp.resource(
    "comfyui://docs/patterns/flux",
    name="FLUX Pattern",
    description="LLM reference pattern for FLUX workflow generation",
    mime_type="text/markdown",
)
def resource_pattern_flux() -> str:
    """FLUX pattern documentation."""
    result = reference_docs.get_model_pattern("flux")
    return result.get("pattern", json.dumps(result))


@mcp.resource(
    "comfyui://docs/patterns/wan",
    name="Wan 2.6 Pattern",
    description="LLM reference pattern for Wan 2.6 workflow generation",
    mime_type="text/markdown",
)
def resource_pattern_wan() -> str:
    """Wan 2.6 pattern documentation."""
    result = reference_docs.get_model_pattern("wan")
    return result.get("pattern", json.dumps(result))


@mcp.resource(
    "comfyui://docs/patterns/qwen",
    name="Qwen Pattern",
    description="LLM reference pattern for Qwen workflow generation",
    mime_type="text/markdown",
)
def resource_pattern_qwen() -> str:
    """Qwen pattern documentation."""
    result = reference_docs.get_model_pattern("qwen")
    return result.get("pattern", json.dumps(result))


@mcp.resource(
    "comfyui://docs/rules",
    name="Parameter Rules",
    description="Validation constraints for LLM-generated workflows",
    mime_type="text/markdown",
)
def resource_parameter_rules() -> str:
    """Parameter validation rules."""
    result = reference_docs.get_parameter_rules()
    return result.get("rules", json.dumps(result))


@mcp.resource(
    "comfyui://docs/system-prompt",
    name="LLM System Prompt",
    description="System prompt guide for LLM workflow generation",
    mime_type="text/markdown",
)
def resource_system_prompt() -> str:
    """System prompt for LLM workflow generation."""
    result = reference_docs.get_system_prompt()
    return result.get("system_prompt", json.dumps(result))


@mcp.resource(
    "comfyui://docs/skeletons/ltx-t2v",
    name="LTX T2V Skeleton",
    description="Token-optimized skeleton for LTX text-to-video",
    mime_type="application/json",
)
def resource_skeleton_ltx_t2v() -> str:
    """LTX text-to-video skeleton."""
    result = reference_docs.get_skeleton("ltx", "t2v")
    return json.dumps(result.get("skeleton", result), indent=2)


@mcp.resource(
    "comfyui://docs/skeletons/ltx-i2v",
    name="LTX I2V Skeleton",
    description="Token-optimized skeleton for LTX image-to-video",
    mime_type="application/json",
)
def resource_skeleton_ltx_i2v() -> str:
    """LTX image-to-video skeleton."""
    result = reference_docs.get_skeleton("ltx", "i2v")
    return json.dumps(result.get("skeleton", result), indent=2)


@mcp.resource(
    "comfyui://docs/skeletons/flux-t2i",
    name="FLUX T2I Skeleton",
    description="Token-optimized skeleton for FLUX text-to-image",
    mime_type="application/json",
)
def resource_skeleton_flux_t2i() -> str:
    """FLUX text-to-image skeleton."""
    result = reference_docs.get_skeleton("flux", "t2i")
    return json.dumps(result.get("skeleton", result), indent=2)


@mcp.resource(
    "comfyui://docs/skeletons/wan-t2v",
    name="Wan T2V Skeleton",
    description="Token-optimized skeleton for Wan text-to-video",
    mime_type="application/json",
)
def resource_skeleton_wan_t2v() -> str:
    """Wan text-to-video skeleton."""
    result = reference_docs.get_skeleton("wan", "t2v")
    return json.dumps(result.get("skeleton", result), indent=2)


@mcp.resource(
    "comfyui://docs/skeletons/qwen-t2i",
    name="Qwen T2I Skeleton",
    description="Token-optimized skeleton for Qwen text-to-image",
    mime_type="application/json",
)
def resource_skeleton_qwen_t2i() -> str:
    """Qwen text-to-image skeleton."""
    result = reference_docs.get_skeleton("qwen", "t2i")
    return json.dumps(result.get("skeleton", result), indent=2)


@mcp.resource(
    "comfyui://docs/patterns/available",
    name="Available Patterns",
    description="List of all workflow patterns and supported models",
    mime_type="application/json",
)
def resource_available_patterns() -> str:
    """All available workflow patterns."""
    result = patterns.list_available_patterns()
    return json.dumps(result, indent=2)


@mcp.resource(
    "comfyui://workflows/supported",
    name="Supported Workflows",
    description="List of supported model+workflow_type combinations",
    mime_type="application/json",
)
def resource_supported_workflows() -> str:
    """Supported workflow types."""
    result = workflow_generator.list_supported_workflows()
    return json.dumps(result, indent=2)


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
