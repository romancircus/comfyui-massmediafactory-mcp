"""ComfyUI MassMediaFactory MCP Server - Main entry point."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

from . import discovery
from . import execution
from . import persistence
from . import vram
from . import validation
from . import sota
from . import templates
from . import patterns
from . import workflow_builder
from . import batch
from . import pipeline
from . import publish
from . import qa
from . import models
from . import analysis
from . import style_learning
from . import schemas
from . import annotations
from . import node_specs
from . import reference_docs
from . import topology_validator
from . import workflow_generator
from . import websocket_client
from . import visualization
from . import rate_limiter
from .mcp_utils import (
    mcp_error,
    mcp_success,
    not_found_error,
    validation_error,
    timeout_error,
    connection_error,
    rate_limit_error,
    mcp_tool_wrapper,
    paginate,
    validate_required,
    validate_range,
    logger,
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
        if not str(resolved).startswith(str(allowed)):
            return False, f"Path '{path}' is outside allowed directory '{allowed_base}'"
        return True, str(resolved)
    except Exception as e:
        return False, f"Invalid path: {e}"


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL is from allowed domains. Prevents arbitrary downloads."""
    import re
    allowed_domains = [r"civitai\.com", r"huggingface\.co", r"github\.com", r"raw\.githubusercontent\.com"]
    for domain in allowed_domains:
        if re.search(rf"https?://[^/]*{domain}", url, re.IGNORECASE):
            return True, ""
    return False, f"URL '{url}' is not from allowed domains: civitai.com, huggingface.co"


def _escape_user_content(text: str) -> str:
    """Escape user content for safe inclusion in prompts. Prevents injection."""
    cleaned = "".join(char for char in text if char.isprintable() or char in "\n\t")
    return cleaned[:10000]


# Discovery Tools

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
def search_nodes(query: str) -> dict:
    """Search ComfyUI nodes by name/category."""
    return discovery.search_nodes(query)


# Execution Tools

@mcp.tool()
@mcp_tool_wrapper
def execute_workflow(workflow: dict, client_id: str = "massmediafactory") -> dict:
    """Queue workflow for execution. Returns prompt_id."""
    return _to_mcp_response(execution.execute_workflow(workflow, client_id))


@mcp.tool()
def get_workflow_status(prompt_id: str = None) -> dict:
    """Check workflow/queue status. With prompt_id: single job. Without: all jobs."""
    if prompt_id:
        return execution.get_workflow_status(prompt_id)
    return execution.get_queue_status()


@mcp.tool()
@mcp_tool_wrapper
def wait_for_completion(prompt_id: str, timeout_seconds: int = 600) -> dict:
    """Wait for workflow completion. Returns outputs."""
    return _to_mcp_response(execution.wait_for_completion(prompt_id, timeout_seconds))


@mcp.tool()
def get_progress(prompt_id: str) -> dict:
    """Get current progress for a workflow. Returns stage, percent, eta, nodes.
    
    Returns:
        {
            "stage": "queued|running|progress|completed|error",
            "percent": 0.0-100.0,
            "eta_seconds": float or null,
            "nodes_completed": int,
            "nodes_total": int,
            "message": str or null,
            "timestamp": ISO datetime
        }
    """
    progress = websocket_client.get_progress_sync(prompt_id)
    if progress:
        return {"isError": False, **progress}
    return {
        "isError": True,
        "error": f"No progress found for prompt_id: {prompt_id}",
        "hint": "Check if prompt_id is valid using get_workflow_status()"
    }


@mcp.tool()
def get_system_stats() -> dict:
    """Get GPU VRAM and system stats."""
    return execution.get_system_stats()


@mcp.tool()
def free_memory(unload_models: bool = False) -> dict:
    """Free GPU memory. unload_models=True to clear all."""
    return execution.free_memory(unload_models)


@mcp.tool()
def interrupt() -> dict:
    """Stop currently running workflow."""
    return execution.interrupt_execution()


# Asset Tools

@mcp.tool()
def regenerate(
    asset_id: str,
    prompt: str = None,
    negative_prompt: str = None,
    seed: int = None,
    steps: int = None,
    cfg: float = None,
) -> dict:
    """Re-run workflow with modified params. Returns new prompt_id."""
    return execution.regenerate(
        asset_id=asset_id, prompt=prompt, negative_prompt=negative_prompt,
        seed=seed, steps=steps, cfg=cfg,
    )


@mcp.tool()
@mcp_tool_wrapper
def list_assets(
    session_id: str = None,
    asset_type: str = None,
    limit: int = 20,
    cursor: str = None,
) -> dict:
    """List generated assets. type: images|video|audio"""
    result = execution.list_assets(session_id=session_id, asset_type=asset_type, limit=1000)
    if "error" in result:
        return _to_mcp_response(result)
    assets = result.get("assets", [])
    paginated = paginate(assets, cursor=cursor, limit=limit)
    return {"assets": paginated["items"], "nextCursor": paginated["nextCursor"], "total": paginated["total"]}


@mcp.tool()
@mcp_tool_wrapper
def get_asset_metadata(asset_id: str) -> dict:
    """Get full asset metadata including workflow."""
    return _to_mcp_response(execution.get_asset_metadata(asset_id))


@mcp.tool()
def view_output(asset_id: str, mode: str = "thumb") -> dict:
    """View asset. mode: thumb|metadata"""
    return execution.view_output(asset_id, mode)


@mcp.tool()
def cleanup_assets() -> dict:
    """Remove expired assets (default 24h TTL)."""
    return execution.cleanup_expired_assets()


@mcp.tool()
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
def download_output(asset_id: str, output_path: str) -> dict:
    """Download asset to local file."""
    home_dir = os.path.expanduser("~")
    valid, resolved_path = _validate_path(output_path, home_dir)
    if not valid:
        return mcp_error(resolved_path, "VALIDATION_ERROR")
    return execution.download_output(asset_id, resolved_path)


# Publishing Tools

@mcp.tool()
def publish_asset(
    asset_id: str,
    target_filename: str = None,
    manifest_key: str = None,
    publish_dir: str = None,
    web_optimize: bool = True,
) -> dict:
    """Publish asset to web dir. Use target_filename or manifest_key."""
    return publish.publish_asset(
        asset_id=asset_id, target_filename=target_filename,
        manifest_key=manifest_key, publish_dir=publish_dir, web_optimize=web_optimize,
    )


@mcp.tool()
def get_publish_info() -> dict:
    """Get current publish directory config."""
    return publish.get_publish_info()


@mcp.tool()
def set_publish_dir(publish_dir: str) -> dict:
    """Set publish directory path."""
    return publish.set_publish_dir(publish_dir)


# Workflow Library

@mcp.tool()
@mcp_tool_wrapper
def workflow_library(
    action: str,
    name: str = None,
    workflow: dict = None,
    description: str = "",
    tags: Optional[List[str]] = None,
    source_name: str = None,
    new_name: str = None,
    tag_filter: str = None,
    limit: int = 20,
    cursor: str = None,
) -> dict:
    """Workflow library. action: save|load|list|delete|duplicate|export|import"""
    if action == "save":
        if not name or not workflow:
            return mcp_error("name and workflow required for action='save'", "INVALID_PARAMS")
        return persistence.save_workflow(name, workflow, description, tags or [])
    elif action == "load":
        if not name:
            return mcp_error("name required for action='load'", "INVALID_PARAMS")
        return persistence.load_workflow(name)
    elif action == "list":
        result = persistence.list_workflows(tag_filter)
        if "error" in result:
            return _to_mcp_response(result)
        workflows = result.get("workflows", [])
        paginated = paginate(workflows, cursor=cursor, limit=limit)
        return {"workflows": paginated["items"], "nextCursor": paginated["nextCursor"], "total": paginated["total"]}
    elif action == "delete":
        if not name:
            return mcp_error("name required for action='delete'", "INVALID_PARAMS")
        return persistence.delete_workflow(name)
    elif action == "duplicate":
        if not source_name or not new_name:
            return mcp_error("source_name and new_name required for action='duplicate'", "INVALID_PARAMS")
        return persistence.duplicate_workflow(source_name, new_name)
    elif action == "export":
        if not name:
            return mcp_error("name required for action='export'", "INVALID_PARAMS")
        return persistence.export_workflow(name)
    elif action == "import":
        if not name or not workflow:
            return mcp_error("name and workflow required for action='import'", "INVALID_PARAMS")
        return persistence.import_workflow(name, workflow, description, tags)
    else:
        return mcp_error(f"Invalid action: {action}. Use: save|load|list|delete|duplicate|export|import", "INVALID_PARAMS")


# VRAM Tools

@mcp.tool()
def estimate_vram(workflow: dict) -> dict:
    """Estimate VRAM usage for workflow."""
    return vram.estimate_workflow_vram(workflow)


@mcp.tool()
def check_model_fits(model_name: str, precision: str = "default") -> dict:
    """Check if model fits in VRAM. precision: fp32|fp16|bf16|fp8|default"""
    return vram.check_model_fits(model_name, precision)


# Validation Tools

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


# SOTA Recommendations

@mcp.tool()
def sota_query(
    mode: str,
    category: str = None,
    task: str = None,
    model_name: str = None,
    available_vram_gb: float = None,
) -> dict:
    """SOTA queries. mode: category|recommend|check|settings|installed"""
    if mode == "category":
        if not category:
            return mcp_error("category required for mode='category'", "INVALID_PARAMS")
        return sota.get_sota_for_category(category)
    elif mode == "recommend":
        if not task:
            return mcp_error("task required for mode='recommend'", "INVALID_PARAMS")
        return sota.recommend_model_for_task(task, available_vram_gb)
    elif mode == "check":
        if not model_name:
            return mcp_error("model_name required for mode='check'", "INVALID_PARAMS")
        return sota.check_model_is_sota(model_name)
    elif mode == "settings":
        if not model_name:
            return mcp_error("model_name required for mode='settings'", "INVALID_PARAMS")
        return sota.get_optimal_settings(model_name)
    elif mode == "installed":
        return sota.get_available_sota_models()
    else:
        return mcp_error(f"Invalid mode: {mode}. Use: category|recommend|check|settings|installed", "INVALID_PARAMS")


# Workflow Templates

@mcp.tool()
@mcp_tool_wrapper
def list_workflow_templates(limit: int = 50, cursor: str = None) -> dict:
    """List available workflow templates. Paginated."""
    result = templates.list_templates()
    if "error" in result:
        return _to_mcp_response(result)
    template_list = result.get("templates", [])
    paginated = paginate(template_list, cursor=cursor, limit=limit)
    return {"templates": paginated["items"], "nextCursor": paginated["nextCursor"], "total": paginated["total"]}


@mcp.tool()
@mcp_tool_wrapper
def get_template(name: str) -> dict:
    """Get workflow template by name."""
    return _to_mcp_response(templates.load_template(name))


@mcp.tool()
def create_workflow_from_template(template_name: str, parameters: dict) -> dict:
    """Create workflow from template. Injects {{PLACEHOLDER}} values."""
    template = templates.load_template(template_name)
    if "error" in template:
        return template
    defaults = template.get("_meta", {}).get("defaults", {})
    final_params = {**defaults, **parameters}
    workflow = templates.inject_parameters(template, final_params)
    return {"workflow": workflow, "template": template_name, "parameters_used": final_params}


# Workflow Patterns

@mcp.tool()
def get_workflow_skeleton(model: str, task: str) -> dict:
    """Get tested workflow structure for model+task."""
    return patterns.get_workflow_skeleton(model, task)


@mcp.tool()
def get_model_constraints(model: str) -> dict:
    """Get model constraints (CFG, resolution, frames, required nodes)."""
    return patterns.get_model_constraints(model)


@mcp.tool()
def get_node_chain(model: str, task: str) -> dict:
    """Get ordered nodes with exact connection slots."""
    result = patterns.get_node_chain(model, task)
    if isinstance(result, dict) and "error" in result:
        return result
    return {"nodes": result, "model": model, "task": task, "count": len(result)}


# Converted to MCP resource - see comfyui://patterns/available


# Batch Execution

@mcp.tool()
def batch_execute(
    workflow: dict,
    mode: str,
    parameter_sets: Optional[List[Dict[str, Any]]] = None,
    sweep_params: dict = None,
    fixed_params: dict = None,
    num_variations: int = 4,
    start_seed: int = 42,
    parallel: int = 1,
    timeout_per_job: int = 600,
) -> dict:
    """Batch execution. mode: batch|sweep|seeds"""
    if mode == "batch":
        if not parameter_sets:
            return mcp_error("parameter_sets required for mode='batch'", "INVALID_PARAMS")
        return batch.execute_batch(workflow, parameter_sets, parallel, timeout_per_job)
    elif mode == "sweep":
        if not sweep_params:
            return mcp_error("sweep_params required for mode='sweep'", "INVALID_PARAMS")
        return batch.execute_sweep(workflow, sweep_params, fixed_params, parallel)
    elif mode == "seeds":
        base_params = parameter_sets[0] if parameter_sets else {}
        return batch.execute_seed_variations(workflow, base_params, num_variations, start_seed, parallel)
    else:
        return mcp_error(f"Invalid mode: {mode}. Use: batch|sweep|seeds", "INVALID_PARAMS")


# Pipelines

@mcp.tool()
def execute_pipeline_stages(stages: List[Dict[str, Any]], initial_params: Dict[str, Any], timeout_per_stage: int = 600) -> dict:
    """Run multi-stage pipeline (e.g., image→upscale→video)."""
    return pipeline.execute_pipeline(stages, initial_params, timeout_per_stage)


@mcp.tool()
def run_image_to_video_pipeline(image_workflow: dict, video_workflow: dict, prompt: str, video_prompt: str = None, seed: int = 42) -> dict:
    """Generate image then animate to video."""
    return pipeline.create_image_to_video_pipeline(image_workflow, video_workflow, prompt, video_prompt, seed)


@mcp.tool()
def run_upscale_pipeline(base_workflow: dict, upscale_workflow: dict, prompt: str, upscale_factor: float = 2.0, seed: int = 42) -> dict:
    """Generate image then upscale."""
    return pipeline.create_upscale_pipeline(base_workflow, upscale_workflow, prompt, upscale_factor, seed)


# Model Management Tools

@mcp.tool()
def search_civitai(query: str, model_type: str = None, nsfw: bool = False, limit: int = 10) -> dict:
    """Search Civitai for models. type: checkpoint|lora|embedding|controlnet|upscaler"""
    return models.search_civitai(query, model_type, nsfw, limit)


@mcp.tool()
@mcp_tool_wrapper
def download_model(url: str, model_type: str, filename: str = None, overwrite: bool = False) -> dict:
    """Download model from Civitai/HF. type: checkpoint|unet|lora|vae|controlnet|clip"""
    valid, error_msg = _validate_url(url)
    if not valid:
        return mcp_error(error_msg, "VALIDATION_ERROR")
    return _to_mcp_response(models.download_model(url, model_type, filename, overwrite))


@mcp.tool()
def get_model_info(model_path: str) -> dict:
    """Get info about installed model."""
    return models.get_model_info(model_path)


@mcp.tool()
def list_installed_models(model_type: str = None) -> dict:
    """List installed models by type."""
    return models.list_installed_models(model_type)


# Asset Analysis

@mcp.tool()
def get_image_dimensions(asset_id: str) -> dict:
    """Get image dimensions and recommended video size."""
    return analysis.get_image_dimensions(asset_id)


@mcp.tool()
@mcp_tool_wrapper
def detect_objects(asset_id: str, objects: List[str], vlm_model: Optional[str] = None) -> dict:
    """Detect objects in image via VLM."""
    return _to_mcp_response(analysis.detect_objects(asset_id, objects, vlm_model))


@mcp.tool()
def get_video_info(asset_id: str) -> dict:
    """Get video duration, fps, frame count."""
    return analysis.get_video_info(asset_id)


# Quality Assurance

@mcp.tool()
@mcp_tool_wrapper
def qa_output(asset_id: str, prompt: str, checks: Optional[List[str]] = None, vlm_model: Optional[str] = None) -> dict:
    """QA check via VLM. checks: prompt_match|artifacts|faces|text|composition"""
    safe_prompt = _escape_user_content(prompt)
    return _to_mcp_response(qa.qa_output(asset_id=asset_id, prompt=safe_prompt, checks=checks, vlm_model=vlm_model))


@mcp.tool()
def check_vlm_available(vlm_model: str = None) -> dict:
    """Check if VLM (Ollama) is available for QA."""
    return qa.check_vlm_available(vlm_model)


# Style Learning Tools

@mcp.tool()
def record_generation(
    prompt: str,
    model: str,
    seed: int,
    parameters: Dict[str, Any] = None,
    negative_prompt: str = "",
    rating: Optional[float] = None,
    tags: Optional[List[str]] = None,
    outcome: str = "success",
    qa_score: float = None,
    notes: str = "",
) -> dict:
    """Record a generation for style learning. Returns record_id."""
    record_id = style_learning.record_generation(
        prompt=prompt, model=model, seed=seed, parameters=parameters,
        negative_prompt=negative_prompt, rating=rating, tags=tags,
        outcome=outcome, qa_score=qa_score, notes=notes,
    )
    return {"record_id": record_id, "success": True}


@mcp.tool()
def rate_generation(record_id: str, rating: float, notes: str = None) -> dict:
    """Rate a generation 0.0-1.0. Returns success status."""
    return {"success": style_learning.rate_generation(record_id, rating, notes)}


@mcp.tool()
def style_suggest(
    mode: str,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    tags: Optional[List[str]] = None,
    min_rating: float = 0.7,
    limit: int = 5,
) -> dict:
    """Style suggestions. mode: prompt|seeds|similar"""
    if mode == "prompt":
        if not prompt:
            return mcp_error("prompt required for mode=prompt")
        return style_learning.suggest_prompt_enhancement(prompt=prompt, model=model, style_tags=tags)
    elif mode == "seeds":
        if not tags:
            return mcp_error("tags required for mode=seeds")
        results = style_learning.get_best_seeds_for_style(tags=tags, model=model, limit=limit)
        return {"seeds": results, "count": len(results)}
    elif mode == "similar":
        if not prompt:
            return mcp_error("prompt required for mode=similar")
        results = style_learning.find_similar_prompts(prompt=prompt, model=model, min_rating=min_rating, limit=limit)
        return {"similar": results, "count": len(results)}
    else:
        return mcp_error(f"Unknown mode: {mode}. Use: prompt|seeds|similar")


@mcp.tool()
def manage_presets(
    action: str,
    name: str = None,
    description: str = None,
    prompt_additions: str = None,
    negative_additions: str = "",
    recommended_model: str = None,
    recommended_params: dict = None,
) -> dict:
    """Manage style presets. action: list|get|save|delete"""
    if action == "list":
        presets = style_learning.list_style_presets()
        return {"presets": presets, "count": len(presets)}
    elif action == "get":
        if not name:
            return mcp_error("name required for action=get")
        preset = style_learning.get_style_preset(name)
        return preset if preset else mcp_error(f"Preset '{name}' not found", "NOT_FOUND")
    elif action == "save":
        if not name or not description or not prompt_additions:
            return mcp_error("name, description, prompt_additions required for action=save")
        success = style_learning.save_style_preset(
            name=name, description=description, prompt_additions=prompt_additions,
            negative_additions=negative_additions, recommended_model=recommended_model,
            recommended_params=recommended_params,
        )
        return {"success": success}
    elif action == "delete":
        if not name:
            return mcp_error("name required for action=delete")
        success = style_learning.delete_style_preset(name)
        return {"success": success}
    else:
        return mcp_error(f"Unknown action: {action}. Use: list|get|save|delete")





# Workflow Generation

@mcp.tool()
def generate_workflow(
    model: str, workflow_type: str, prompt: str, negative_prompt: str = "",
    width: int = None, height: int = None, frames: int = None,
    seed: int = None, steps: int = None, cfg: float = None, guidance: float = None,
) -> dict:
    """Generate validated workflow. model: ltx|flux|wan|qwen. type: t2v|i2v|t2i"""
    return workflow_generator.generate_workflow(
        model=model, workflow_type=workflow_type, prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, frames=frames, seed=seed, steps=steps, cfg=cfg, guidance=guidance,
    )


# =============================================================================
# Workflow Visualization Tools
# =============================================================================

@mcp.tool()
def visualize_workflow(workflow: dict) -> dict:
    """Generate Mermaid diagram from workflow for visualization."""
    return visualization.visualize_workflow(workflow)


@mcp.tool()
def get_workflow_summary(workflow: dict) -> dict:
    """Get text summary of workflow structure (node types, parameters)."""
    return visualization.get_workflow_summary(workflow)


# =============================================================================
# Rate Limiting Dashboard Tools
# =============================================================================

@mcp.tool()
def get_rate_limit_status(tool_name: str = None) -> dict:
    """Get current rate limiting status. Shows requests remaining, reset time, usage."""
    return rate_limiter.get_rate_limit_status(tool_name)


@mcp.tool()
def get_all_tools_rate_status() -> dict:
    """Get rate limiting status for all tools."""
    return rate_limiter.get_all_tools_rate_status()


@mcp.tool()
def get_rate_limit_summary() -> dict:
    """Get brief summary of rate limit status for dashboard display."""
    return rate_limiter.get_rate_limit_summary()


# Converted to MCP resource - see comfyui://workflows/supported


# MCP Resources

@mcp.resource(
    "comfyui://docs/patterns/ltx",
    name="LTX Video Pattern",
    description="LLM reference pattern for LTX-Video workflow generation",
    mime_type="text/markdown"
)
def resource_pattern_ltx() -> str:
    """LTX Video pattern documentation."""
    result = reference_docs.get_model_pattern("ltx")
    return result.get("pattern", json.dumps(result))


@mcp.resource(
    "comfyui://docs/patterns/flux",
    name="FLUX Pattern",
    description="LLM reference pattern for FLUX workflow generation",
    mime_type="text/markdown"
)
def resource_pattern_flux() -> str:
    """FLUX pattern documentation."""
    result = reference_docs.get_model_pattern("flux")
    return result.get("pattern", json.dumps(result))


@mcp.resource(
    "comfyui://docs/patterns/wan",
    name="Wan 2.1 Pattern",
    description="LLM reference pattern for Wan 2.1 workflow generation",
    mime_type="text/markdown"
)
def resource_pattern_wan() -> str:
    """Wan 2.1 pattern documentation."""
    result = reference_docs.get_model_pattern("wan")
    return result.get("pattern", json.dumps(result))


@mcp.resource(
    "comfyui://docs/patterns/qwen",
    name="Qwen Pattern",
    description="LLM reference pattern for Qwen workflow generation",
    mime_type="text/markdown"
)
def resource_pattern_qwen() -> str:
    """Qwen pattern documentation."""
    result = reference_docs.get_model_pattern("qwen")
    return result.get("pattern", json.dumps(result))


@mcp.resource(
    "comfyui://docs/rules",
    name="Parameter Rules",
    description="Validation constraints for LLM-generated workflows",
    mime_type="text/markdown"
)
def resource_parameter_rules() -> str:
    """Parameter validation rules."""
    result = reference_docs.get_parameter_rules()
    return result.get("rules", json.dumps(result))


@mcp.resource(
    "comfyui://docs/system-prompt",
    name="LLM System Prompt",
    description="System prompt guide for LLM workflow generation",
    mime_type="text/markdown"
)
def resource_system_prompt() -> str:
    """System prompt for LLM workflow generation."""
    result = reference_docs.get_system_prompt()
    return result.get("system_prompt", json.dumps(result))


@mcp.resource(
    "comfyui://docs/skeletons/ltx-t2v",
    name="LTX T2V Skeleton",
    description="Token-optimized skeleton for LTX text-to-video",
    mime_type="application/json"
)
def resource_skeleton_ltx_t2v() -> str:
    """LTX text-to-video skeleton."""
    result = reference_docs.get_skeleton("ltx", "t2v")
    return json.dumps(result.get("skeleton", result), indent=2)


@mcp.resource(
    "comfyui://docs/skeletons/ltx-i2v",
    name="LTX I2V Skeleton",
    description="Token-optimized skeleton for LTX image-to-video",
    mime_type="application/json"
)
def resource_skeleton_ltx_i2v() -> str:
    """LTX image-to-video skeleton."""
    result = reference_docs.get_skeleton("ltx", "i2v")
    return json.dumps(result.get("skeleton", result), indent=2)


@mcp.resource(
    "comfyui://docs/skeletons/flux-t2i",
    name="FLUX T2I Skeleton",
    description="Token-optimized skeleton for FLUX text-to-image",
    mime_type="application/json"
)
def resource_skeleton_flux_t2i() -> str:
    """FLUX text-to-image skeleton."""
    result = reference_docs.get_skeleton("flux", "t2i")
    return json.dumps(result.get("skeleton", result), indent=2)


@mcp.resource(
    "comfyui://docs/skeletons/wan-t2v",
    name="Wan T2V Skeleton",
    description="Token-optimized skeleton for Wan text-to-video",
    mime_type="application/json"
)
def resource_skeleton_wan_t2v() -> str:
    """Wan text-to-video skeleton."""
    result = reference_docs.get_skeleton("wan", "t2v")
    return json.dumps(result.get("skeleton", result), indent=2)


@mcp.resource(
    "comfyui://docs/skeletons/qwen-t2i",
    name="Qwen T2I Skeleton",
    description="Token-optimized skeleton for Qwen text-to-image",
    mime_type="application/json"
)
def resource_skeleton_qwen_t2i() -> str:
    """Qwen text-to-image skeleton."""
    result = reference_docs.get_skeleton("qwen", "t2i")
    return json.dumps(result.get("skeleton", result), indent=2)


@mcp.resource(
    "comfyui://patterns/available",
    name="Available Patterns",
    description="List of all workflow patterns and supported models",
    mime_type="application/json"
)
def resource_available_patterns() -> str:
    """All available workflow patterns."""
    result = patterns.list_available_patterns()
    return json.dumps(result, indent=2)


@mcp.resource(
    "comfyui://workflows/supported",
    name="Supported Workflows",
    description="List of supported model+workflow_type combinations",
    mime_type="application/json"
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
