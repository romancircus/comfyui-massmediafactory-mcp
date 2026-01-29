"""
ComfyUI MassMediaFactory MCP Server

Main entry point that exposes all tools via MCP protocol.

MCP Compliance:
- JSON-RPC 2.0 via FastMCP
- isError flag for error responses
- Rate limiting on tool invocations
- Structured logging with correlation IDs
- Cursor-based pagination support
"""

import json
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
    """
    Convert a result dict to MCP-compliant format.

    If result contains "error" key without "isError", adds MCP compliance.
    """
    if isinstance(result, dict) and "error" in result and "isError" not in result:
        return {
            **result,
            "isError": True,
            "code": result.get("code", "TOOL_ERROR"),
        }
    return result


# =============================================================================
# Discovery Tools
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
def search_nodes(query: str) -> dict:
    """Search ComfyUI nodes by name/category."""
    return discovery.search_nodes(query)


# =============================================================================
# Execution Tools
# =============================================================================

@mcp.tool()
@mcp_tool_wrapper
def execute_workflow(workflow: dict, client_id: str = "massmediafactory") -> dict:
    """Queue workflow for execution. Returns prompt_id."""
    return _to_mcp_response(execution.execute_workflow(workflow, client_id))


@mcp.tool()
def get_workflow_status(prompt_id: str) -> dict:
    """Check workflow status. Returns queued/running/completed/error."""
    return execution.get_workflow_status(prompt_id)


@mcp.tool()
@mcp_tool_wrapper
def wait_for_completion(prompt_id: str, timeout_seconds: int = 600) -> dict:
    """Wait for workflow completion. Returns outputs."""
    return _to_mcp_response(execution.wait_for_completion(prompt_id, timeout_seconds))


@mcp.tool()
def get_system_stats() -> dict:
    """Get GPU VRAM and system stats."""
    return execution.get_system_stats()


@mcp.tool()
def free_memory(unload_models: bool = False) -> dict:
    """Free GPU memory. unload_models=True to clear all."""
    return execution.free_memory(unload_models)


@mcp.tool()
def interrupt_execution() -> dict:
    """Stop currently running workflow."""
    return execution.interrupt_execution()


@mcp.tool()
def get_queue_status() -> dict:
    """Get queue status (running/pending jobs)."""
    return execution.get_queue_status()


# =============================================================================
# Asset Iteration Tools
# =============================================================================

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
def cleanup_expired_assets() -> dict:
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
    return execution.upload_image(image_path, filename, subfolder, overwrite)


@mcp.tool()
def download_output(asset_id: str, output_path: str) -> dict:
    """Download asset to local file."""
    return execution.download_output(asset_id, output_path)


# =============================================================================
# Publishing Tools
# =============================================================================

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


# =============================================================================
# Persistence Tools (Workflow Library)
# =============================================================================

@mcp.tool()
def save_workflow(name: str, workflow: dict, description: str = "", tags: list = None) -> dict:
    """Save workflow to library."""
    return persistence.save_workflow(name, workflow, description, tags or [])


@mcp.tool()
def load_workflow(name: str) -> dict:
    """Load workflow from library."""
    return persistence.load_workflow(name)


@mcp.tool()
@mcp_tool_wrapper
def list_saved_workflows(tag: str = None, limit: int = 20, cursor: str = None) -> dict:
    """List saved workflows. Paginated."""
    result = persistence.list_workflows(tag)
    if "error" in result:
        return _to_mcp_response(result)
    workflows = result.get("workflows", [])
    paginated = paginate(workflows, cursor=cursor, limit=limit)
    return {"workflows": paginated["items"], "nextCursor": paginated["nextCursor"], "total": paginated["total"]}


@mcp.tool()
def delete_workflow(name: str) -> dict:
    """Delete workflow from library."""
    return persistence.delete_workflow(name)


@mcp.tool()
def duplicate_workflow(source_name: str, new_name: str) -> dict:
    """Duplicate workflow with new name."""
    return persistence.duplicate_workflow(source_name, new_name)


@mcp.tool()
def export_workflow(name: str) -> dict:
    """Export workflow as raw JSON."""
    return persistence.export_workflow(name)


@mcp.tool()
def import_workflow(name: str, workflow_json: dict, description: str = "", tags: list = None) -> dict:
    """Import raw workflow JSON."""
    return persistence.import_workflow(name, workflow_json, description, tags)


# =============================================================================
# VRAM Estimation Tools
# =============================================================================

@mcp.tool()
def estimate_vram(workflow: dict) -> dict:
    """Estimate VRAM usage for workflow."""
    return vram.estimate_workflow_vram(workflow)


@mcp.tool()
def check_model_fits(model_name: str, precision: str = "default") -> dict:
    """Check if model fits in VRAM. precision: fp32|fp16|bf16|fp8|default"""
    return vram.check_model_fits(model_name, precision)


# =============================================================================
# Validation Tools
# =============================================================================

@mcp.tool()
@mcp_tool_wrapper
def validate_workflow(workflow: dict) -> dict:
    """Validate workflow for errors before execution."""
    return _to_mcp_response(validation.validate_workflow(workflow))


@mcp.tool()
def validate_and_fix_workflow(workflow: dict) -> dict:
    """Validate and auto-fix common workflow issues."""
    return validation.validate_and_fix(workflow)


@mcp.tool()
def check_connection_compatibility(source_type: str, source_slot: int, target_type: str, target_input: str) -> dict:
    """Check if node connection is type-compatible."""
    return validation.check_node_compatibility(source_type, source_slot, target_type, target_input)


# =============================================================================
# SOTA Model Recommendations
# =============================================================================

@mcp.tool()
def get_sota_models(category: str) -> dict:
    """Get SOTA models. category: image_gen|video_gen|controlnet"""
    return sota.get_sota_for_category(category)


@mcp.tool()
def recommend_model(task: str, available_vram_gb: float = None) -> dict:
    """Recommend model for task. task: portrait|text_in_image|fast_iteration|etc."""
    return sota.recommend_model_for_task(task, available_vram_gb)


@mcp.tool()
def check_model_freshness(model_name: str) -> dict:
    """Check if model is current SOTA or deprecated."""
    return sota.check_model_is_sota(model_name)


@mcp.tool()
def get_model_settings(model_name: str) -> dict:
    """Get optimal CFG/steps/sampler settings for model."""
    return sota.get_optimal_settings(model_name)


@mcp.tool()
def check_installed_sota() -> dict:
    """List installed SOTA models and missing ones."""
    return sota.get_available_sota_models()


# =============================================================================
# Workflow Templates
# =============================================================================

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


# =============================================================================
# Workflow Pattern Tools (Prevent Drift)
# =============================================================================

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


@mcp.tool()
def validate_against_pattern(workflow: dict, model: str) -> dict:
    """Validate workflow against known working patterns."""
    return patterns.validate_against_pattern(workflow, model)


@mcp.tool()
def list_available_patterns() -> dict:
    """List all workflow patterns and supported models."""
    return patterns.list_available_patterns()


# NOTE: Workflow Builder Tools (explain_workflow, get_required_nodes, get_connection_pattern,
# compare_workflows) removed in token reduction. Use MCP Resources instead.

# NOTE: Template Discovery Tools (find_template_for_task, get_templates_by_type,
# get_templates_by_model, validate_all_templates) removed in token reduction.
# Use list_templates() and filter client-side.


# =============================================================================
# Batch Execution
# =============================================================================

@mcp.tool()
def execute_batch_workflows(workflow: dict, parameter_sets: list, parallel: int = 1, timeout_per_job: int = 600) -> dict:
    """Run workflow with multiple parameter sets."""
    return batch.execute_batch(workflow, parameter_sets, parallel, timeout_per_job)


@mcp.tool()
def execute_parameter_sweep(workflow: dict, sweep_params: dict, fixed_params: dict = None, parallel: int = 1) -> dict:
    """Grid search over parameter values."""
    return batch.execute_sweep(workflow, sweep_params, fixed_params, parallel)


@mcp.tool()
def generate_seed_variations(workflow: dict, parameters: dict, num_variations: int = 4, start_seed: int = 42, parallel: int = 2) -> dict:
    """Generate multiple outputs with different seeds."""
    return batch.execute_seed_variations(workflow, parameters, num_variations, start_seed, parallel)


# =============================================================================
# Multi-Stage Pipelines
# =============================================================================

@mcp.tool()
def execute_pipeline_stages(stages: list, initial_params: dict, timeout_per_stage: int = 600) -> dict:
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


# =============================================================================
# Templates as MCP Resources
# =============================================================================

# Define template resources with proper MCP-compliant URIs
# Format: comfyui://templates/{template_name}

@mcp.resource(
    "comfyui://templates/flux2_txt2img",
    name="FLUX.2 Text-to-Image",
    description="FLUX.2 text-to-image workflow with {{PROMPT}}, {{SEED}}, {{WIDTH}}, {{HEIGHT}} placeholders",
    mime_type="application/json"
)
def resource_flux2_txt2img() -> str:
    """FLUX.2 text-to-image workflow template."""
    template = templates.load_template("flux2_txt2img")
    return json.dumps(template, indent=2)


@mcp.resource(
    "comfyui://templates/qwen_txt2img",
    name="Qwen Text-to-Image",
    description="Qwen image generation workflow optimized for text rendering",
    mime_type="application/json"
)
def resource_qwen_txt2img() -> str:
    """Qwen text-to-image workflow template."""
    template = templates.load_template("qwen_txt2img")
    return json.dumps(template, indent=2)


@mcp.resource(
    "comfyui://templates/ltx2_txt2vid",
    name="LTX-2 Text-to-Video",
    description="LTX-2 text-to-video workflow with audio sync support",
    mime_type="application/json"
)
def resource_ltx2_txt2vid() -> str:
    """LTX-2 text-to-video workflow template."""
    template = templates.load_template("ltx2_txt2vid")
    return json.dumps(template, indent=2)


@mcp.resource(
    "comfyui://templates/wan26_img2vid",
    name="Wan 2.6 Image-to-Video",
    description="Wan 2.6 image-to-video workflow for animating still images",
    mime_type="application/json"
)
def resource_wan26_img2vid() -> str:
    """Wan 2.6 image-to-video workflow template."""
    template = templates.load_template("wan26_img2vid")
    return json.dumps(template, indent=2)


@mcp.resource(
    "comfyui://templates/flux2_ultimate_upscale",
    name="FLUX.2 Ultimate Upscale",
    description="4K/8K upscaling with neural network enhancement",
    mime_type="application/json"
)
def resource_flux2_ultimate_upscale() -> str:
    """FLUX.2 upscaling workflow template."""
    template = templates.load_template("flux2_ultimate_upscale")
    return json.dumps(template, indent=2)


@mcp.resource(
    "comfyui://templates/flux2_face_id",
    name="FLUX.2 Face ID",
    description="Generate images with consistent face identity (--cref replacement)",
    mime_type="application/json"
)
def resource_flux2_face_id() -> str:
    """FLUX.2 Face ID workflow template."""
    template = templates.load_template("flux2_face_id")
    return json.dumps(template, indent=2)


@mcp.resource(
    "comfyui://templates/qwen3_tts_custom_voice",
    name="Qwen3-TTS Custom Voice",
    description="Text-to-speech with 9 premium preset voices",
    mime_type="application/json"
)
def resource_qwen3_tts_custom_voice() -> str:
    """Qwen3-TTS custom voice workflow template."""
    template = templates.load_template("qwen3_tts_custom_voice")
    return json.dumps(template, indent=2)


@mcp.resource(
    "comfyui://templates/chatterbox_tts",
    name="Chatterbox TTS",
    description="Expressive TTS with emotion tags ([laugh], [sigh], etc.)",
    mime_type="application/json"
)
def resource_chatterbox_tts() -> str:
    """Chatterbox TTS workflow template."""
    template = templates.load_template("chatterbox_tts")
    return json.dumps(template, indent=2)


# Legacy resource for backwards compatibility
@mcp.resource("template://flux-txt2img")
def flux_txt2img_template_legacy() -> str:
    """[DEPRECATED] Use comfyui://templates/flux2_txt2img instead."""
    return resource_flux2_txt2img()


# =============================================================================
# Model Management Tools
# =============================================================================

@mcp.tool()
def search_civitai(query: str, model_type: str = None, nsfw: bool = False, limit: int = 10) -> dict:
    """Search Civitai for models. type: checkpoint|lora|embedding|controlnet|upscaler"""
    return models.search_civitai(query, model_type, nsfw, limit)


@mcp.tool()
@mcp_tool_wrapper
def download_model(url: str, model_type: str, filename: str = None, overwrite: bool = False) -> dict:
    """Download model from Civitai/HF. type: checkpoint|unet|lora|vae|controlnet|clip"""
    return _to_mcp_response(models.download_model(url, model_type, filename, overwrite))


@mcp.tool()
def get_model_info(model_path: str) -> dict:
    """Get info about installed model."""
    return models.get_model_info(model_path)


@mcp.tool()
def list_installed_models(model_type: str = None) -> dict:
    """List installed models by type."""
    return models.list_installed_models(model_type)


# =============================================================================
# Asset Analysis Tools
# =============================================================================

@mcp.tool()
def get_image_dimensions(asset_id: str) -> dict:
    """Get image dimensions and recommended video size."""
    return analysis.get_image_dimensions(asset_id)


@mcp.tool()
@mcp_tool_wrapper
def detect_objects(asset_id: str, objects: list, vlm_model: str = None) -> dict:
    """Detect objects in image via VLM."""
    return _to_mcp_response(analysis.detect_objects(asset_id, objects, vlm_model))


@mcp.tool()
def get_video_info(asset_id: str) -> dict:
    """Get video duration, fps, frame count."""
    return analysis.get_video_info(asset_id)


# =============================================================================
# Quality Assurance Tools
# =============================================================================

@mcp.tool()
@mcp_tool_wrapper
def qa_output(asset_id: str, prompt: str, checks: list = None, vlm_model: str = None) -> dict:
    """QA check via VLM. checks: prompt_match|artifacts|faces|text|composition"""
    return _to_mcp_response(qa.qa_output(asset_id=asset_id, prompt=prompt, checks=checks, vlm_model=vlm_model))


@mcp.tool()
def check_vlm_available(vlm_model: str = None) -> dict:
    """Check if VLM (Ollama) is available for QA."""
    return qa.check_vlm_available(vlm_model)


# =============================================================================
# Style Learning Tools (consolidated from 10 → 4 tools)
# =============================================================================

@mcp.tool()
def record_generation(
    prompt: str,
    model: str,
    seed: int,
    parameters: dict = None,
    negative_prompt: str = "",
    rating: float = None,
    tags: list = None,
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
    prompt: str = None,
    style: str = None,
    model: str = None,
    tags: list = None,
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
        return preset if preset else {"error": f"Preset '{name}' not found"}
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


# NOTE: Schema Documentation Tools (get_tool_schema, list_tool_schemas, get_tool_output_schema)
# removed in token reduction. MCP handles tool schemas natively.

# NOTE: Tool Annotation Tools (get_tool_annotation, list_user_facing_tools, list_tools_by_category)
# removed in token reduction. Available via annotations module for internal use.


# NOTE: LLM Reference Documentation Tools removed in token reduction.
# Use MCP Resources instead: comfy://docs/patterns/{model}, comfy://docs/rules, etc.
# Tools removed: get_node_spec, list_node_specs, get_model_pattern, get_workflow_skeleton_json,
# get_parameter_rules, search_patterns, get_llm_system_prompt, list_workflow_skeletons


# =============================================================================
# Workflow Generation & Validation Tools
# =============================================================================

@mcp.tool()
@mcp_tool_wrapper
def validate_topology(workflow_json: str, model: str = None) -> dict:
    """Validate workflow against model constraints (resolution, frames, CFG)."""
    return _to_mcp_response(topology_validator.validate_topology(workflow_json, model))


@mcp.tool()
def auto_correct_workflow(workflow: dict, model: str = None) -> dict:
    """Auto-fix invalid params (resolution, frames, CFG)."""
    return topology_validator.auto_correct_parameters(workflow, model)


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


@mcp.tool()
def list_supported_workflows() -> dict:
    """List supported model+workflow_type combinations."""
    return workflow_generator.list_supported_workflows()


# =============================================================================
# LLM Reference Documentation Resources
# =============================================================================

@mcp.resource(
    "comfy://docs/patterns/ltx",
    name="LTX Video Pattern",
    description="LLM reference pattern for LTX-Video workflow generation",
    mime_type="text/markdown"
)
def resource_pattern_ltx() -> str:
    """LTX Video pattern documentation."""
    result = reference_docs.get_model_pattern("ltx")
    return result.get("pattern", json.dumps(result))


@mcp.resource(
    "comfy://docs/patterns/flux",
    name="FLUX Pattern",
    description="LLM reference pattern for FLUX workflow generation",
    mime_type="text/markdown"
)
def resource_pattern_flux() -> str:
    """FLUX pattern documentation."""
    result = reference_docs.get_model_pattern("flux")
    return result.get("pattern", json.dumps(result))


@mcp.resource(
    "comfy://docs/patterns/wan",
    name="Wan 2.1 Pattern",
    description="LLM reference pattern for Wan 2.1 workflow generation",
    mime_type="text/markdown"
)
def resource_pattern_wan() -> str:
    """Wan 2.1 pattern documentation."""
    result = reference_docs.get_model_pattern("wan")
    return result.get("pattern", json.dumps(result))


@mcp.resource(
    "comfy://docs/patterns/qwen",
    name="Qwen Pattern",
    description="LLM reference pattern for Qwen workflow generation",
    mime_type="text/markdown"
)
def resource_pattern_qwen() -> str:
    """Qwen pattern documentation."""
    result = reference_docs.get_model_pattern("qwen")
    return result.get("pattern", json.dumps(result))


@mcp.resource(
    "comfy://docs/rules",
    name="Parameter Rules",
    description="Validation constraints for LLM-generated workflows",
    mime_type="text/markdown"
)
def resource_parameter_rules() -> str:
    """Parameter validation rules."""
    result = reference_docs.get_parameter_rules()
    return result.get("rules", json.dumps(result))


@mcp.resource(
    "comfy://docs/system-prompt",
    name="LLM System Prompt",
    description="System prompt guide for LLM workflow generation",
    mime_type="text/markdown"
)
def resource_system_prompt() -> str:
    """System prompt for LLM workflow generation."""
    result = reference_docs.get_system_prompt()
    return result.get("system_prompt", json.dumps(result))


@mcp.resource(
    "comfy://docs/skeletons/ltx-t2v",
    name="LTX T2V Skeleton",
    description="Token-optimized skeleton for LTX text-to-video",
    mime_type="application/json"
)
def resource_skeleton_ltx_t2v() -> str:
    """LTX text-to-video skeleton."""
    result = reference_docs.get_skeleton("ltx", "t2v")
    return json.dumps(result.get("skeleton", result), indent=2)


@mcp.resource(
    "comfy://docs/skeletons/ltx-i2v",
    name="LTX I2V Skeleton",
    description="Token-optimized skeleton for LTX image-to-video",
    mime_type="application/json"
)
def resource_skeleton_ltx_i2v() -> str:
    """LTX image-to-video skeleton."""
    result = reference_docs.get_skeleton("ltx", "i2v")
    return json.dumps(result.get("skeleton", result), indent=2)


@mcp.resource(
    "comfy://docs/skeletons/flux-t2i",
    name="FLUX T2I Skeleton",
    description="Token-optimized skeleton for FLUX text-to-image",
    mime_type="application/json"
)
def resource_skeleton_flux_t2i() -> str:
    """FLUX text-to-image skeleton."""
    result = reference_docs.get_skeleton("flux", "t2i")
    return json.dumps(result.get("skeleton", result), indent=2)


@mcp.resource(
    "comfy://docs/skeletons/wan-t2v",
    name="Wan T2V Skeleton",
    description="Token-optimized skeleton for Wan text-to-video",
    mime_type="application/json"
)
def resource_skeleton_wan_t2v() -> str:
    """Wan text-to-video skeleton."""
    result = reference_docs.get_skeleton("wan", "t2v")
    return json.dumps(result.get("skeleton", result), indent=2)


@mcp.resource(
    "comfy://docs/skeletons/qwen-t2i",
    name="Qwen T2I Skeleton",
    description="Token-optimized skeleton for Qwen text-to-image",
    mime_type="application/json"
)
def resource_skeleton_qwen_t2i() -> str:
    """Qwen text-to-image skeleton."""
    result = reference_docs.get_skeleton("qwen", "t2i")
    return json.dumps(result.get("skeleton", result), indent=2)


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
