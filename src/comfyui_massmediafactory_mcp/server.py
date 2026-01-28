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
def list_checkpoints() -> dict:
    """
    List all available checkpoint models in ComfyUI.
    Returns model filenames for CheckpointLoaderSimple or UNETLoader.
    """
    return _to_mcp_response(discovery.list_checkpoints())


@mcp.tool()
def list_unets() -> dict:
    """
    List all available UNET models (for Flux, SD3, etc.).
    """
    return discovery.list_unets()


@mcp.tool()
def list_loras() -> dict:
    """
    List all available LoRA models in ComfyUI.
    """
    return discovery.list_loras()


@mcp.tool()
def list_vaes() -> dict:
    """
    List all available VAE models in ComfyUI.
    """
    return discovery.list_vaes()


@mcp.tool()
def list_controlnets() -> dict:
    """
    List all available ControlNet models.
    """
    return discovery.list_controlnets()


@mcp.tool()
@mcp_tool_wrapper
def get_node_info(node_type: str) -> dict:
    """
    Get detailed information about a specific ComfyUI node type.

    Args:
        node_type: The node class name (e.g., "KSampler", "CLIPTextEncode", "UNETLoader")

    Returns:
        Node schema including inputs, outputs, and their types.
    """
    return _to_mcp_response(discovery.get_node_info(node_type))


@mcp.tool()
def search_nodes(query: str) -> dict:
    """
    Search for ComfyUI nodes by name or category.

    Args:
        query: Search term (e.g., "sampler", "image", "video", "flux", "wan")

    Returns:
        List of matching node types sorted by relevance.
    """
    return discovery.search_nodes(query)


@mcp.tool()
def get_all_models() -> dict:
    """
    Get a summary of all available models (checkpoints, UNETs, LoRAs, VAEs, etc.).
    Useful for understanding what's installed before building workflows.
    """
    return discovery.get_all_models()


# =============================================================================
# Execution Tools
# =============================================================================

@mcp.tool()
@mcp_tool_wrapper
def execute_workflow(workflow: dict, client_id: str = "massmediafactory") -> dict:
    """
    Execute a ComfyUI workflow and return the prompt_id for tracking.

    Args:
        workflow: The workflow JSON with node definitions.
                  Each key is a node ID, value is {"class_type": "...", "inputs": {...}}
        client_id: Optional identifier for tracking.

    Returns:
        prompt_id for polling results.

    Example:
        {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a dragon", "clip": ["1", 1]}}
        }
    """
    return _to_mcp_response(execution.execute_workflow(workflow, client_id))


@mcp.tool()
def get_workflow_status(prompt_id: str) -> dict:
    """
    Check the status of a queued/running workflow.

    Args:
        prompt_id: The prompt_id from execute_workflow.

    Returns:
        Status (queued/running/completed/error) and outputs if completed.
    """
    return execution.get_workflow_status(prompt_id)


@mcp.tool()
@mcp_tool_wrapper
def wait_for_completion(prompt_id: str, timeout_seconds: int = 600) -> dict:
    """
    Wait for a workflow to complete and return outputs.

    Args:
        prompt_id: The prompt_id to wait for.
        timeout_seconds: Maximum wait time (default 600s / 10 minutes).

    Returns:
        Final status with output file paths.
    """
    return _to_mcp_response(execution.wait_for_completion(prompt_id, timeout_seconds))


@mcp.tool()
def get_system_stats() -> dict:
    """
    Get ComfyUI system statistics including GPU VRAM usage.
    Use this to check available memory before running large workflows.
    """
    return execution.get_system_stats()


@mcp.tool()
def free_memory(unload_models: bool = False) -> dict:
    """
    Free GPU memory in ComfyUI.

    Args:
        unload_models: If True, also unload all loaded models from VRAM.
    """
    return execution.free_memory(unload_models)


@mcp.tool()
def interrupt_execution() -> dict:
    """
    Interrupt the currently running workflow.
    """
    return execution.interrupt_execution()


@mcp.tool()
def get_queue_status() -> dict:
    """
    Get current queue status (running and pending jobs).
    """
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
    """
    Regenerate an asset with parameter overrides.

    Enables quick iteration: tweak CFG, change prompt, try new seed.

    Args:
        asset_id: The asset to regenerate from.
        prompt: New prompt (optional).
        negative_prompt: New negative prompt (optional).
        seed: New seed. Use -1 to keep original, None/omit for random.
        steps: New step count (optional).
        cfg: New CFG scale (optional).

    Returns:
        New prompt_id for the regenerated workflow.

    Example:
        # Generate initial image
        result = execute_workflow(workflow)
        output = wait_for_completion(result["prompt_id"])
        asset_id = output["outputs"][0]["asset_id"]

        # Iterate with higher CFG
        result = regenerate(asset_id, cfg=4.5)
        output = wait_for_completion(result["prompt_id"])
    """
    return execution.regenerate(
        asset_id=asset_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        cfg=cfg,
    )


@mcp.tool()
@mcp_tool_wrapper
def list_assets(
    session_id: str = None,
    asset_type: str = None,
    limit: int = 20,
    cursor: str = None,
) -> dict:
    """
    List recent generated assets with cursor-based pagination.

    Args:
        session_id: Filter by session (optional).
        asset_type: Filter by type: "images", "video", "audio" (optional).
        limit: Maximum results per page (default 20).
        cursor: Pagination cursor from previous response (optional).

    Returns:
        {
            "assets": [...],
            "nextCursor": "..." or null,
            "total": N
        }
    """
    # Get all matching assets (internal function returns full list)
    result = execution.list_assets(
        session_id=session_id,
        asset_type=asset_type,
        limit=1000,  # Get all for pagination
    )

    if "error" in result:
        return _to_mcp_response(result)

    # Apply pagination
    assets = result.get("assets", [])
    paginated = paginate(assets, cursor=cursor, limit=limit)

    return {
        "assets": paginated["items"],
        "nextCursor": paginated["nextCursor"],
        "total": paginated["total"],
    }


@mcp.tool()
@mcp_tool_wrapper
def get_asset_metadata(asset_id: str) -> dict:
    """
    Get full metadata for an asset including workflow and parameters.

    Useful for debugging or understanding how an asset was generated.

    Args:
        asset_id: The asset ID to retrieve.

    Returns:
        Full asset metadata including original workflow.
    """
    return _to_mcp_response(execution.get_asset_metadata(asset_id))


@mcp.tool()
def view_output(asset_id: str, mode: str = "thumb") -> dict:
    """
    View a generated asset.

    Args:
        asset_id: The asset to view.
        mode: "thumb" for preview info, "metadata" for full details.

    Returns:
        Asset URL and preview information.
    """
    return execution.view_output(asset_id, mode)


@mcp.tool()
def cleanup_expired_assets() -> dict:
    """
    Clean up expired assets from the registry.

    Assets expire after 24 hours by default (configurable via
    COMFY_MCP_ASSET_TTL_HOURS environment variable).

    Returns:
        Number of assets removed.
    """
    return execution.cleanup_expired_assets()


@mcp.tool()
def upload_image(
    image_path: str,
    filename: str = None,
    subfolder: str = "",
    overwrite: bool = True,
) -> dict:
    """
    Upload an image to ComfyUI for use in workflows.

    Use this to upload reference images for ControlNet, IP-Adapter,
    or Image-to-Video workflows.

    Args:
        image_path: Local path to the image file.
        filename: Target filename in ComfyUI (optional, uses original name).
        subfolder: Subfolder within ComfyUI input directory.
        overwrite: Whether to overwrite existing files (default True).

    Returns:
        {"name": "filename.png", "subfolder": "", "type": "input"}

    Example:
        result = upload_image("/path/to/reference.png")
        # Use result["name"] in workflow LoadImage node:
        # {"class_type": "LoadImage", "inputs": {"image": result["name"]}}
    """
    return execution.upload_image(image_path, filename, subfolder, overwrite)


@mcp.tool()
def download_output(asset_id: str, output_path: str) -> dict:
    """
    Download a generated asset to a local file.

    Args:
        asset_id: The asset ID to download.
        output_path: Local path to save the file.

    Returns:
        {"success": True, "path": "/path/to/file", "bytes": 12345}
    """
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
    """
    Publish a generated asset to a web directory.

    Two modes:
    - Demo mode: Provide target_filename for explicit naming (e.g., "hero.png")
    - Library mode: Provide manifest_key for auto-naming + manifest.json update

    Args:
        asset_id: The asset to publish.
        target_filename: Explicit filename (demo mode).
        manifest_key: Key for manifest.json (library mode).
        publish_dir: Target directory (auto-detected from public/gen, static/gen).
        web_optimize: Apply compression (default True).

    Returns:
        Published asset info with URL and path.

    Example:
        # Demo mode - explicit filename
        publish_asset(asset_id, target_filename="product_hero.png")

        # Library mode - auto filename + manifest
        publish_asset(asset_id, manifest_key="product_shot_1")
    """
    return publish.publish_asset(
        asset_id=asset_id,
        target_filename=target_filename,
        manifest_key=manifest_key,
        publish_dir=publish_dir,
        web_optimize=web_optimize,
    )


@mcp.tool()
def get_publish_info() -> dict:
    """
    Get current publish configuration.

    Returns:
        ComfyUI output dir, publish dir, and whether auto-detected.
    """
    return publish.get_publish_info()


@mcp.tool()
def set_publish_dir(publish_dir: str) -> dict:
    """
    Set the publish directory for assets.

    Args:
        publish_dir: Path to publish directory (will be created if needed).

    Returns:
        Success status.
    """
    return publish.set_publish_dir(publish_dir)


# =============================================================================
# Persistence Tools (Workflow Library)
# =============================================================================

@mcp.tool()
def save_workflow(
    name: str,
    workflow: dict,
    description: str = "",
    tags: list = None,
) -> dict:
    """
    Save a workflow to the local library for reuse.

    Args:
        name: Unique name (e.g., "flux-portrait", "qwen-landscape")
        workflow: The workflow JSON object
        description: What this workflow does
        tags: List of tags (e.g., ["image", "flux", "portrait"])

    Returns:
        Success status and file path.
    """
    return persistence.save_workflow(name, workflow, description, tags or [])


@mcp.tool()
def load_workflow(name: str) -> dict:
    """
    Load a workflow from the local library.

    Args:
        name: The workflow name to load.

    Returns:
        The workflow object with metadata.
    """
    return persistence.load_workflow(name)


@mcp.tool()
@mcp_tool_wrapper
def list_saved_workflows(
    tag: str = None,
    limit: int = 20,
    cursor: str = None,
) -> dict:
    """
    List all saved workflows in the library with cursor-based pagination.

    Args:
        tag: Optional tag to filter by.
        limit: Maximum results per page (default 20).
        cursor: Pagination cursor from previous response (optional).

    Returns:
        {
            "workflows": [...],
            "nextCursor": "..." or null,
            "total": N
        }
    """
    result = persistence.list_workflows(tag)

    if "error" in result:
        return _to_mcp_response(result)

    workflows = result.get("workflows", [])
    paginated = paginate(workflows, cursor=cursor, limit=limit)

    return {
        "workflows": paginated["items"],
        "nextCursor": paginated["nextCursor"],
        "total": paginated["total"],
    }


@mcp.tool()
def delete_workflow(name: str) -> dict:
    """
    Delete a workflow from the library.

    Args:
        name: The workflow name to delete.
    """
    return persistence.delete_workflow(name)


@mcp.tool()
def duplicate_workflow(source_name: str, new_name: str) -> dict:
    """
    Duplicate an existing workflow with a new name.

    Args:
        source_name: The workflow to copy.
        new_name: Name for the new workflow.
    """
    return persistence.duplicate_workflow(source_name, new_name)


@mcp.tool()
def export_workflow(name: str) -> dict:
    """
    Export a workflow as raw JSON (without metadata).
    Useful for sharing or using in ComfyUI directly.

    Args:
        name: The workflow name.
    """
    return persistence.export_workflow(name)


@mcp.tool()
def import_workflow(
    name: str,
    workflow_json: dict,
    description: str = "",
    tags: list = None,
) -> dict:
    """
    Import a raw workflow JSON (e.g., exported from ComfyUI).

    Args:
        name: Name for the workflow.
        workflow_json: The raw workflow JSON.
        description: Optional description.
        tags: Optional tags.
    """
    return persistence.import_workflow(name, workflow_json, description, tags)


# =============================================================================
# VRAM Estimation Tools
# =============================================================================

@mcp.tool()
def estimate_vram(workflow: dict) -> dict:
    """
    Estimate GPU VRAM usage for a workflow before execution.

    Use this to check if a workflow will fit in available memory,
    and get recommendations for optimization if it won't.

    Args:
        workflow: The workflow JSON to analyze.

    Returns:
        Estimated VRAM, available VRAM, will_fit flag, and recommendations.
    """
    return vram.estimate_workflow_vram(workflow)


@mcp.tool()
def check_model_fits(model_name: str, precision: str = "default") -> dict:
    """
    Quick check if a specific model will fit in available VRAM.

    Args:
        model_name: Model filename (e.g., "flux1-dev-fp8.safetensors")
        precision: Override precision (fp32, fp16, bf16, fp8, default)

    Returns:
        Whether model fits, estimated VRAM, and alternatives if it doesn't.
    """
    return vram.check_model_fits(model_name, precision)


# =============================================================================
# Validation Tools
# =============================================================================

@mcp.tool()
@mcp_tool_wrapper
def validate_workflow(workflow: dict) -> dict:
    """
    Validate a workflow before execution to catch errors early.

    Checks:
    - All node types exist in ComfyUI
    - All model files referenced exist
    - Node connections are valid
    - Required inputs are provided
    - Has output node (SaveImage, etc.)

    Args:
        workflow: The workflow JSON to validate.

    Returns:
        Validation result with errors, warnings, and suggestions.
    """
    return _to_mcp_response(validation.validate_workflow(workflow))


@mcp.tool()
def validate_and_fix_workflow(workflow: dict) -> dict:
    """
    Validate workflow and attempt to fix common issues.

    Fixes applied:
    - Ensures all node IDs are strings
    - Fixes connection references

    Args:
        workflow: The workflow JSON to validate and fix.

    Returns:
        Fixed workflow and validation results.
    """
    return validation.validate_and_fix(workflow)


@mcp.tool()
def check_connection_compatibility(
    source_type: str,
    source_slot: int,
    target_type: str,
    target_input: str,
) -> dict:
    """
    Check if a connection between two nodes is type-compatible.

    Args:
        source_type: Source node class type (e.g., "UNETLoader")
        source_slot: Output slot index on source node
        target_type: Target node class type (e.g., "KSampler")
        target_input: Input name on target node (e.g., "model")

    Returns:
        Compatibility result with type information.
    """
    return validation.check_node_compatibility(
        source_type, source_slot, target_type, target_input
    )


# =============================================================================
# SOTA Model Recommendations
# =============================================================================

@mcp.tool()
def get_sota_models(category: str) -> dict:
    """
    Get current State-of-the-Art models for a category.

    Args:
        category: One of "image_gen", "video_gen", "controlnet"

    Returns:
        Current SOTA models and deprecated models to avoid.
    """
    return sota.get_sota_for_category(category)


@mcp.tool()
def recommend_model(task: str, available_vram_gb: float = None) -> dict:
    """
    Get the best model recommendation for a specific task.

    Args:
        task: The task type. Options:
              - portrait: Face/person images
              - text_in_image: Images with text/logos
              - fast_iteration: Quick drafts
              - talking_head: Video with speech
              - image_to_video: Animate an image
              - cinematic_video: High-quality video
              - style_transfer: Change image style
        available_vram_gb: GPU memory (auto-detected if not provided)

    Returns:
        Model recommendation with settings and VRAM check.
    """
    return sota.recommend_model_for_task(task, available_vram_gb)


@mcp.tool()
def check_model_freshness(model_name: str) -> dict:
    """
    Check if a model is current SOTA or deprecated.

    Args:
        model_name: The model filename or name.

    Returns:
        Status (CURRENT/DEPRECATED/UNKNOWN) with replacement suggestions.
    """
    return sota.check_model_is_sota(model_name)


@mcp.tool()
def get_model_settings(model_name: str) -> dict:
    """
    Get optimal ComfyUI settings for a model.

    Args:
        model_name: The model name or filename.

    Returns:
        Recommended CFG, steps, sampler, scheduler, and notes.
    """
    return sota.get_optimal_settings(model_name)


@mcp.tool()
def check_installed_sota() -> dict:
    """
    Check which SOTA models are installed in your ComfyUI.

    Returns:
        List of installed SOTA models and what's missing.
    """
    return sota.get_available_sota_models()


# =============================================================================
# Workflow Templates
# =============================================================================

@mcp.tool()
@mcp_tool_wrapper
def list_workflow_templates(
    limit: int = 50,
    cursor: str = None,
) -> dict:
    """
    List all available workflow templates with cursor-based pagination.

    Returns templates for SOTA models with {{PLACEHOLDER}} syntax for customization.

    Args:
        limit: Maximum results per page (default 50).
        cursor: Pagination cursor from previous response (optional).

    Returns:
        {
            "templates": [...],
            "nextCursor": "..." or null,
            "total": N
        }
    """
    result = templates.list_templates()

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
    """
    Get a workflow template by name.

    Args:
        name: Template name (e.g., "qwen_txt2img", "flux2_txt2img", "ltx2_txt2vid")

    Returns:
        Template with metadata and {{PLACEHOLDER}} fields.
    """
    return _to_mcp_response(templates.load_template(name))


@mcp.tool()
def create_workflow_from_template(
    template_name: str,
    parameters: dict,
) -> dict:
    """
    Create a ready-to-execute workflow from a template.

    Replaces {{PLACEHOLDER}} fields with actual values.

    Args:
        template_name: Name of the template (e.g., "qwen_txt2img")
        parameters: Dict of parameter values (e.g., {"PROMPT": "a dragon", "SEED": 123})

    Returns:
        Complete workflow ready for execute_workflow().

    Example:
        create_workflow_from_template(
            "qwen_txt2img",
            {"PROMPT": "a majestic dragon", "WIDTH": 1024, "HEIGHT": 1024}
        )
    """
    template = templates.load_template(template_name)
    if "error" in template:
        return template

    # Get defaults from metadata
    defaults = template.get("_meta", {}).get("defaults", {})

    # Merge defaults with provided parameters
    final_params = {**defaults, **parameters}

    # Inject parameters into template
    workflow = templates.inject_parameters(template, final_params)

    return {
        "workflow": workflow,
        "template": template_name,
        "parameters_used": final_params,
        "note": "Use execute_workflow() with the 'workflow' field to run this.",
    }


# =============================================================================
# Workflow Pattern Tools (Prevent Drift)
# =============================================================================

@mcp.tool()
def get_workflow_skeleton(
    model: str,
    task: str,
) -> dict:
    """
    Get exact working workflow JSON for a model+task combination.

    IMPORTANT: Use this instead of building workflows from scratch to prevent
    drift from working patterns. Returns complete, tested workflow structures.

    Args:
        model: Model identifier - "ltx2", "flux2", "wan26", "qwen", "sdxl"
        task: Task type - "txt2vid", "img2vid", "txt2img", "txt2vid_distilled"

    Returns:
        Complete workflow JSON with {{PLACEHOLDER}} parameters ready for injection.

    Example:
        skeleton = get_workflow_skeleton("ltx2", "txt2vid")
        # Returns exact 10-node workflow with LTXVLoader, LTXVConditioning,
        # SamplerCustom, etc. - not the broken KSampler pattern.
    """
    return patterns.get_workflow_skeleton(model, task)


@mcp.tool()
def get_model_constraints(model: str) -> dict:
    """
    Get hard constraints for a model that must not be violated.

    Returns critical parameters like:
    - CFG ranges (LTX-2 must be 3.0, not 7.0)
    - Resolution divisibility (FLUX: 16, others: 8)
    - Frame count rules (LTX-2: must be 8n+1)
    - Required nodes (SamplerCustom not KSampler for video)
    - Forbidden nodes (CheckpointLoaderSimple for FLUX)

    Args:
        model: Model identifier - "ltx2", "flux2", "wan26", "qwen", "sdxl", "hunyuan15"

    Returns:
        Constraints dict with cfg, resolution, frames, required_nodes, forbidden_nodes.

    Example:
        constraints = get_model_constraints("ltx2")
        # Returns: {"cfg": {"default": 3.0}, "frames": {"formula": "8n+1"}, ...}
    """
    return patterns.get_model_constraints(model)


@mcp.tool()
def get_node_chain(
    model: str,
    task: str,
) -> dict:
    """
    Get ordered list of required nodes with exact connection information.

    Shows the precise slot indices for each connection, preventing errors like
    connecting to slot [1, 1] when it should be [2, 0].

    Args:
        model: Model identifier
        task: Task type

    Returns:
        List of nodes in execution order with input/output slot mappings.

    Example:
        chain = get_node_chain("flux2", "txt2img")
        # Returns 13 nodes showing UNETLoader→DualCLIPLoader→FluxGuidance→...
        # with exact slot indices for each connection.
    """
    result = patterns.get_node_chain(model, task)
    if isinstance(result, dict) and "error" in result:
        return result
    return {"nodes": result, "model": model, "task": task, "count": len(result)}


@mcp.tool()
def validate_against_pattern(
    workflow: dict,
    model: str,
) -> dict:
    """
    Validate a workflow against known working patterns to detect drift.

    Catches common mistakes like:
    - Using KSampler instead of SamplerCustom
    - Missing LTXVConditioning or FluxGuidance
    - Wrong CFG values (7.0 instead of 3.0 for LTX-2)
    - Wrong loader nodes
    - Invalid resolution or frame counts

    Args:
        workflow: Workflow JSON to validate
        model: Model identifier for constraint lookup

    Returns:
        {
            "valid": True/False,
            "errors": ["Using KSampler - should use SamplerCustom", ...],
            "warnings": [...],
            "suggestions": [...]
        }

    Example:
        result = validate_against_pattern(my_workflow, "ltx2")
        if not result["valid"]:
            print("Errors:", result["errors"])
    """
    return patterns.validate_against_pattern(workflow, model)


@mcp.tool()
def list_available_patterns() -> dict:
    """
    List all available workflow patterns and supported models.

    Returns:
        {
            "skeletons": [{"model": "ltx2", "task": "txt2vid", "description": "..."}],
            "models": ["ltx2", "flux2", "wan26", "qwen", "sdxl", "hunyuan15"],
            "total": count
        }
    """
    return patterns.list_available_patterns()


# =============================================================================
# Workflow Builder Tools
# =============================================================================

@mcp.tool()
def explain_workflow(workflow: dict) -> dict:
    """
    Generate natural language description of a workflow.

    Analyzes the workflow structure and describes what each node does
    and how they're connected. Useful for understanding existing workflows.

    Args:
        workflow: Workflow JSON to explain

    Returns:
        Human-readable description of the workflow structure.
    """
    explanation = workflow_builder.explain_workflow(workflow)
    return {"explanation": explanation}


@mcp.tool()
def get_required_nodes(model: str) -> dict:
    """
    Get required and forbidden nodes for a specific model.

    Use this to understand what nodes are mandatory and which to avoid
    when building workflows for a particular model.

    Args:
        model: Model identifier (ltx2, flux2, wan26, qwen, sdxl)

    Returns:
        Dict with required_nodes and forbidden_nodes for the model.

    Example:
        nodes = get_required_nodes("ltx2")
        # Returns: {"required_nodes": {"sampler": "SamplerCustom", ...},
        #           "forbidden_nodes": {"KSampler": "Use SamplerCustom..."}}
    """
    return workflow_builder.get_required_nodes_for_model(model)


@mcp.tool()
def get_connection_pattern(
    source_node: str,
    target_node: str,
) -> dict:
    """
    Get the correct connection pattern between two node types.

    Shows the exact output slot and input name to use when connecting nodes.
    Prevents common mistakes like wrong slot indices.

    Args:
        source_node: Source node class_type (e.g., "LTXVLoader")
        target_node: Target node class_type (e.g., "CLIPTextEncode")

    Returns:
        Connection info with output_slot and input_name.

    Example:
        pattern = get_connection_pattern("LTXVLoader", "CLIPTextEncode")
        # Returns: {"output_slot": 1, "input_name": "clip", "note": "CLIP is at slot 1"}
    """
    return workflow_builder.get_connection_pattern(source_node, target_node)


@mcp.tool()
def compare_workflows(
    workflow1: dict,
    workflow2: dict,
) -> dict:
    """
    Compare two workflows and show differences.

    Useful for understanding what changed between versions or comparing
    a generated workflow against a known working template.

    Args:
        workflow1: First workflow (e.g., reference)
        workflow2: Second workflow (e.g., generated)

    Returns:
        Dict with added, removed, and modified nodes.
    """
    return workflow_builder.get_workflow_diff(workflow1, workflow2)


# =============================================================================
# Template Discovery Tools
# =============================================================================

@mcp.tool()
def find_template_for_task(
    task_description: str,
) -> dict:
    """
    Find the best template for a given task description.

    Matches the task description against available templates based on
    model type, task type, and capabilities.

    Args:
        task_description: Natural language description of what you want to do.
                          E.g., "generate a video from text", "create an image with text"

    Returns:
        List of matching templates ranked by relevance.

    Example:
        templates = find_template_for_task("fast video from an image")
        # Returns templates for img2vid sorted by speed
    """
    all_templates = templates.list_templates()["templates"]

    # Simple keyword matching (could be enhanced with semantic search)
    keywords = task_description.lower().split()

    scored = []
    for t in all_templates:
        score = 0
        template_text = f"{t['description']} {t['type']} {t['model']}".lower()

        for keyword in keywords:
            if keyword in template_text:
                score += 1

        # Boost for exact type matches
        if "video" in keywords and "vid" in t["type"]:
            score += 2
        if "image" in keywords and "img" in t["type"]:
            score += 2
        if "fast" in keywords and "distilled" in t["name"]:
            score += 3
        if "text" in keywords and "txt" in t["type"]:
            score += 1

        if score > 0:
            scored.append({"template": t, "score": score})

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": task_description,
        "matches": scored[:5],
        "total_templates": len(all_templates)
    }


@mcp.tool()
def get_templates_by_type(
    template_type: str,
) -> dict:
    """
    Filter templates by type.

    Args:
        template_type: One of "txt2img", "img2vid", "txt2vid", "img2img"

    Returns:
        List of templates matching the type.
    """
    all_templates = templates.list_templates()["templates"]
    filtered = [t for t in all_templates if t["type"] == template_type]

    return {
        "type": template_type,
        "templates": filtered,
        "count": len(filtered)
    }


@mcp.tool()
def get_templates_by_model(
    model: str,
) -> dict:
    """
    Filter templates by model.

    Args:
        model: Model name to filter by (e.g., "LTX-2", "FLUX", "Wan")

    Returns:
        List of templates for that model.
    """
    all_templates = templates.list_templates()["templates"]
    model_lower = model.lower()
    filtered = [t for t in all_templates if model_lower in t["model"].lower() or model_lower in t["name"]]

    return {
        "model": model,
        "templates": filtered,
        "count": len(filtered)
    }


@mcp.tool()
def validate_all_templates() -> dict:
    """
    Validate all templates in the library.

    Checks each template for:
    - Valid _meta section
    - Declared parameters match placeholders
    - Valid node structure
    - Valid connection formats

    Returns:
        Validation results for each template.
    """
    return templates.validate_all_templates()


# =============================================================================
# Batch Execution
# =============================================================================

@mcp.tool()
def execute_batch_workflows(
    workflow: dict,
    parameter_sets: list,
    parallel: int = 1,
    timeout_per_job: int = 600,
) -> dict:
    """
    Execute the same workflow with multiple parameter sets.

    Use this to generate multiple variations efficiently.

    Args:
        workflow: Base workflow with {{PLACEHOLDER}} fields.
        parameter_sets: List of parameter dicts.
                        e.g., [{"PROMPT": "a cat", "SEED": 1}, {"PROMPT": "a dog", "SEED": 2}]
        parallel: Max concurrent executions (default 1).
        timeout_per_job: Timeout per job in seconds.

    Returns:
        Results for each parameter set with output files.
    """
    return batch.execute_batch(workflow, parameter_sets, parallel, timeout_per_job)


@mcp.tool()
def execute_parameter_sweep(
    workflow: dict,
    sweep_params: dict,
    fixed_params: dict = None,
    parallel: int = 1,
) -> dict:
    """
    Execute a parameter sweep over specified values.

    Useful for testing different CFG values, steps, guidance, etc.

    Args:
        workflow: Base workflow with {{PLACEHOLDER}} fields.
        sweep_params: Parameters to sweep over.
                      e.g., {"CFG": [3.0, 3.5, 4.0], "STEPS": [20, 30]}
                      This would run 6 combinations.
        fixed_params: Parameters to keep constant across all runs.
        parallel: Max concurrent executions.

    Returns:
        Results organized by parameter combination.
    """
    return batch.execute_sweep(workflow, sweep_params, fixed_params, parallel)


@mcp.tool()
def generate_seed_variations(
    workflow: dict,
    parameters: dict,
    num_variations: int = 4,
    start_seed: int = 42,
    parallel: int = 2,
) -> dict:
    """
    Generate multiple outputs from the same prompt with different seeds.

    Quick way to explore variations without changing the prompt.

    Args:
        workflow: Workflow with {{SEED}} placeholder.
        parameters: Base parameters (PROMPT, etc.).
        num_variations: Number of variations (default 4).
        start_seed: Starting seed value.
        parallel: Concurrent executions (default 2).

    Returns:
        Results for each seed variation.
    """
    return batch.execute_seed_variations(
        workflow, parameters, num_variations, start_seed, parallel
    )


# =============================================================================
# Multi-Stage Pipelines
# =============================================================================

@mcp.tool()
def execute_pipeline_stages(
    stages: list,
    initial_params: dict,
    timeout_per_stage: int = 600,
) -> dict:
    """
    Execute a multi-stage pipeline where outputs flow between stages.

    Use this for complex workflows like: generate image -> upscale -> create video.

    Args:
        stages: List of stage definitions. Each stage has:
                - name: Stage identifier
                - workflow: Workflow JSON with {{PLACEHOLDER}} fields
                - output_mapping: Optional dict mapping output type to next stage's param
                  e.g., {"images": "IMAGE_PATH"} passes images to next stage's IMAGE_PATH
        initial_params: Starting parameters for first stage.
        timeout_per_stage: Max time per stage in seconds.

    Returns:
        Complete pipeline results with all stage outputs.

    Example:
        execute_pipeline_stages(
            stages=[
                {
                    "name": "generate",
                    "workflow": qwen_workflow,
                    "output_mapping": {"images": "IMAGE_PATH"}
                },
                {
                    "name": "animate",
                    "workflow": wan_workflow
                }
            ],
            initial_params={"PROMPT": "a dragon", "SEED": 42}
        )
    """
    return pipeline.execute_pipeline(stages, initial_params, timeout_per_stage)


@mcp.tool()
def run_image_to_video_pipeline(
    image_workflow: dict,
    video_workflow: dict,
    prompt: str,
    video_prompt: str = None,
    seed: int = 42,
) -> dict:
    """
    Convenient pipeline: Generate image then animate to video.

    Common workflow for creating AI videos from text prompts.

    Args:
        image_workflow: Text-to-image workflow (e.g., Qwen, FLUX.2).
        video_workflow: Image-to-video workflow (e.g., Wan 2.6).
        prompt: Image generation prompt.
        video_prompt: Optional motion prompt (defaults to image prompt).
        seed: Random seed for reproducibility.

    Returns:
        Pipeline results with final video output.
    """
    return pipeline.create_image_to_video_pipeline(
        image_workflow, video_workflow, prompt, video_prompt, seed
    )


@mcp.tool()
def run_upscale_pipeline(
    base_workflow: dict,
    upscale_workflow: dict,
    prompt: str,
    upscale_factor: float = 2.0,
    seed: int = 42,
) -> dict:
    """
    Convenient pipeline: Generate image then upscale.

    Use for high-resolution outputs.

    Args:
        base_workflow: Base image generation workflow.
        upscale_workflow: Upscaling workflow (uses IMAGE_PATH parameter).
        prompt: Generation prompt.
        upscale_factor: How much to upscale (2.0 = double resolution).
        seed: Random seed.

    Returns:
        Pipeline results with upscaled output.
    """
    return pipeline.create_upscale_pipeline(
        base_workflow, upscale_workflow, prompt, upscale_factor, seed
    )


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
def search_civitai(
    query: str,
    model_type: str = None,
    nsfw: bool = False,
    limit: int = 10,
) -> dict:
    """
    Search Civitai for models by query.

    Use this to find LoRAs, checkpoints, and other models when a workflow
    requires a model that isn't installed. Returns download URLs and
    trigger words.

    Args:
        query: Search term (e.g., "anime style", "flux lora", "cinematic")
        model_type: Filter by type: "checkpoint", "lora", "embedding",
                   "controlnet", "upscaler" (optional)
        nsfw: Include NSFW models (default False)
        limit: Maximum results to return (default 10)

    Returns:
        {
            "models": [
                {
                    "name": "Model Name",
                    "type": "LORA",
                    "download_url": "https://...",
                    "filename": "model.safetensors",
                    "trigger_words": ["trigger1", "trigger2"],
                    "base_model": "SDXL 1.0",
                    "rating": 4.8,
                    "downloads": 12345
                }
            ],
            "total": 50
        }

    Example:
        # Find anime style LoRAs
        result = search_civitai("anime style", model_type="lora")
        for model in result["models"]:
            print(f"{model['name']}: {model['trigger_words']}")
    """
    return models.search_civitai(query, model_type, nsfw, limit)


@mcp.tool()
@mcp_tool_wrapper
def download_model(
    url: str,
    model_type: str,
    filename: str = None,
    overwrite: bool = False,
) -> dict:
    """
    Download a model to the appropriate ComfyUI directory.

    Use this for self-healing when a workflow fails due to missing models.
    Supports Civitai and HuggingFace download URLs.

    Args:
        url: Download URL (from search_civitai or HuggingFace)
        model_type: Where to save: "checkpoint", "unet", "lora", "vae",
                   "controlnet", "clip", "upscaler", "embedding"
        filename: Target filename (auto-detected if not provided)
        overwrite: Replace existing file (default False)

    Returns:
        {
            "success": True,
            "path": "/path/to/model.safetensors",
            "size_mb": 2048.5
        }

    Example:
        # Self-healing: workflow failed because LoRA missing
        search = search_civitai("anime style", model_type="lora")
        if search["models"]:
            result = download_model(
                url=search["models"][0]["download_url"],
                model_type="lora"
            )
            if result["success"]:
                # Retry workflow
                execute_workflow(workflow)

    Environment:
        Set CIVITAI_API_TOKEN for authenticated downloads (higher rate limits)
    """
    return _to_mcp_response(models.download_model(url, model_type, filename, overwrite))


@mcp.tool()
def get_model_info(model_path: str) -> dict:
    """
    Get information about an installed model.

    Args:
        model_path: Path to model (can be relative like "anime_style.safetensors")

    Returns:
        {
            "exists": True,
            "filename": "model.safetensors",
            "size_mb": 2048.5,
            "type": "safetensors",
            "full_path": "/path/to/model.safetensors"
        }
    """
    return models.get_model_info(model_path)


@mcp.tool()
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

    Example:
        # See what LoRAs are installed
        result = list_installed_models("lora")
        print(f"Found {result['total']} LoRAs")
    """
    return models.list_installed_models(model_type)


# =============================================================================
# Asset Analysis Tools
# =============================================================================

@mcp.tool()
def get_image_dimensions(asset_id: str) -> dict:
    """
    Get dimensions of an image asset.

    Essential for Image-to-Video workflows where agent needs to know
    aspect ratio to set video width/height correctly.

    Args:
        asset_id: The asset ID to analyze.

    Returns:
        {
            "width": 1024,
            "height": 768,
            "aspect_ratio": "4:3",
            "aspect_ratio_decimal": 1.333,
            "orientation": "landscape",
            "recommended_video_size": {"width": 1280, "height": 720}
        }

    Example:
        # After generating an image, check dimensions for video
        dims = get_image_dimensions(image_asset_id)
        video_workflow = inject_parameters(workflow, {
            "WIDTH": dims["recommended_video_size"]["width"],
            "HEIGHT": dims["recommended_video_size"]["height"]
        })
    """
    return analysis.get_image_dimensions(asset_id)


@mcp.tool()
@mcp_tool_wrapper
def detect_objects(
    asset_id: str,
    objects: list,
    vlm_model: str = None,
) -> dict:
    """
    Detect if specific objects exist in a generated image.

    Use this to validate output before expensive operations like video
    generation. E.g., verify "a cat" actually exists in output before
    generating a 5-second video of it.

    Args:
        asset_id: The asset to analyze.
        objects: List of objects to detect (e.g., ["cat", "dog", "person"]).
        vlm_model: VLM to use (default: qwen2.5-vl:7b via Ollama).

    Returns:
        {
            "detected": ["cat", "person"],
            "not_detected": ["dog"],
            "confidence": {
                "cat": 0.95,
                "person": 0.87,
                "dog": 0.12
            },
            "description": "The image shows a cat sitting next to a person..."
        }

    Example:
        # Validate before video generation
        result = detect_objects(image_id, ["dragon", "fire"])
        if "dragon" not in result["detected"]:
            # Regenerate with different seed
            regenerate(image_id, seed=None)
    """
    return _to_mcp_response(analysis.detect_objects(asset_id, objects, vlm_model))


@mcp.tool()
def get_video_info(asset_id: str) -> dict:
    """
    Get information about a video asset.

    Args:
        asset_id: The video asset ID.

    Returns:
        {
            "width": 1280,
            "height": 720,
            "duration_seconds": 5.0,
            "frame_count": 120,
            "fps": 24,
            "aspect_ratio": "16:9"
        }
    """
    return analysis.get_video_info(asset_id)


# =============================================================================
# Quality Assurance Tools
# =============================================================================

@mcp.tool()
@mcp_tool_wrapper
def qa_output(
    asset_id: str,
    prompt: str,
    checks: list = None,
    vlm_model: str = None,
) -> dict:
    """
    Run quality assurance on a generated asset using a Vision-Language Model.

    Automatically analyzes generated images to check for issues before
    presenting to the user. Uses a local VLM (via Ollama) or API-based VLM.

    Args:
        asset_id: The asset to evaluate.
        prompt: Original generation prompt for comparison.
        checks: List of checks to perform. Options:
                - "prompt_match": Does image match the prompt?
                - "artifacts": Visual artifacts, distortions, blur?
                - "faces": Face/hand issues (extra fingers, asymmetry)?
                - "text": Text rendering issues?
                - "composition": Overall composition quality?
                Default: ["prompt_match", "artifacts", "composition"]
        vlm_model: VLM to use (default: qwen2.5-vl:7b via Ollama).

    Returns:
        {
            "passed": True/False,
            "score": 0.0-1.0,
            "issues": ["list", "of", "issues"],
            "recommendation": "accept" | "regenerate" | "tweak_prompt"
        }

    Example:
        # After generating an image
        result = qa_output(
            asset_id="abc123",
            prompt="a majestic dragon breathing fire",
            checks=["prompt_match", "artifacts", "composition"]
        )
        if not result["passed"]:
            # Auto-regenerate with new seed
            regenerate(asset_id, seed=None)
    """
    return _to_mcp_response(qa.qa_output(
        asset_id=asset_id,
        prompt=prompt,
        checks=checks,
        vlm_model=vlm_model,
    ))


@mcp.tool()
def check_vlm_available(vlm_model: str = None) -> dict:
    """
    Check if VLM is available for QA operations.

    Args:
        vlm_model: Specific model to check for (default: qwen2.5-vl:7b).

    Returns:
        {
            "available": True/False,
            "ollama_url": "http://localhost:11434",
            "models": ["list", "of", "installed", "models"],
            "model_available": True/False
        }

    Example:
        status = check_vlm_available()
        if not status["available"]:
            print("Start Ollama with: ollama serve")
        if not status["model_available"]:
            print(f"Pull model with: ollama pull {status['requested_model']}")
    """
    return qa.check_vlm_available(vlm_model)


# =============================================================================
# Style Learning Tools
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
    """
    Record a generation for style learning.

    Store prompts, seeds, and ratings to learn from successful generations
    and improve future prompt engineering.

    Args:
        prompt: The generation prompt.
        model: Model used (e.g., "flux2-dev").
        seed: Random seed.
        parameters: Full parameter dict.
        negative_prompt: Negative prompt if any.
        rating: User rating 0.0-1.0 (None if not rated yet).
        tags: Style tags (e.g., ["anime", "portrait"]).
        outcome: "success", "failed", or "regenerated".
        qa_score: Automated QA score if available.
        notes: Any additional notes.

    Returns:
        {"record_id": "gen_xxx", "success": True}

    Example:
        # After a successful generation
        record_generation(
            prompt="a majestic dragon",
            model="flux2-dev",
            seed=42,
            rating=0.9,
            tags=["fantasy", "creature"]
        )
    """
    record_id = style_learning.record_generation(
        prompt=prompt,
        model=model,
        seed=seed,
        parameters=parameters,
        negative_prompt=negative_prompt,
        rating=rating,
        tags=tags,
        outcome=outcome,
        qa_score=qa_score,
        notes=notes,
    )
    return {"record_id": record_id, "success": True}


@mcp.tool()
def rate_generation(record_id: str, rating: float, notes: str = None) -> dict:
    """
    Rate a past generation (0.0-1.0).

    Args:
        record_id: The generation record ID.
        rating: Rating between 0.0 (bad) and 1.0 (excellent).
        notes: Optional notes about what was good/bad.

    Returns:
        {"success": True/False}
    """
    success = style_learning.rate_generation(record_id, rating, notes)
    return {"success": success}


@mcp.tool()
def suggest_prompt_enhancement(
    prompt: str,
    model: str = None,
    style_tags: list = None,
) -> dict:
    """
    Get prompt enhancement suggestions based on past successful generations.

    Analyzes your generation history to suggest improvements.

    Args:
        prompt: The base prompt to enhance.
        model: Target model.
        style_tags: Desired style tags.

    Returns:
        {
            "recommended_additions": ["words", "to", "add"],
            "negative_suggestions": ["suggested negative prompts"],
            "similar_successful": [...],
            "best_seeds": [...]
        }

    Example:
        suggestions = suggest_prompt_enhancement(
            prompt="a portrait of a woman",
            model="flux2-dev",
            style_tags=["portrait", "realistic"]
        )
        enhanced_prompt = prompt + ", " + ", ".join(suggestions["recommended_additions"])
    """
    return style_learning.suggest_prompt_enhancement(
        prompt=prompt,
        model=model,
        style_tags=style_tags,
    )


@mcp.tool()
def find_similar_prompts(
    prompt: str,
    model: str = None,
    min_rating: float = 0.7,
    limit: int = 5,
) -> dict:
    """
    Find similar past generations with good ratings.

    Args:
        prompt: The prompt to match against.
        model: Optional model filter.
        min_rating: Minimum rating threshold (default 0.7).
        limit: Maximum results (default 5).

    Returns:
        {"similar": [...], "count": N}
    """
    results = style_learning.find_similar_prompts(
        prompt=prompt,
        model=model,
        min_rating=min_rating,
        limit=limit,
    )
    return {"similar": results, "count": len(results)}


@mcp.tool()
def get_best_seeds_for_style(
    tags: list,
    model: str = None,
    limit: int = 10,
) -> dict:
    """
    Get best-rated seeds for a specific style.

    Args:
        tags: Style tags to match (e.g., ["anime", "portrait"]).
        model: Optional model filter.
        limit: Maximum results.

    Returns:
        {"seeds": [...], "count": N}

    Example:
        # Find seeds that worked well for anime portraits
        result = get_best_seeds_for_style(["anime", "portrait"])
        for item in result["seeds"]:
            print(f"Seed {item['seed']} rated {item['rating']}")
    """
    results = style_learning.get_best_seeds_for_style(
        tags=tags,
        model=model,
        limit=limit,
    )
    return {"seeds": results, "count": len(results)}


@mcp.tool()
def save_style_preset(
    name: str,
    description: str,
    prompt_additions: str,
    negative_additions: str = "",
    recommended_model: str = None,
    recommended_params: dict = None,
) -> dict:
    """
    Save a reusable style preset.

    Args:
        name: Preset name (e.g., "cinematic_portrait").
        description: Description of the style.
        prompt_additions: Text to add to prompts.
        negative_additions: Text to add to negative prompts.
        recommended_model: Best model for this style.
        recommended_params: Recommended parameters.

    Returns:
        {"success": True}

    Example:
        save_style_preset(
            name="anime_portrait",
            description="High-quality anime character portraits",
            prompt_additions="anime style, detailed eyes, soft shading",
            negative_additions="realistic, photograph, 3d render",
            recommended_model="flux2-dev"
        )
    """
    success = style_learning.save_style_preset(
        name=name,
        description=description,
        prompt_additions=prompt_additions,
        negative_additions=negative_additions,
        recommended_model=recommended_model,
        recommended_params=recommended_params,
    )
    return {"success": success}


@mcp.tool()
def get_style_preset(name: str) -> dict:
    """
    Get a saved style preset.

    Args:
        name: Preset name.

    Returns:
        Style preset dict or {"error": "Not found"}
    """
    preset = style_learning.get_style_preset(name)
    if preset:
        return preset
    return {"error": f"Style preset '{name}' not found"}


@mcp.tool()
@mcp_tool_wrapper
def list_style_presets(
    limit: int = 20,
    cursor: str = None,
) -> dict:
    """
    List all saved style presets with cursor-based pagination.

    Args:
        limit: Maximum results per page (default 20).
        cursor: Pagination cursor from previous response (optional).

    Returns:
        {
            "presets": [...],
            "nextCursor": "..." or null,
            "total": N
        }
    """
    presets = style_learning.list_style_presets()
    paginated = paginate(presets, cursor=cursor, limit=limit)

    return {
        "presets": paginated["items"],
        "nextCursor": paginated["nextCursor"],
        "total": paginated["total"],
    }


@mcp.tool()
def get_style_learning_stats() -> dict:
    """
    Get statistics about stored generations and learning data.

    Returns:
        {
            "total_generations": N,
            "average_rating": 0.X,
            "high_rated_count": N,
            "style_presets": N
        }
    """
    return style_learning.get_statistics()


# =============================================================================
# Schema Documentation Tools
# =============================================================================

@mcp.tool()
def get_tool_schema(tool_name: str) -> dict:
    """
    Get the JSON Schema for a tool's input parameters.

    Useful for understanding the expected structure of complex inputs
    like workflows and parameter sets.

    Args:
        tool_name: Name of the tool (e.g., "execute_workflow", "qa_output")

    Returns:
        JSON Schema definition for the tool's inputs.
    """
    schema = schemas.get_schema(tool_name)
    if schema:
        return {"schema": schema, "tool": tool_name}
    return mcp_error(
        f"No schema found for tool: {tool_name}",
        "NOT_FOUND",
        {"available": schemas.list_schemas()}
    )


@mcp.tool()
def list_tool_schemas() -> dict:
    """
    List all tools with explicit JSON Schema definitions.

    Returns:
        {"input_schemas": [...], "output_schemas": [...]}
    """
    return {
        "input_schemas": schemas.list_schemas(),
        "output_schemas": schemas.list_output_schemas(),
    }


@mcp.tool()
def get_tool_output_schema(tool_name: str) -> dict:
    """
    Get the JSON Schema for a tool's output (response structure).

    Per June 2025 MCP spec, output schemas define expected response structure.

    Args:
        tool_name: Name of the tool

    Returns:
        JSON Schema definition for the tool's outputs.
    """
    schema = schemas.get_output_schema(tool_name)
    if schema:
        return {"schema": schema, "tool": tool_name, "type": "output"}
    return mcp_error(
        f"No output schema found for tool: {tool_name}",
        "NOT_FOUND",
        {"available": schemas.list_output_schemas()}
    )


# =============================================================================
# Tool Annotation Tools
# =============================================================================

@mcp.tool()
def get_tool_annotation(tool_name: str) -> dict:
    """
    Get MCP annotations for a tool.

    Annotations describe:
    - audience: Who should see results (["user"], ["assistant"], or both)
    - priority: Importance 0.0-1.0
    - category: Tool category for grouping

    Args:
        tool_name: Name of the tool

    Returns:
        Annotation metadata for the tool.
    """
    annotation = annotations.get_annotation(tool_name)
    if annotation:
        return {"tool": tool_name, "annotation": annotation}
    return mcp_error(
        f"No annotation found for tool: {tool_name}",
        "NOT_FOUND"
    )


@mcp.tool()
def list_user_facing_tools() -> dict:
    """
    List tools whose results should be shown to users.

    These are tools with audience: ["user"] or ["user", "assistant"].

    Returns:
        {"tools": [...], "count": N}
    """
    tools = annotations.get_user_facing_tools()
    return {"tools": tools, "count": len(tools)}


@mcp.tool()
def list_tools_by_category(category: str) -> dict:
    """
    List tools in a specific category.

    Categories: discovery, execution, assets, publishing, library,
    vram, validation, sota, templates, batch, pipeline, models,
    analysis, qa, style_learning

    Args:
        category: Tool category name

    Returns:
        {"tools": [...], "category": "...", "count": N}
    """
    tools = annotations.get_tools_by_category(category)
    if tools:
        return {"tools": tools, "category": category, "count": len(tools)}
    return mcp_error(
        f"Unknown category: {category}",
        "NOT_FOUND",
        {"available_categories": [
            "discovery", "execution", "assets", "publishing",
            "library", "vram", "validation", "sota", "templates",
            "batch", "pipeline", "models", "analysis", "qa", "style_learning"
        ]}
    )


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
