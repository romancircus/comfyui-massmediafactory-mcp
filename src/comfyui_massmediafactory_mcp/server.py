"""
ComfyUI MassMediaFactory MCP Server

Main entry point that exposes all tools via MCP protocol.
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
from . import batch
from . import pipeline
from . import publish
from . import qa
from . import models

# Initialize MCP server
mcp = FastMCP(
    "comfyui-massmediafactory",
    instructions="ComfyUI workflow orchestration for image and video generation",
)


# =============================================================================
# Discovery Tools
# =============================================================================

@mcp.tool()
def list_checkpoints() -> dict:
    """
    List all available checkpoint models in ComfyUI.
    Returns model filenames for CheckpointLoaderSimple or UNETLoader.
    """
    return discovery.list_checkpoints()


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
def get_node_info(node_type: str) -> dict:
    """
    Get detailed information about a specific ComfyUI node type.

    Args:
        node_type: The node class name (e.g., "KSampler", "CLIPTextEncode", "UNETLoader")

    Returns:
        Node schema including inputs, outputs, and their types.
    """
    return discovery.get_node_info(node_type)


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
    return execution.execute_workflow(workflow, client_id)


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
def wait_for_completion(prompt_id: str, timeout_seconds: int = 600) -> dict:
    """
    Wait for a workflow to complete and return outputs.

    Args:
        prompt_id: The prompt_id to wait for.
        timeout_seconds: Maximum wait time (default 600s / 10 minutes).

    Returns:
        Final status with output file paths.
    """
    return execution.wait_for_completion(prompt_id, timeout_seconds)


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
def list_assets(
    session_id: str = None,
    asset_type: str = None,
    limit: int = 20,
) -> dict:
    """
    List recent generated assets.

    Args:
        session_id: Filter by session (optional).
        asset_type: Filter by type: "images", "video", "audio" (optional).
        limit: Maximum results (default 20).

    Returns:
        List of asset summaries with asset_ids.
    """
    return execution.list_assets(
        session_id=session_id,
        asset_type=asset_type,
        limit=limit,
    )


@mcp.tool()
def get_asset_metadata(asset_id: str) -> dict:
    """
    Get full metadata for an asset including workflow and parameters.

    Useful for debugging or understanding how an asset was generated.

    Args:
        asset_id: The asset ID to retrieve.

    Returns:
        Full asset metadata including original workflow.
    """
    return execution.get_asset_metadata(asset_id)


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
def list_saved_workflows(tag: str = None) -> dict:
    """
    List all saved workflows in the library.

    Args:
        tag: Optional tag to filter by.

    Returns:
        List of workflow summaries.
    """
    return persistence.list_workflows(tag)


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
    return validation.validate_workflow(workflow)


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
def list_workflow_templates() -> dict:
    """
    List all available workflow templates.

    Returns templates for SOTA models with {{PLACEHOLDER}} syntax for customization.
    """
    return templates.list_templates()


@mcp.tool()
def get_template(name: str) -> dict:
    """
    Get a workflow template by name.

    Args:
        name: Template name (e.g., "qwen_txt2img", "flux2_txt2img", "ltx2_txt2vid")

    Returns:
        Template with metadata and {{PLACEHOLDER}} fields.
    """
    return templates.load_template(name)


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
# Templates as Resources (Legacy)
# =============================================================================

@mcp.resource("template://flux-txt2img")
def flux_txt2img_template() -> str:
    """Flux text-to-image workflow template with {{PROMPT}}, {{SEED}} placeholders."""
    template = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "{{MODEL}}", "weight_dtype": "default"}
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp16.safetensors",
                "type": "flux"
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"}
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": "{{PROMPT}}"}
        },
        "5": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["4", 0], "guidance": 3.5}
        },
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1}
        },
        "7": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"}
        },
        "8": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}
        },
        "9": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": "{{SEED}}"}
        },
        "10": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}
        },
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["10", 0],
                "sampler": ["7", 0],
                "sigmas": ["8", 0],
                "latent_image": ["6", 0]
            }
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["3", 0]}
        },
        "13": {
            "class_type": "SaveImage",
            "inputs": {"images": ["12", 0], "filename_prefix": "flux_output"}
        }
    }
    return json.dumps(template, indent=2)


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
    return models.download_model(url, model_type, filename, overwrite)


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
# Quality Assurance Tools
# =============================================================================

@mcp.tool()
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
    return qa.qa_output(
        asset_id=asset_id,
        prompt=prompt,
        checks=checks,
        vlm_model=vlm_model,
    )


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


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
