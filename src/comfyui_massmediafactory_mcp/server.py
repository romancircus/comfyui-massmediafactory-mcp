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

# Initialize MCP server
mcp = FastMCP(
    "comfyui-massmediafactory",
    description="ComfyUI workflow orchestration for image and video generation",
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
# Templates as Resources
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


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
