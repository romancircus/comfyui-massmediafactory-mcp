"""
Execution Tools

Tools for executing workflows and managing ComfyUI operations.
"""

import time
import copy
import random
from typing import Optional, Any
from .client import get_client
from .assets import get_registry
from .mcp_utils import log_structured, get_correlation_id


def execute_workflow(workflow: dict, client_id: str = "massmediafactory") -> dict:
    """
    Execute a ComfyUI workflow and return the prompt_id.

    Args:
        workflow: The workflow JSON object with node definitions.
                  Each key is a node ID, value is {"class_type": "...", "inputs": {...}}
        client_id: Optional client identifier for tracking.

    Returns:
        prompt_id for polling results, or error message.

    Example workflow:
        {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["1", 1]}},
            ...
        }
    """
    client = get_client()
    result = client.queue_prompt(workflow, client_id)

    if "error" in result:
        return result

    # Structured logging for workflow queued
    log_structured(
        "info",
        "workflow_queued",
        prompt_id=result.get("prompt_id"),
        client_id=client_id,
        node_count=len(workflow),
        queue_position=result.get("number", 0),
        correlation_id=get_correlation_id(),
    )

    return {
        "prompt_id": result.get("prompt_id"),
        "queue_position": result.get("number"),
        "status": "queued",
    }


def get_workflow_status(prompt_id: str) -> dict:
    """
    Check the status of a queued/running workflow.

    Args:
        prompt_id: The prompt_id returned from execute_workflow.

    Returns:
        Status (queued/running/completed/error) and outputs if completed.
    """
    client = get_client()
    result = client.get_history(prompt_id)

    if "error" in result:
        return result

    if prompt_id not in result:
        # Check if still in queue
        queue = client.get_queue()
        running = queue.get("queue_running", [])
        pending = queue.get("queue_pending", [])

        for item in running:
            if len(item) > 1 and item[1] == prompt_id:
                return {"status": "running", "prompt_id": prompt_id}

        for item in pending:
            if len(item) > 1 and item[1] == prompt_id:
                return {"status": "queued", "prompt_id": prompt_id}

        return {"status": "unknown", "prompt_id": prompt_id}

    entry = result[prompt_id]

    # Check for error
    if entry.get("status", {}).get("status_str") == "error":
        return {
            "status": "error",
            "prompt_id": prompt_id,
            "error": entry["status"].get("messages", []),
        }

    # Check for outputs
    if "outputs" in entry:
        outputs = []
        for node_id, output in entry["outputs"].items():
            for key in ["images", "video", "videos", "gifs", "audio"]:
                if key in output:
                    for item in output[key]:
                        outputs.append(
                            {
                                "type": key,
                                "filename": item.get("filename"),
                                "subfolder": item.get("subfolder", ""),
                                "node_id": node_id,
                            }
                        )

        return {
            "status": "completed",
            "prompt_id": prompt_id,
            "outputs": outputs,
        }

    return {"status": "running", "prompt_id": prompt_id}


def wait_for_completion(
    prompt_id: str,
    timeout_seconds: int = 600,
    poll_interval: float = 2.0,
    workflow: dict = None,
    parameters: dict = None,
    session_id: str = None,
) -> dict:
    """
    Wait for a workflow to complete and return outputs.

    Uses WebSocket for event-driven waiting (1 connection vs 300 polls).
    Falls back to HTTP polling if WebSocket connection fails.

    Args:
        prompt_id: The prompt_id to wait for.
        timeout_seconds: Maximum time to wait (default 600s / 10 minutes).
        poll_interval: Seconds between status checks (only used in fallback).
        workflow: Original workflow (for asset registration).
        parameters: Template parameters used (for regeneration).
        session_id: Session ID for grouping related generations.

    Returns:
        Final status and output files with asset_ids.
    """
    start_time = time.time()
    cid = get_correlation_id()
    log_structured(
        "info",
        "waiting_for_completion",
        prompt_id=prompt_id,
        timeout_seconds=timeout_seconds,
        correlation_id=cid,
    )

    # Try WebSocket first (event-driven, no polling)
    ws_stage = _wait_via_websocket(prompt_id, timeout_seconds)

    if ws_stage is not None:
        # WebSocket told us the terminal state, now fetch full status once
        log_structured(
            "info",
            "ws_wait_complete",
            prompt_id=prompt_id,
            ws_stage=ws_stage,
            correlation_id=cid,
        )
        status = get_workflow_status(prompt_id)
        # If WS said completed but history isn't ready yet, give it a moment
        if status.get("status") not in ("completed", "error"):
            time.sleep(1.0)
            status = get_workflow_status(prompt_id)
    else:
        # WebSocket failed, fall back to polling
        log_structured(
            "info",
            "ws_fallback_to_polling",
            prompt_id=prompt_id,
            correlation_id=cid,
        )
        status = _poll_for_completion(prompt_id, timeout_seconds, start_time, poll_interval)

    if status is None:
        # Timeout
        elapsed = time.time() - start_time
        log_structured(
            "warning",
            "workflow_timeout",
            prompt_id=prompt_id,
            timeout_seconds=timeout_seconds,
            elapsed_seconds=round(elapsed, 2),
            correlation_id=cid,
        )
        return {
            "status": "timeout",
            "prompt_id": prompt_id,
            "elapsed_seconds": timeout_seconds,
        }

    elapsed = time.time() - start_time
    status["elapsed_seconds"] = round(elapsed, 1)

    # Register assets if workflow provided and completed
    if status.get("status") == "completed" and workflow:
        status = _register_outputs_as_assets(status, workflow, parameters, session_id)
        outputs = status.get("outputs", [])
        asset_ids = [a.get("asset_id") for a in outputs if a.get("asset_id")]
        log_structured(
            "info",
            "workflow_completed",
            prompt_id=prompt_id,
            status="completed",
            elapsed_seconds=round(elapsed, 2),
            output_count=len(outputs) if outputs else 0,
            asset_ids=asset_ids,
            correlation_id=cid,
        )
    elif status.get("status") == "error":
        error_details = status.get("error")
        log_structured(
            "error",
            "workflow_error",
            prompt_id=prompt_id,
            status=status["status"],
            error=str(error_details) if error_details else None,
            elapsed_seconds=round(elapsed, 2),
            correlation_id=cid,
        )

    return status


def _wait_via_websocket(prompt_id: str, timeout_seconds: int) -> Optional[str]:
    """Try to wait for completion via WebSocket.

    Returns:
        Terminal stage ("completed"/"error") or None if WS unavailable.
    """
    try:
        from .websocket_client import wait_for_prompt_ws

        return wait_for_prompt_ws(prompt_id, timeout_seconds=timeout_seconds)
    except Exception:
        return None


def _poll_for_completion(
    prompt_id: str,
    timeout_seconds: int,
    start_time: float,
    poll_interval: float,
) -> Optional[dict]:
    """Fall back to polling get_workflow_status(). Returns status or None on timeout."""
    while time.time() - start_time < timeout_seconds:
        status = get_workflow_status(prompt_id)
        if status["status"] in ["completed", "error"]:
            return status
        time.sleep(poll_interval)
    return None


def _register_outputs_as_assets(
    status: dict,
    workflow: dict,
    parameters: dict = None,
    session_id: str = None,
) -> dict:
    """Register completed outputs as assets in the registry."""
    registry = get_registry()
    outputs = status.get("outputs", [])
    assets = []

    for output in outputs:
        asset = registry.register_asset(
            filename=output.get("filename"),
            subfolder=output.get("subfolder", ""),
            asset_type=output.get("type", "images"),
            workflow=workflow,
            parameters=parameters,
            session_id=session_id,
            node_id=output.get("node_id"),
        )
        output["asset_id"] = asset.asset_id
        assets.append(asset)

    # Structured logging for asset registration
    asset_ids = [asset.asset_id for asset in assets]
    log_structured(
        "info",
        "assets_registered",
        count=len(assets),
        asset_ids=asset_ids,
        session_id=session_id,
        correlation_id=get_correlation_id(),
    )

    return status


def get_system_stats() -> dict:
    """
    Get ComfyUI system statistics including VRAM usage.

    Returns:
        System info including GPU memory, devices, etc.
    """
    client = get_client()
    result = client.get_system_stats()

    if "error" in result:
        return result

    # Parse and simplify the response
    devices = result.get("devices", [])
    gpu_info = []

    for device in devices:
        gpu_info.append(
            {
                "name": device.get("name", "Unknown"),
                "type": device.get("type", "Unknown"),
                "vram_total_gb": round(device.get("vram_total", 0) / (1024**3), 2),
                "vram_free_gb": round(device.get("vram_free", 0) / (1024**3), 2),
                "vram_used_gb": round(
                    (device.get("vram_total", 0) - device.get("vram_free", 0)) / (1024**3),
                    2,
                ),
            }
        )

    return {
        "devices": gpu_info,
        "system": result.get("system", {}),
    }


def free_memory(unload_models: bool = False) -> dict:
    """
    Free GPU memory in ComfyUI.

    Args:
        unload_models: If True, also unload all loaded models from VRAM.

    Returns:
        Success status.
    """
    client = get_client()
    result = client.free_memory(unload_models)

    if "error" in result:
        return result

    log_structured(
        "info",
        "memory_freed",
        unload_models=unload_models,
    )

    return {
        "success": True,
        "unloaded_models": unload_models,
    }


def interrupt_execution() -> dict:
    """
    Interrupt the currently running workflow.

    Returns:
        Success status.
    """
    client = get_client()
    result = client.interrupt()

    if "error" in result:
        return result

    log_structured(
        "info",
        "execution_interrupted",
        prompt_id=None,
    )

    return {"success": True, "message": "Execution interrupted"}


def get_queue_status() -> dict:
    """
    Get current queue status.

    Returns:
        Number of running and pending jobs.
    """
    client = get_client()
    result = client.get_queue()

    if "error" in result:
        return result

    return {
        "running": len(result.get("queue_running", [])),
        "pending": len(result.get("queue_pending", [])),
    }


# =============================================================================
# Asset Iteration Tools
# =============================================================================


def regenerate(
    asset_id: str,
    prompt: str = None,
    negative_prompt: str = None,
    seed: int = None,
    steps: int = None,
    cfg: float = None,
    **extra_overrides,
) -> dict:
    """
    Regenerate an asset with parameter overrides.

    Args:
        asset_id: The asset to regenerate from.
        prompt: New prompt (optional).
        negative_prompt: New negative prompt (optional).
        seed: New seed. Use -1 to keep original, None for random.
        steps: New step count (optional).
        cfg: New CFG scale (optional).
        **extra_overrides: Additional parameter overrides.

    Returns:
        New prompt_id for the regenerated workflow.
    """
    registry = get_registry()
    asset = registry.get_asset(asset_id)

    if asset is None:
        return {"error": "ASSET_NOT_FOUND_OR_EXPIRED", "asset_id": asset_id}

    if not asset.workflow:
        return {"error": "NO_WORKFLOW_DATA", "asset_id": asset_id}

    # Deep copy to avoid mutation
    workflow = copy.deepcopy(asset.workflow)
    new_params = copy.deepcopy(asset.parameters)

    # Log regeneration start
    cid = get_correlation_id()
    override_keys = list(extra_overrides.keys()) if extra_overrides else []
    log_structured(
        "info",
        "regeneration_started",
        asset_id=asset_id,
        overrides=override_keys,
        correlation_id=cid,
    )

    # Collect overrides
    overrides = {}
    if prompt is not None:
        overrides["prompt"] = prompt
        new_params["PROMPT"] = prompt
    if negative_prompt is not None:
        overrides["negative_prompt"] = negative_prompt
        new_params["NEGATIVE_PROMPT"] = negative_prompt
    if steps is not None:
        overrides["steps"] = steps
        new_params["STEPS"] = steps
    if cfg is not None:
        overrides["cfg"] = cfg
        new_params["CFG"] = cfg
    overrides.update(extra_overrides)
    new_params.update(extra_overrides)

    # Apply overrides to workflow
    workflow = _apply_workflow_overrides(workflow, overrides)

    # Handle seed
    if seed is None:
        # Random seed
        new_seed = random.randint(0, 2**32 - 1)
        workflow = _update_workflow_seed(workflow, new_seed)
        new_params["SEED"] = new_seed
    elif seed != -1:
        # Use provided seed (not -1 which means keep original)
        workflow = _update_workflow_seed(workflow, seed)
        new_params["SEED"] = seed

    # Execute regenerated workflow
    result = execute_workflow(workflow)

    if "error" in result:
        return result

    # Log regeneration completed
    log_structured(
        "info",
        "regeneration_completed",
        asset_id=asset_id,
        new_prompt_id=result.get("prompt_id"),
        correlation_id=cid,
    )

    # Return with context for wait_for_completion
    return {
        "prompt_id": result["prompt_id"],
        "status": "queued",
        "regenerated_from": asset_id,
        "session_id": asset.session_id,
        "_workflow": workflow,
        "_parameters": new_params,
    }


def _find_conditioning_nodes(workflow: dict) -> dict:
    """
    Trace sampler connections to identify positive vs negative conditioning nodes.

    Returns:
        {"positive": [node_ids], "negative": [node_ids]}
    """
    result = {"positive": [], "negative": []}

    # Find sampler nodes
    sampler_classes = ["KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced"]
    sampler_nodes = [
        node_id
        for node_id, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type") in sampler_classes
    ]

    if not sampler_nodes:
        return result

    # Trace connections for each sampler
    for sampler_id in sampler_nodes:
        sampler = workflow[sampler_id]
        inputs = sampler.get("inputs", {})

        # Trace positive conditioning
        if "positive" in inputs:
            pos_input = inputs["positive"]
            if isinstance(pos_input, list) and len(pos_input) >= 1:
                pos_node_id = str(pos_input[0])
                if pos_node_id in workflow:
                    result["positive"].append(pos_node_id)

        # Trace negative conditioning
        if "negative" in inputs:
            neg_input = inputs["negative"]
            if isinstance(neg_input, list) and len(neg_input) >= 1:
                neg_node_id = str(neg_input[0])
                if neg_node_id in workflow:
                    result["negative"].append(neg_node_id)

    return result


def _apply_workflow_overrides(workflow: dict, overrides: dict) -> dict:
    """Apply parameter overrides to matching workflow nodes."""
    # Find conditioning nodes by tracing sampler connections
    conditioning_nodes = _find_conditioning_nodes(workflow)

    override_map = {
        "steps": [
            ("KSampler", "steps"),
            ("KSamplerAdvanced", "steps"),
            ("SamplerCustomAdvanced", "steps"),
        ],
        "cfg": [
            ("KSampler", "cfg"),
            ("KSamplerAdvanced", "cfg"),
            ("FluxGuidance", "guidance"),
        ],
    }

    for key, value in overrides.items():
        # Handle prompt/negative_prompt with connection-aware targeting
        if key == "prompt":
            # Update positive conditioning nodes
            for node_id in conditioning_nodes.get("positive", []):
                if node_id in workflow:
                    node = workflow[node_id]
                    if node.get("class_type") == "CLIPTextEncode":
                        node["inputs"]["text"] = value
                    elif node.get("class_type") == "CLIPTextEncodeSDXL":
                        node["inputs"]["text_g"] = value
        elif key == "negative_prompt":
            # Update negative conditioning nodes
            for node_id in conditioning_nodes.get("negative", []):
                if node_id in workflow:
                    node = workflow[node_id]
                    if node.get("class_type") == "CLIPTextEncode":
                        node["inputs"]["text"] = value
        elif key in override_map:
            targets = override_map[key]
            for class_type, input_name in targets:
                _update_nodes_by_class(workflow, class_type, input_name, value)
        else:
            # Try as direct input name across all nodes
            for node_id, node in workflow.items():
                if isinstance(node, dict) and "inputs" in node:
                    if key in node["inputs"]:
                        node["inputs"][key] = value

    return workflow


def _update_nodes_by_class(workflow: dict, class_type: str, input_name: str, value: Any) -> None:
    """Update all nodes of a given class type."""
    for node_id, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            if "inputs" in node and input_name in node["inputs"]:
                node["inputs"][input_name] = value


def _update_workflow_seed(workflow: dict, seed: int) -> dict:
    """Update seed in all relevant nodes."""
    seed_nodes = [
        ("KSampler", "seed"),
        ("KSamplerAdvanced", "noise_seed"),
        ("RandomNoise", "noise_seed"),
        ("SamplerCustomAdvanced", "noise_seed"),
    ]

    for class_type, input_name in seed_nodes:
        _update_nodes_by_class(workflow, class_type, input_name, seed)

    return workflow


def list_assets(
    session_id: str = None,
    asset_type: str = None,
    limit: int = 20,
    include_expired: bool = False,
) -> dict:
    """
    List recent generated assets.

    Args:
        session_id: Filter by session (optional).
        asset_type: Filter by type: "images", "video", "audio" (optional).
        limit: Maximum results (default 20).
        include_expired: Include expired assets (default False).

    Returns:
        List of asset summaries.
    """
    registry = get_registry()
    assets = registry.list_assets(
        session_id=session_id,
        asset_type=asset_type,
        limit=limit,
        include_expired=include_expired,
    )

    return {
        "assets": [asset.to_dict() for asset in assets],
        "count": len(assets),
        "has_more": len(assets) == limit,
    }


def get_asset_metadata(asset_id: str) -> dict:
    """
    Get full metadata for an asset including workflow and parameters.

    Args:
        asset_id: The asset ID to retrieve.

    Returns:
        Full asset metadata for debugging/inspection.
    """
    registry = get_registry()
    asset = registry.get_asset(asset_id)

    if asset is None:
        return {"error": "ASSET_NOT_FOUND_OR_EXPIRED", "asset_id": asset_id}

    result = asset.to_full_dict()
    result["url"] = registry.get_asset_url(asset)
    return result


def view_output(
    asset_id: str,
    mode: str = "thumb",
    max_dim: int = 512,
) -> dict:
    """
    View a generated asset.

    Args:
        asset_id: The asset to view.
        mode: "thumb" for inline preview, "metadata" for info only.
        max_dim: Maximum dimension for thumbnail (default 512px).

    Returns:
        Asset preview or metadata.
    """
    registry = get_registry()
    asset = registry.get_asset(asset_id)

    if asset is None:
        return {"error": "ASSET_NOT_FOUND_OR_EXPIRED", "asset_id": asset_id}

    # Check supported types for viewing
    viewable_types = ["image/png", "image/jpeg", "image/webp", "image/gif"]
    if asset.mime_type not in viewable_types:
        return {
            "error": "UNSUPPORTED_TYPE",
            "mime_type": asset.mime_type,
            "url": registry.get_asset_url(asset),
            "message": f"Cannot preview {asset.mime_type}. Use the URL to download.",
        }

    if mode == "metadata":
        return {
            "asset_id": asset.asset_id,
            "filename": asset.filename,
            "type": asset.asset_type,
            "mime_type": asset.mime_type,
            "dimensions": [asset.width, asset.height] if asset.width else None,
            "created_at": asset.created_at,
            "url": registry.get_asset_url(asset),
            "prompt_preview": asset.prompt_preview,
        }

    # Thumb mode - return URL for now
    # Full base64 inline preview would require reading the file
    # which needs ComfyUI output dir access
    return {
        "asset_id": asset.asset_id,
        "url": registry.get_asset_url(asset),
        "mime_type": asset.mime_type,
        "dimensions": [asset.width, asset.height] if asset.width else None,
        "prompt_preview": asset.prompt_preview,
        "note": "Use the URL to view the image. Full inline preview requires output dir access.",
    }


def cleanup_expired_assets() -> dict:
    """
    Clean up expired assets from the registry.

    Returns:
        Number of assets removed.
    """
    registry = get_registry()
    removed_count = registry.cleanup_expired()

    log_structured(
        "info",
        "assets_cleanup",
        removed_count=removed_count,
    )

    return {"removed": removed_count}


# =============================================================================
# Image Upload and Download Tools
# =============================================================================


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
        Use the returned name in LoadImage nodes.

    Example:
        result = upload_image("/path/to/reference.png")
        # Use result["name"] in workflow:
        # {"class_type": "LoadImage", "inputs": {"image": result["name"]}}
    """
    client = get_client()
    return client.upload_image(image_path, filename, subfolder, overwrite)


def download_output(
    asset_id: str,
    output_path: str,
) -> dict:
    """
    Download a generated asset to a local file.

    Args:
        asset_id: The asset ID to download.
        output_path: Local path to save the file.

    Returns:
        {"success": True, "path": "/path/to/file", "bytes": 12345}
    """
    from pathlib import Path

    registry = get_registry()
    asset = registry.get_asset(asset_id)

    if asset is None:
        return {"error": "ASSET_NOT_FOUND_OR_EXPIRED", "asset_id": asset_id}

    client = get_client()
    result = client.download_file(
        filename=asset.filename,
        subfolder=asset.subfolder or "",
        folder_type="output",
    )

    if isinstance(result, dict) and "error" in result:
        return result

    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as f:
        f.write(result)

    return {
        "success": True,
        "path": str(output_file.absolute()),
        "bytes": len(result),
        "asset_id": asset_id,
    }
