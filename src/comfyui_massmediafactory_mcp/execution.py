"""
Execution Tools

Tools for executing workflows and managing ComfyUI operations.
"""

import time
from typing import Optional
from .client import get_client


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
                        outputs.append({
                            "type": key,
                            "filename": item.get("filename"),
                            "subfolder": item.get("subfolder", ""),
                            "node_id": node_id,
                        })

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
) -> dict:
    """
    Wait for a workflow to complete and return outputs.

    Args:
        prompt_id: The prompt_id to wait for.
        timeout_seconds: Maximum time to wait (default 600s / 10 minutes).
        poll_interval: Seconds between status checks.

    Returns:
        Final status and output files.
    """
    start = time.time()

    while time.time() - start < timeout_seconds:
        status = get_workflow_status(prompt_id)

        if status["status"] in ["completed", "error"]:
            status["elapsed_seconds"] = round(time.time() - start, 1)
            return status

        time.sleep(poll_interval)

    return {
        "status": "timeout",
        "prompt_id": prompt_id,
        "elapsed_seconds": timeout_seconds,
    }


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
        gpu_info.append({
            "name": device.get("name", "Unknown"),
            "type": device.get("type", "Unknown"),
            "vram_total_gb": round(device.get("vram_total", 0) / (1024**3), 2),
            "vram_free_gb": round(device.get("vram_free", 0) / (1024**3), 2),
            "vram_used_gb": round(
                (device.get("vram_total", 0) - device.get("vram_free", 0)) / (1024**3), 2
            ),
        })

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
