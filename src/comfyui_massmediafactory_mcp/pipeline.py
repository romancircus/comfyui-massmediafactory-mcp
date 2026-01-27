"""
Multi-Stage Pipeline Orchestration

Chain multiple workflows together where outputs from one stage
become inputs to the next stage.
"""

import time
from typing import Dict, List, Any, Optional, Callable
from .client import get_client
from .execution import wait_for_completion


class PipelineStage:
    """Represents a single stage in a pipeline."""

    def __init__(
        self,
        name: str,
        workflow: Dict[str, Any],
        output_mapping: Optional[Dict[str, str]] = None,
        condition: Optional[Callable[[Dict], bool]] = None,
    ):
        """
        Initialize a pipeline stage.

        Args:
            name: Human-readable stage name.
            workflow: Workflow JSON with {{PLACEHOLDER}} fields.
            output_mapping: Map output files to next stage's input parameters.
                           e.g., {"images": "IMAGE_PATH"} means the images output
                           becomes the IMAGE_PATH parameter for the next stage.
            condition: Optional function to determine if stage should run.
                      Receives previous stage results, returns bool.
        """
        self.name = name
        self.workflow = workflow
        self.output_mapping = output_mapping or {}
        self.condition = condition


def execute_pipeline(
    stages: List[Dict[str, Any]],
    initial_params: Dict[str, Any],
    timeout_per_stage: int = 600,
) -> Dict[str, Any]:
    """
    Execute a multi-stage pipeline.

    Each stage's outputs can be automatically passed to the next stage.

    Args:
        stages: List of stage definitions, each with:
                - name: Stage name
                - workflow: Workflow JSON
                - output_mapping: Optional dict mapping output type to next stage's param
                - skip_if: Optional condition dict for skipping
        initial_params: Starting parameters for the first stage.
        timeout_per_stage: Timeout per stage in seconds.

    Returns:
        Pipeline results including all stage outputs.

    Example:
        execute_pipeline(
            stages=[
                {
                    "name": "generate_image",
                    "workflow": txt2img_workflow,
                    "output_mapping": {"images": "IMAGE_PATH"}
                },
                {
                    "name": "upscale",
                    "workflow": upscale_workflow,
                    "output_mapping": {"images": "IMAGE_PATH"}
                },
                {
                    "name": "create_video",
                    "workflow": img2vid_workflow,
                }
            ],
            initial_params={"PROMPT": "a dragon", "SEED": 42}
        )
    """
    client = get_client()
    results = {
        "stages": [],
        "total_duration": 0,
        "status": "running",
    }

    current_params = initial_params.copy()
    start_time = time.time()

    for i, stage_def in enumerate(stages):
        stage_name = stage_def.get("name", f"stage_{i}")
        stage_workflow = stage_def.get("workflow", {})
        output_mapping = stage_def.get("output_mapping", {})
        skip_condition = stage_def.get("skip_if")

        stage_result = {
            "name": stage_name,
            "index": i,
            "status": "pending",
            "parameters": current_params.copy(),
        }

        # Check skip condition
        if skip_condition and _should_skip(skip_condition, results):
            stage_result["status"] = "skipped"
            stage_result["skip_reason"] = skip_condition.get("reason", "Condition not met")
            results["stages"].append(stage_result)
            continue

        # Substitute parameters in workflow
        try:
            prepared_workflow = _substitute_params(stage_workflow, current_params)
        except Exception as e:
            stage_result["status"] = "error"
            stage_result["error"] = f"Parameter substitution failed: {str(e)}"
            results["stages"].append(stage_result)
            results["status"] = "error"
            break

        # Execute the stage
        stage_start = time.time()
        queue_result = client.queue_prompt(prepared_workflow)

        if "error" in queue_result:
            stage_result["status"] = "error"
            stage_result["error"] = queue_result["error"]
            results["stages"].append(stage_result)
            results["status"] = "error"
            break

        prompt_id = queue_result.get("prompt_id")
        stage_result["prompt_id"] = prompt_id

        # Wait for completion
        completion = wait_for_completion(prompt_id, timeout_per_stage)

        if completion.get("status") == "error":
            stage_result["status"] = "error"
            stage_result["error"] = completion.get("error")
            results["stages"].append(stage_result)
            results["status"] = "error"
            break

        # Extract outputs
        outputs = completion.get("outputs", {})
        stage_result["status"] = "completed"
        stage_result["outputs"] = outputs
        stage_result["duration"] = time.time() - stage_start

        # Map outputs to next stage's parameters
        for output_type, param_name in output_mapping.items():
            output_files = _extract_output_files(outputs, output_type)
            if output_files:
                # Use first file for single input, list for batch
                current_params[param_name] = output_files[0] if len(output_files) == 1 else output_files

        results["stages"].append(stage_result)

    results["total_duration"] = time.time() - start_time
    if results["status"] == "running":
        results["status"] = "completed"

    # Collect final outputs
    if results["stages"]:
        last_completed = None
        for stage in reversed(results["stages"]):
            if stage["status"] == "completed":
                last_completed = stage
                break
        if last_completed:
            results["final_outputs"] = last_completed.get("outputs", {})

    return results


def create_image_to_video_pipeline(
    txt2img_workflow: Dict[str, Any],
    img2vid_workflow: Dict[str, Any],
    prompt: str,
    video_prompt: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Convenience function for common image-to-video pipeline.

    Args:
        txt2img_workflow: Text-to-image workflow template.
        img2vid_workflow: Image-to-video workflow template.
        prompt: Image generation prompt.
        video_prompt: Optional video motion prompt (defaults to image prompt).
        seed: Random seed.

    Returns:
        Pipeline execution results.
    """
    stages = [
        {
            "name": "generate_image",
            "workflow": txt2img_workflow,
            "output_mapping": {"images": "IMAGE_PATH"},
        },
        {
            "name": "animate_to_video",
            "workflow": img2vid_workflow,
        },
    ]

    initial_params = {
        "PROMPT": prompt,
        "VIDEO_PROMPT": video_prompt or prompt,
        "SEED": seed,
    }

    return execute_pipeline(stages, initial_params)


def create_upscale_pipeline(
    base_workflow: Dict[str, Any],
    upscale_workflow: Dict[str, Any],
    prompt: str,
    upscale_factor: float = 2.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Convenience function for generate + upscale pipeline.

    Args:
        base_workflow: Base image generation workflow.
        upscale_workflow: Upscaling workflow.
        prompt: Generation prompt.
        upscale_factor: Upscale multiplier.
        seed: Random seed.

    Returns:
        Pipeline execution results.
    """
    stages = [
        {
            "name": "generate_base",
            "workflow": base_workflow,
            "output_mapping": {"images": "IMAGE_PATH"},
        },
        {
            "name": "upscale",
            "workflow": upscale_workflow,
        },
    ]

    initial_params = {
        "PROMPT": prompt,
        "SEED": seed,
        "UPSCALE_FACTOR": upscale_factor,
    }

    return execute_pipeline(stages, initial_params)


def _substitute_params(workflow: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Substitute parameters in workflow."""
    import json

    workflow_str = json.dumps(workflow)

    for param_name, param_value in params.items():
        placeholder = f"{{{{{param_name}}}}}"

        if isinstance(param_value, (int, float)):
            workflow_str = workflow_str.replace(f'"{placeholder}"', str(param_value))
            workflow_str = workflow_str.replace(placeholder, str(param_value))
        elif isinstance(param_value, bool):
            workflow_str = workflow_str.replace(f'"{placeholder}"', str(param_value).lower())
        else:
            escaped = json.dumps(str(param_value))[1:-1]
            workflow_str = workflow_str.replace(placeholder, escaped)

    return json.loads(workflow_str)


def _should_skip(condition: Dict[str, Any], results: Dict[str, Any]) -> bool:
    """Check if a stage should be skipped based on condition."""
    if "previous_status" in condition:
        if results["stages"]:
            last_stage = results["stages"][-1]
            return last_stage.get("status") != condition["previous_status"]
    return False


def _extract_output_files(outputs: Dict[str, Any], output_type: str) -> List[str]:
    """Extract output files of a specific type from workflow outputs."""
    files = []

    for node_outputs in outputs.values():
        if output_type in node_outputs:
            for item in node_outputs[output_type]:
                if isinstance(item, dict):
                    filename = item.get("filename", "")
                    subfolder = item.get("subfolder", "")
                    if filename:
                        path = f"{subfolder}/{filename}" if subfolder else filename
                        files.append(path)
                elif isinstance(item, str):
                    files.append(item)

    return files
