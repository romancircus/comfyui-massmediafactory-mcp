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


# =============================================================================
# Audio-Video Sync Automation
# =============================================================================

def calculate_video_frames(
    audio_duration_seconds: float,
    fps: int = 24,
    buffer_frames: int = 1,
) -> int:
    """
    Calculate required video frames for audio sync.

    Critical formula: frames = (audio_duration * fps) + buffer

    Args:
        audio_duration_seconds: Duration of audio in seconds.
        fps: Target video framerate (default 24).
        buffer_frames: Extra frames to ensure full coverage (default 1).

    Returns:
        Number of frames needed.

    Example:
        frames = calculate_video_frames(5.5, fps=24)
        # Returns 133 (5.5 * 24 + 1)
    """
    return int(audio_duration_seconds * fps) + buffer_frames


def create_tts_to_video_pipeline(
    tts_workflow: Dict[str, Any],
    img2vid_workflow: Dict[str, Any],
    text: str,
    portrait_image: str,
    fps: int = 24,
    voice_params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Create a complete TTS to talking-head video pipeline.

    Automatically handles audio-video frame synchronization.

    Args:
        tts_workflow: TTS workflow template (Qwen3-TTS or Chatterbox).
        img2vid_workflow: Image-to-video workflow (LTX-2 audio-reactive).
        text: Text to speak.
        portrait_image: Path to portrait image.
        fps: Target video framerate.
        voice_params: Optional voice parameters (SPEAKER, INSTRUCT, etc.).
        seed: Random seed.

    Returns:
        Pipeline execution results with synced video.

    Example:
        result = create_tts_to_video_pipeline(
            tts_workflow=load_template("qwen3_tts_custom_voice"),
            img2vid_workflow=load_template("ltx2_audio_reactive"),
            text="Hello, welcome to our channel!",
            portrait_image="character.png",
            voice_params={"SPEAKER": "Ryan", "INSTRUCT": "Excited"}
        )
    """
    voice_params = voice_params or {}

    # Stage 1: Generate TTS audio
    # Stage 2: Calculate frames from audio duration
    # Stage 3: Generate lip-synced video

    def _get_audio_duration(stage_outputs: Dict) -> float:
        """Extract audio duration from TTS stage output."""
        # Look for duration in metadata or calculate from audio file
        for node_outputs in stage_outputs.values():
            if "audio" in node_outputs:
                for item in node_outputs["audio"]:
                    if isinstance(item, dict) and "duration" in item:
                        return item["duration"]
        # Default fallback: estimate from text length
        # ~150 words per minute = 2.5 words per second
        word_count = len(text.split())
        return word_count / 2.5

    stages = [
        {
            "name": "generate_audio",
            "workflow": tts_workflow,
            "output_mapping": {"audio": "AUDIO_PATH"},
        },
        {
            "name": "generate_video",
            "workflow": img2vid_workflow,
            # FRAMES will be calculated dynamically
        },
    ]

    initial_params = {
        "TEXT": text,
        "IMAGE_PATH": portrait_image,
        "PROMPT": "speaking naturally, subtle head movements, lip sync",
        "SEED": seed,
        "FPS": fps,
        **voice_params,
    }

    # Execute with custom frame calculation between stages
    client = get_client()
    results = {
        "stages": [],
        "total_duration": 0,
        "status": "running",
    }

    current_params = initial_params.copy()
    start_time = time.time()

    # Stage 1: TTS
    tts_workflow_prepared = _substitute_params(tts_workflow, current_params)
    queue_result = client.queue_prompt(tts_workflow_prepared)

    if "error" in queue_result:
        return {"status": "error", "error": queue_result["error"]}

    tts_completion = wait_for_completion(queue_result["prompt_id"], timeout=300)

    if tts_completion.get("status") == "error":
        return {"status": "error", "error": tts_completion.get("error")}

    results["stages"].append({
        "name": "generate_audio",
        "status": "completed",
        "outputs": tts_completion.get("outputs", {}),
    })

    # Extract audio path and duration
    audio_path = _extract_output_files(tts_completion.get("outputs", {}), "audio")
    if audio_path:
        current_params["AUDIO_PATH"] = audio_path[0]

    # Calculate frames from audio duration
    audio_duration = _get_audio_duration(tts_completion.get("outputs", {}))
    current_params["FRAMES"] = calculate_video_frames(audio_duration, fps)

    # Stage 2: Video generation with calculated frames
    video_workflow_prepared = _substitute_params(img2vid_workflow, current_params)
    queue_result = client.queue_prompt(video_workflow_prepared)

    if "error" in queue_result:
        return {"status": "error", "error": queue_result["error"]}

    video_completion = wait_for_completion(queue_result["prompt_id"], timeout=600)

    if video_completion.get("status") == "error":
        return {"status": "error", "error": video_completion.get("error")}

    results["stages"].append({
        "name": "generate_video",
        "status": "completed",
        "outputs": video_completion.get("outputs", {}),
        "audio_duration": audio_duration,
        "calculated_frames": current_params["FRAMES"],
    })

    results["total_duration"] = time.time() - start_time
    results["status"] = "completed"
    results["final_outputs"] = video_completion.get("outputs", {})
    results["sync_info"] = {
        "audio_duration_seconds": audio_duration,
        "video_frames": current_params["FRAMES"],
        "fps": fps,
        "video_duration_seconds": current_params["FRAMES"] / fps,
    }

    return results
