"""
Batch Execution

Execute multiple workflows or parameter variations efficiently.
"""

import copy
import time
import uuid
from typing import Dict, List, Any, Optional
from .client import get_client
from .execution import wait_for_completion
from .mcp_utils import log_structured, get_correlation_id


def execute_batch(
    workflow: Dict[str, Any],
    parameter_sets: List[Dict[str, Any]],
    parallel: int = 1,
    timeout_per_job: int = 600,
) -> Dict[str, Any]:
    """
    Execute the same workflow with multiple parameter sets.

    Args:
        workflow: Base workflow JSON with {{PLACEHOLDER}} fields.
        parameter_sets: List of parameter dicts to substitute.
        parallel: Max concurrent executions (default 1 for sequential).
        timeout_per_job: Timeout per job in seconds.

    Returns:
        Results for each parameter set.
    """
    cid = get_correlation_id() or str(uuid.uuid4())[:8]
    batch_id = str(uuid.uuid4())[:8]

    log_structured("info", "batch_started",
        batch_id=batch_id,
        correlation_id=cid,
        total_jobs=len(parameter_sets),
        mode="batch",
        parallel=parallel,
    )

    client = get_client()
    results = []
    pending_jobs = []

    for i, params in enumerate(parameter_sets):
        # Create workflow copy with substituted parameters
        job_workflow = _substitute_parameters(workflow, params)

        # Queue the job
        result = client.queue_prompt(job_workflow)

        if "error" in result:
            results.append({
                "index": i,
                "parameters": params,
                "status": "error",
                "error": result["error"],
            })
            continue

        prompt_id = result.get("prompt_id")
        pending_jobs.append({
            "index": i,
            "prompt_id": prompt_id,
            "parameters": params,
            "queued_at": time.time(),
        })

        # If we've hit parallel limit, wait for some to complete
        if parallel > 0 and len(pending_jobs) >= parallel:
            completed = _wait_for_jobs(client, pending_jobs, timeout_per_job, 1)
            results.extend(completed)
            pending_jobs = [j for j in pending_jobs if j["prompt_id"] not in
                           [c["prompt_id"] for c in completed]]

    # Wait for remaining jobs
    if pending_jobs:
        completed = _wait_for_jobs(client, pending_jobs, timeout_per_job, len(pending_jobs))
        results.extend(completed)

    # Sort by original index
    results.sort(key=lambda x: x["index"])

    completed_count = sum(1 for r in results if r["status"] == "completed")
    errored_count = len(parameter_sets) - completed_count

    log_structured("info", "batch_completed",
        batch_id=batch_id,
        correlation_id=cid,
        total_jobs=len(parameter_sets),
        completed=completed_count,
        errors=errored_count,
    )

    return {
        "total_jobs": len(parameter_sets),
        "completed": completed_count,
        "errors": errored_count,
        "results": results,
    }


def execute_sweep(
    workflow: Dict[str, Any],
    sweep_params: Dict[str, List[Any]],
    fixed_params: Optional[Dict[str, Any]] = None,
    parallel: int = 1,
    timeout_per_job: int = 600,
) -> Dict[str, Any]:
    """
    Execute a parameter sweep over specified parameters.

    Args:
        workflow: Base workflow with {{PLACEHOLDER}} fields.
        sweep_params: Dict mapping parameter names to lists of values to try.
                      e.g., {"SEED": [1, 2, 3], "CFG": [3.0, 3.5, 4.0]}
        fixed_params: Optional fixed parameters to include in all runs.
        parallel: Max concurrent executions.
        timeout_per_job: Timeout per job.

    Returns:
        Results organized by parameter combination.
    """
    batch_id = str(uuid.uuid4())[:8]
    cid = get_correlation_id()

    # Generate all combinations
    param_combinations = _generate_combinations(sweep_params)
    
    # Add fixed params to each combination
    if fixed_params:
        param_combinations = [
            {**fixed_params, **combo} for combo in param_combinations
        ]

    results = execute_batch(workflow, param_combinations, parallel, timeout_per_job)

    log_structured("info", "sweep_completed",
        batch_id=batch_id,
        correlation_id=cid,
        total_jobs=len(param_combinations),
        completed=results.get("completed", 0),
        errors=results.get("errors", 0),
    )

    return results


def execute_seed_variations(
    workflow: Dict[str, Any],
    base_params: Dict[str, Any],
    num_variations: int = 4,
    start_seed: int = 42,
    parallel: int = 2,
    timeout_per_job: int = 600,
) -> Dict[str, Any]:
    """
    Generate multiple variations of the same workflow with different seeds.

    Common use case for exploring different outputs from the same prompt.

    Args:
        workflow: Base workflow with {{SEED}} placeholder.
        base_params: Parameters to use (except SEED).
        num_variations: Number of seed variations to generate.
        start_seed: Starting seed value.
        parallel: Max concurrent executions.
        timeout_per_job: Timeout per job.

    Returns:
        Results for each seed variation.
    """
    batch_id = str(uuid.uuid4())[:8]
    cid = get_correlation_id()

    log_structured("info", "seed_variations_started",
        batch_id=batch_id,
        correlation_id=cid,
        num_variations=num_variations,
        start_seed=start_seed,
    )

    param_sets = []
    for i in range(num_variations):
        params = {**base_params, "SEED": start_seed + i}
        param_sets.append(params)

    results = execute_batch(workflow, param_sets, parallel, timeout_per_job)

    log_structured("info", "seed_variations_completed",
        batch_id=batch_id,
        correlation_id=cid,
        num_variations=num_variations,
        completed=results.get("completed", 0),
        errors=results.get("errors", 0),
    )

    return results


def _substitute_parameters(workflow: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Substitute {{PARAM}} placeholders in workflow with actual values."""
    import json

    # Convert to JSON string for easy replacement
    workflow_str = json.dumps(workflow)

    for param_name, param_value in params.items():
        placeholder = f"{{{{{param_name}}}}}"

        if isinstance(param_value, (int, float)):
            # Numeric: remove quotes around placeholder
            workflow_str = workflow_str.replace(f'"{placeholder}"', str(param_value))
            workflow_str = workflow_str.replace(placeholder, str(param_value))
        elif isinstance(param_value, bool):
            workflow_str = workflow_str.replace(f'"{placeholder}"', str(param_value).lower())
            workflow_str = workflow_str.replace(placeholder, str(param_value).lower())
        else:
            # String: escape for JSON
            escaped = json.dumps(str(param_value))[1:-1]
            workflow_str = workflow_str.replace(placeholder, escaped)

    return json.loads(workflow_str)


def _wait_for_jobs(
    client,
    jobs: List[Dict[str, Any]],
    timeout: int,
    min_complete: int = 1,
) -> List[Dict[str, Any]]:
    """Wait for jobs to complete."""
    completed = []
    start_time = time.time()

    while len(completed) < min_complete and (time.time() - start_time) < timeout:
        for job in jobs:
            if job["prompt_id"] in [c["prompt_id"] for c in completed]:
                continue

            history = client.get_history(job["prompt_id"])
            if job["prompt_id"] in history:
                job_history = history[job["prompt_id"]]

                if job_history.get("status", {}).get("completed"):
                    outputs = job_history.get("outputs", {})
                    output_files = []

                    for node_output in outputs.values():
                        if "images" in node_output:
                            output_files.extend(node_output["images"])
                        if "videos" in node_output:
                            output_files.extend(node_output["videos"])

                    completed.append({
                        "index": job["index"],
                        "prompt_id": job["prompt_id"],
                        "parameters": job["parameters"],
                        "status": "completed",
                        "outputs": output_files,
                        "duration": time.time() - job["queued_at"],
                    })

                elif job_history.get("status", {}).get("status_str") == "error":
                    completed.append({
                        "index": job["index"],
                        "prompt_id": job["prompt_id"],
                        "parameters": job["parameters"],
                        "status": "error",
                        "error": job_history.get("status", {}).get("messages", []),
                    })

        if len(completed) < min_complete:
            time.sleep(1)

    return completed


def _generate_combinations(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameter values."""
    import itertools

    keys = list(params.keys())
    values = [params[k] for k in keys]

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations
