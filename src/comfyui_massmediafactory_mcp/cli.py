"""
mmf CLI — Single ComfyUI interface for all repos.

Usage:
    mmf run --model flux --type t2i --prompt "a dragon"
    mmf run --template qwen_txt2img --params '{"PROMPT":"test"}'
    mmf execute workflow.json
    mmf wait <prompt_id>
    mmf status [prompt_id]
    mmf progress <prompt_id>
    mmf regenerate <asset_id> --seed 42
    mmf upload image.png
    mmf download <asset_id> output.png
    mmf batch seeds --count 4 < workflow.json
    mmf pipeline i2v --image photo.png --prompt "motion"
    mmf templates list
    mmf models constraints wan
    mmf search-model "anime lora"
    mmf install-model <url> --type lora
    mmf workflow-lib list
    mmf stats
    mmf --url http://host:8188 stats
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional


# Exit codes
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_TIMEOUT = 2
EXIT_VALIDATION = 3
EXIT_PARTIAL = 4
EXIT_CONNECTION = 5
EXIT_NOT_FOUND = 6
EXIT_VRAM = 7


# ─── Error classification ────────────────────────────────────────────

# Error codes that are transient and may succeed on retry
_TRANSIENT_ERRORS = {"VRAM_EXHAUSTED", "TIMEOUT", "CONNECTION_ERROR"}

# Error codes that are permanent and should never be retried
_PERMANENT_ERRORS = {"VALIDATION_ERROR", "NOT_FOUND", "INVALID_PARAMS"}


def _classify_error(result: dict) -> str:
    """Classify an error result into a category for exit code and retry decisions."""
    code = result.get("code", "")
    error_msg = str(result.get("error", "")).lower()

    # Explicit error codes
    if code in _TRANSIENT_ERRORS or code in _PERMANENT_ERRORS:
        return code

    # Heuristic classification from error messages
    if any(kw in error_msg for kw in ("vram", "out of memory", "oom", "cuda out of memory")):
        return "VRAM_EXHAUSTED"
    if any(kw in error_msg for kw in ("timeout", "timed out")):
        return "TIMEOUT"
    if any(kw in error_msg for kw in ("connection", "unreachable", "refused", "connect")):
        return "CONNECTION_ERROR"
    if any(kw in error_msg for kw in ("not found", "no such", "missing")):
        return "NOT_FOUND"
    if any(kw in error_msg for kw in ("invalid", "validation")):
        return "VALIDATION_ERROR"

    return code or "UNKNOWN"


def _exit_code_for_error(error_class: str) -> int:
    """Map an error classification to an exit code."""
    return {
        "VRAM_EXHAUSTED": EXIT_VRAM,
        "TIMEOUT": EXIT_TIMEOUT,
        "CONNECTION_ERROR": EXIT_CONNECTION,
        "NOT_FOUND": EXIT_NOT_FOUND,
        "VALIDATION_ERROR": EXIT_VALIDATION,
        "INVALID_PARAMS": EXIT_VALIDATION,
    }.get(error_class, EXIT_ERROR)


def _output(data: dict, pretty: bool = False) -> None:
    """Write JSON data to stdout (results/data only)."""
    if pretty:
        json.dump(data, sys.stdout, indent=2, default=str)
    else:
        json.dump(data, sys.stdout, default=str)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _msg(text: str) -> None:
    """Write a status/progress message to stderr."""
    sys.stderr.write(text)
    if not text.endswith("\n"):
        sys.stderr.write("\n")
    sys.stderr.flush()


def _error(message: str, code: str = "CLI_ERROR") -> dict:
    return {"error": message, "code": code}


def _is_pretty() -> bool:
    return os.environ.get("MMF_PRETTY", "").lower() in ("1", "true", "yes")


def _parse_json_arg(value: str) -> dict:
    """Parse a JSON string argument, supporting both raw JSON and @file references."""
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.exists():
            _msg(json.dumps(_error(f"File not found: {path}", "INVALID_PARAMS")))
            sys.exit(EXIT_VALIDATION)
        return json.loads(path.read_text())
    return json.loads(value)


def _read_workflow(args) -> Optional[dict]:
    """Read workflow from --workflow file, positional arg, or stdin."""
    path = getattr(args, "workflow_file", None) or getattr(args, "workflow", None)
    if path and path != "-":
        p = Path(path)
        if not p.exists():
            _msg(json.dumps(_error(f"Workflow file not found: {path}", "INVALID_PARAMS")))
            sys.exit(EXIT_VALIDATION)
        return json.loads(p.read_text())
    if not sys.stdin.isatty():
        return json.loads(sys.stdin.read())
    return None


# ─── Retry logic ─────────────────────────────────────────────────────


def _retry_loop(fn, max_retries: int, retry_on: str = "vram,timeout,connection"):
    """Execute fn with retry logic for transient errors.

    Args:
        fn: Callable that returns (exit_code, result_dict). May raise exceptions.
        max_retries: Maximum number of retries (0 = no retry).
        retry_on: Comma-separated list of transient error types to retry on.

    Returns:
        Tuple of (exit_code, result_dict) from the last attempt.
    """
    from . import execution

    retry_types = {t.strip().upper() for t in retry_on.split(",")} if retry_on else set()

    # Map shorthand names to error codes
    _type_map = {
        "VRAM": "VRAM_EXHAUSTED",
        "TIMEOUT": "TIMEOUT",
        "CONNECTION": "CONNECTION_ERROR",
    }
    retry_codes = set()
    for t in retry_types:
        retry_codes.add(_type_map.get(t, t))

    last_code = EXIT_ERROR
    last_result = {}

    for attempt in range(1 + max_retries):
        try:
            last_code, last_result = fn()
        except Exception as e:
            last_result = _error(str(e), "CLI_ERROR")
            last_code = EXIT_ERROR

        # Success -- no retry needed
        if last_code == EXIT_OK:
            return last_code, last_result

        # Classify the error
        error_class = _classify_error(last_result)

        # Permanent errors -- never retry
        if error_class in _PERMANENT_ERRORS:
            return last_code, last_result

        # Not in retry list -- don't retry
        if error_class not in retry_codes:
            return last_code, last_result

        # Last attempt -- don't retry
        if attempt >= max_retries:
            return last_code, last_result

        # Recovery actions before retry
        _msg(f"Attempt {attempt + 1}/{1 + max_retries} failed ({error_class}), retrying...")

        if error_class == "VRAM_EXHAUSTED":
            _msg("Freeing GPU memory before retry...")
            try:
                execution.free_memory(unload_models=True)
            except Exception:
                pass
            time.sleep(5)
        elif error_class == "TIMEOUT":
            _msg("Interrupting timed-out workflow before retry...")
            try:
                execution.interrupt_execution()
            except Exception:
                pass
            time.sleep(3)
        elif error_class == "CONNECTION_ERROR":
            backoff = min(2**attempt, 30)
            _msg(f"Connection error, waiting {backoff}s before retry...")
            time.sleep(backoff)

    return last_code, last_result


# ─── Command handlers ────────────────────────────────────────────────


def cmd_run(args):
    """Generate + execute + wait (one command)."""
    from . import execution, templates as tmpl
    from .workflow_generator import generate_workflow

    pretty = args.pretty or _is_pretty()

    # Input validation
    if args.timeout and (args.timeout < 5 or args.timeout > 3600):
        _output(_error("Timeout must be 5-3600 seconds", "VALIDATION_ERROR"), pretty)
        return EXIT_VALIDATION

    # Template-based run
    if args.template:
        params = _parse_json_arg(args.params) if args.params else {}
        wf = tmpl.inject_parameters(tmpl.load_template(args.template), params)
        if "error" in wf:
            _output(wf, pretty)
            return EXIT_ERROR
    else:
        # Auto-generated run
        if not args.model or not args.type:
            _output(_error("--model and --type required (or use --template)", "INVALID_PARAMS"), pretty)
            return EXIT_VALIDATION
        if not args.prompt:
            _output(_error("--prompt required", "INVALID_PARAMS"), pretty)
            return EXIT_VALIDATION

        extra = {}
        if args.image:
            if not Path(args.image).exists():
                _output(_error(f"Image not found: {args.image}", "NOT_FOUND"), pretty)
                return EXIT_NOT_FOUND
            # Upload image first for i2v
            upload_result = execution.upload_image(args.image)
            if "error" in upload_result:
                _output(upload_result, pretty)
                return EXIT_ERROR
            extra["image_path"] = upload_result.get("name", "")

        result = generate_workflow(
            model=args.model,
            workflow_type=args.type,
            prompt=args.prompt,
            negative_prompt=args.negative or "",
            width=args.width,
            height=args.height,
            frames=args.frames,
            seed=args.seed,
            steps=args.steps,
            cfg=args.cfg,
            guidance=args.guidance,
            **extra,
        )
        if "error" in result:
            _output(result, pretty)
            return EXIT_VALIDATION if result.get("code") == "VALIDATION_ERROR" else EXIT_ERROR
        wf = result["workflow"]

    # --dry-run: generate workflow but don't execute
    if getattr(args, "dry_run", False):
        dry_result = {
            "workflow": wf,
            "model": args.model or args.template,
            "dry_run": True,
        }
        # Try to include hardware optimization info
        try:
            from .optimization import get_optimal_workflow_params

            model_key = args.model or "flux"
            task_key = args.type or "t2i"
            hw_params = get_optimal_workflow_params(model_key, task=task_key)
            if "error" not in hw_params:
                dry_result["hardware"] = hw_params
        except Exception:
            pass
        _output(dry_result, pretty)
        return EXIT_OK

    # Execute + wait (possibly with retry)
    def _execute_and_wait():
        exec_result = execution.execute_workflow(wf)
        if "error" in exec_result:
            error_class = _classify_error(exec_result)
            return _exit_code_for_error(error_class), exec_result

        prompt_id = exec_result["prompt_id"]
        timeout = args.timeout or 600

        # --no-wait: return prompt_id immediately
        if getattr(args, "no_wait", False):
            return EXIT_OK, {"prompt_id": prompt_id, "status": "queued"}

        # Wait for completion
        output = execution.wait_for_completion(prompt_id, timeout_seconds=timeout, workflow=wf)

        if output.get("status") == "timeout":
            return EXIT_TIMEOUT, output

        if output.get("status") == "error":
            error_class = _classify_error(output)
            return _exit_code_for_error(error_class), output

        # Auto-download if --output specified
        if args.output and output.get("outputs"):
            first_asset = output["outputs"][0]
            asset_id = first_asset.get("asset_id")
            if asset_id:
                dl = execution.download_output(asset_id, args.output)
                if "error" in dl:
                    output["download_error"] = dl["error"]
                else:
                    output["downloaded"] = dl

        return EXIT_OK, output

    max_retries = getattr(args, "retry", 0) or 0
    retry_on = getattr(args, "retry_on", "vram,timeout,connection") or "vram,timeout,connection"

    if max_retries > 0:
        exit_code, output = _retry_loop(_execute_and_wait, max_retries, retry_on)
    else:
        exit_code, output = _execute_and_wait()

    _output(output, pretty)
    return exit_code


def cmd_execute(args):
    """Execute a pre-built workflow file."""
    from . import execution

    pretty = args.pretty or _is_pretty()
    wf = _read_workflow(args)
    if wf is None:
        _output(_error("Provide workflow file path or pipe JSON via stdin", "INVALID_PARAMS"), pretty)
        return EXIT_ERROR

    result = execution.execute_workflow(wf)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_wait(args):
    """Block until workflow completes."""
    from . import execution

    pretty = args.pretty or _is_pretty()
    result = execution.wait_for_completion(args.prompt_id, timeout_seconds=args.timeout)

    if result.get("status") == "timeout":
        _output(result, pretty)
        return EXIT_TIMEOUT

    _output(result, pretty)
    return EXIT_OK if result.get("status") == "completed" else EXIT_ERROR


def cmd_upload(args):
    """Upload image for I2V/ControlNet."""
    from . import execution

    pretty = args.pretty or _is_pretty()
    result = execution.upload_image(args.image_path, filename=args.filename, subfolder=args.subfolder or "")
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_download(args):
    """Download asset to local file."""
    from . import execution

    pretty = args.pretty or _is_pretty()
    result = execution.download_output(args.asset_id, args.output_path)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


# ─── Batch commands ──────────────────────────────────────────────────


def _batch_exit_code(result: dict) -> int:
    """Determine exit code for batch operations based on success/failure counts."""
    if "error" in result:
        return EXIT_ERROR

    # Check for partial failures in batch results
    errors = result.get("errors", 0)
    completed = result.get("completed", 0)

    # Also check 'failed' key used by batch queue
    failed = result.get("failed", 0)
    queued = result.get("queued", 0)

    if errors > 0 or failed > 0:
        # Some succeeded, some failed
        if completed > 0 or queued > 0:
            return EXIT_PARTIAL
        # All failed
        return EXIT_ERROR

    # Check results list for mixed outcomes
    results_list = result.get("results", [])
    if results_list:
        has_errors = any("error" in r for r in results_list)
        has_success = any("error" not in r for r in results_list)
        if has_errors and has_success:
            return EXIT_PARTIAL
        if has_errors and not has_success:
            return EXIT_ERROR

    return EXIT_OK


def cmd_batch_seeds(args):
    """Generate seed variations."""
    from . import batch

    pretty = args.pretty or _is_pretty()
    wf = _read_workflow(args)
    if wf is None:
        _output(_error("Provide workflow file or pipe JSON via stdin", "INVALID_PARAMS"), pretty)
        return EXIT_ERROR

    fixed = _parse_json_arg(args.fixed) if args.fixed else {}
    result = batch.execute_seed_variations(
        workflow=wf,
        base_params=fixed,
        num_variations=args.count,
        start_seed=args.start_seed,
        parallel=args.parallel,
        timeout_per_job=args.timeout,
    )
    _output(result, pretty)
    return _batch_exit_code(result)


def cmd_batch_sweep(args):
    """Parameter sweep."""
    from . import batch

    pretty = args.pretty or _is_pretty()
    wf = _read_workflow(args)
    if wf is None:
        _output(_error("Provide workflow file or pipe JSON via stdin", "INVALID_PARAMS"), pretty)
        return EXIT_ERROR

    sweep = _parse_json_arg(args.sweep)
    fixed = _parse_json_arg(args.fixed) if args.fixed else None
    result = batch.execute_sweep(
        workflow=wf,
        sweep_params=sweep,
        fixed_params=fixed,
        parallel=args.parallel,
        timeout_per_job=args.timeout,
    )
    _output(result, pretty)
    return _batch_exit_code(result)


def cmd_batch_queue(args):
    """Queue batch from manifest (fire-and-forget)."""
    from . import execution, templates as tmpl

    pretty = args.pretty or _is_pretty()
    manifest = _parse_json_arg(args.manifest)

    jobs = manifest.get("jobs", manifest) if isinstance(manifest, dict) else manifest
    if not isinstance(jobs, list):
        _output(_error("Manifest must contain a 'jobs' list or be a JSON array", "INVALID_PARAMS"), pretty)
        return EXIT_ERROR

    template_name = args.template
    prompt_ids = []

    for job in jobs:
        params = job.get("params", job)
        if template_name:
            template = tmpl.load_template(template_name)
            if "error" in template:
                _output(template, pretty)
                return EXIT_ERROR
            wf = tmpl.inject_parameters(template, params)
        else:
            wf = params.get("workflow", params)

        result = execution.execute_workflow(wf)
        if "error" in result:
            prompt_ids.append({"error": result["error"], "params": params})
        else:
            prompt_ids.append({"prompt_id": result["prompt_id"], "params": params})

    queued_count = len([p for p in prompt_ids if "prompt_id" in p])
    failed_count = len([p for p in prompt_ids if "error" in p])
    output_data = {"queued": queued_count, "failed": failed_count, "jobs": prompt_ids}
    _output(output_data, pretty)
    return _batch_exit_code(output_data)


def cmd_batch_dir(args):
    """Batch from directory of images."""
    from . import execution, templates as tmpl

    pretty = args.pretty or _is_pretty()
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        _output(_error(f"Input directory not found: {args.input}", "INVALID_PARAMS"), pretty)
        return EXIT_ERROR

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    template = tmpl.load_template(args.template)
    if "error" in template:
        _output(template, pretty)
        return EXIT_ERROR

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in image_exts)

    if not images:
        _output(_error(f"No image files found in {args.input}", "VALIDATION_ERROR"), pretty)
        return EXIT_VALIDATION

    results = []
    for img_path in images:
        # Upload image
        upload = execution.upload_image(str(img_path))
        if "error" in upload:
            results.append({"image": img_path.name, "error": upload["error"]})
            continue

        params = {"IMAGE_PATH": upload["name"]}
        if args.prompt:
            params["PROMPT"] = args.prompt
        if args.seed is not None:
            params["SEED"] = args.seed

        wf = tmpl.inject_parameters(template, params)
        exec_result = execution.execute_workflow(wf)
        if "error" in exec_result:
            results.append({"image": img_path.name, "error": exec_result["error"]})
            continue

        # Wait and optionally download
        output = execution.wait_for_completion(exec_result["prompt_id"], timeout_seconds=args.timeout)
        entry = {"image": img_path.name, "prompt_id": exec_result["prompt_id"], "status": output.get("status")}

        if output_dir and output.get("outputs"):
            asset_id = output["outputs"][0].get("asset_id")
            if asset_id:
                out_path = output_dir / f"{img_path.stem}_output{img_path.suffix}"
                dl = execution.download_output(asset_id, str(out_path))
                entry["output"] = dl.get("path")

        results.append(entry)

    output_data = {"total": len(images), "results": results}
    _output(output_data, pretty)
    return _batch_exit_code(output_data)


# ─── Template commands ───────────────────────────────────────────────


def cmd_templates_list(args):
    """List available templates."""
    from . import templates as tmpl

    pretty = args.pretty or _is_pretty()
    tags = args.tags.split(",") if args.tags else None
    result = tmpl.list_templates(only_installed=args.installed, model_type=args.model, tags=tags)
    _output(result, pretty)
    return EXIT_OK


def cmd_templates_get(args):
    """Get raw template JSON."""
    from . import templates as tmpl

    pretty = args.pretty or _is_pretty()
    result = tmpl.load_template(args.name)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_templates_create(args):
    """Create workflow from template with injected params."""
    from . import templates as tmpl

    pretty = args.pretty or _is_pretty()
    template = tmpl.load_template(args.name)
    if "error" in template:
        _output(template, pretty)
        return EXIT_ERROR

    params = _parse_json_arg(args.params) if args.params else {}
    wf = tmpl.inject_parameters(template, params)
    _output(wf, pretty)
    return EXIT_OK


# ─── Model commands ──────────────────────────────────────────────────


def cmd_models_list(args):
    """List installed models."""
    from . import discovery

    pretty = args.pretty or _is_pretty()
    model_type = args.type or "all"
    result = discovery.list_models(model_type)
    _output(result, pretty)
    return EXIT_OK


def cmd_models_constraints(args):
    """Get model constraints."""
    from .model_registry import get_model_constraints

    pretty = args.pretty or _is_pretty()
    result = get_model_constraints(args.model)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_models_compatibility(args):
    """Model compatibility matrix."""
    from .compatibility import get_compatibility_matrix

    pretty = args.pretty or _is_pretty()
    result = get_compatibility_matrix()
    _output(result, pretty)
    return EXIT_OK


def cmd_models_optimize(args):
    """Hardware-optimized params."""
    from .optimization import get_optimal_workflow_params

    pretty = args.pretty or _is_pretty()
    result = get_optimal_workflow_params(args.model, task=args.task or "i2v")
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


# ─── Quality & System commands ───────────────────────────────────────


def cmd_stats(args):
    """GPU VRAM and system info."""
    from . import execution

    pretty = args.pretty or _is_pretty()
    result = execution.get_system_stats()
    _output(result, pretty)
    return EXIT_OK


def cmd_free(args):
    """Free GPU memory."""
    from . import execution

    pretty = args.pretty or _is_pretty()
    result = execution.free_memory(unload_models=args.unload)
    _output(result, pretty)
    return EXIT_OK


def cmd_interrupt(args):
    """Stop current workflow."""
    from . import execution

    pretty = args.pretty or _is_pretty()
    result = execution.interrupt_execution()
    _output(result, pretty)
    return EXIT_OK


def cmd_enhance(args):
    """LLM-enhanced prompt."""
    from .prompt_enhance import enhance_prompt

    pretty = args.pretty or _is_pretty()
    result = enhance_prompt(
        prompt=args.prompt,
        model=args.model or "flux",
        style=args.style,
        use_llm=not args.no_llm,
    )
    _output(result, pretty)
    return EXIT_OK


def cmd_qa(args):
    """VLM quality check."""
    from .qa import qa_output

    pretty = args.pretty or _is_pretty()
    checks = args.checks.split(",") if args.checks else None
    result = qa_output(asset_id=args.asset_id, prompt=args.prompt, checks=checks)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_profile(args):
    """Per-node execution timing."""
    from .profiling import get_execution_profile

    pretty = args.pretty or _is_pretty()
    result = get_execution_profile(args.prompt_id)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_regenerate(args):
    """Regenerate asset with tweaked params."""
    from . import execution

    pretty = args.pretty or _is_pretty()
    result = execution.regenerate(
        asset_id=args.asset_id,
        prompt=args.prompt,
        seed=args.seed,
        cfg=args.cfg,
        steps=args.steps,
    )

    if "error" in result:
        _output(result, pretty)
        return EXIT_ERROR

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        _output(result, pretty)
        return EXIT_OK

    timeout = args.timeout or 600
    output = execution.wait_for_completion(prompt_id, timeout_seconds=timeout)

    if output.get("status") == "timeout":
        _output(output, pretty)
        return EXIT_TIMEOUT

    if output.get("status") == "error":
        _output(output, pretty)
        return EXIT_ERROR

    _output(output, pretty)
    return EXIT_OK


def cmd_status(args):
    """Workflow/queue status."""
    from . import execution

    pretty = args.pretty or _is_pretty()
    if args.prompt_id:
        result = execution.get_workflow_status(args.prompt_id)
    else:
        result = execution.get_queue_status()

    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_progress(args):
    """Real-time workflow progress."""
    from .websocket_client import get_progress_sync

    pretty = args.pretty or _is_pretty()
    progress = get_progress_sync(args.prompt_id)
    if progress:
        _output(progress, pretty)
        return EXIT_OK
    else:
        _output(_error(f"No progress found for prompt_id: {args.prompt_id}", "NOT_FOUND"), pretty)
        return EXIT_NOT_FOUND


def cmd_search_model(args):
    """Search Civitai for models."""
    from . import models as models_mod

    pretty = args.pretty or _is_pretty()
    result = models_mod.search_civitai(args.query, model_type=args.type, limit=args.limit)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_install_model(args):
    """Download and install a model."""
    from . import models as models_mod

    pretty = args.pretty or _is_pretty()
    result = models_mod.download_model(args.url, model_type=args.type, filename=args.filename)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_workflow_lib(args):
    """Workflow library operations."""
    from . import persistence

    pretty = args.pretty or _is_pretty()
    action = args.action

    workflow_data = None
    if args.workflow:
        workflow_data = _parse_json_arg(args.workflow)

    if action == "save":
        if not args.name or not workflow_data:
            _output(_error("--name and --workflow required for action 'save'", "INVALID_PARAMS"), pretty)
            return EXIT_ERROR
        result = persistence.save_workflow(args.name, workflow_data)
    elif action == "load":
        if not args.name:
            _output(_error("--name required for action 'load'", "INVALID_PARAMS"), pretty)
            return EXIT_ERROR
        result = persistence.load_workflow(args.name)
    elif action == "list":
        result = persistence.list_workflows()
    elif action == "delete":
        if not args.name:
            _output(_error("--name required for action 'delete'", "INVALID_PARAMS"), pretty)
            return EXIT_ERROR
        result = persistence.delete_workflow(args.name)
    else:
        _output(_error(f"Unknown action: {action}. Use save|load|list|delete", "INVALID_PARAMS"), pretty)
        return EXIT_ERROR

    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_validate(args):
    """Validate workflow."""
    from .validation import validate_workflow, validate_and_fix

    pretty = args.pretty or _is_pretty()
    wf = _read_workflow(args)
    if wf is None:
        _output(_error("Provide workflow file or pipe JSON via stdin", "INVALID_PARAMS"), pretty)
        return EXIT_ERROR

    if args.auto_fix:
        result = validate_and_fix(wf)
    else:
        result = validate_workflow(wf)

    _output(result, pretty)
    has_errors = bool(result.get("errors"))
    return EXIT_VALIDATION if has_errors else EXIT_OK


# ─── Asset commands ──────────────────────────────────────────────────


def cmd_assets_list(args):
    """List generated assets."""
    from . import execution

    pretty = args.pretty or _is_pretty()
    result = execution.list_assets(asset_type=args.type, limit=args.limit)
    _output(result, pretty)
    return EXIT_OK


def cmd_assets_metadata(args):
    """Get full asset metadata."""
    from . import execution

    pretty = args.pretty or _is_pretty()
    result = execution.get_asset_metadata(args.asset_id)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_publish(args):
    """Publish asset to web directory."""
    from .publish import publish_asset

    pretty = args.pretty or _is_pretty()
    result = publish_asset(
        asset_id=args.asset_id,
        target_filename=args.filename,
        manifest_key=args.manifest_key,
    )
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


# ─── Pipeline commands ───────────────────────────────────────────────


def cmd_pipeline(args):
    """Run a pre-tested pipeline."""
    from .cli_pipelines import run_pipeline

    pretty = args.pretty or _is_pretty()
    result = run_pipeline(args.pipeline_name, args)
    _output(result, pretty)

    if result.get("status") == "timeout":
        return EXIT_TIMEOUT
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_telestyle(args):
    """TeleStyle transfer (image or video)."""
    from .cli_pipelines import run_telestyle

    pretty = args.pretty or _is_pretty()
    result = run_telestyle(args.mode, args)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


# ─── Parser construction ─────────────────────────────────────────────


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add --pretty and --timeout flags."""
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")


def _add_retry_args(parser: argparse.ArgumentParser) -> None:
    """Add --retry and --retry-on flags."""
    parser.add_argument("--retry", type=int, default=0, help="Number of retries for transient errors (default: 0)")
    parser.add_argument(
        "--retry-on",
        default="vram,timeout,connection",
        help="Comma-separated error types to retry on (default: vram,timeout,connection)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmf",
        description="MassMediaFactory CLI — single ComfyUI interface",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--url", help="ComfyUI server URL (overrides COMFYUI_URL env)")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ──
    p_run = sub.add_parser("run", help="Generate + execute + wait")
    p_run.add_argument("--model", "-m", help="Model: flux, ltx, wan, qwen")
    p_run.add_argument("--type", "-t", dest="type", help="Workflow type: t2i, t2v, i2v")
    p_run.add_argument("--prompt", "-p", help="Generation prompt")
    p_run.add_argument("--negative", help="Negative prompt")
    p_run.add_argument("--image", help="Input image for i2v")
    p_run.add_argument("--template", help="Use template instead of model+type")
    p_run.add_argument("--params", help="JSON params for template (or @file.json)")
    p_run.add_argument("--width", type=int)
    p_run.add_argument("--height", type=int)
    p_run.add_argument("--frames", type=int)
    p_run.add_argument("--seed", type=int)
    p_run.add_argument("--steps", type=int)
    p_run.add_argument("--cfg", type=float)
    p_run.add_argument("--guidance", type=float)
    p_run.add_argument("--timeout", type=int, default=600, help="Wait timeout in seconds")
    p_run.add_argument("--output", "-o", help="Auto-download result to this path")
    p_run.add_argument("--no-wait", action="store_true", default=False, help="Skip waiting, return prompt_id only")
    p_run.add_argument("--dry-run", action="store_true", default=False, help="Generate workflow without executing")
    _add_retry_args(p_run)
    _add_common_args(p_run)
    p_run.set_defaults(func=cmd_run)

    # ── execute ──
    p_exec = sub.add_parser("execute", help="Execute a pre-built workflow")
    p_exec.add_argument("workflow", nargs="?", default="-", help="Workflow JSON file (or - for stdin)")
    _add_common_args(p_exec)
    p_exec.set_defaults(func=cmd_execute)

    # ── wait ──
    p_wait = sub.add_parser("wait", help="Block until workflow completes")
    p_wait.add_argument("prompt_id", help="Prompt ID to wait for")
    p_wait.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    _add_common_args(p_wait)
    p_wait.set_defaults(func=cmd_wait)

    # ── upload ──
    p_upload = sub.add_parser("upload", help="Upload image for I2V/ControlNet")
    p_upload.add_argument("image_path", help="Local image path")
    p_upload.add_argument("--filename", help="Target filename in ComfyUI")
    p_upload.add_argument("--subfolder", help="Subfolder within ComfyUI input dir")
    _add_common_args(p_upload)
    p_upload.set_defaults(func=cmd_upload)

    # ── download ──
    p_dl = sub.add_parser("download", help="Download asset to local file")
    p_dl.add_argument("asset_id", help="Asset ID")
    p_dl.add_argument("output_path", help="Local output path")
    _add_common_args(p_dl)
    p_dl.set_defaults(func=cmd_download)

    # ── batch ──
    p_batch = sub.add_parser("batch", help="Batch operations")
    batch_sub = p_batch.add_subparsers(dest="batch_command")

    # batch seeds
    p_bs = batch_sub.add_parser("seeds", help="Seed variations")
    p_bs.add_argument("workflow", nargs="?", default="-", help="Workflow file (or - for stdin)")
    p_bs.add_argument("--count", type=int, default=4, help="Number of variations")
    p_bs.add_argument("--start-seed", type=int, default=42, help="Starting seed")
    p_bs.add_argument("--fixed", help="Fixed params JSON (or @file.json)")
    p_bs.add_argument("--parallel", type=int, default=1)
    p_bs.add_argument("--timeout", type=int, default=600)
    _add_common_args(p_bs)
    p_bs.set_defaults(func=cmd_batch_seeds)

    # batch sweep
    p_bsw = batch_sub.add_parser("sweep", help="Parameter sweep")
    p_bsw.add_argument("workflow", nargs="?", default="-", help="Workflow file (or - for stdin)")
    p_bsw.add_argument("--sweep", required=True, help="Sweep params JSON")
    p_bsw.add_argument("--fixed", help="Fixed params JSON")
    p_bsw.add_argument("--parallel", type=int, default=1)
    p_bsw.add_argument("--timeout", type=int, default=600)
    _add_common_args(p_bsw)
    p_bsw.set_defaults(func=cmd_batch_sweep)

    # batch queue
    p_bq = batch_sub.add_parser("queue", help="Queue batch from manifest (fire-and-forget)")
    p_bq.add_argument("--manifest", required=True, help="Manifest JSON file or @file.json")
    p_bq.add_argument("--template", help="Template name for all jobs")
    _add_common_args(p_bq)
    p_bq.set_defaults(func=cmd_batch_queue)

    # batch dir
    p_bd = batch_sub.add_parser("dir", help="Batch from directory of images")
    p_bd.add_argument("--input", required=True, help="Input directory of images")
    p_bd.add_argument("--template", required=True, help="Template name")
    p_bd.add_argument("--prompt", help="Prompt for all images")
    p_bd.add_argument("--seed", type=int, help="Seed")
    p_bd.add_argument("--output", help="Output directory")
    p_bd.add_argument("--timeout", type=int, default=600)
    _add_common_args(p_bd)
    p_bd.set_defaults(func=cmd_batch_dir)

    # ── templates ──
    p_tmpl = sub.add_parser("templates", help="Template operations")
    tmpl_sub = p_tmpl.add_subparsers(dest="templates_command")

    p_tl = tmpl_sub.add_parser("list", help="List templates")
    p_tl.add_argument("--installed", action="store_true", help="Only installed models")
    p_tl.add_argument("--model", help="Filter by model type")
    p_tl.add_argument("--tags", help="Filter by tags (comma-separated)")
    _add_common_args(p_tl)
    p_tl.set_defaults(func=cmd_templates_list)

    p_tg = tmpl_sub.add_parser("get", help="Get raw template JSON")
    p_tg.add_argument("name", help="Template name")
    _add_common_args(p_tg)
    p_tg.set_defaults(func=cmd_templates_get)

    p_tc = tmpl_sub.add_parser("create", help="Create workflow from template")
    p_tc.add_argument("name", help="Template name")
    p_tc.add_argument("--params", help="JSON params (or @file.json)")
    _add_common_args(p_tc)
    p_tc.set_defaults(func=cmd_templates_create)

    # ── models ──
    p_models = sub.add_parser("models", help="Model operations")
    models_sub = p_models.add_subparsers(dest="models_command")

    p_ml = models_sub.add_parser("list", help="List installed models")
    p_ml.add_argument("--type", help="Model type: checkpoint, lora, unet, etc.")
    _add_common_args(p_ml)
    p_ml.set_defaults(func=cmd_models_list)

    p_mc = models_sub.add_parser("constraints", help="Model constraints")
    p_mc.add_argument("model", help="Model name: flux, wan, ltx, qwen")
    _add_common_args(p_mc)
    p_mc.set_defaults(func=cmd_models_constraints)

    p_mcompat = models_sub.add_parser("compatibility", help="Compatibility matrix")
    _add_common_args(p_mcompat)
    p_mcompat.set_defaults(func=cmd_models_compatibility)

    p_mo = models_sub.add_parser("optimize", help="Hardware-optimal params")
    p_mo.add_argument("model", help="Model name")
    p_mo.add_argument("--task", help="Task type: i2v, t2v, t2i")
    _add_common_args(p_mo)
    p_mo.set_defaults(func=cmd_models_optimize)

    # ── quality & system ──
    p_stats = sub.add_parser("stats", help="GPU VRAM and system info")
    _add_common_args(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    p_free = sub.add_parser("free", help="Free GPU memory")
    p_free.add_argument("--unload", action="store_true", help="Unload all models")
    _add_common_args(p_free)
    p_free.set_defaults(func=cmd_free)

    p_int = sub.add_parser("interrupt", help="Stop current workflow")
    _add_common_args(p_int)
    p_int.set_defaults(func=cmd_interrupt)

    p_enh = sub.add_parser("enhance", help="LLM-enhanced prompt")
    p_enh.add_argument("--prompt", "-p", required=True, help="Prompt to enhance")
    p_enh.add_argument("--model", "-m", help="Target model (default: flux)")
    p_enh.add_argument("--style", help="Style: cinematic, anime, photorealistic")
    p_enh.add_argument("--no-llm", action="store_true", help="Skip LLM, token injection only")
    _add_common_args(p_enh)
    p_enh.set_defaults(func=cmd_enhance)

    p_qa = sub.add_parser("qa", help="VLM quality check")
    p_qa.add_argument("asset_id", help="Asset ID to check")
    p_qa.add_argument("--prompt", "-p", required=True, help="Original generation prompt")
    p_qa.add_argument("--checks", help="Checks: prompt_match,artifacts,composition,faces,text")
    _add_common_args(p_qa)
    p_qa.set_defaults(func=cmd_qa)

    p_prof = sub.add_parser("profile", help="Per-node execution timing")
    p_prof.add_argument("prompt_id", help="Prompt ID to profile")
    _add_common_args(p_prof)
    p_prof.set_defaults(func=cmd_profile)

    p_val = sub.add_parser("validate", help="Validate workflow")
    p_val.add_argument("workflow", nargs="?", default="-", help="Workflow file (or - for stdin)")
    p_val.add_argument("--auto-fix", action="store_true", help="Auto-correct common issues")
    _add_common_args(p_val)
    p_val.set_defaults(func=cmd_validate)

    # ── regenerate ──
    p_regen = sub.add_parser("regenerate", help="Regenerate asset with tweaked params")
    p_regen.add_argument("asset_id", help="Asset ID to regenerate from")
    p_regen.add_argument("--prompt", "-p", help="New prompt")
    p_regen.add_argument("--seed", type=int, help="New seed")
    p_regen.add_argument("--cfg", type=float, help="New CFG scale")
    p_regen.add_argument("--steps", type=int, help="New step count")
    p_regen.add_argument("--timeout", type=int, default=600, help="Wait timeout in seconds")
    _add_common_args(p_regen)
    p_regen.set_defaults(func=cmd_regenerate)

    # ── status ──
    p_status = sub.add_parser("status", help="Workflow/queue status")
    p_status.add_argument("prompt_id", nargs="?", default=None, help="Prompt ID (omit for queue status)")
    _add_common_args(p_status)
    p_status.set_defaults(func=cmd_status)

    # ── progress ──
    p_prog = sub.add_parser("progress", help="Real-time workflow progress")
    p_prog.add_argument("prompt_id", help="Prompt ID to check progress for")
    _add_common_args(p_prog)
    p_prog.set_defaults(func=cmd_progress)

    # ── search-model ──
    p_sm = sub.add_parser("search-model", help="Search Civitai for models")
    p_sm.add_argument("query", help="Search query")
    p_sm.add_argument("--type", help="Model type: checkpoint, lora, controlnet, etc.")
    p_sm.add_argument("--limit", type=int, default=10, help="Max results (default 10)")
    _add_common_args(p_sm)
    p_sm.set_defaults(func=cmd_search_model)

    # ── install-model ──
    p_im = sub.add_parser("install-model", help="Download and install a model")
    p_im.add_argument("url", help="Download URL (Civitai or HuggingFace)")
    p_im.add_argument("--type", required=True, help="Model type: checkpoint|unet|lora|vae|controlnet|clip")
    p_im.add_argument("--filename", help="Target filename (auto-detected if omitted)")
    _add_common_args(p_im)
    p_im.set_defaults(func=cmd_install_model)

    # ── workflow-lib ──
    p_wl = sub.add_parser("workflow-lib", help="Workflow library operations")
    p_wl.add_argument("action", choices=["save", "load", "list", "delete"], help="Library action")
    p_wl.add_argument("--name", help="Workflow name")
    p_wl.add_argument("--workflow", help="Workflow JSON string or @file.json (for save)")
    _add_common_args(p_wl)
    p_wl.set_defaults(func=cmd_workflow_lib)

    # ── assets ──
    p_assets = sub.add_parser("assets", help="Asset operations")
    assets_sub = p_assets.add_subparsers(dest="assets_command")

    p_al = assets_sub.add_parser("list", help="List generated assets")
    p_al.add_argument("--type", help="Asset type: images, video, audio")
    p_al.add_argument("--limit", type=int, default=20)
    _add_common_args(p_al)
    p_al.set_defaults(func=cmd_assets_list)

    p_am = assets_sub.add_parser("metadata", help="Full asset metadata")
    p_am.add_argument("asset_id", help="Asset ID")
    _add_common_args(p_am)
    p_am.set_defaults(func=cmd_assets_metadata)

    # ── publish ──
    p_pub = sub.add_parser("publish", help="Publish asset to web directory")
    p_pub.add_argument("asset_id", help="Asset ID")
    p_pub.add_argument("--filename", help="Target filename")
    p_pub.add_argument("--manifest-key", help="Manifest key for tracking")
    _add_common_args(p_pub)
    p_pub.set_defaults(func=cmd_publish)

    # ── pipeline ──
    p_pipe = sub.add_parser("pipeline", help="Run pre-tested pipelines")
    p_pipe.add_argument("pipeline_name", help="Pipeline: i2v, upscale, viral-short, t2v-styled, bio-to-video")
    p_pipe.add_argument("--model", "-m", help="Model override")
    p_pipe.add_argument("--prompt", "-p", help="Generation prompt")
    p_pipe.add_argument("--image", help="Input image")
    p_pipe.add_argument("--output", "-o", help="Output path")
    p_pipe.add_argument("--seed", type=int)
    p_pipe.add_argument("--timeout", type=int, default=600)
    # Pipeline-specific args (flexible)
    p_pipe.add_argument("--style-image", help="Style reference image (telestyle)")
    p_pipe.add_argument("--character", help="Character name (viral-short)")
    p_pipe.add_argument("--style", help="Style preset (viral-short)")
    p_pipe.add_argument("--bio-prompt", help="Bio prompt (bio-to-video)")
    p_pipe.add_argument("--shiny-colors", help="Shiny colors JSON (bio-to-video)")
    p_pipe.add_argument("--motion-prompt", help="Motion prompt (bio-to-video)")
    p_pipe.add_argument("--factor", type=float, default=2.0, help="Upscale factor")
    _add_retry_args(p_pipe)
    _add_common_args(p_pipe)
    p_pipe.set_defaults(func=cmd_pipeline)

    # ── telestyle ──
    p_ts = sub.add_parser("telestyle", help="TeleStyle transfer")
    p_ts.add_argument("mode", choices=["image", "video"], help="TeleStyle mode")
    p_ts.add_argument("--content", required=True, help="Content image/video path")
    p_ts.add_argument("--style", required=True, help="Style reference image")
    p_ts.add_argument("--output", "-o", help="Output path")
    p_ts.add_argument("--cfg", type=float, help="CFG override")
    p_ts.add_argument("--steps", type=int, help="Steps override")
    p_ts.add_argument("--seed", type=int)
    p_ts.add_argument("--timeout", type=int, default=600)
    _add_common_args(p_ts)
    p_ts.set_defaults(func=cmd_telestyle)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "url") and args.url:
        os.environ["COMFYUI_URL"] = args.url

    if not args.command:
        parser.print_help()
        sys.exit(EXIT_ERROR)

    func = getattr(args, "func", None)
    if func is None:
        # Subcommand group without subcommand (e.g., "mmf batch" without seeds/sweep/etc.)
        # Find the subparser and print its help
        parser.parse_args([args.command, "--help"])
        sys.exit(EXIT_ERROR)

    try:
        exit_code = func(args)
        sys.exit(exit_code or EXIT_OK)
    except json.JSONDecodeError as e:
        _output(_error(f"Invalid JSON: {e}", "INVALID_PARAMS"), args.pretty or _is_pretty())
        sys.exit(EXIT_VALIDATION)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        _output(_error(str(e), "CLI_ERROR"), args.pretty or _is_pretty())
        sys.exit(EXIT_ERROR)


if __name__ == "__main__":
    main()
