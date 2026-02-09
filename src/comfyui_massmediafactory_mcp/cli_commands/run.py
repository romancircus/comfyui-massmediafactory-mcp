"""Run, execute, wait, upload, download commands."""

import os
from pathlib import Path

from ..cli_utils import (
    EXIT_ERROR,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_TIMEOUT,
    EXIT_VALIDATION,
    TIMEOUTS,
    _add_common_args,
    _add_retry_args,
    _classify_error,
    _error,
    _exit_code_for_error,
    _is_pretty,
    _output,
    _parse_json_arg,
    _read_workflow,
    _retry_loop,
)


def cmd_run(args):
    """Generate + execute + wait (one command)."""
    from .. import execution, templates as tmpl
    from ..workflow_generator import generate_workflow

    pretty = args.pretty or _is_pretty()

    if args.timeout and (args.timeout < 5 or args.timeout > 3600):
        _output(_error("Timeout must be 5-3600 seconds", "VALIDATION_ERROR"), pretty)
        return EXIT_VALIDATION

    # Template-based run
    if args.template:
        params = _parse_json_arg(args.params) if args.params else {}
        raw_template = tmpl.load_template(args.template)
        skip_hw = raw_template.get("_meta", {}).get("skip_hardware_optimization", False)
        wf = tmpl.inject_parameters(raw_template, params)
        if "error" in wf:
            _output(wf, pretty)
            return EXIT_ERROR
        if not skip_hw:
            try:
                from ..optimization import get_optimal_workflow_params, apply_hardware_overrides

                hw = get_optimal_workflow_params(args.template, "generic")
                overrides = hw.get("workflow_overrides", {})
                if overrides:
                    apply_hardware_overrides(wf, overrides)
            except Exception:
                pass
    else:
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
            fps=args.fps,
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

    # --dry-run
    if getattr(args, "dry_run", False):
        dry_result = {
            "workflow": wf,
            "model": args.model or args.template,
            "dry_run": True,
        }
        try:
            from ..optimization import get_optimal_workflow_params

            model_key = args.model or "flux"
            task_key = args.type or "t2i"
            hw_params = get_optimal_workflow_params(model_key, task=task_key)
            if "error" not in hw_params:
                dry_result["hardware"] = hw_params
        except Exception:
            pass
        _output(dry_result, pretty)
        return EXIT_OK

    # Execute + wait
    def _execute_and_wait():
        exec_result = execution.execute_workflow(wf)
        if "error" in exec_result:
            error_class = _classify_error(exec_result)
            return _exit_code_for_error(error_class), exec_result

        prompt_id = exec_result["prompt_id"]
        is_video = args.type in ("t2v", "i2v") if args.type else False
        default_timeout = TIMEOUTS["batch"] if is_video else TIMEOUTS["run"]
        timeout = args.timeout or default_timeout

        if getattr(args, "no_wait", False):
            return EXIT_OK, {"prompt_id": prompt_id, "status": "queued"}

        output = execution.wait_for_completion(prompt_id, timeout_seconds=timeout, workflow=wf)

        if output.get("status") == "timeout":
            return EXIT_TIMEOUT, output

        if output.get("status") == "error":
            error_class = _classify_error(output)
            return _exit_code_for_error(error_class), output

        if args.output and output.get("outputs"):
            first_asset = output["outputs"][0]
            asset_id = first_asset.get("asset_id")
            if asset_id:
                dl = execution.download_output(asset_id, args.output)
                if "error" in dl:
                    output["download_error"] = dl["error"]
                else:
                    output["downloaded"] = dl

        if args.output and os.path.exists(args.output):
            file_size = os.path.getsize(args.output)
            is_image = False
            if args.template:
                is_image = any(x in args.template for x in ["txt2img", "txt2vid", "t2i", "img2img"])
            elif args.model:
                is_image = args.type == "t2i" or args.model == "flux" or args.model == "qwen"

            if is_image and file_size < 100000:
                return EXIT_VALIDATION, _error(
                    f"Corrupted output: image too small ({file_size} bytes, minimum 100KB required)", "VALIDATION_ERROR"
                )
            elif not is_image and file_size < 200000:
                return EXIT_VALIDATION, _error(
                    f"Corrupted output: video too small ({file_size} bytes, minimum 200KB required)", "VALIDATION_ERROR"
                )

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
    from .. import execution

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
    from .. import execution

    pretty = args.pretty or _is_pretty()
    result = execution.wait_for_completion(args.prompt_id, timeout_seconds=args.timeout)

    if result.get("status") == "timeout":
        _output(result, pretty)
        return EXIT_TIMEOUT

    _output(result, pretty)
    return EXIT_OK if result.get("status") == "completed" else EXIT_ERROR


def cmd_upload(args):
    """Upload image for I2V/ControlNet."""
    from .. import execution

    pretty = args.pretty or _is_pretty()
    result = execution.upload_image(args.image_path, filename=args.filename, subfolder=args.subfolder or "")
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_download(args):
    """Download asset to local file."""
    from .. import execution

    pretty = args.pretty or _is_pretty()
    result = execution.download_output(args.asset_id, args.output_path)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def register_commands(sub, add_common=_add_common_args, add_retry=_add_retry_args):
    """Register run, execute, wait, upload, download subcommands."""
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
    p_run.add_argument("--fps", type=int, help="Frames per second for video output")
    p_run.add_argument("--seed", type=int)
    p_run.add_argument("--steps", type=int)
    p_run.add_argument("--cfg", type=float)
    p_run.add_argument("--guidance", type=float)
    p_run.add_argument(
        "--timeout", type=int, default=None, help="Wait timeout in seconds (default: 660s for images, 900s for video)"
    )
    p_run.add_argument("--output", "-o", help="Auto-download result to this path")
    p_run.add_argument("--no-wait", action="store_true", default=False, help="Skip waiting, return prompt_id only")
    p_run.add_argument("--dry-run", action="store_true", default=False, help="Generate workflow without executing")
    add_retry(p_run)
    add_common(p_run)
    p_run.set_defaults(func=cmd_run)

    # ── execute ──
    p_exec = sub.add_parser("execute", help="Execute a pre-built workflow")
    p_exec.add_argument("workflow", nargs="?", default="-", help="Workflow JSON file (or - for stdin)")
    add_common(p_exec)
    p_exec.set_defaults(func=cmd_execute)

    # ── wait ──
    p_wait = sub.add_parser("wait", help="Block until workflow completes")
    p_wait.add_argument("prompt_id", help="Prompt ID to wait for")
    p_wait.add_argument("--timeout", type=int, default=None, help="Timeout in seconds (default: 660s)")
    add_common(p_wait)
    p_wait.set_defaults(func=cmd_wait)

    # ── upload ──
    p_upload = sub.add_parser("upload", help="Upload image for I2V/ControlNet")
    p_upload.add_argument("image_path", help="Local image path")
    p_upload.add_argument("--filename", help="Target filename in ComfyUI")
    p_upload.add_argument("--subfolder", help="Subfolder within ComfyUI input dir")
    add_common(p_upload)
    p_upload.set_defaults(func=cmd_upload)

    # ── download ──
    p_dl = sub.add_parser("download", help="Download asset to local file")
    p_dl.add_argument("asset_id", help="Asset ID")
    p_dl.add_argument("output_path", help="Local output path")
    add_common(p_dl)
    p_dl.set_defaults(func=cmd_download)
