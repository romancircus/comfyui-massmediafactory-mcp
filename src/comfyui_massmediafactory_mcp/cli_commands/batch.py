"""Batch commands: seeds, sweep, queue, dir."""

from pathlib import Path

from ..cli_utils import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_PARTIAL,
    EXIT_VALIDATION,
    _add_common_args,
    _error,
    _is_pretty,
    _output,
    _parse_json_arg,
    _read_workflow,
)


def _batch_exit_code(result: dict) -> int:
    """Determine exit code for batch operations based on success/failure counts."""
    if "error" in result:
        return EXIT_ERROR

    errors = result.get("errors", 0)
    completed = result.get("completed", 0)
    failed = result.get("failed", 0)
    queued = result.get("queued", 0)

    if errors > 0 or failed > 0:
        if completed > 0 or queued > 0:
            return EXIT_PARTIAL
        return EXIT_ERROR

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
    from .. import batch

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
    from .. import batch

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
    from .. import execution, templates as tmpl

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
    from .. import execution, templates as tmpl

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


def register_commands(sub, add_common=_add_common_args, **_kwargs):
    """Register batch subcommands."""
    p_batch = sub.add_parser("batch", help="Batch operations")
    batch_sub = p_batch.add_subparsers(dest="batch_command")

    # batch seeds
    p_bs = batch_sub.add_parser("seeds", help="Seed variations")
    p_bs.add_argument("workflow", nargs="?", default="-", help="Workflow file (or - for stdin)")
    p_bs.add_argument("--count", type=int, default=4, help="Number of variations")
    p_bs.add_argument("--start-seed", type=int, default=42, help="Starting seed")
    p_bs.add_argument("--fixed", help="Fixed params JSON (or @file.json)")
    p_bs.add_argument("--parallel", type=int, default=1)
    p_bs.add_argument("--timeout", type=int, default=None, help="Timeout per job (default: 900s)")
    add_common(p_bs)
    p_bs.set_defaults(func=cmd_batch_seeds)

    # batch sweep
    p_bsw = batch_sub.add_parser("sweep", help="Parameter sweep")
    p_bsw.add_argument("workflow", nargs="?", default="-", help="Workflow file (or - for stdin)")
    p_bsw.add_argument("--sweep", required=True, help="Sweep params JSON")
    p_bsw.add_argument("--fixed", help="Fixed params JSON")
    p_bsw.add_argument("--parallel", type=int, default=1)
    p_bsw.add_argument("--timeout", type=int, default=None, help="Timeout per job (default: 900s)")
    add_common(p_bsw)
    p_bsw.set_defaults(func=cmd_batch_sweep)

    # batch queue
    p_bq = batch_sub.add_parser("queue", help="Queue batch from manifest (fire-and-forget)")
    p_bq.add_argument("--manifest", required=True, help="Manifest JSON file or @file.json")
    p_bq.add_argument("--template", help="Template name for all jobs")
    add_common(p_bq)
    p_bq.set_defaults(func=cmd_batch_queue)

    # batch dir
    p_bd = batch_sub.add_parser("dir", help="Batch from directory of images")
    p_bd.add_argument("--input", required=True, help="Input directory of images")
    p_bd.add_argument("--template", required=True, help="Template name")
    p_bd.add_argument("--prompt", help="Prompt for all images")
    p_bd.add_argument("--seed", type=int, help="Seed")
    p_bd.add_argument("--output", help="Output directory")
    p_bd.add_argument("--timeout", type=int, default=None, help="Timeout per job (default: 900s)")
    add_common(p_bd)
    p_bd.set_defaults(func=cmd_batch_dir)
