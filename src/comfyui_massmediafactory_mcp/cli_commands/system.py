"""System commands: stats, free, interrupt, enhance, qa, profile, validate, status, progress, regenerate, sota, diff."""

from ..cli_utils import (
    EXIT_ERROR,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_TIMEOUT,
    EXIT_VALIDATION,
    TIMEOUTS,
    _add_common_args,
    _error,
    _is_pretty,
    _output,
    _parse_json_arg,
    _read_workflow,
)


def cmd_stats(args):
    """GPU VRAM and system info."""
    from .. import execution

    pretty = args.pretty or _is_pretty()
    result = execution.get_system_stats()
    _output(result, pretty)
    return EXIT_OK


def cmd_free(args):
    """Free GPU memory."""
    from .. import execution

    pretty = args.pretty or _is_pretty()
    result = execution.free_memory(unload_models=args.unload)
    _output(result, pretty)
    return EXIT_OK


def cmd_interrupt(args):
    """Stop current workflow."""
    from .. import execution

    pretty = args.pretty or _is_pretty()
    result = execution.interrupt_execution()
    _output(result, pretty)
    return EXIT_OK


def cmd_enhance(args):
    """LLM-enhanced prompt."""
    from ..prompt_enhance import enhance_prompt

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
    from ..qa import qa_output

    pretty = args.pretty or _is_pretty()
    checks = args.checks.split(",") if args.checks else None
    result = qa_output(asset_id=args.asset_id, prompt=args.prompt, checks=checks)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_profile(args):
    """Per-node execution timing."""
    from ..profiling import get_execution_profile

    pretty = args.pretty or _is_pretty()
    result = get_execution_profile(args.prompt_id)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_validate(args):
    """Validate workflow."""
    from ..validation import validate_workflow, validate_and_fix

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


def cmd_status(args):
    """Workflow/queue status."""
    from .. import execution

    pretty = args.pretty or _is_pretty()
    if args.prompt_id:
        result = execution.get_workflow_status(args.prompt_id)
    else:
        result = execution.get_queue_status()

    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_progress(args):
    """Real-time workflow progress."""
    from ..websocket_client import get_progress_sync

    pretty = args.pretty or _is_pretty()
    progress = get_progress_sync(args.prompt_id)
    if progress:
        _output(progress, pretty)
        return EXIT_OK
    else:
        _output(_error(f"No progress found for prompt_id: {args.prompt_id}", "NOT_FOUND"), pretty)
        return EXIT_NOT_FOUND


def cmd_regenerate(args):
    """Regenerate asset with tweaked params."""
    from .. import execution

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

    timeout = args.timeout or TIMEOUTS["run"]
    output = execution.wait_for_completion(prompt_id, timeout_seconds=timeout)

    if output.get("status") == "timeout":
        _output(output, pretty)
        return EXIT_TIMEOUT

    if output.get("status") == "error":
        _output(output, pretty)
        return EXIT_ERROR

    _output(output, pretty)
    return EXIT_OK


def cmd_sota(args):
    """SOTA model queries."""
    from .. import sota

    pretty = args.pretty or _is_pretty()
    mode = args.mode

    if mode == "category":
        if not args.category:
            _output(_error("--category required for mode 'category'", "INVALID_PARAMS"), pretty)
            return EXIT_ERROR
        result = sota.get_sota_for_category(args.category)
    elif mode == "recommend":
        if not args.task:
            _output(_error("--task required for mode 'recommend'", "INVALID_PARAMS"), pretty)
            return EXIT_ERROR
        result = sota.recommend_model_for_task(args.task, args.vram)
    elif mode == "check":
        if not args.model_name:
            _output(_error("--model-name required for mode 'check'", "INVALID_PARAMS"), pretty)
            return EXIT_ERROR
        result = sota.check_model_is_sota(args.model_name)
    elif mode == "settings":
        if not args.model_name:
            _output(_error("--model-name required for mode 'settings'", "INVALID_PARAMS"), pretty)
            return EXIT_ERROR
        result = sota.get_optimal_settings(args.model_name)
    elif mode == "installed":
        result = sota.get_available_sota_models()
    else:
        _output(
            _error(f"Unknown mode: {mode}. Use: category|recommend|check|settings|installed", "INVALID_PARAMS"), pretty
        )
        return EXIT_ERROR

    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_diff(args):
    """Compare two workflows."""
    from ..workflow_diff import diff_workflows

    pretty = args.pretty or _is_pretty()
    wf_a = _parse_json_arg(args.workflow_a)
    wf_b = _parse_json_arg(args.workflow_b)
    result = diff_workflows(wf_a, wf_b)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_search_workflow(args):
    """Search CivitAI for ComfyUI workflows."""
    from ..civitai import search_workflows

    pretty = args.pretty or _is_pretty()
    result = search_workflows(
        query=args.query,
        limit=args.limit,
        sort=args.sort,
        period=args.period,
    )
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_import_workflow(args):
    """Import a CivitAI workflow as a local template."""
    import json as _json
    from ..civitai import fetch_workflow_from_url, convert_to_template
    from pathlib import Path

    pretty = args.pretty or _is_pretty()

    # Fetch the workflow
    fetch_result = fetch_workflow_from_url(args.url)
    if "error" in fetch_result:
        _output(fetch_result, pretty)
        return EXIT_ERROR

    workflow = fetch_result["workflow"]

    # Convert to template
    result = convert_to_template(
        workflow=workflow,
        name=args.name,
        model=args.model or "unknown",
        task=args.task or "unknown",
        description=args.description or "",
    )

    template = result["template"]

    # Save to templates directory if --save flag
    if args.save:
        # Validate template name (prevent path traversal)
        import re as _re

        if not _re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$", args.name):
            _output(
                _error("Template name must be alphanumeric/underscore/dash, 1-64 chars, no path separators", "INVALID_PARAMS"),
                pretty,
            )
            return EXIT_ERROR

        templates_dir = Path(__file__).parent.parent / "templates"
        out_path = (templates_dir / f"{args.name}.json").resolve()
        if not out_path.is_relative_to(templates_dir.resolve()):
            _output(_error("Path traversal detected in template name", "INVALID_PARAMS"), pretty)
            return EXIT_ERROR
        out_path.write_text(_json.dumps(template, indent=2))
        result["saved_to"] = str(out_path)

    _output(result, pretty)
    return EXIT_OK


def cmd_node_registry(args):
    """Show dynamic node registry stats or dump the full registry."""
    from ..node_registry import get_node_output_types, get_registry_stats

    pretty = args.pretty or _is_pretty()

    if args.dump:
        output_types = get_node_output_types(force_refresh=args.refresh)
        result = {"nodes": output_types, "count": len(output_types)}
    else:
        if args.refresh:
            get_node_output_types(force_refresh=True)
        result = get_registry_stats()

    _output(result, pretty)
    return EXIT_OK


def register_commands(sub, add_common=_add_common_args, **_kwargs):
    """Register system subcommands."""
    p_stats = sub.add_parser("stats", help="GPU VRAM and system info")
    add_common(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    p_free = sub.add_parser("free", help="Free GPU memory")
    p_free.add_argument("--unload", action="store_true", help="Unload all models")
    add_common(p_free)
    p_free.set_defaults(func=cmd_free)

    p_int = sub.add_parser("interrupt", help="Stop current workflow")
    add_common(p_int)
    p_int.set_defaults(func=cmd_interrupt)

    p_enh = sub.add_parser("enhance", help="LLM-enhanced prompt")
    p_enh.add_argument("--prompt", "-p", required=True, help="Prompt to enhance")
    p_enh.add_argument("--model", "-m", help="Target model (default: flux)")
    p_enh.add_argument("--style", help="Style: cinematic, anime, photorealistic")
    p_enh.add_argument("--no-llm", action="store_true", help="Skip LLM, token injection only")
    add_common(p_enh)
    p_enh.set_defaults(func=cmd_enhance)

    p_qa = sub.add_parser("qa", help="VLM quality check")
    p_qa.add_argument("asset_id", help="Asset ID to check")
    p_qa.add_argument("--prompt", "-p", required=True, help="Original generation prompt")
    p_qa.add_argument("--checks", help="Checks: prompt_match,artifacts,composition,faces,text")
    add_common(p_qa)
    p_qa.set_defaults(func=cmd_qa)

    p_prof = sub.add_parser("profile", help="Per-node execution timing")
    p_prof.add_argument("prompt_id", help="Prompt ID to profile")
    add_common(p_prof)
    p_prof.set_defaults(func=cmd_profile)

    p_val = sub.add_parser("validate", help="Validate workflow")
    p_val.add_argument("workflow", nargs="?", default="-", help="Workflow file (or - for stdin)")
    p_val.add_argument("--auto-fix", action="store_true", help="Auto-correct common issues")
    add_common(p_val)
    p_val.set_defaults(func=cmd_validate)

    p_regen = sub.add_parser("regenerate", help="Regenerate asset with tweaked params")
    p_regen.add_argument("asset_id", help="Asset ID to regenerate from")
    p_regen.add_argument("--prompt", "-p", help="New prompt")
    p_regen.add_argument("--seed", type=int, help="New seed")
    p_regen.add_argument("--cfg", type=float, help="New CFG scale")
    p_regen.add_argument("--steps", type=int, help="New step count")
    p_regen.add_argument("--timeout", type=int, default=None, help="Wait timeout in seconds (default: 660s)")
    add_common(p_regen)
    p_regen.set_defaults(func=cmd_regenerate)

    p_status = sub.add_parser("status", help="Workflow/queue status")
    p_status.add_argument("prompt_id", nargs="?", default=None, help="Prompt ID (omit for queue status)")
    add_common(p_status)
    p_status.set_defaults(func=cmd_status)

    p_prog = sub.add_parser("progress", help="Real-time workflow progress")
    p_prog.add_argument("prompt_id", help="Prompt ID to check progress for")
    add_common(p_prog)
    p_prog.set_defaults(func=cmd_progress)

    p_sota = sub.add_parser("sota", help="SOTA model queries")
    p_sota.add_argument("mode", choices=["category", "recommend", "check", "settings", "installed"], help="Query mode")
    p_sota.add_argument("--category", help="Model category (for mode=category)")
    p_sota.add_argument("--task", help="Task type (for mode=recommend)")
    p_sota.add_argument("--model-name", help="Model name (for mode=check|settings)")
    p_sota.add_argument("--vram", type=float, help="Available VRAM in GB (for mode=recommend)")
    add_common(p_sota)
    p_sota.set_defaults(func=cmd_sota)

    p_diff = sub.add_parser("diff", help="Compare two workflows")
    p_diff.add_argument("workflow_a", help="First workflow JSON or @file.json")
    p_diff.add_argument("workflow_b", help="Second workflow JSON or @file.json")
    add_common(p_diff)
    p_diff.set_defaults(func=cmd_diff)

    p_sw = sub.add_parser("search-workflow", help="Search CivitAI for ComfyUI workflows")
    p_sw.add_argument("query", help="Search query (e.g., 'flux portrait')")
    p_sw.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    p_sw.add_argument("--sort", default="Most Reactions", help="Sort: 'Most Reactions', 'Newest', 'Most Comments'")
    p_sw.add_argument("--period", default="Month", help="Period: Day, Week, Month, Year, AllTime")
    add_common(p_sw)
    p_sw.set_defaults(func=cmd_search_workflow)

    p_iw = sub.add_parser("import-workflow", help="Import CivitAI workflow as local template")
    p_iw.add_argument("url", help="CivitAI URL or direct workflow JSON URL")
    p_iw.add_argument("--name", required=True, help="Template name (e.g., flux_portrait_custom)")
    p_iw.add_argument("--model", help="Model type (auto-detected if omitted)")
    p_iw.add_argument("--task", help="Task type (auto-detected if omitted)")
    p_iw.add_argument("--description", help="Template description")
    p_iw.add_argument("--save", action="store_true", help="Save to templates directory")
    add_common(p_iw)
    p_iw.set_defaults(func=cmd_import_workflow)

    p_nr = sub.add_parser("node-registry", help="Dynamic node output type registry")
    p_nr.add_argument("--dump", action="store_true", help="Dump full registry (all node output types)")
    p_nr.add_argument("--refresh", action="store_true", help="Force re-fetch from ComfyUI")
    add_common(p_nr)
    p_nr.set_defaults(func=cmd_node_registry)
