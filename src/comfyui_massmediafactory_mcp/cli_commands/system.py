"""System commands: stats, free, interrupt, enhance, qa, profile, validate, status, progress, regenerate."""

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
