"""Pipeline commands: pipeline, telestyle."""

from ..cli_utils import EXIT_ERROR, EXIT_OK, EXIT_TIMEOUT, _add_common_args, _add_retry_args, _is_pretty, _output


def cmd_pipeline(args):
    """Run a pre-tested pipeline."""
    from ..cli_pipelines import run_pipeline

    pretty = args.pretty or _is_pretty()
    result = run_pipeline(args.pipeline_name, args)
    _output(result, pretty)

    if result.get("status") == "timeout":
        return EXIT_TIMEOUT
    return EXIT_OK if "error" not in result else EXIT_ERROR


def cmd_telestyle(args):
    """TeleStyle transfer (image or video)."""
    from ..cli_pipelines import run_telestyle

    pretty = args.pretty or _is_pretty()
    result = run_telestyle(args.mode, args)
    _output(result, pretty)
    return EXIT_OK if "error" not in result else EXIT_ERROR


def register_commands(sub, add_common=_add_common_args, add_retry=_add_retry_args, **_kwargs):
    """Register pipeline subcommands."""
    p_pipe = sub.add_parser("pipeline", help="Run pre-tested pipelines")
    p_pipe.add_argument("pipeline_name", help="Pipeline: i2v, upscale, viral-short, t2v-styled, bio-to-video")
    p_pipe.add_argument("--model", "-m", help="Model override")
    p_pipe.add_argument("--prompt", "-p", help="Generation prompt")
    p_pipe.add_argument("--image", help="Input image")
    p_pipe.add_argument("--output", "-o", help="Output path")
    p_pipe.add_argument("--seed", type=int)
    p_pipe.add_argument("--timeout", type=int, default=None, help="Timeout in seconds (default: 1800s)")
    p_pipe.add_argument("--style-image", help="Style reference image (telestyle)")
    p_pipe.add_argument("--character", help="Character name (viral-short)")
    p_pipe.add_argument("--style", help="Style preset (viral-short)")
    p_pipe.add_argument("--bio-prompt", help="Bio prompt (bio-to-video)")
    p_pipe.add_argument("--shiny-colors", help="Shiny colors JSON (bio-to-video)")
    p_pipe.add_argument("--motion-prompt", help="Motion prompt (bio-to-video)")
    p_pipe.add_argument("--factor", type=float, default=2.0, help="Upscale factor")
    add_retry(p_pipe)
    add_common(p_pipe)
    p_pipe.set_defaults(func=cmd_pipeline)

    p_ts = sub.add_parser("telestyle", help="TeleStyle transfer")
    p_ts.add_argument("mode", choices=["image", "video"], help="TeleStyle mode")
    p_ts.add_argument("--content", required=True, help="Content image/video path")
    p_ts.add_argument("--style", required=True, help="Style reference image")
    p_ts.add_argument("--output", "-o", help="Output path")
    p_ts.add_argument("--cfg", type=float, help="CFG override")
    p_ts.add_argument("--steps", type=int, help="Steps override")
    p_ts.add_argument("--seed", type=int)
    p_ts.add_argument(
        "--timeout", type=int, default=None, help="Timeout in seconds (default: 660s for image, 900s for video)"
    )
    add_common(p_ts)
    p_ts.set_defaults(func=cmd_telestyle)
