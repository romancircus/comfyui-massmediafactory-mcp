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

# Re-export shared utilities for backward compatibility (tests import from here)
from .cli_utils import (  # noqa: F401
    EXIT_CONNECTION,
    EXIT_ERROR,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_PARTIAL,
    EXIT_TIMEOUT,
    EXIT_VALIDATION,
    EXIT_VRAM,
    TIMEOUTS,
    _add_common_args,
    _add_retry_args,
    _classify_error,
    _error,
    _exit_code_for_error,
    _is_pretty,
    _msg,
    _output,
    _parse_json_arg,
    _read_workflow,
    _retry_loop,
)

# Re-export command handlers for backward compatibility (tests import from here)
from .cli_commands.run import cmd_run, cmd_execute, cmd_wait, cmd_upload, cmd_download  # noqa: F401
from .cli_commands.batch import cmd_batch_seeds, cmd_batch_sweep, cmd_batch_queue, cmd_batch_dir, _batch_exit_code  # noqa: F401
from .cli_commands.templates import cmd_templates_list, cmd_templates_get, cmd_templates_create  # noqa: F401
from .cli_commands.models import (  # noqa: F401
    cmd_models_list,
    cmd_models_constraints,
    cmd_models_compatibility,
    cmd_models_optimize,
    cmd_search_model,
    cmd_install_model,
)
from .cli_commands.system import (  # noqa: F401
    cmd_stats,
    cmd_free,
    cmd_interrupt,
    cmd_enhance,
    cmd_qa,
    cmd_profile,
    cmd_validate,
    cmd_status,
    cmd_progress,
    cmd_regenerate,
)
from .cli_commands.assets import cmd_assets_list, cmd_assets_metadata, cmd_publish, cmd_workflow_lib  # noqa: F401
from .cli_commands.pipeline import cmd_pipeline, cmd_telestyle  # noqa: F401

# Import register functions
from .cli_commands import run as _run_mod
from .cli_commands import batch as _batch_mod
from .cli_commands import templates as _tmpl_mod
from .cli_commands import models as _models_mod
from .cli_commands import system as _system_mod
from .cli_commands import assets as _assets_mod
from .cli_commands import pipeline as _pipe_mod


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmf",
        description="MassMediaFactory CLI — single ComfyUI interface",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--url", help="ComfyUI server URL (overrides COMFYUI_URL env)")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # Register all command groups
    _run_mod.register_commands(sub)
    _batch_mod.register_commands(sub)
    _tmpl_mod.register_commands(sub)
    _models_mod.register_commands(sub)
    _system_mod.register_commands(sub)
    _assets_mod.register_commands(sub)
    _pipe_mod.register_commands(sub)

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
