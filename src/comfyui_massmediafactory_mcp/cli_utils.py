"""Shared CLI utilities — exit codes, output helpers, error classification, retry logic."""

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

# Operation-specific default timeouts
TIMEOUTS = {
    "run": 660,  # 11 min for image generation
    "batch": 900,  # 15 min for video operations
    "pipeline": 1800,  # 30 min for multi-stage pipelines
    "system": 30,  # 30 sec for system commands
}

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


def _retry_loop(fn, max_retries: int, retry_on: str = "vram,timeout,connection"):
    """Execute fn with retry logic for transient errors."""
    from comfyui_massmediafactory_mcp import execution

    retry_types = {t.strip().upper() for t in retry_on.split(",")} if retry_on else set()

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

        if last_code == EXIT_OK:
            return last_code, last_result

        error_class = _classify_error(last_result)

        if error_class in _PERMANENT_ERRORS:
            return last_code, last_result

        if error_class not in retry_codes:
            return last_code, last_result

        if attempt >= max_retries:
            return last_code, last_result

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


def _add_common_args(parser) -> None:
    """Add --pretty flag."""
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")


def _add_retry_args(parser) -> None:
    """Add --retry and --retry-on flags."""
    parser.add_argument("--retry", type=int, default=0, help="Number of retries for transient errors (default: 0)")
    parser.add_argument(
        "--retry-on",
        default="vram,timeout,connection",
        help="Comma-separated error types to retry on (default: vram,timeout,connection)",
    )
