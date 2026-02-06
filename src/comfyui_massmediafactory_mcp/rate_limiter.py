"""
Rate Limiting Module

Exposes rate limiting statistics and status for the MCP dashboard.

Provides visibility into current rate limits, usage, and reset times.
"""

from typing import Dict, Any
from datetime import datetime, timedelta

# Import the rate limiter from mcp_utils
from .mcp_utils import _rate_limiter


def get_rate_limit_status(tool_name: str = None) -> Dict[str, Any]:
    """
    Get current rate limiting status.

    Returns information about rate limits, remaining requests,
    reset time, and current usage.

    Args:
        tool_name: Optional specific tool name. If None, returns global stats.

    Returns:
        {
            "requests_per_minute": int,
            "requests_remaining": int,
            "current_usage": int,
            "reset_at": str (ISO 8601 timestamp),
            "reset_in_seconds": float,
            "window_seconds": int,
            "per_tool": bool,
            "tool": str (optional),
            "warning": str (if approaching limit),
        }
    """
    # Get stats from the rate limiter
    if tool_name:
        remaining = _rate_limiter.get_remaining(tool_name)
        reset_seconds = _rate_limiter.get_reset_time(tool_name)
        current_usage = _rate_limiter.max_calls - remaining
    else:
        # Global stats
        remaining = _rate_limiter.get_remaining("global")
        reset_seconds = _rate_limiter.get_reset_time("global")
        current_usage = _rate_limiter.max_calls - remaining

    # Calculate reset timestamp
    reset_at = datetime.utcnow() + timedelta(seconds=reset_seconds)
    reset_at_iso = reset_at.isoformat() + "Z"

    # Determine warning level
    usage_percent = (current_usage / _rate_limiter.max_calls) * 100
    warning = None
    if usage_percent >= 90:
        warning = "CRITICAL: Rate limit nearly exhausted"
    elif usage_percent >= 75:
        warning = "WARNING: Approaching rate limit"
    elif usage_percent >= 50:
        warning = "INFO: Moderate usage"

    result = {
        "requests_per_minute": _rate_limiter.max_calls,
        "requests_remaining": remaining,
        "current_usage": current_usage,
        "reset_at": reset_at_iso,
        "reset_in_seconds": round(reset_seconds, 1),
        "window_seconds": _rate_limiter.window_seconds,
        "per_tool": _rate_limiter.per_tool,
        "usage_percent": round(usage_percent, 1),
    }

    if tool_name:
        result["tool"] = tool_name

    if warning:
        result["warning"] = warning

    return result


def get_all_tools_rate_status() -> Dict[str, Any]:
    """
    Get rate limiting status for all tools.

    Returns status for every tool that has been tracked.

    Returns:
        {
            "tools": [
                {
                    "tool": str,
                    "requests_remaining": int,
                    "current_usage": int,
                    "reset_in_seconds": float,
                    "usage_percent": float,
                }
            ],
            "global_limit": int,
            "window_seconds": int,
        }
    """
    # Access the internal calls dict to get all tracked tools
    tools_data = []

    for tool_name in _rate_limiter._calls.keys():
        remaining = _rate_limiter.get_remaining(tool_name)
        reset_seconds = _rate_limiter.get_reset_time(tool_name)
        current_usage = _rate_limiter.max_calls - remaining
        usage_percent = (current_usage / _rate_limiter.max_calls) * 100

        tools_data.append(
            {
                "tool": tool_name,
                "requests_remaining": remaining,
                "current_usage": current_usage,
                "reset_in_seconds": round(reset_seconds, 1),
                "usage_percent": round(usage_percent, 1),
            }
        )

    # Sort by usage percent descending
    tools_data.sort(key=lambda x: x["usage_percent"], reverse=True)

    return {
        "tools": tools_data,
        "global_limit": _rate_limiter.max_calls,
        "window_seconds": _rate_limiter.window_seconds,
        "per_tool": _rate_limiter.per_tool,
        "total_tools_tracked": len(tools_data),
    }


def get_rate_limit_summary() -> Dict[str, Any]:
    """
    Get a brief summary of rate limit status.

    Quick check for dashboard display.

    Returns:
        {
            "status": str,  # "ok", "warning", "critical"
            "message": str,
            "requests_remaining": int,
            "reset_in_seconds": float,
        }
    """
    remaining = _rate_limiter.get_remaining("global")
    reset_seconds = _rate_limiter.get_reset_time("global")
    current_usage = _rate_limiter.max_calls - remaining
    usage_percent = (current_usage / _rate_limiter.max_calls) * 100

    if usage_percent >= 90:
        status = "critical"
        message = f"Rate limit nearly exhausted ({current_usage}/{_rate_limiter.max_calls})"
    elif usage_percent >= 75:
        status = "warning"
        message = f"Approaching rate limit ({current_usage}/{_rate_limiter.max_calls})"
    elif usage_percent >= 50:
        status = "ok"
        message = f"Moderate usage ({current_usage}/{_rate_limiter.max_calls})"
    else:
        status = "ok"
        message = f"Healthy ({current_usage}/{_rate_limiter.max_calls})"

    return {
        "status": status,
        "message": message,
        "requests_remaining": remaining,
        "reset_in_seconds": round(reset_seconds, 1),
        "limit": _rate_limiter.max_calls,
        "window_seconds": _rate_limiter.window_seconds,
    }
