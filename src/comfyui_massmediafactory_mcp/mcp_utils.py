"""
MCP Utilities

Utilities for MCP-compliant responses, error handling, logging, and rate limiting.
"""

import time
import uuid
import logging
import functools
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict

# =============================================================================
# Structured Logging
# =============================================================================

logger = logging.getLogger("comfyui-mcp")
logger.setLevel(logging.INFO)

# Add handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)


@dataclass
class ToolInvocation:
    """Track a tool invocation for logging."""
    tool_name: str
    invocation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_time: float = field(default_factory=time.time)

    def complete(self, status: str = "success", error: str = None) -> Dict[str, Any]:
        """Log completion and return log entry."""
        latency_ms = (time.time() - self.start_time) * 1000
        log_entry = {
            "tool": self.tool_name,
            "invocation_id": self.invocation_id,
            "latency_ms": round(latency_ms, 2),
            "status": status,
        }
        if error:
            log_entry["error"] = error

        if status == "success":
            logger.info(f"Tool call: {log_entry}")
        else:
            logger.error(f"Tool call: {log_entry}")

        return log_entry


# =============================================================================
# MCP-Compliant Error Responses
# =============================================================================

@dataclass
class MCPError:
    """
    MCP-compliant error response.

    Per MCP spec, tool execution errors should include isError: true
    """
    message: str
    code: str = "TOOL_ERROR"
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP-compliant error dict."""
        result = {
            "error": self.message,
            "code": self.code,
            "isError": True,
        }
        if self.details:
            result["details"] = self.details
        return result


def mcp_error(
    message: str,
    code: str = "TOOL_ERROR",
    details: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Create an MCP-compliant error response.

    Args:
        message: Human-readable error message
        code: Error code (e.g., "NOT_FOUND", "VALIDATION_ERROR", "TIMEOUT")
        details: Additional error context

    Returns:
        Dict with error, code, isError=True

    Example:
        return mcp_error("Asset not found", "NOT_FOUND", {"asset_id": "abc123"})
    """
    return MCPError(message, code, details).to_dict()


# Common error helpers
def not_found_error(resource_type: str, identifier: str) -> Dict[str, Any]:
    """Resource not found error."""
    return mcp_error(
        f"{resource_type} not found: {identifier}",
        "NOT_FOUND",
        {resource_type.lower(): identifier}
    )


def validation_error(message: str, field: str = None) -> Dict[str, Any]:
    """Input validation error."""
    details = {"field": field} if field else None
    return mcp_error(message, "VALIDATION_ERROR", details)


def timeout_error(operation: str, timeout_seconds: int) -> Dict[str, Any]:
    """Operation timeout error."""
    return mcp_error(
        f"{operation} timed out after {timeout_seconds}s",
        "TIMEOUT",
        {"timeout_seconds": timeout_seconds}
    )


def connection_error(service: str, url: str = None) -> Dict[str, Any]:
    """Service connection error."""
    details = {"url": url} if url else None
    return mcp_error(f"Cannot connect to {service}", "CONNECTION_ERROR", details)


# =============================================================================
# MCP-Compliant Success Responses
# =============================================================================

def mcp_success(
    data: Any,
    message: str = None,
    metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Create an MCP-compliant success response.

    Args:
        data: The response data
        message: Optional success message
        metadata: Optional metadata (stored in _meta)

    Returns:
        Response dict with data and optional metadata
    """
    if isinstance(data, dict):
        result = data.copy()
    else:
        result = {"data": data}

    if message:
        result["message"] = message

    if metadata:
        result["_meta"] = metadata

    return result


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """
    Simple rate limiter for MCP tools.

    Per MCP security requirements, servers MUST rate limit tool invocations.
    """

    def __init__(
        self,
        max_calls: int = 100,
        window_seconds: int = 60,
        per_tool: bool = True,
    ):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed in window
            window_seconds: Time window in seconds
            per_tool: If True, limits are per-tool; if False, global
        """
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.per_tool = per_tool
        self._calls: Dict[str, List[float]] = defaultdict(list)

    def check(self, tool_name: str = "global") -> bool:
        """
        Check if a call is allowed.

        Args:
            tool_name: Tool name (used if per_tool=True)

        Returns:
            True if allowed, False if rate limited
        """
        key = tool_name if self.per_tool else "global"
        now = time.time()

        # Clean old entries
        self._calls[key] = [
            t for t in self._calls[key]
            if now - t < self.window_seconds
        ]

        # Check limit
        if len(self._calls[key]) >= self.max_calls:
            return False

        # Record call
        self._calls[key].append(now)
        return True

    def get_remaining(self, tool_name: str = "global") -> int:
        """Get remaining calls in current window."""
        key = tool_name if self.per_tool else "global"
        now = time.time()

        self._calls[key] = [
            t for t in self._calls[key]
            if now - t < self.window_seconds
        ]

        return max(0, self.max_calls - len(self._calls[key]))

    def get_reset_time(self, tool_name: str = "global") -> float:
        """Get seconds until rate limit resets."""
        key = tool_name if self.per_tool else "global"
        if not self._calls[key]:
            return 0

        oldest = min(self._calls[key])
        return max(0, self.window_seconds - (time.time() - oldest))


# Global rate limiter instance
_rate_limiter = RateLimiter(max_calls=100, window_seconds=60, per_tool=True)


def rate_limit_error(tool_name: str) -> Dict[str, Any]:
    """Rate limit exceeded error."""
    reset_time = _rate_limiter.get_reset_time(tool_name)
    return mcp_error(
        f"Rate limit exceeded for {tool_name}",
        "RATE_LIMITED",
        {
            "retry_after_seconds": round(reset_time, 1),
            "limit": _rate_limiter.max_calls,
            "window_seconds": _rate_limiter.window_seconds,
        }
    )


# =============================================================================
# Tool Decorator with Logging and Rate Limiting
# =============================================================================

def mcp_tool_wrapper(func):
    """
    Decorator that adds MCP-compliant logging and rate limiting to tools.

    Wrap tool functions with this to get:
    - Structured logging with correlation IDs
    - Rate limiting
    - Automatic error formatting

    Example:
        @mcp.tool()
        @mcp_tool_wrapper
        def my_tool(param: str) -> dict:
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        invocation = ToolInvocation(tool_name)

        # Check rate limit
        if not _rate_limiter.check(tool_name):
            invocation.complete("rate_limited")
            return rate_limit_error(tool_name)

        try:
            result = func(*args, **kwargs)

            # Check if result is an error
            if isinstance(result, dict) and result.get("isError"):
                invocation.complete("error", result.get("error"))
            else:
                invocation.complete("success")

            return result

        except Exception as e:
            invocation.complete("error", str(e))
            return mcp_error(str(e), "INTERNAL_ERROR")

    return wrapper


# =============================================================================
# Pagination Utilities
# =============================================================================

def paginate(
    items: List[Any],
    cursor: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Apply cursor-based pagination to a list.

    Args:
        items: Full list of items
        cursor: Cursor from previous request (base64 encoded index)
        limit: Maximum items to return

    Returns:
        {
            "items": [...],
            "nextCursor": "..." or None,
            "total": N
        }
    """
    import base64

    total = len(items)

    # Decode cursor to get start index
    start_idx = 0
    if cursor:
        try:
            start_idx = int(base64.b64decode(cursor).decode())
        except (ValueError, Exception):
            start_idx = 0

    # Slice items
    end_idx = min(start_idx + limit, total)
    page_items = items[start_idx:end_idx]

    # Encode next cursor
    next_cursor = None
    if end_idx < total:
        next_cursor = base64.b64encode(str(end_idx).encode()).decode()

    return {
        "items": page_items,
        "nextCursor": next_cursor,
        "total": total,
    }


# =============================================================================
# Input Validation Helpers
# =============================================================================

def validate_required(params: Dict[str, Any], required: List[str]) -> Optional[Dict[str, Any]]:
    """
    Validate required parameters are present.

    Returns None if valid, error dict if invalid.
    """
    missing = [p for p in required if p not in params or params[p] is None]
    if missing:
        return validation_error(
            f"Missing required parameters: {', '.join(missing)}",
            field=missing[0]
        )
    return None


def validate_type(value: Any, expected_type: type, param_name: str) -> Optional[Dict[str, Any]]:
    """
    Validate parameter type.

    Returns None if valid, error dict if invalid.
    """
    if not isinstance(value, expected_type):
        return validation_error(
            f"Parameter '{param_name}' must be {expected_type.__name__}, got {type(value).__name__}",
            field=param_name
        )
    return None


def validate_range(
    value: Union[int, float],
    param_name: str,
    min_val: Union[int, float] = None,
    max_val: Union[int, float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Validate numeric parameter is within range.

    Returns None if valid, error dict if invalid.
    """
    if min_val is not None and value < min_val:
        return validation_error(
            f"Parameter '{param_name}' must be >= {min_val}, got {value}",
            field=param_name
        )
    if max_val is not None and value > max_val:
        return validation_error(
            f"Parameter '{param_name}' must be <= {max_val}, got {value}",
            field=param_name
        )
    return None
