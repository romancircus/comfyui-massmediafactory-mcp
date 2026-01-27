"""
Structured Error Codes for MassMediaFactory MCP

Provides consistent error handling across all modules.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class ErrorCode(Enum):
    """
    Structured error codes organized by category.

    E1xx: Assets & Resources
    E2xx: Models & Discovery
    E3xx: Connection & Communication
    E4xx: Validation & Input
    E5xx: Execution & Runtime
    """

    # E1xx: Assets & Resources
    ASSET_NOT_FOUND = ("E100", "Asset not found or expired", False)
    ASSET_EXPIRED = ("E101", "Asset has expired", False)
    FILE_NOT_FOUND = ("E102", "File not found on disk", False)
    LOAD_FAILED = ("E103", "Failed to load file", False)
    UNSUPPORTED_TYPE = ("E104", "Unsupported asset type", False)
    PUBLISH_FAILED = ("E105", "Failed to publish asset", False)

    # E2xx: Models & Discovery
    MODEL_NOT_FOUND = ("E200", "Model file not found in ComfyUI", False)
    NODE_NOT_FOUND = ("E201", "Node type not found in ComfyUI", False)
    WORKFLOW_NOT_FOUND = ("E202", "Saved workflow not found", False)
    TEMPLATE_NOT_FOUND = ("E203", "Workflow template not found", False)

    # E3xx: Connection & Communication
    CONNECTION_FAILED = ("E300", "Cannot connect to ComfyUI", True)
    CONNECTION_TIMEOUT = ("E301", "Connection to ComfyUI timed out", True)
    INVALID_RESPONSE = ("E302", "Invalid response from ComfyUI", True)
    UPLOAD_FAILED = ("E303", "Failed to upload file to ComfyUI", True)
    VLM_UNAVAILABLE = ("E310", "VLM service unavailable", True)

    # E4xx: Validation & Input
    VALIDATION_FAILED = ("E400", "Workflow validation failed", False)
    INVALID_FORMAT = ("E401", "Invalid workflow format", False)
    MISSING_INPUT = ("E402", "Required input missing", False)
    INVALID_CONNECTION = ("E403", "Invalid node connection", False)
    CYCLE_DETECTED = ("E404", "Circular dependency in workflow", False)
    RESOLUTION_INVALID = ("E405", "Invalid resolution for model", False)
    NO_WORKFLOW_DATA = ("E406", "Asset has no workflow data for regeneration", False)

    # E5xx: Execution & Runtime
    EXECUTION_FAILED = ("E500", "Workflow execution failed", False)
    EXECUTION_TIMEOUT = ("E501", "Workflow execution timed out", True)
    EXECUTION_INTERRUPTED = ("E502", "Workflow execution was interrupted", True)
    QUEUE_FULL = ("E503", "ComfyUI queue is full", True)
    VRAM_INSUFFICIENT = ("E504", "Insufficient GPU VRAM", False)

    def __init__(self, code: str, message: str, retryable: bool):
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass
class MCPError:
    """Structured error response."""

    error_code: ErrorCode
    details: Dict[str, Any]

    def to_dict(self) -> dict:
        """Convert to MCP response dict."""
        return {
            "error": self.error_code.code,
            "error_name": self.error_code.name,
            "message": self.error_code.message,
            "details": self.details,
            "retryable": self.error_code.retryable,
        }


def make_error(error_code: ErrorCode, **details) -> dict:
    """
    Create a structured error response.

    Args:
        error_code: The ErrorCode enum value.
        **details: Additional context (asset_id, node_id, etc.)

    Returns:
        Structured error dict for MCP response.

    Example:
        return make_error(ErrorCode.ASSET_NOT_FOUND, asset_id="abc-123")
    """
    return MCPError(error_code, details).to_dict()


# List of retryable error codes
RETRYABLE_ERRORS = {
    ErrorCode.CONNECTION_FAILED.code,
    ErrorCode.CONNECTION_TIMEOUT.code,
    ErrorCode.INVALID_RESPONSE.code,
    ErrorCode.UPLOAD_FAILED.code,
    ErrorCode.VLM_UNAVAILABLE.code,
    ErrorCode.EXECUTION_TIMEOUT.code,
    ErrorCode.EXECUTION_INTERRUPTED.code,
}


def is_retryable(error_response: dict) -> bool:
    """Check if an error response is retryable."""
    if "error" not in error_response:
        return False
    return error_response.get("error") in RETRYABLE_ERRORS or error_response.get("retryable", False)
