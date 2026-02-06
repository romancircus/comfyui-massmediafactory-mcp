"""
Core Error Handling System

Centralized error classes with actionable guidance for MCP errors.

All errors follow MCP specification:
- Include "isError": true
- Include "code" for error categorization
- Include "suggestion" for actionable guidance
- Include "details" for additional context
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class RichMCPError:
    """
    Rich MCP-compliant error response with actionable guidance.

    Extended version with suggestion + troubleshooting fields.
    The simpler MCPError in mcp_utils.py is used by @mcp_tool_wrapper.
    This class is used by domain-specific error subclasses below.

    Per MCP spec, tool execution errors should include:
    - isError: true (required)
    - code: error category (required)
    - error: human-readable message (required)
    - suggestion: actionable guidance (recommended)
    - details: additional context (optional)
    """

    code: str = ""
    error: str = ""
    suggestion: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    troubleshooting: Optional[str | List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP-compliant error dict."""
        result: Dict[str, Any] = {
            "isError": True,
            "code": self.code,
            "error": self.error,
            "suggestion": self.suggestion,
        }
        if self.details:
            result["details"] = self.details
        if self.troubleshooting:
            result["troubleshooting"] = self.troubleshooting
        return result


# Backward compatibility alias
MCPError = RichMCPError


@dataclass
class ModelNotFoundError(RichMCPError):
    """
    Model file not found in ComfyUI models directory.

    Example:
        ModelNotFoundError(
            model_name="flux1-dev.safetensors",
            model_type="checkpoints",
            available_models=["flux2-dev.safetensors"],
            path="~/ComfyUI/models/checkpoints/"
        ).to_dict()
    """

    model_name: str = ""
    model_type: str = ""
    available_models: List[str] = field(default_factory=list)
    path: str = ""

    def __post_init__(self):
        self.code = "MODEL_NOT_FOUND"
        self.error = f"Model '{self.model_name}' not found in models/{self.model_type}/"
        self.suggestion = (
            f"Run list_models('{self.model_type}') to see available models. "
            f"Common issues: typo in model name, model not downloaded, or wrong model_type."
        )
        self.details = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "available": self.available_models[:10],
        }
        if self.path:
            self.details["path"] = self.path
        self.troubleshooting = f"Check {self.path} for the file."


@dataclass
class CustomNodeMissingError(RichMCPError):
    """
    Required ComfyUI custom node not installed.

    Example:
        CustomNodeMissingError(
            node_type="Wan2VideoNode",
            package="comfyui-wan2-video",
            install_cmd="pip install comfyui-wan2-video"
        ).to_dict()
    """

    node_type: str = ""
    package: Optional[str] = None
    install_cmd: Optional[str] = None

    def __post_init__(self):
        self.code = "CUSTOM_NODE_MISSING"
        self.error = f"Custom node '{self.node_type}' not installed or not loaded."
        self.suggestion = "Install the required custom node package or restart ComfyUI to reload nodes."
        self.details = {"node_type": self.node_type}
        if self.package:
            self.details["package"] = self.package
        if self.install_cmd:
            self.details["install_cmd"] = self.install_cmd
            self.suggestion = f"Run: {self.install_cmd}"
        self.troubleshooting = (
            "1. Install package: pip install <package>\n"
            "2. Restart ComfyUI: sudo systemctl restart comfyui\n"
            "3. Check nodes are loaded in ComfyUI UI (missing nodes panel)"
        )


@dataclass
class TemplateMetadataError(RichMCPError):
    """
    Template missing required metadata or has invalid structure.

    Example:
        TemplateMetadataError(
            template_name="flux_basic",
            missing_fields=["description", "model", "type"],
            template_path="~/ComfyUI/comfyui-massmediafactory-mcp/src/comfyui_massmediafactory_mcp/templates/flux_basic.json"
        ).to_dict()
    """

    template_name: str = ""
    missing_fields: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    template_path: Optional[str] = None

    def __post_init__(self):
        self.code = "TEMPLATE_METADATA"
        self.error = f"Template '{self.template_name}' has invalid metadata structure."
        if self.missing_fields:
            self.error = f"Template '{self.template_name}' missing required fields: {', '.join(self.missing_fields)}"
        if self.errors:
            self.error = f"Template '{self.template_name}' has {len(self.errors)} validation errors."
        self.suggestion = (
            "Ensure template has _meta section with: description, model, type, parameters, defaults. "
            "Use list_workflow_templates() to see valid templates."
        )
        self.details = {
            "template_name": self.template_name,
            "missing_fields": self.missing_fields,
            "errors": self.errors[:10],
        }
        if self.template_path:
            self.details["template_path"] = self.template_path
        self.troubleshooting = (
            "1. Open template file\n"
            "2. Ensure _meta section exists with required fields\n"
            "3. Validate: validate_all_templates() or test at https://jsonlint.com"
        )


@dataclass
class TemplateParameterError(RichMCPError):
    """
    Template parameter validation error (missing required, invalid type, etc.).

    Example:
        TemplateParameterError(
            template_name="ltx_t2v",
            parameter="FRAMES",
            expected="integer divisible by 8",
            provided=97,
            suggestion_override="Use 81, 89, 97, 105, etc. (8n+1)"
        ).to_dict()
    """

    template_name: str = ""
    parameter: str = ""
    expected: str = ""
    provided: Optional[Any] = None
    suggestion_override: Optional[str] = None

    def __post_init__(self):
        self.code = "TEMPLATE_PARAMETER"
        self.error = f"Parameter '{self.parameter}' invalid for template '{self.template_name}'."
        self.suggestion = self.suggestion_override or (
            f"Expected: {self.expected}. " f"Run get_template('{self.template_name}') to see valid parameters."
        )
        self.details = {
            "template_name": self.template_name,
            "parameter": self.parameter,
            "expected": self.expected,
            "provided": self.provided,
        }
        self.troubleshooting = (
            "1. Check template parameters: get_template('<name>')\n"
            "2. Use list_workflow_templates() to see all templates\n"
            "3. See docs/ERROR_RECOVERY.md for parameter reference"
        )


# =============================================================================
# Error Factory Functions (Backward Compatibility)
# =============================================================================


def format_model_not_found(model_name: str, model_type: str, available_models: List[str]) -> Dict[str, Any]:
    """Format error for model not found with actionable suggestions."""
    error = ModelNotFoundError(
        model_name=model_name,
        model_type=model_type,
        available_models=available_models,
    )
    error.code = "MODEL_NOT_FOUND"
    error.error = f"Model '{model_name}' not found in models/{model_type}/"
    error.suggestion = (
        f"Run list_models('{model_type}') to see available models. "
        f"Common issues: typo in model name, model not downloaded, or wrong model_type."
    )
    error.details = {
        "model_name": model_name,
        "model_type": model_type,
        "available": available_models[:10],
    }
    error.troubleshooting = f"Check ~/ComfyUI/models/{model_type}/ for the file."
    return error.to_dict()


def format_custom_node_missing(
    node_type: str, package: Optional[str] = None, install_cmd: Optional[str] = None
) -> Dict[str, Any]:
    """Format error for missing custom node."""
    error = CustomNodeMissingError(
        node_type=node_type,
        package=package,
        install_cmd=install_cmd,
    )
    error.code = "CUSTOM_NODE_MISSING"
    error.error = f"Custom node '{node_type}' not installed or not loaded."
    error.details = {"node_type": node_type}
    if package:
        error.details["package"] = package
    if install_cmd:
        error.details["install_cmd"] = install_cmd
        error.suggestion = f"Run: {install_cmd}"
    error.troubleshooting = (
        "1. Install package: pip install <package>\n"
        "2. Restart ComfyUI: sudo systemctl restart comfyui\n"
        "3. Check nodes are loaded in ComfyUI UI (missing nodes panel)"
    )
    return error.to_dict()


def format_template_metadata(
    template_name: str,
    missing_fields: Optional[List[str]] = None,
    errors: Optional[List[str]] = None,
    template_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Format error for invalid template metadata."""
    error = TemplateMetadataError(
        template_name=template_name,
        missing_fields=missing_fields or [],
        errors=errors or [],
        template_path=template_path,
    )
    error.code = "TEMPLATE_METADATA"
    if missing_fields:
        error.error = f"Template '{template_name}' missing required fields: {', '.join(missing_fields)}"
    elif errors:
        error.error = f"Template '{template_name}' has {len(errors)} validation errors."
    else:
        error.error = f"Template '{template_name}' has invalid metadata structure."
    error.suggestion = (
        "Ensure template has _meta section with: description, model, type, parameters, defaults. "
        "Use list_workflow_templates() to see valid templates."
    )
    error.details = {
        "template_name": template_name,
        "missing_fields": missing_fields or [],
        "errors": (errors or [])[:10],
    }
    if template_path:
        error.details["template_path"] = template_path
    error.troubleshooting = (
        "1. Open template file\n"
        "2. Ensure _meta section exists with required fields\n"
        "3. Validate: validate_all_templates() or test at https://jsonlint.com"
    )
    return error.to_dict()


def format_template_parameter(
    template_name: str,
    parameter: str,
    expected: str,
    provided: Optional[Any] = None,
    suggestion_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Format error for invalid template parameter."""
    error = TemplateParameterError(
        template_name=template_name,
        parameter=parameter,
        expected=expected,
        provided=provided,
        suggestion_override=suggestion_override,
    )
    error.code = "TEMPLATE_PARAMETER"
    error.error = f"Parameter '{parameter}' invalid for template '{template_name}'."
    error.suggestion = suggestion_override or (
        f"Expected: {expected}. " f"Run get_template('{template_name}') to see valid parameters."
    )
    error.details = {
        "template_name": template_name,
        "parameter": parameter,
        "expected": expected,
        "provided": provided,
    }
    error.troubleshooting = (
        "1. Check template parameters: get_template('<name>')\n"
        "2. Use list_workflow_templates() to see all templates\n"
        "3. See docs/ERROR_RECOVERY.md for parameter reference"
    )
    return error.to_dict()
