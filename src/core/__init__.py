"""
Core Error Handling
"""

from .errors import (
    RichMCPError,
    MCPError,  # backward compat alias for RichMCPError
    ModelNotFoundError,
    CustomNodeMissingError,
    TemplateMetadataError,
    TemplateParameterError,
    format_model_not_found,
    format_custom_node_missing,
    format_template_metadata,
    format_template_parameter,
)

__all__ = [
    "RichMCPError",
    "MCPError",
    "ModelNotFoundError",
    "CustomNodeMissingError",
    "TemplateMetadataError",
    "TemplateParameterError",
    "format_model_not_found",
    "format_custom_node_missing",
    "format_template_metadata",
    "format_template_parameter",
]
