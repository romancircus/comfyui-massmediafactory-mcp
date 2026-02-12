"""
ComfyUI MassMediaFactory MCP Server

A Model Context Protocol server for ComfyUI workflow orchestration.
Enables Claude and other AI assistants to create, iterate, and maintain
image and video generation pipelines.
"""

__version__ = "0.1.0"

from .server import mcp, main
from .ltx_planner import (
    LTXVariant,
    LTXSampler,
    BranchSelection,
    select_ltx_variant,
    get_ltx_workflow_with_trace,
    list_available_ltx_variants,
    get_ltx_model_constraints,
    generate_compile_report,
    get_advanced_ltx_branches,
    LTX_CANONICAL_NODES,
)

__all__ = [
    "mcp",
    "main",
    "__version__",
    "LTXVariant",
    "LTXSampler",
    "BranchSelection",
    "select_ltx_variant",
    "get_ltx_workflow_with_trace",
    "list_available_ltx_variants",
    "get_ltx_model_constraints",
    "generate_compile_report",
    "get_advanced_ltx_branches",
    "LTX_CANONICAL_NODES",
]
