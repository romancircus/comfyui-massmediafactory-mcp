"""
ComfyUI MassMediaFactory MCP Server

A Model Context Protocol server for ComfyUI workflow orchestration.
Enables Claude and other AI assistants to create, iterate, and maintain
image and video generation pipelines.
"""

__version__ = "0.1.0"

from .server import mcp, main

__all__ = ["mcp", "main", "__version__"]
