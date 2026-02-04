"""
MCP Tool Annotations

Per MCP spec, tools can have annotations for:
- audience: ["user"] and/or ["assistant"] - who should see results
- priority: 0.0-1.0 - importance/necessity indicator

These help clients understand how to present tool results.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ToolAnnotation:
    """Annotation metadata for an MCP tool."""
    audience: List[str]  # ["user"], ["assistant"], or ["user", "assistant"]
    priority: float  # 0.0 (low) to 1.0 (high)
    category: str  # Tool category for grouping
    description: Optional[str] = None


# =============================================================================
# Tool Annotation Definitions (Updated for consolidated tool set)
# =============================================================================

TOOL_ANNOTATIONS: Dict[str, ToolAnnotation] = {
    # Discovery Tools (consolidated: list_models replaces 5 tools)
    "list_models": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="discovery",
        description="List models by type"
    ),
    "get_node_info": ToolAnnotation(
        audience=["assistant"],
        priority=0.6,
        category="discovery"
    ),
    "search_nodes": ToolAnnotation(
        audience=["assistant"],
        priority=0.6,
        category="discovery"
    ),

    # Execution Tools
    "execute_workflow": ToolAnnotation(
        audience=["assistant"],
        priority=0.9,
        category="execution"
    ),
    "get_workflow_status": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.8,
        category="execution"
    ),
    "wait_for_completion": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.9,
        category="execution"
    ),
    "get_system_stats": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="execution"
    ),
    "free_memory": ToolAnnotation(
        audience=["assistant"],
        priority=0.4,
        category="execution"
    ),
    "interrupt_execution": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.8,
        category="execution"
    ),
    "get_queue_status": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="execution"
    ),

    # Asset Tools
    "regenerate": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.8,
        category="assets"
    ),
    "list_assets": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.7,
        category="assets"
    ),
    "get_asset_metadata": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="assets"
    ),
    "view_output": ToolAnnotation(
        audience=["user"],
        priority=0.9,
        category="assets"
    ),
    "cleanup_expired_assets": ToolAnnotation(
        audience=["assistant"],
        priority=0.3,
        category="assets"
    ),
    "upload_image": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="assets"
    ),
    "download_output": ToolAnnotation(
        audience=["user"],
        priority=0.8,
        category="assets"
    ),

    # Publishing Tools
    "publish_asset": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.8,
        category="publishing"
    ),
    "get_publish_info": ToolAnnotation(
        audience=["assistant"],
        priority=0.4,
        category="publishing"
    ),
    "set_publish_dir": ToolAnnotation(
        audience=["assistant"],
        priority=0.4,
        category="publishing"
    ),

    # Workflow Library (consolidated: workflow_library replaces 5 tools)
    "workflow_library": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.7,
        category="library",
        description="save|load|list|delete|duplicate workflows"
    ),
    "export_workflow": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.5,
        category="library"
    ),
    "import_workflow": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="library"
    ),

    # VRAM Tools
    "estimate_vram": ToolAnnotation(
        audience=["assistant"],
        priority=0.6,
        category="vram"
    ),
    "check_model_fits": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="vram"
    ),

    # Validation (consolidated: validate_workflow replaces 4 tools)
    "validate_workflow": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="validation",
        description="Unified validation with auto_fix and check_pattern options"
    ),
    "check_connection": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="validation"
    ),

    # SOTA Recommendations (consolidated: sota_query replaces 5 tools)
    "sota_query": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.8,
        category="sota",
        description="category|recommend|check|settings|installed"
    ),

    # Templates
    "list_workflow_templates": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="templates",
        params={
            "only_installed": {
                "type": "boolean",
                "description": "Filter to only templates with installed models",
                "default": False
            },
            "model_type": {
                "type": "string",
                "description": "Filter by model type (flux2, ltx2, wan26, qwen, hunyuan15, audio, utility)",
                "enum": ["flux2", "ltx2", "wan26", "qwen", "hunyuan15", "sdxl", "telestyle", "audio", "utility"]
            },
            "tags": {
                "type": "list",
                "description": "Filter templates with specific tags (e.g., ['priority:recommended', 'type:audio2vid'])",
                "default": []
            }
        }
    ),
    "get_template": ToolAnnotation(
        audience=["assistant"],
        priority=0.8,
        category="templates"
    ),
    "create_workflow_from_template": ToolAnnotation(
        audience=["assistant"],
        priority=0.9,
        category="templates"
    ),

    # Workflow Patterns
    "get_workflow_skeleton": ToolAnnotation(
        audience=["assistant"],
        priority=0.8,
        category="patterns"
    ),
    "get_model_constraints": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="patterns"
    ),
    "get_node_chain": ToolAnnotation(
        audience=["assistant"],
        priority=0.6,
        category="patterns"
    ),
    "list_available_patterns": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="patterns"
    ),

    # Batch Execution (consolidated: batch_execute replaces 3 tools)
    "batch_execute": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.7,
        category="batch",
        description="batch|sweep|seeds execution modes"
    ),

    # Pipelines
    "execute_pipeline_stages": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.8,
        category="pipeline"
    ),
    "run_image_to_video_pipeline": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.8,
        category="pipeline"
    ),
    "run_upscale_pipeline": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.7,
        category="pipeline"
    ),

    # Model Management
    "search_civitai": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.6,
        category="models"
    ),
    "download_model": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="models"
    ),
    "get_model_info": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="models"
    ),
    "list_installed_models": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.5,
        category="models"
    ),

    # Analysis
    "get_image_dimensions": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="analysis"
    ),
    "detect_objects": ToolAnnotation(
        audience=["assistant"],
        priority=0.6,
        category="analysis"
    ),
    "get_video_info": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="analysis"
    ),

    # QA
    "qa_output": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="qa"
    ),
    "check_vlm_available": ToolAnnotation(
        audience=["assistant"],
        priority=0.4,
        category="qa"
    ),

    # Style Learning (consolidated: style_suggest replaces 3, manage_presets replaces 4)
    "record_generation": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="style_learning"
    ),
    "rate_generation": ToolAnnotation(
        audience=["user"],
        priority=0.6,
        category="style_learning"
    ),
    "style_suggest": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.7,
        category="style_learning",
        description="Prompt enhancement and similar prompts"
    ),
    "manage_presets": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.5,
        category="style_learning",
        description="CRUD for style presets"
    ),

    # Workflow Generation
    "generate_workflow": ToolAnnotation(
        audience=["assistant"],
        priority=0.9,
        category="generation",
        description="Primary workflow generation"
    ),
    "list_supported_workflows": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="generation"
    ),
}


def get_annotation(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get annotation for a tool."""
    annotation = TOOL_ANNOTATIONS.get(tool_name)
    if annotation:
        return {
            "audience": annotation.audience,
            "priority": annotation.priority,
            "category": annotation.category,
            "description": annotation.description,
        }
    return None


def get_all_annotations() -> Dict[str, Dict[str, Any]]:
    """Get all tool annotations."""
    return {
        name: get_annotation(name)
        for name in TOOL_ANNOTATIONS
    }


def get_tools_by_category(category: str) -> List[str]:
    """Get all tools in a category."""
    return [
        name for name, ann in TOOL_ANNOTATIONS.items()
        if ann.category == category
    ]


def get_user_facing_tools() -> List[str]:
    """Get tools whose results should be shown to users."""
    return [
        name for name, ann in TOOL_ANNOTATIONS.items()
        if "user" in ann.audience
    ]


def get_high_priority_tools(threshold: float = 0.7) -> List[str]:
    """Get tools with priority >= threshold."""
    return [
        name for name, ann in TOOL_ANNOTATIONS.items()
        if ann.priority >= threshold
    ]
