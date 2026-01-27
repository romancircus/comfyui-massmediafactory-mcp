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
# Tool Annotation Definitions
# =============================================================================

TOOL_ANNOTATIONS: Dict[str, ToolAnnotation] = {
    # Discovery Tools - primarily for assistant
    "list_checkpoints": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="discovery",
        description="List available models - assistant uses to build workflows"
    ),
    "list_unets": ToolAnnotation(
        audience=["assistant"],
        priority=0.6,
        category="discovery"
    ),
    "list_loras": ToolAnnotation(
        audience=["assistant"],
        priority=0.6,
        category="discovery"
    ),
    "list_vaes": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="discovery"
    ),
    "list_controlnets": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="discovery"
    ),
    "get_node_info": ToolAnnotation(
        audience=["assistant"],
        priority=0.6,
        category="discovery",
        description="Get node details for workflow building"
    ),
    "search_nodes": ToolAnnotation(
        audience=["assistant"],
        priority=0.6,
        category="discovery"
    ),
    "get_all_models": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.7,
        category="discovery",
        description="Summary of all models - useful for user to see capabilities"
    ),

    # Execution Tools - results for both user and assistant
    "execute_workflow": ToolAnnotation(
        audience=["assistant"],
        priority=0.9,
        category="execution",
        description="Core workflow execution - assistant orchestrates"
    ),
    "get_workflow_status": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.8,
        category="execution",
        description="Status updates shown to user during generation"
    ),
    "wait_for_completion": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.9,
        category="execution",
        description="Final results shown to user"
    ),
    "get_system_stats": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="execution",
        description="VRAM check before running workflows"
    ),
    "free_memory": ToolAnnotation(
        audience=["assistant"],
        priority=0.4,
        category="execution"
    ),
    "interrupt_execution": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.8,
        category="execution",
        description="User may want to cancel generation"
    ),

    # Asset Tools - results often shown to user
    "regenerate": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.8,
        category="assets",
        description="Iteration results shown to user"
    ),
    "list_assets": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.7,
        category="assets",
        description="User browses generated assets"
    ),
    "get_asset_metadata": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="assets",
        description="Internal use for debugging/iteration"
    ),
    "view_output": ToolAnnotation(
        audience=["user"],
        priority=0.9,
        category="assets",
        description="User views generated content"
    ),
    "upload_image": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="assets"
    ),
    "download_output": ToolAnnotation(
        audience=["user"],
        priority=0.8,
        category="assets",
        description="User downloads final assets"
    ),

    # Publishing Tools - results for user
    "publish_asset": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.8,
        category="publishing",
        description="Published URL shown to user"
    ),
    "get_publish_info": ToolAnnotation(
        audience=["assistant"],
        priority=0.4,
        category="publishing"
    ),

    # Workflow Library - both audiences
    "save_workflow": ToolAnnotation(
        audience=["assistant"],
        priority=0.6,
        category="library"
    ),
    "load_workflow": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="library"
    ),
    "list_saved_workflows": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.6,
        category="library"
    ),

    # VRAM Tools - assistant use
    "estimate_vram": ToolAnnotation(
        audience=["assistant"],
        priority=0.6,
        category="vram",
        description="Check before running expensive workflows"
    ),
    "check_model_fits": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="vram"
    ),

    # Validation - assistant use
    "validate_workflow": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="validation",
        description="Catch errors before execution"
    ),
    "validate_and_fix_workflow": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="validation"
    ),

    # SOTA Recommendations - both audiences
    "get_sota_models": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.7,
        category="sota",
        description="Best models for user's task"
    ),
    "recommend_model": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.8,
        category="sota",
        description="Model recommendation shown to user"
    ),
    "check_model_freshness": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="sota"
    ),

    # Templates - assistant use
    "list_workflow_templates": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="templates"
    ),
    "get_template": ToolAnnotation(
        audience=["assistant"],
        priority=0.8,
        category="templates"
    ),
    "create_workflow_from_template": ToolAnnotation(
        audience=["assistant"],
        priority=0.9,
        category="templates",
        description="Primary way to create workflows"
    ),

    # Batch Execution - results for user
    "execute_batch_workflows": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.7,
        category="batch",
        description="Batch results shown to user"
    ),
    "execute_parameter_sweep": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.6,
        category="batch"
    ),
    "generate_seed_variations": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.7,
        category="batch",
        description="Variations shown to user for selection"
    ),

    # Pipelines - results for user
    "execute_pipeline_stages": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.8,
        category="pipeline"
    ),
    "run_image_to_video_pipeline": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.8,
        category="pipeline",
        description="Final video shown to user"
    ),

    # Model Management - assistant use
    "search_civitai": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.6,
        category="models",
        description="Search results may be shown to user"
    ),
    "download_model": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="models",
        description="Self-healing when models missing"
    ),

    # Analysis Tools - mixed use
    "get_image_dimensions": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="analysis"
    ),
    "detect_objects": ToolAnnotation(
        audience=["assistant"],
        priority=0.6,
        category="analysis",
        description="Validation before expensive operations"
    ),
    "get_video_info": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="analysis"
    ),

    # QA Tools - assistant use primarily
    "qa_output": ToolAnnotation(
        audience=["assistant"],
        priority=0.7,
        category="qa",
        description="Auto-check quality before showing to user"
    ),
    "check_vlm_available": ToolAnnotation(
        audience=["assistant"],
        priority=0.4,
        category="qa"
    ),

    # Style Learning - both audiences
    "record_generation": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="style_learning"
    ),
    "rate_generation": ToolAnnotation(
        audience=["user"],
        priority=0.6,
        category="style_learning",
        description="User provides ratings"
    ),
    "suggest_prompt_enhancement": ToolAnnotation(
        audience=["assistant", "user"],
        priority=0.7,
        category="style_learning",
        description="Suggestions shown to user"
    ),
    "find_similar_prompts": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="style_learning"
    ),
    "get_best_seeds_for_style": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="style_learning"
    ),
    "save_style_preset": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="style_learning"
    ),
    "get_style_preset": ToolAnnotation(
        audience=["assistant"],
        priority=0.5,
        category="style_learning"
    ),
    "list_style_presets": ToolAnnotation(
        audience=["user", "assistant"],
        priority=0.5,
        category="style_learning"
    ),
    "get_style_learning_stats": ToolAnnotation(
        audience=["user"],
        priority=0.4,
        category="style_learning",
        description="Stats shown to user"
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
