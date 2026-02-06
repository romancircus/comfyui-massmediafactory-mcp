"""
Comprehensive Type Definitions for ComfyUI MassMediaFactory MCP

Provides full TypedDict definitions for all return types, enabling:
- Complete IDE autocomplete
- mypy strict mode compliance
- Better documentation
- Type safety

Usage:
    from comfyui_massmediafactory_mcp.types import WorkflowResult, AssetOutput

    def my_function() -> WorkflowResult:
        return {"prompt_id": "abc", "status": "completed", "outputs": []}
"""

from typing import (
    TypedDict,
    NotRequired,
    Literal,
    List,
    Dict,
    Any,
    Union,
)


# =============================================================================
# Core Asset Types
# =============================================================================


class AssetOutput(TypedDict):
    """Output asset from workflow execution."""

    asset_id: str
    asset_type: Literal["image", "video", "audio", "model"]
    url: str
    filename: str
    width: NotRequired[int]
    height: NotRequired[int]
    duration: NotRequired[float]  # For video/audio
    fps: NotRequired[float]  # For video


class AssetMetadata(TypedDict):
    """Full metadata for an asset including workflow."""

    asset_id: str
    asset_type: Literal["image", "video", "audio", "model"]
    created_at: str  # ISO 8601 timestamp
    workflow: Dict[str, Any]
    parameters: Dict[str, Any]
    prompt: NotRequired[str]
    model: NotRequired[str]
    seed: NotRequired[int]


# =============================================================================
# Workflow Types
# =============================================================================


class WorkflowResult(TypedDict):
    """Result from workflow execution."""

    prompt_id: str
    status: Literal["queued", "running", "completed", "error"]
    outputs: List[AssetOutput]
    error: NotRequired[str]
    message: NotRequired[str]


class WorkflowStatus(TypedDict):
    """Status of a workflow execution."""

    prompt_id: str
    status: Literal["pending", "running", "completed", "error", "cancelled"]
    progress: NotRequired[float]  # 0.0 to 1.0
    outputs: NotRequired[List[AssetOutput]]
    error: NotRequired[str]
    execution_time_seconds: NotRequired[float]


class QueueStatus(TypedDict):
    """Status of the execution queue."""

    queue_running: List[str]  # prompt_ids
    queue_pending: List[str]  # prompt_ids
    queue_completed: int
    queue_failed: int


# =============================================================================
# Model Types
# =============================================================================


class ModelInfo(TypedDict):
    """Information about a model."""

    name: str
    type: Literal["checkpoint", "unet", "lora", "vae", "controlnet", "clip"]
    path: str
    size_bytes: int
    modified: str  # ISO 8601 timestamp


class ModelConstraints(TypedDict):
    """Constraints for a specific model."""

    min_cfg: NotRequired[float]
    max_cfg: NotRequired[float]
    default_cfg: NotRequired[float]
    min_resolution: NotRequired[int]
    max_resolution: NotRequired[int]
    resolution_step: NotRequired[int]
    min_frames: NotRequired[int]
    max_frames: NotRequired[int]
    frame_step: NotRequired[int]


# =============================================================================
# Node Types
# =============================================================================


class NodeInfo(TypedDict):
    """Information about a ComfyUI node type."""

    class_type: str
    category: str
    description: str
    inputs: Dict[str, Any]
    outputs: List[Dict[str, Any]]


class NodeChainEntry(TypedDict):
    """Entry in a node chain."""

    id: str
    class_type: str
    inputs: Dict[str, List]  # input_name -> [source_id, slot]
    outputs: Dict[str, int]  # output_name -> slot_index


# =============================================================================
# Template Types
# =============================================================================


class TemplateMetadata(TypedDict):
    """Metadata for a workflow template."""

    name: str
    description: str
    model: str
    type: Literal["t2i", "t2v", "i2v", "edit"]
    parameters: List[str]
    defaults: Dict[str, Any]


class Template(TypedDict):
    """Complete workflow template."""

    name: str
    workflow: Dict[str, Any]
    _meta: TemplateMetadata


# =============================================================================
# Pattern Types
# =============================================================================


class PatternInfo(TypedDict):
    """Information about a workflow pattern."""

    model: str
    task: str
    description: str
    type: str
    parameters: List[str]


class ValidationResult(TypedDict):
    """Result of workflow validation."""

    valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    model_detected: NotRequired[str]
    workflow: NotRequired[Dict[str, Any]]  # If auto_fix applied


# =============================================================================
# VRAM Types
# =============================================================================


class VRAMEstimate(TypedDict):
    """VRAM usage estimate."""

    estimated_vram_gb: float
    model_vram_gb: float
    overhead_gb: float
    peak_vram_gb: float
    fits_available: bool
    available_vram_gb: float


class ModelFitCheck(TypedDict):
    """Result of model fit check."""

    fits: bool
    model_size_gb: float
    available_vram_gb: float
    precision: str
    notes: NotRequired[str]


# =============================================================================
# SOTA Types
# =============================================================================


class SOTAModel(TypedDict):
    """SOTA model recommendation."""

    name: str
    category: Literal["image", "video", "audio", "text"]
    vram_gb: float
    best_for: str
    recommended: bool


class SOTASettings(TypedDict):
    """Optimal settings for a model."""

    model: str
    recommended_cfg: float
    recommended_steps: int
    recommended_resolution: str
    notes: str


# =============================================================================
# Rate Limiting Types
# =============================================================================


class RateLimitStatus(TypedDict):
    """Rate limiting status."""

    requests_per_minute: int
    requests_remaining: int
    current_usage: int
    reset_at: str  # ISO 8601 timestamp
    reset_in_seconds: float
    window_seconds: int
    per_tool: bool
    usage_percent: float
    warning: NotRequired[str]
    tool: NotRequired[str]


class ToolRateStatus(TypedDict):
    """Rate status for a specific tool."""

    tool: str
    requests_remaining: int
    current_usage: int
    reset_in_seconds: float
    usage_percent: float


class AllToolsRateStatus(TypedDict):
    """Rate status for all tools."""

    tools: List[ToolRateStatus]
    global_limit: int
    window_seconds: int
    per_tool: bool
    total_tools_tracked: int


class RateLimitSummary(TypedDict):
    """Brief rate limit summary for dashboards."""

    status: Literal["ok", "warning", "critical"]
    message: str
    requests_remaining: int
    reset_in_seconds: float
    limit: int
    window_seconds: int


# =============================================================================
# Execution Types
# =============================================================================


class ExecutionResult(TypedDict):
    """Result of a single execution."""

    prompt_id: str
    status: Literal["queued", "running", "completed", "error"]
    outputs: List[AssetOutput]
    error: NotRequired[str]


class BatchResult(TypedDict):
    """Result of batch execution."""

    results: List[ExecutionResult]
    total: int
    succeeded: int
    failed: int
    total_time_seconds: float


# =============================================================================
# QA Types
# =============================================================================


class QAResult(TypedDict):
    """Quality assurance check result."""

    passed: bool
    score: float  # 0.0 to 1.0
    issues: List[str]
    details: Dict[str, Any]


# =============================================================================
# Style Learning Types
# =============================================================================


class GenerationRecord(TypedDict):
    """Recorded generation for style learning."""

    record_id: str
    prompt: str
    model: str
    seed: int
    parameters: Dict[str, Any]
    rating: NotRequired[float]
    tags: NotRequired[List[str]]
    outcome: str
    created_at: str


class StylePreset(TypedDict):
    """Style preset for prompt enhancement."""

    name: str
    description: str
    prompt_additions: str
    negative_additions: NotRequired[str]
    recommended_model: NotRequired[str]
    recommended_params: NotRequired[Dict[str, Any]]


class SimilarPrompt(TypedDict):
    """Similar prompt from style learning."""

    prompt: str
    similarity_score: float
    rating: float
    tags: List[str]


# =============================================================================
# Visualization Types
# =============================================================================


class WorkflowVisualization(TypedDict):
    """Workflow visualization result."""

    mermaid: str  # Mermaid diagram syntax
    url: str  # Mermaid Live Editor URL
    node_count: int
    edge_count: int


class WorkflowSummary(TypedDict):
    """Summary of workflow structure."""

    node_types: Dict[str, int]  # class_type -> count
    total_nodes: int
    parameters: List[str]
    unique_node_types: int


# =============================================================================
# Semantic Search Types
# =============================================================================


class SemanticSearchResult(TypedDict):
    """Result from semantic pattern search."""

    results: List[Dict[str, Any]]  # Pattern matches with scores
    query: str
    query_time_ms: float
    method: Literal["semantic", "keyword_fallback"]
    total_available: int
    cache_stats: NotRequired[Dict[str, Any]]
    note: NotRequired[str]


class PatternMatch(TypedDict):
    """Single pattern match from semantic search."""

    pattern_id: str
    score: float
    description: str
    tags: List[str]
    use_cases: List[str]


# =============================================================================
# Error Types
# =============================================================================


class MCPError(TypedDict):
    """MCP-compliant error response."""

    error: str
    code: str
    isError: Literal[True]
    details: NotRequired[Dict[str, Any]]
    retry_after_seconds: NotRequired[float]


class SuccessResult(TypedDict):
    """Generic success result."""

    success: Literal[True]
    message: NotRequired[str]
    data: NotRequired[Any]


# =============================================================================
# API Response Unions
# =============================================================================

# Common result types that can be success or error
WorkflowResponse = Union[WorkflowResult, MCPError]
AssetResponse = Union[AssetMetadata, MCPError]
ValidationResponse = Union[ValidationResult, MCPError]
BatchResponse = Union[BatchResult, MCPError]
ListResponse = Union[Dict[str, List[Any]], MCPError]  # For list operations
