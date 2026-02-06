"""
JSON Schemas for MCP Tool Inputs

Explicit JSON Schema definitions per MCP specification requirements.
These schemas define the expected structure of tool inputs.
"""

from typing import TypedDict, Optional, List, Dict, Any
from typing_extensions import Required, NotRequired

# =============================================================================
# Workflow Schema
# =============================================================================


class WorkflowNodeInput(TypedDict, total=False):
    """Input connection or value for a workflow node."""

    pass  # Can be str, int, float, bool, or [node_id, slot_index]


class WorkflowNode(TypedDict):
    """A single node in a ComfyUI workflow."""

    class_type: Required[str]
    inputs: Required[Dict[str, Any]]
    _meta: NotRequired[Dict[str, Any]]


# Workflow is a dict of node_id -> WorkflowNode
Workflow = Dict[str, WorkflowNode]


# =============================================================================
# JSON Schema Definitions
# =============================================================================

WORKFLOW_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "description": "ComfyUI workflow JSON with node definitions",
    "additionalProperties": {
        "type": "object",
        "required": ["class_type", "inputs"],
        "properties": {
            "class_type": {"type": "string", "description": "Node type (e.g., 'KSampler', 'UNETLoader')"},
            "inputs": {
                "type": "object",
                "description": "Node inputs - can be values or links [node_id, slot]",
                "additionalProperties": True,
            },
            "_meta": {"type": "object", "description": "Optional metadata (stripped on submit)"},
        },
    },
}


EXECUTE_WORKFLOW_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["workflow"],
    "properties": {
        "workflow": {**WORKFLOW_SCHEMA, "description": "The workflow JSON with node definitions"},
        "client_id": {
            "type": "string",
            "default": "massmediafactory",
            "description": "Optional identifier for tracking",
        },
    },
}


REGENERATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["asset_id"],
    "properties": {
        "asset_id": {"type": "string", "description": "The asset ID to regenerate from"},
        "prompt": {"type": "string", "description": "New prompt (optional)"},
        "negative_prompt": {"type": "string", "description": "New negative prompt (optional)"},
        "seed": {"type": "integer", "description": "New seed. -1 to keep original, null for random"},
        "steps": {"type": "integer", "minimum": 1, "maximum": 150, "description": "New step count (optional)"},
        "cfg": {"type": "number", "minimum": 0, "maximum": 30, "description": "New CFG scale (optional)"},
    },
}


LIST_ASSETS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "session_id": {"type": "string", "description": "Filter by session (optional)"},
        "asset_type": {
            "type": "string",
            "enum": ["images", "video", "audio"],
            "description": "Filter by type (optional)",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "default": 20,
            "description": "Maximum results per page",
        },
        "cursor": {"type": "string", "description": "Pagination cursor from previous response"},
    },
}


VALIDATE_WORKFLOW_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["workflow"],
    "properties": {"workflow": {**WORKFLOW_SCHEMA, "description": "The workflow JSON to validate"}},
}


CREATE_WORKFLOW_FROM_TEMPLATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["template_name", "parameters"],
    "properties": {
        "template_name": {"type": "string", "description": "Name of the template (e.g., 'qwen_txt2img')"},
        "parameters": {
            "type": "object",
            "description": "Dict of parameter values (e.g., {'PROMPT': 'a dragon', 'SEED': 123})",
            "additionalProperties": True,
        },
    },
}


EXECUTE_BATCH_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["workflow", "parameter_sets"],
    "properties": {
        "workflow": {**WORKFLOW_SCHEMA, "description": "Base workflow with {{PLACEHOLDER}} fields"},
        "parameter_sets": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
            "description": "List of parameter dicts for each execution",
        },
        "parallel": {
            "type": "integer",
            "minimum": 1,
            "maximum": 4,
            "default": 1,
            "description": "Max concurrent executions",
        },
        "timeout_per_job": {
            "type": "integer",
            "minimum": 60,
            "maximum": 3600,
            "default": 600,
            "description": "Timeout per job in seconds",
        },
    },
}


QA_OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["asset_id", "prompt"],
    "properties": {
        "asset_id": {"type": "string", "description": "The asset to evaluate"},
        "prompt": {"type": "string", "description": "Original generation prompt for comparison"},
        "checks": {
            "type": "array",
            "items": {"type": "string", "enum": ["prompt_match", "artifacts", "faces", "text", "composition"]},
            "description": "List of checks to perform",
        },
        "vlm_model": {"type": "string", "description": "VLM to use (default: qwen2.5-vl:7b)"},
    },
}


RECORD_GENERATION_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["prompt", "model", "seed"],
    "properties": {
        "prompt": {"type": "string", "description": "The generation prompt"},
        "model": {"type": "string", "description": "Model used (e.g., 'flux2-dev')"},
        "seed": {"type": "integer", "description": "Random seed"},
        "parameters": {"type": "object", "additionalProperties": True, "description": "Full parameter dict"},
        "negative_prompt": {"type": "string", "default": "", "description": "Negative prompt if any"},
        "rating": {"type": "number", "minimum": 0, "maximum": 1, "description": "User rating 0.0-1.0"},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Style tags (e.g., ['anime', 'portrait'])",
        },
        "outcome": {
            "type": "string",
            "enum": ["success", "failed", "regenerated"],
            "default": "success",
            "description": "Generation outcome",
        },
        "qa_score": {"type": "number", "minimum": 0, "maximum": 1, "description": "Automated QA score if available"},
        "notes": {"type": "string", "description": "Additional notes"},
    },
}


DOWNLOAD_MODEL_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["url", "model_type"],
    "properties": {
        "url": {"type": "string", "format": "uri", "description": "Download URL (Civitai or HuggingFace)"},
        "model_type": {
            "type": "string",
            "enum": ["checkpoint", "unet", "lora", "vae", "controlnet", "clip", "upscaler", "embedding"],
            "description": "Where to save the model",
        },
        "filename": {"type": "string", "description": "Target filename (auto-detected if not provided)"},
        "overwrite": {"type": "boolean", "default": False, "description": "Replace existing file"},
    },
}


# =============================================================================
# Output Schemas (June 2025 MCP Spec)
# =============================================================================

EXECUTE_WORKFLOW_OUTPUT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "prompt_id": {"type": "string", "description": "Unique identifier for tracking the workflow"},
        "status": {"type": "string", "enum": ["queued", "running", "completed", "error"]},
    },
    "required": ["prompt_id"],
}


WAIT_FOR_COMPLETION_OUTPUT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["completed", "error", "timeout"]},
        "outputs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["image", "video", "audio"]},
                    "filename": {"type": "string"},
                    "asset_id": {"type": "string"},
                    "url": {"type": "string", "format": "uri"},
                },
            },
        },
        "duration_seconds": {"type": "number"},
    },
}


LIST_ASSETS_OUTPUT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "assets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "type": {"type": "string"},
                    "filename": {"type": "string"},
                    "created_at": {"type": "string", "format": "date-time"},
                },
            },
        },
        "nextCursor": {"type": ["string", "null"]},
        "total": {"type": "integer"},
    },
    "required": ["assets", "total"],
}


VALIDATE_WORKFLOW_OUTPUT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "error": {"type": "string"},
                    "severity": {"type": "string", "enum": ["error", "warning"]},
                },
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "suggestions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["valid"],
}


QA_OUTPUT_OUTPUT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "issues": {"type": "array", "items": {"type": "string"}},
        "recommendation": {"type": "string", "enum": ["accept", "regenerate", "tweak_prompt"]},
    },
    "required": ["passed", "score", "recommendation"],
}


DETECT_OBJECTS_OUTPUT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "detected": {"type": "array", "items": {"type": "string"}},
        "not_detected": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1}},
        "description": {"type": "string"},
    },
    "required": ["detected", "not_detected", "confidence"],
}


IMAGE_DIMENSIONS_OUTPUT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "width": {"type": "integer"},
        "height": {"type": "integer"},
        "aspect_ratio": {"type": "string"},
        "aspect_ratio_decimal": {"type": "number"},
        "orientation": {"type": "string", "enum": ["landscape", "portrait", "square"]},
        "recommended_video_size": {
            "type": "object",
            "properties": {"width": {"type": "integer"}, "height": {"type": "integer"}},
        },
    },
    "required": ["width", "height", "aspect_ratio"],
}


RECOMMEND_MODEL_OUTPUT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "model": {"type": "string"},
        "model_file": {"type": "string"},
        "settings": {
            "type": "object",
            "properties": {
                "cfg": {"type": "number"},
                "steps": {"type": "integer"},
                "sampler": {"type": "string"},
                "scheduler": {"type": "string"},
            },
        },
        "fits_vram": {"type": "boolean"},
        "vram_required_gb": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["model", "settings"],
}


MCP_ERROR_OUTPUT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "error": {"type": "string", "description": "Human-readable error message"},
        "code": {"type": "string", "description": "Error code"},
        "isError": {"type": "boolean", "const": True},
        "details": {"type": "object", "additionalProperties": True},
    },
    "required": ["error", "isError"],
}


# =============================================================================
# Schema Registry
# =============================================================================

TOOL_SCHEMAS = {
    "execute_workflow": EXECUTE_WORKFLOW_SCHEMA,
    "regenerate": REGENERATE_SCHEMA,
    "list_assets": LIST_ASSETS_SCHEMA,
    "validate_workflow": VALIDATE_WORKFLOW_SCHEMA,
    "create_workflow_from_template": CREATE_WORKFLOW_FROM_TEMPLATE_SCHEMA,
    "execute_batch_workflows": EXECUTE_BATCH_SCHEMA,
    "qa_output": QA_OUTPUT_SCHEMA,
    "record_generation": RECORD_GENERATION_SCHEMA,
    "download_model": DOWNLOAD_MODEL_SCHEMA,
}


OUTPUT_SCHEMAS = {
    "execute_workflow": EXECUTE_WORKFLOW_OUTPUT,
    "wait_for_completion": WAIT_FOR_COMPLETION_OUTPUT,
    "list_assets": LIST_ASSETS_OUTPUT,
    "validate_workflow": VALIDATE_WORKFLOW_OUTPUT,
    "qa_output": QA_OUTPUT_OUTPUT,
    "detect_objects": DETECT_OBJECTS_OUTPUT,
    "get_image_dimensions": IMAGE_DIMENSIONS_OUTPUT,
    "recommend_model": RECOMMEND_MODEL_OUTPUT,
    "_error": MCP_ERROR_OUTPUT,
}


def get_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get JSON schema for a tool's inputs."""
    return TOOL_SCHEMAS.get(tool_name)


def get_output_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get JSON schema for a tool's outputs."""
    return OUTPUT_SCHEMAS.get(tool_name)


def list_schemas() -> List[str]:
    """List all tools with defined input schemas."""
    return list(TOOL_SCHEMAS.keys())


def list_output_schemas() -> List[str]:
    """List all tools with defined output schemas."""
    return list(OUTPUT_SCHEMAS.keys())
