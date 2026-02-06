# Meta-Template System Design Specification

## Overview

The Meta-Template system generates complete, validated ComfyUI workflow JSONs from high-level parameters. It bridges the gap between LLM understanding and valid workflow output.

## Problem Statement

LLMs often generate invalid workflows due to:
1. Wrong node names (hallucination)
2. Invalid connections
3. Wrong parameter values (frame count, resolution, CFG)
4. Using KSampler for video models

The meta-template system solves this by:
1. Using pre-validated skeleton templates
2. Injecting parameters into placeholders
3. Validating before returning
4. Auto-correcting where possible

## API Design

### Primary Function: `generate_workflow`

```python
generate_workflow(
    model: str,           # "ltx", "flux", "wan", "qwen"
    workflow_type: str,   # "t2v", "i2v", "t2i"
    prompt: str,
    negative_prompt: str = "",
    width: int = None,    # Auto-set from skeleton defaults
    height: int = None,
    frames: int = None,   # For video
    seed: int = None,     # Random if not provided
    steps: int = None,
    cfg: float = None,
    **extra_params        # Model-specific params
) -> dict
```

### Return Format

```json
{
    "workflow": { ... complete JSON ... },
    "parameters_used": {
        "PROMPT": "...",
        "WIDTH": 1024,
        ...
    },
    "auto_corrections": [
        {"field": "WIDTH", "from": 1000, "to": 1024, "reason": "..."}
    ],
    "validation": {
        "valid": true,
        "warnings": []
    },
    "skeleton_used": "ltx_video_t2v.json"
}
```

## Implementation Architecture

### 1. Skeleton Loading

```
docs/library/skeletons/
├── ltx_video_t2v.json
├── ltx_video_i2v.json
├── flux_dev_t2i.json
├── wan_t2v.json
└── qwen_t2i.json
```

Each skeleton contains:
- `_meta`: Metadata including model, workflow type, notes
- `nodes`: List of node definitions with types
- `connections`: Connection patterns
- `defaults`: Default parameter values

### 2. Parameter Resolution

Order of precedence:
1. User-provided parameters
2. Skeleton defaults
3. Model constraints (from PARAMETER_RULES)

### 3. Placeholder Injection

Skeletons use `{{PLACEHOLDER}}` syntax. Generator replaces:
- `{{PROMPT}}` → user prompt
- `{{NEGATIVE}}` → negative prompt
- `{{SEED}}` → seed value
- `{{WIDTH}}` / `{{HEIGHT}}` → dimensions
- `{{FRAMES}}` / `{{LENGTH}}` → frame count
- `{{STEPS}}` → sampling steps
- `{{CFG}}` → CFG scale

### 4. Workflow Expansion

Convert skeleton format to ComfyUI API format:

```python
# From skeleton:
{
    "nodes": [
        {"id": "1", "type": "UNETLoader", ...}
    ],
    "connections": [
        {"from": "1.MODEL", "to": "2.model"}
    ],
    "defaults": {"1.unet_name": "flux1-dev.safetensors"}
}

# To API format:
{
    "1": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "flux1-dev.safetensors"
        }
    },
    "2": {
        "class_type": "SamplerCustom",
        "inputs": {
            "model": ["1", 0],
            ...
        }
    }
}
```

### 5. Pre-flight Validation

Before returning, run through `topology_validator.validate_topology()`:
- If errors found, attempt auto-correction
- If still invalid, return error with suggestions

### 6. Auto-Correction Rules

| Parameter | Rule | Example |
|-----------|------|---------|
| Width/Height | Round to nearest divisor | 1000 → 1024 (FLUX /16) |
| LTX Frames | Round to nearest 8n+1 | 100 → 97 |
| CFG | Clamp to model range | 7.5 → 3.0 (LTX) |
| Seed | Generate random if -1 or None | -1 → 123456789 |

## Model-Specific Behaviors

### FLUX
- Uses `DualCLIPLoader` not `CLIPLoader`
- Uses `FluxGuidance` not `cfg` parameter
- Resolution divisible by 16
- Uses `EmptySD3LatentImage`

### LTX-Video
- Frames must be 8n+1
- Uses `SamplerCustom` not `KSampler`
- Uses `LTXVScheduler` not `BasicScheduler`
- Uses `LTXVConditioning` wrapper
- CFG around 3.0

### Wan 2.1/2.6
- Uses `WanVideoModelLoader`
- Uses `WanVideoDecode` (NOT `WanVAEDecode`)
- Uses `WanVideoSampler` (NOT `SamplerCustom`)
- CFG around 5.0
- `negative_default` from model_registry: temporal negatives (anti-static, anti-morph)

### Qwen
- Uses `ModelSamplingAuraFlow`
- KSampler OK for images
- CFG around 3.0

## Error Handling

### Recoverable Errors
- Invalid resolution → auto-correct
- Invalid frame count → auto-correct
- Missing optional params → use defaults

### Non-Recoverable Errors
- Unknown model type
- Missing required param (prompt)
- Skeleton file not found

## Usage Examples

### Simple Text-to-Video

```python
result = generate_workflow(
    model="ltx",
    workflow_type="t2v",
    prompt="A cat walking on a beach at sunset"
)
# Returns complete 97-frame LTX workflow
```

### High-Resolution Image

```python
result = generate_workflow(
    model="flux",
    workflow_type="t2i",
    prompt="A detailed portrait",
    width=1024,
    height=1536
)
# Returns FLUX workflow with proper FluxGuidance
```

### Image-to-Video with Custom Params

```python
result = generate_workflow(
    model="ltx",
    workflow_type="i2v",
    prompt="The cat starts running",
    frames=121,  # 5 seconds
    seed=42
)
# Returns LTX I2V workflow
```

## Integration with MCP

### Tool Definition

```python
@mcp.tool()
def generate_workflow(
    model: str,
    workflow_type: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = None,
    height: int = None,
    frames: int = None,
    seed: int = None,
    steps: int = None,
    cfg: float = None,
) -> dict:
    """Generate a complete, validated workflow from parameters."""
    ...
```

### Workflow

1. LLM calls `generate_workflow()` with high-level params
2. System returns complete workflow JSON
3. LLM can optionally validate with `validate_topology()`
4. LLM executes with `execute_workflow()`

## Future Enhancements

1. **Chained Generation**: Generate image then video in one call
2. **Style Presets**: Apply saved style presets to prompts
3. **Dynamic LoRA**: Auto-inject relevant LoRAs based on prompt
4. **Quality Tiers**: Quick/Standard/High quality presets

## Changelog

- **January 2026**: Initial design specification
