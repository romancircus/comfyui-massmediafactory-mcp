# IMPLEMENTATION_PLAN.md - First Feature Implementation

**Author:** KIMI
**Repository:** comfyui-massmediafactory-mcp
**Date:** February 2026
**Feature:** Type Safety Overhaul + Model Registry Consolidation
**Effort:** 14 hours

---

## Why This Feature First?

### The Critical Path Argument

1. **Foundation for Everything:** Type safety and centralized model definitions are prerequisites for nearly all other improvements
2. **Prevents Technical Debt:** Every new feature added without types adds more debt
3. **Immediate Value:** Fixes 100+ type errors preventing crashes
4. **Developer Experience:** Enables IDE autocomplete and refactoring
5. **Low Risk:** No behavioral changes, only type annotations

### Comparison with Alternatives

| Alternative | Why Not First |
|-------------|---------------|
| Token reduction | Already in progress (Claude's plan) |
| New model support | Requires model registry first |
| Progress streaming | Nice-to-have, not blocking |
| LoRA support | Complex, needs solid foundation |
| Error messages | Depends on proper error types |

---

## Implementation Overview

### Two-Phase Approach

**Phase 1:** Type Safety Overhaul (8 hours)
**Phase 2:** Model Registry Consolidation (6 hours)

---

## Phase 1: Type Safety Overhaul

### Current Problems

```python
# server.py - 50+ type errors
@mcp.tool()
def regenerate(
    asset_id: str,
    prompt: str = None,  # ERROR: None not assignable to str
    negative_prompt: str = None,
    seed: int = None,    # ERROR: None not assignable to int
    steps: int = None,
    cfg: float = None,   # ERROR: None not assignable to float
):
```

### Step-by-Step Implementation

#### Step 1.1: Create Type Definitions Module (1 hour)

**File:** `src/comfyui_massmediafactory_mcp/types.py`

```python
"""Type definitions for comfyui-massmediafactory-mcp."""

from typing import TypedDict, Optional, Literal, Union, Any
from typing import List, Dict

# MCP Response Types
class MCPError(TypedDict):
    error: str
    isError: Literal[True]
    code: str

class MCPSuccess(TypedDict):
    isError: Literal[False]

MCPResponse = Union[MCPError, MCPSuccess, Dict[str, Any]]

# Workflow Types
class WorkflowNode(TypedDict):
    class_type: str
    inputs: Dict[str, Any]

Workflow = Dict[str, WorkflowNode]

# Model Types
ModelType = Literal["checkpoint", "unet", "lora", "vae", "controlnet"]
WorkflowType = Literal["t2i", "t2v", "i2v", "edit"]

# Asset Types
AssetType = Literal["images", "video", "audio"]

# Generation Parameters
class GenerationParams(TypedDict, total=False):
    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    cfg: float
    guidance: float
    width: int
    height: int
    frames: int

# Model Constraints
class CFGConstraint(TypedDict):
    min: float
    max: float
    default: float
    note: str

class ResolutionConstraint(TypedDict):
    divisible_by: int
    native: List[int]
    max: Optional[List[int]]
    note: str

class ModelConstraint(TypedDict):
    display_name: str
    type: str
    cfg: CFGConstraint
    resolution: ResolutionConstraint
```

#### Step 1.2: Fix server.py Type Annotations (3 hours)

**Changes to make:**

```python
# Before
@mcp.tool()
def regenerate(
    asset_id: str,
    prompt: str = None,
    negative_prompt: str = None,
    seed: int = None,
    steps: int = None,
    cfg: float = None,
):

# After
from typing import Optional
from .types import MCPResponse

@mcp.tool()
def regenerate(
    asset_id: str,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
) -> MCPResponse:
    """Re-run workflow with modified params. Returns new prompt_id."""
```

**Tools to update (with line numbers from server.py):**

| Tool | Line | Change |
|------|------|--------|
| `regenerate` | 159 | Add Optional[], return type |
| `list_assets` | 175 | Add return type |
| `get_asset_metadata` | 193 | Add return type |
| `workflow_library` | 263 | Add return type |
| `batch_execute` | 512 | Add return type |
| `execute_pipeline_stages` | 544 | Add return type |
| `run_image_to_video_pipeline` | 550 | Add return type |
| `run_upscale_pipeline` | 556 | Add return type |
| `qa_output` | 620 | Add return type |
| `style_suggest` | 665 | Add return type |
| `manage_presets` | 693 | Add return type |
| `generate_workflow` | 747 | Add return type |

**Process:**
1. Add imports at top of file (15 min)
2. Update each tool function signature (2.5h)
3. Fix any new type errors that emerge (45 min)

#### Step 1.3: Fix workflow_generator.py Types (2 hours)

**Key changes:**

```python
# Line 193 - resolve_parameters function
def resolve_parameters(
    skeleton: dict,
    model: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    frames: Optional[int] = None,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    guidance: Optional[float] = None,
    **extra_params: Any
) -> Dict[str, Any]:

# Line 441 - generate_workflow function
def generate_workflow(
    model: str,
    workflow_type: str,
    prompt: str,
    negative_prompt: str = "",
    width: Optional[int] = None,
    height: Optional[int] = None,
    frames: Optional[int] = None,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    guidance: Optional[float] = None,
    auto_correct: bool = True,
    validate: bool = True,
    **extra_params: Any
) -> Dict[str, Any]:
```

#### Step 1.4: Fix patterns.py Return Types (1.5 hours)

**Line 1142 issue:**
```python
# Current (error)
def get_node_chain(model: str, task: str) -> List[Dict[str, Any]]:
    # Returns error dict on failure, not list

# Fixed
from typing import Union

def get_node_chain(model: str, task: str) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    # Or raise exception on error
```

**Alternative - use exceptions:**
```python
class PatternNotFoundError(Exception):
    pass

def get_node_chain(model: str, task: str) -> List[Dict[str, Any]]:
    key = (model.lower(), task.lower())
    if key not in NODE_CHAINS:
        raise PatternNotFoundError(f"No node chain for {model}/{task}")
    return copy.deepcopy(NODE_CHAINS[key])
```

#### Step 1.5: Add mypy to CI (0.5 hours)

**File:** `.github/workflows/ci.yml` (or create)

```yaml
name: CI

on: [push, pull_request]

jobs:
  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install mypy
      - run: pip install -e .
      - run: mypy src/comfyui_massmediafactory_mcp/ --strict
```

---

## Phase 2: Model Registry Consolidation

### Current State

**Model constraints in 3 places:**
1. `patterns.py` lines 22-303: MODEL_CONSTRAINTS
2. `topology_validator.py`: MODEL_CONSTRAINTS (duplicated)
3. `workflow_generator.py` lines 77-144: MODEL_DEFAULTS

**Problems:**
- Adding a model requires 7+ file changes
- Risk of inconsistencies
- No validation that all pieces exist

### Step-by-Step Implementation

#### Step 2.1: Create Unified Model Registry (2 hours)

**File:** `src/comfyui_massmediafactory_mcp/model_registry.py`

```python
"""Centralized model definitions - single source of truth."""

from typing import TypedDict, Optional, List, Dict, Any
from pathlib import Path

class ModelDefinition(TypedDict):
    """Complete definition for a supported model."""

    # Identity
    id: str
    display_name: str
    model_type: str  # "image", "video", "edit"

    # Constraints
    cfg_min: float
    cfg_max: float
    cfg_default: float
    resolution_divisor: int
    resolution_min: int
    resolution_max: int
    resolution_native: List[int]

    # Video-specific (optional)
    frames_default: Optional[int]
    frames_max: Optional[int]
    frames_formula: Optional[str]

    # Generation defaults
    steps_default: int
    steps_min: int
    steps_max: int
    width_default: int
    height_default: int

    # Workflow
    skeleton_key: str  # Key in WORKFLOW_SKELETONS
    required_nodes: List[str]
    forbidden_nodes: Dict[str, str]

    # Aliases
    aliases: List[str]  # Alternative names ("flux", "flux2")

# Registry
MODELS: Dict[str, ModelDefinition] = {
    "flux2": {
        "id": "flux2",
        "display_name": "FLUX.2",
        "model_type": "image",
        "cfg_min": 2.5,
        "cfg_max": 5.0,
        "cfg_default": 3.5,
        "resolution_divisor": 16,
        "resolution_min": 256,
        "resolution_max": 2048,
        "resolution_native": [1024, 1024],
        "frames_default": None,
        "frames_max": None,
        "frames_formula": None,
        "steps_default": 20,
        "steps_min": 15,
        "steps_max": 50,
        "width_default": 1024,
        "height_default": 1024,
        "skeleton_key": "flux2_txt2img",
        "required_nodes": ["UNETLoader", "DualCLIPLoader", "VAELoader"],
        "forbidden_nodes": {
            "KSampler": "Use SamplerCustomAdvanced",
            "CheckpointLoaderSimple": "Use UNETLoader"
        },
        "aliases": ["flux", "flux2", "flux.2"]
    },

    "ltx2": {
        "id": "ltx2",
        "display_name": "LTX-Video 2.0",
        "model_type": "video",
        "cfg_min": 2.5,
        "cfg_max": 4.0,
        "cfg_default": 3.0,
        "resolution_divisor": 8,
        "resolution_min": 256,
        "resolution_max": 1920,
        "resolution_native": [768, 512],
        "frames_default": 97,
        "frames_max": 121,
        "frames_formula": "8n+1",
        "steps_default": 30,
        "steps_min": 25,
        "steps_max": 35,
        "width_default": 768,
        "height_default": 512,
        "skeleton_key": "ltx2_txt2vid",
        "required_nodes": ["LTXVLoader", "SamplerCustom", "LTXVScheduler"],
        "forbidden_nodes": {
            "KSampler": "Use SamplerCustom",
            "CheckpointLoaderSimple": "Use LTXVLoader"
        },
        "aliases": ["ltx", "ltx2", "ltx-video"]
    },

    # ... more models
}

# Build alias lookup
ALIAS_TO_MODEL: Dict[str, str] = {}
for model_id, definition in MODELS.items():
    for alias in definition["aliases"]:
        ALIAS_TO_MODEL[alias.lower()] = model_id

def get_model(model_id_or_alias: str) -> Optional[ModelDefinition]:
    """Get model definition by ID or alias."""
    key = model_id_or_alias.lower()

    # Direct lookup
    if key in MODELS:
        return MODELS[key]

    # Alias lookup
    if key in ALIAS_TO_MODEL:
        return MODELS[ALIAS_TO_MODEL[key]]

    return None

def list_models_by_type(model_type: Optional[str] = None) -> List[ModelDefinition]:
    """List all models, optionally filtered by type."""
    if model_type is None:
        return list(MODELS.values())
    return [m for m in MODELS.values() if m["model_type"] == model_type]

def validate_model_params(model_id: str, params: Dict[str, Any]) -> List[str]:
    """Validate parameters against model constraints."""
    errors = []
    model = get_model(model_id)

    if not model:
        errors.append(f"Unknown model: {model_id}")
        return errors

    # Check CFG
    if "cfg" in params:
        cfg = params["cfg"]
        if cfg < model["cfg_min"] or cfg > model["cfg_max"]:
            errors.append(
                f"CFG {cfg} out of range [{model['cfg_min']}, {model['cfg_max']}]"
            )

    # Check resolution
    for dim in ["width", "height"]:
        if dim in params:
            val = params[dim]
            divisor = model["resolution_divisor"]
            if val % divisor != 0:
                errors.append(f"{dim} {val} not divisible by {divisor}")

    # Check frames for video
    if model["model_type"] == "video" and "frames" in params:
        frames = params["frames"]
        if model["frames_formula"] == "8n+1":
            if (frames - 1) % 8 != 0:
                errors.append(f"Frames {frames} must be 8n+1")

    return errors
```

#### Step 2.2: Refactor patterns.py to Use Registry (1.5 hours)

**Changes:**

```python
# patterns.py
from .model_registry import get_model, MODELS

# Replace MODEL_CONSTRAINTS with:
def get_model_constraints(model: str) -> Dict[str, Any]:
    """Get constraints from unified registry."""
    model_def = get_model(model)
    if not model_def:
        return {"error": f"Unknown model: {model}"}

    # Convert ModelDefinition to old format for compatibility
    return {
        "display_name": model_def["display_name"],
        "type": model_def["model_type"],
        "cfg": {
            "min": model_def["cfg_min"],
            "max": model_def["cfg_max"],
            "default": model_def["cfg_default"]
        },
        "resolution": {
            "divisible_by": model_def["resolution_divisor"],
            "native": model_def["resolution_native"]
        }
        # ... etc
    }
```

#### Step 2.3: Refactor workflow_generator.py (1.5 hours)

**Changes:**

```python
# workflow_generator.py
from .model_registry import get_model, validate_model_params

# Replace MODEL_DEFAULTS usage:
def resolve_parameters(skeleton: dict, model: str, ...) -> Dict[str, Any]:
    model_def = get_model(model)
    if not model_def:
        raise ValueError(f"Unknown model: {model}")

    # Use model_def instead of MODEL_DEFAULTS
    defaults = {
        "width": model_def["width_default"],
        "height": model_def["height_default"],
        "steps": model_def["steps_default"],
        "cfg": model_def["cfg_default"]
    }

    # ... rest of function
```

#### Step 2.4: Refactor topology_validator.py (1 hour)

**Changes:**

```python
# topology_validator.py
from .model_registry import get_model

# Replace MODEL_CONSTRAINTS lookups:
def validate_topology(workflow_json: str, model: Optional[str] = None):
    model_def = get_model(model) if model else None

    if model_def:
        # Use model_def for validation
        errors = validate_model_params(model, extracted_params)
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_types.py
import pytest
from comfyui_massmediafactory_mcp.types import GenerationParams

def test_generation_params_optional():
    """Test that optional params work correctly."""
    params: GenerationParams = {"prompt": "test"}
    assert params["prompt"] == "test"
    assert "seed" not in params

# tests/test_model_registry.py
from comfyui_massmediafactory_mcp.model_registry import get_model, validate_model_params

def test_get_model_by_alias():
    """Test that aliases resolve correctly."""
    assert get_model("flux") == get_model("flux2")
    assert get_model("ltx") == get_model("ltx2")

def test_validate_model_params():
    """Test parameter validation."""
    errors = validate_model_params("ltx2", {"frames": 100})  # Not 8n+1
    assert len(errors) == 1
    assert "8n+1" in errors[0]
```

### Integration Tests

```python
# tests/test_type_integration.py
import subprocess

def test_mypy_passes():
    """Ensure mypy finds no errors."""
    result = subprocess.run(
        ["mypy", "src/comfyui_massmediafactory_mcp/", "--strict"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"mypy errors:\n{result.stdout}"
```

---

## Verification Checklist

### Phase 1 Verification

- [ ] `mypy src/comfyui_massmediafactory_mcp/ --strict` passes with 0 errors
- [ ] All tool functions have proper type annotations
- [ ] `Optional[]` used for all optional parameters
- [ ] Return types specified for all public functions
- [ ] CI includes mypy check

### Phase 2 Verification

- [ ] All model info in single registry file
- [ ] `patterns.py` uses registry
- [ ] `workflow_generator.py` uses registry
- [ ] `topology_validator.py` uses registry
- [ ] Aliases resolve correctly
- [ ] Adding new model requires only registry update

### Integration Verification

- [ ] All existing tests pass
- [ ] New type-related tests pass
- [ ] No behavioral changes (only types)
- [ ] Documentation updated

---

## Rollback Plan

If issues arise:

1. **Git revert:** `git revert HEAD` to undo changes
2. **Type errors only:** Remove `--strict` from mypy temporarily
3. **Registry issues:** Keep old definitions as fallback

---

## Success Criteria

| Metric | Before | After |
|--------|--------|-------|
| mypy errors | 100+ | 0 |
| Files with types | 3 | 15+ |
| Model definition locations | 3 | 1 |
| Time to add new model | 2 hours | 15 minutes |
| IDE autocomplete | Partial | Full |
| CI type checking | No | Yes |

---

## Time Summary

| Phase | Task | Hours |
|-------|------|-------|
| 1.1 | Create types module | 1 |
| 1.2 | Fix server.py types | 3 |
| 1.3 | Fix workflow_generator.py | 2 |
| 1.4 | Fix patterns.py | 1.5 |
| 1.5 | Add CI | 0.5 |
| 2.1 | Create model registry | 2 |
| 2.2 | Refactor patterns.py | 1.5 |
| 2.3 | Refactor workflow_generator.py | 1.5 |
| 2.4 | Refactor topology_validator.py | 1 |
| **Total** | | **14** |

---

## Post-Implementation

### Immediate Benefits
1. **No more type-related crashes**
2. **Full IDE support** (autocomplete, refactoring)
3. **Easier model additions**
4. **CI catches type errors**

### Enables Next Features
- New model support (uses registry)
- Better error messages (typed errors)
- Plugin system (typed interfaces)
- API documentation (from types)

### Documentation Updates Needed
- [ ] CLAUDE.md: Add type usage examples
- [ ] README.md: Update contributing guide
- [ ] docs/development.md: Type system guide
