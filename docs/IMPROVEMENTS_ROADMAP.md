# LLM Reference Documentation - Improvements Roadmap

**Created**: January 2026
**Status**: Post-Implementation Review

---

## Executive Summary

Successfully implemented Gemini's recommended LLM reference documentation architecture across 4 phases:
- Phase 1: Documentation structure (01_MODEL_PATTERNS.md, 02_NODE_LIBRARY.md, 03_PARAMETER_RULES.md)
- Phase 2: MCP resources and tools (get_node_spec, search_patterns)
- Phase 3: Topology validation (validate_topology, auto_correct_workflow)
- Phase 4: Meta-template workflow generation (generate_workflow)

This document outlines prioritized improvements for the next iteration.

---

## Priority 1: Critical Improvements

### 1.1 Add Unit Tests

**Status**: Missing
**Impact**: High
**Effort**: Medium

The new modules lack test coverage:
- `node_specs.py` - No tests
- `reference_docs.py` - No tests
- `topology_validator.py` - No tests
- `workflow_generator.py` - No tests

**Recommended Action**:
```python
# tests/test_topology_validator.py
def test_ltx_frame_validation():
    assert validate_ltx_frames(97)[0] == True  # 8*12+1
    assert validate_ltx_frames(100)[0] == False  # Not 8n+1

def test_flux_resolution_validation():
    assert validate_resolution(1024, 1024, "flux")[0] == True  # /16
    assert validate_resolution(1000, 1000, "flux")[0] == False  # Not /16
```

### 1.2 Add Type Hints to All Functions

**Status**: Partial
**Impact**: Medium
**Effort**: Low

Some functions have type hints, others don't. Inconsistent:
```python
# Missing type hints
def get_model_pattern(model: str) -> dict:  # Has hints ✓

# reference_docs.py search_patterns missing return type
def search_patterns(query: str) -> dict:  # Has hints ✓
```

### 1.3 Handle Edge Cases in Workflow Generator

**Status**: Some gaps
**Impact**: High
**Effort**: Medium

Current edge cases not handled:
1. Empty prompt → Should error, not generate
2. Negative seed values other than -1 → Should normalize
3. Width=0 or Height=0 → Should error
4. Frames < 1 for video → Should error

**Recommended Action**:
```python
def generate_workflow(...):
    if not prompt or not prompt.strip():
        return {"error": "Prompt is required"}
    if width is not None and width <= 0:
        return {"error": "Width must be positive"}
```

---

## Priority 2: Feature Enhancements

### 2.1 Add More Model Support

**Status**: 4 models supported
**Impact**: High
**Effort**: Medium

Missing models:
- SD3 / SD3.5
- HunyuanVideo
- CogVideoX
- Stable Video Diffusion (SVD)
- AnimateDiff

**Recommended Action**:
1. Create skeleton files for each model
2. Add to MODEL_SKELETON_MAP
3. Add constraints to MODEL_CONSTRAINTS
4. Add node specs to NODE_SPECS

### 2.2 Add More Workflow Types

**Status**: T2V, I2V, T2I supported
**Impact**: Medium
**Effort**: Medium

Missing workflow types:
- `upscale` - Image upscaling
- `inpaint` - Inpainting
- `outpaint` - Outpainting
- `controlnet` - ControlNet guided
- `lora` - With LoRA injection
- `vid2vid` - Video-to-video

### 2.3 Skeleton Caching

**Status**: No caching
**Impact**: Medium (performance)
**Effort**: Low

Currently reads skeleton files on every call:
```python
def load_skeleton(model: str, workflow_type: str):
    skeleton = json.loads(skeleton_path.read_text())  # File I/O every time
```

**Recommended Action**:
```python
_SKELETON_CACHE = {}

def load_skeleton(model: str, workflow_type: str):
    cache_key = f"{model}:{workflow_type}"
    if cache_key not in _SKELETON_CACHE:
        _SKELETON_CACHE[cache_key] = json.loads(skeleton_path.read_text())
    return _SKELETON_CACHE[cache_key], skeleton_name
```

### 2.4 Connection Type Validation

**Status**: Not implemented
**Impact**: Medium
**Effort**: Medium

The TYPE_COMPATIBILITY matrix in topology_validator.py is defined but not used:
```python
TYPE_COMPATIBILITY = {
    "MODEL": ["model", "unet"],
    "CLIP": ["clip"],
    ...
}
```

**Recommended Action**:
Add function to validate that all connections are type-safe:
```python
def validate_connection_types(workflow: dict) -> List[str]:
    """Check that MODEL outputs only connect to model inputs, etc."""
    errors = []
    for node_id, node_data in workflow.items():
        for input_name, connection in node_data.get("inputs", {}).items():
            if isinstance(connection, list):
                source_node, source_slot = connection
                # Verify source output type matches input type
    return errors
```

---

## Priority 3: Documentation Improvements

### 3.1 Add Example Workflows to Skeletons Directory

**Status**: Only skeletons exist
**Impact**: Medium
**Effort**: Low

The `library/full_examples/` directory is empty. Add complete working examples:
```
library/full_examples/
├── ltx_cat_walking.json    # Complete example with real values
├── flux_portrait.json
├── wan_timelapse.json
└── qwen_poster.json
```

### 3.2 Add Troubleshooting Section to SYSTEM_PROMPT.md

**Status**: Missing
**Impact**: Medium
**Effort**: Low

Common errors LLMs make and how to fix:
- "ModuleNotFoundError" → Model file doesn't exist
- "Connection type mismatch" → Wrong slot index
- "Out of memory" → Reduce resolution or batch size

### 3.3 Create Quick Reference Card

**Status**: Missing
**Impact**: Medium
**Effort**: Low

One-page reference for LLMs:
```markdown
# ComfyUI Quick Reference

## Model → Sampler Type
- LTX, Wan → SamplerCustom
- FLUX → SamplerCustom + FluxGuidance
- Qwen, SDXL → KSampler

## Resolution Rules
- FLUX → divisible by 16
- Others → divisible by 8

## Frame Count
- LTX → 8n+1 (9, 17, 25, ... 97, 121)
```

---

## Priority 4: Code Quality

### 4.1 Consolidate Duplicate Constraint Definitions

**Status**: Duplicated
**Impact**: Low
**Effort**: Low

MODEL_CONSTRAINTS duplicated in:
- `topology_validator.py`
- `workflow_generator.py` (MODEL_DEFAULTS)

**Recommended Action**:
Create single source of truth in `model_constraints.py`:
```python
# model_constraints.py
MODEL_CONSTRAINTS = {...}
MODEL_DEFAULTS = {...}
```

### 4.2 Add Structured Logging

**Status**: No logging
**Impact**: Low
**Effort**: Low

Add logging to track:
- Skeleton loads
- Validation failures
- Auto-corrections made

```python
from .mcp_utils import logger

def validate_topology(workflow_json: str, model: str = None):
    logger.debug(f"Validating workflow for model={model}")
    ...
    if errors:
        logger.warning(f"Validation failed: {errors}")
```

### 4.3 Add Docstrings to All Classes and Functions

**Status**: Partial
**Impact**: Low
**Effort**: Low

Some functions missing docstrings:
- `detect_model_type` - Has docstring ✓
- `expand_skeleton_to_workflow` - Has docstring ✓
- `get_node_categories` - Missing detailed docstring

---

## Priority 5: Future Enhancements

### 5.1 Semantic Search for Patterns

**Status**: Basic keyword matching
**Impact**: Medium
**Effort**: High

Current `search_patterns` uses simple keyword matching. Could use embeddings:
```python
# Future: Use sentence-transformers for semantic search
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(pattern_sections)
```

### 5.2 Dynamic Node Discovery from ComfyUI

**Status**: Static node specs
**Impact**: Medium
**Effort**: High

Currently using static NODE_SPECS. Could query live ComfyUI:
```python
# Future: Fetch from ComfyUI /object_info endpoint
response = requests.get("http://localhost:8188/object_info")
node_info = response.json()
```

### 5.3 Workflow Visualization

**Status**: Not implemented
**Impact**: Low
**Effort**: High

Generate ASCII or Mermaid diagrams from workflows:
```python
def workflow_to_mermaid(workflow: dict) -> str:
    """Generate Mermaid flowchart from workflow."""
    lines = ["graph TD"]
    for node_id, node_data in workflow.items():
        lines.append(f"    {node_id}[{node_data['class_type']}]")
    return "\n".join(lines)
```

---

## Implementation Order

| Priority | Item | Effort | Impact | Sprint |
|----------|------|--------|--------|--------|
| 1.1 | Unit Tests | Medium | High | 1 |
| 1.3 | Edge Case Handling | Medium | High | 1 |
| 2.3 | Skeleton Caching | Low | Medium | 1 |
| 3.3 | Quick Reference Card | Low | Medium | 1 |
| 4.1 | Consolidate Constraints | Low | Low | 1 |
| 2.1 | More Models | Medium | High | 2 |
| 2.2 | More Workflow Types | Medium | Medium | 2 |
| 2.4 | Connection Type Validation | Medium | Medium | 2 |
| 3.1 | Example Workflows | Low | Medium | 2 |
| 3.2 | Troubleshooting Guide | Low | Medium | 2 |

---

## Files Created in This Implementation

| File | Purpose | Lines |
|------|---------|-------|
| `node_specs.py` | Node input/output specifications | 430 |
| `reference_docs.py` | Documentation access module | 200 |
| `topology_validator.py` | Workflow validation | 457 |
| `workflow_generator.py` | Meta-template system | 462 |
| `docs/reference/01_MODEL_PATTERNS.md` | Pattern documentation | 350 |
| `docs/reference/02_NODE_LIBRARY.md` | Node API reference | 280 |
| `docs/reference/03_PARAMETER_RULES.md` | Validation rules | 250 |
| `docs/prompt_guides/SYSTEM_PROMPT.md` | LLM instructions | 200 |
| `docs/library/skeletons/*.json` | 5 skeleton files | 100 each |
| `docs/design/META_TEMPLATE_SPEC.md` | Design spec | 200 |

**Total new code**: ~2,500 lines

---

## Changelog

- **January 2026**: Initial improvements roadmap created post-implementation
