# Model Registry Consolidation - Implementation Notes

**Task:** P0-2: Centralize Model Constraints
**Agent:** Claude
**Date:** 2026-02-01

## Summary

Created a centralized `model_registry.py` module that serves as the single source of truth for all model definitions, constraints, defaults, and aliases. Refactored existing modules to import from this central registry.

## Changes Made

### New File: `src/comfyui_massmediafactory_mcp/model_registry.py`

A comprehensive module containing:

1. **Data Classes** for type-safe model definitions:
   - `CFGSpec` - CFG/guidance constraints
   - `ResolutionSpec` - Resolution constraints
   - `FrameSpec` - Frame count constraints (for video models)
   - `StepsSpec` - Sampling steps constraints
   - `SchedulerSpec` - Scheduler parameters
   - `ModelConstraints` - Complete constraint specification

2. **Central Registry** (`_MODEL_REGISTRY`):
   - `ltx2` - LTX-Video 2.0
   - `flux2` - FLUX.2
   - `wan21` - Wan 2.1
   - `qwen` - Qwen Image
   - `sdxl` - SDXL
   - `hunyuan15` - HunyuanVideo 1.5
   - `qwen_edit` - Qwen Image Edit 2511

3. **Alias Mappings**:
   - `MODEL_ALIASES` - Maps shorthand model names (e.g., "ltx" → "ltx2")
   - `WORKFLOW_TYPE_ALIASES` - Maps shorthand types (e.g., "t2v" → "txt2vid")
   - `MODEL_SKELETON_MAP` - Maps (model, type) tuples to canonical skeleton keys

4. **Default Parameters** (`MODEL_DEFAULTS`):
   - Generation defaults per model (width, height, frames, steps, cfg, etc.)

5. **Public API Functions**:
   - `resolve_model_name(model)` - Resolve alias to canonical name
   - `resolve_workflow_type(workflow_type)` - Resolve type alias
   - `get_canonical_model_key(model, type)` - Get skeleton lookup key
   - `get_model_constraints(model)` - Get constraints dict
   - `get_model_defaults(model)` - Get default parameters
   - `list_supported_models()` - List all canonical model names
   - `is_video_model(model)` - Check if model is video type
   - `validate_model_exists(model)` - Validate model exists

6. **Backwards Compatibility Exports**:
   - `MODEL_CONSTRAINTS` - Dict format for existing code
   - `MODEL_DEFAULTS` - Dict with alias support
   - `MODEL_RESOLUTION_SPECS` - Resolution specs for validation.py

### Refactored Files

1. **`patterns.py`**:
   - Removed ~280 lines of `MODEL_CONSTRAINTS` definition
   - Now imports from `model_registry`
   - `get_model_constraints()` delegates to registry

2. **`workflow_generator.py`**:
   - Removed `MODEL_SKELETON_MAP` (71 lines)
   - Removed `MODEL_DEFAULTS` (68 lines)
   - Now imports from `model_registry`
   - Uses `get_model_defaults()` and `get_canonical_model_key()`

3. **`topology_validator.py`**:
   - Removed `_build_validation_constraints()` reimplementation
   - Now builds validation constraints from `model_registry.MODEL_CONSTRAINTS`
   - Added helper `_flatten_required_nodes()` for list conversion

4. **`validation.py`**:
   - Removed `MODEL_RESOLUTION_SPECS` (8 lines)
   - Now imports `MODEL_RESOLUTION_SPECS` from `model_registry`

### New Test File: `tests/test_model_registry.py`

Comprehensive test suite with **52 tests** covering:
- Model constraints existence and structure
- Model defaults values
- Model alias resolution
- Workflow type alias resolution
- Canonical model key lookup
- List functions
- Utility functions
- Backwards compatibility exports
- Constraint structure consistency
- Single source of truth validation

### Test Fixes

Updated existing tests to include all required nodes in validation tests:
- `test_patterns.py::test_validate_valid_workflow`
- `test_topology_validator.py::test_valid_workflow`

## Verification

All **131 tests pass**:
- 52 new model_registry tests
- 79 existing tests (patterns, workflow_generator, topology_validator)

```
======================== 131 passed, 1 warning in 0.06s ========================
```

## Benefits

1. **Single Source of Truth**: Model definitions exist in one place only
2. **Adding New Models**: Only requires editing `model_registry.py`
3. **Type Safety**: Dataclasses provide structure and documentation
4. **Backwards Compatibility**: Existing code continues to work
5. **Centralized Aliases**: All model/type aliases in one location
6. **Better Testing**: Comprehensive tests ensure consistency

## Files Modified

| File | Lines Removed | Lines Added | Net Change |
|------|---------------|-------------|------------|
| `model_registry.py` (new) | 0 | 666 | +666 |
| `patterns.py` | ~280 | 10 | -270 |
| `workflow_generator.py` | ~140 | 15 | -125 |
| `topology_validator.py` | 76 | 68 | -8 |
| `validation.py` | 8 | 2 | -6 |
| `test_model_registry.py` (new) | 0 | 334 | +334 |

**Net reduction in scattered model definitions**: ~430 lines consolidated

## Acceptance Criteria Met

- [x] All tests pass (`pytest tests/`)
- [x] No duplicate model definitions remain
- [x] Adding a new model requires editing only 1 file (`model_registry.py`)
- [x] Type hints complete (dataclasses provide type information)
