# Improvement Plan - Prioritized

**Date:** February 2026
**Author:** Claude Opus 4.5
**Repository:** comfyui-massmediafactory-mcp
**Baseline:** January 2026 codebase review (9/10 rating)

---

## Executive Summary

This plan categorizes improvements by priority level:
- **P0 (Critical):** Blocking issues that should be fixed before any new features
- **P1 (High Value):** Features that significantly improve utility
- **P2 (Nice to Have):** Polish, optimization, and quality-of-life improvements

---

## P0: Critical Gaps (Fix First)

### P0-1: Complete Token Reduction Plan

**Status:** In Progress (from docs/TOKEN_REDUCTION_PLAN.md)
**Problem:** MCP context at ~23,462 tokens, blocking efficient usage
**Target:** <10,000 tokens (77% reduction)

| Phase | Task | Effort | Impact |
|-------|------|--------|--------|
| Phase 1 | Consolidate discovery tools (5→1) | 2h | -720 tokens |
| Phase 2 | Convert 25 reference tools → MCP Resources | 4h | -4,500 tokens |
| Phase 3 | Consolidate style learning (10→4) | 2h | -1,080 tokens |
| Phase 4 | Minimize all docstrings | 3h | -10,800 tokens |
| Phase 5 | Update annotations.py | 1h | N/A |

**Total Effort:** 12 hours
**Expected Result:** 32 tools, ~5,500 tokens

---

### P0-2: Add Missing Integration Tests

**Status:** Not Started
**Problem:** Core workflow execution paths untested
**Target:** 80% coverage on critical paths

| Test Suite | Coverage Target | Effort |
|------------|-----------------|--------|
| `test_execution.py` | execute_workflow, wait_for_completion, regenerate | 3h |
| `test_client.py` | HTTP retry logic, upload/download | 2h |
| `test_batch.py` | Batch execution, parameter sweeps | 2h |
| `test_assets.py` | Asset lifecycle, TTL expiration | 1.5h |

**Total Effort:** 8.5 hours
**Dependencies:** Running ComfyUI instance

---

### P0-3: Fix Edge Cases in Workflow Generator

**Status:** Partially Implemented (tests exist but gaps remain)
**Problem:** Some edge cases can produce invalid workflows

| Edge Case | Current Behavior | Fix Required |
|-----------|------------------|--------------|
| Negative seed < -1 | Passes through | Normalize to -1 or random |
| Negative CFG | Passes through | Error or normalize |
| Extreme resolution (>4096) | Warning only | Error for unsupported |
| I2V without image_path | Generates anyway | Error with clear message |

**Effort:** 4 hours
**Impact:** Prevents invalid workflow execution failures

---

### P0-4: Security Hardening

**Status:** Not audited
**Problem:** Potential injection vulnerabilities

| Vulnerability | Location | Fix |
|---------------|----------|-----|
| Path traversal | `upload_image()`, `download_output()` | Validate paths, no .. |
| URL injection | `download_model()` Civitai URLs | Whitelist domains |
| Prompt injection | `qa_output()` VLM prompts | Escape user content |

**Effort:** 4 hours
**Impact:** Prevents security exploits

---

## P1: High Value Additions

### P1-1: Add SOTA Model Support

**Status:** Partial (4 models supported: FLUX, LTX, Wan, Qwen)
**Problem:** Missing support for current SOTA models

| Model | Type | ComfyUI Support | Effort |
|-------|------|-----------------|--------|
| **Z-Image-Turbo** | Image | Has nodes | 4h |
| **CogVideoX-5B** | Video | Has nodes | 6h |
| **AnimateDiff v3** | Video | Has nodes | 4h |
| **Stable Diffusion 3.5** | Image | Has nodes | 3h |

**Per-Model Work:**
1. Create skeleton workflow JSON
2. Add constraints to MODEL_CONSTRAINTS
3. Add node specs to NODE_SPECS
4. Add workflow pattern to patterns.py
5. Create template in templates/
6. Add tests

**Total Effort:** 17 hours
**Impact:** Supports current generation models

---

### P1-2: Implement Progress Reporting

**Status:** Not implemented
**Problem:** No visibility into long-running generations
**Solution:** WebSocket connection for progress updates

| Component | Description | Effort |
|-----------|-------------|--------|
| `websocket_client.py` | WS connection to ComfyUI | 3h |
| Progress tool | `get_progress(prompt_id)` | 2h |
| Integration | Update wait_for_completion | 2h |

**Total Effort:** 7 hours
**Impact:** Better UX for video generation (can take 5+ minutes)

---

### P1-3: Add ControlNet/IP-Adapter Support

**Status:** Not implemented
**Problem:** No support for guided generation

| Feature | Use Case | Effort |
|---------|----------|--------|
| ControlNet depth | Image composition | 4h |
| ControlNet canny | Edge-guided generation | 3h |
| IP-Adapter | Style/face transfer | 5h |
| InstantID | Face-guided generation | 4h |

**Per-Feature Work:**
1. Add skeleton workflow
2. Create template with placeholders
3. Document required models
4. Add preprocessing (edge detection, etc.)

**Total Effort:** 16 hours
**Impact:** Enables major new use cases

---

### P1-4: Improve Error Messages

**Status:** Basic error codes implemented
**Problem:** Errors not actionable

| Current Error | Better Error |
|---------------|--------------|
| `"Model not found"` | `"Model not found: flux1-dev.safetensors. Expected in: models/unet/. Run list_models('unet') to see available."` |
| `"Validation failed"` | `"Resolution 1000x1000 invalid for FLUX. Must be divisible by 16. Suggest: 1008x1008 or 1024x1024."` |
| `"Connection failed"` | `"Cannot connect to ComfyUI at http://localhost:8188. Is ComfyUI running? Check: curl http://localhost:8188/system_stats"` |

**Effort:** 6 hours
**Impact:** Reduces debugging time, improves self-service

---

### P1-5: Add Workflow Templates for Common Tasks

**Status:** 35 templates exist
**Problem:** Missing templates for common tasks

| Template | Use Case | Priority |
|----------|----------|----------|
| `flux_portrait_lora` | LoRA-enhanced portraits | HIGH |
| `flux_controlnet_depth` | Depth-guided generation | HIGH |
| `upscale_4x_facerestore` | 4K upscale with face fix | HIGH |
| `ltx_loop_video` | Seamless looping video | MEDIUM |
| `batch_character_poses` | Multiple poses same char | MEDIUM |

**Effort:** 3 hours per template
**Total Effort:** 15 hours for 5 templates

---

## P2: Nice to Have

### P2-1: Skeleton Caching

**Status:** Not implemented (file I/O on every call)
**Problem:** Slight performance overhead
**Solution:** In-memory cache with TTL

```python
_SKELETON_CACHE = {}
_CACHE_TTL = 300  # 5 minutes

def load_skeleton(model, workflow_type):
    cache_key = f"{model}:{workflow_type}"
    if cache_key not in _SKELETON_CACHE or cache_expired(cache_key):
        _SKELETON_CACHE[cache_key] = {
            "data": load_from_file(...),
            "loaded_at": time.time()
        }
    return _SKELETON_CACHE[cache_key]["data"]
```

**Effort:** 2 hours
**Impact:** ~10ms faster per generate_workflow call

---

### P2-2: Consolidate Model Constraints

**Status:** Duplicated in topology_validator.py and workflow_generator.py
**Problem:** Risk of drift between definitions
**Solution:** Single source of truth

```
src/comfyui_massmediafactory_mcp/
├── model_constraints.py  # NEW: Single source
├── topology_validator.py  # Imports from model_constraints
├── workflow_generator.py  # Imports from model_constraints
```

**Effort:** 3 hours
**Impact:** Maintainability, reduces bugs

---

### P2-3: Add Workflow Visualization

**Status:** Not implemented
**Problem:** Hard to debug complex workflows
**Solution:** Generate Mermaid diagrams

```python
def workflow_to_mermaid(workflow: dict) -> str:
    """Generate Mermaid flowchart from workflow."""
    lines = ["graph TD"]
    for node_id, node in workflow.items():
        class_type = node["class_type"]
        lines.append(f'    {node_id}["{class_type}"]')
        for input_name, connection in node.get("inputs", {}).items():
            if isinstance(connection, list):
                source_id, slot = connection
                lines.append(f"    {source_id} -->|{input_name}| {node_id}")
    return "\n".join(lines)
```

**Effort:** 4 hours
**Impact:** Debugging, documentation

---

### P2-4: Implement Semantic Pattern Search

**Status:** Basic keyword search
**Problem:** Hard to find relevant patterns
**Solution:** Embedding-based semantic search

**Effort:** 8 hours (requires embedding model integration)
**Impact:** Better discoverability

---

### P2-5: Add Structured Logging

**Status:** Basic logging via mcp_utils.logger
**Problem:** No correlation IDs, inconsistent format
**Solution:** Structured JSON logging

```python
logger.info("workflow_executed", extra={
    "prompt_id": prompt_id,
    "model": model_type,
    "duration_ms": duration,
    "correlation_id": correlation_id
})
```

**Effort:** 4 hours
**Impact:** Better observability, debugging

---

### P2-6: Add Rate Limiting Dashboard

**Status:** Rate limiting exists but no visibility
**Problem:** Users don't know when rate limited
**Solution:** Tool to check rate limit status

**Effort:** 2 hours
**Impact:** Better UX

---

### P2-7: Improve Type Hints

**Status:** Partial coverage
**Problem:** IDE support incomplete
**Solution:** Full TypedDict for complex returns, @overload decorators

**Effort:** 6 hours
**Impact:** Better developer experience

---

## Effort Summary by Priority

| Priority | Items | Total Effort |
|----------|-------|--------------|
| **P0** | 4 items | 28.5 hours |
| **P1** | 5 items | 61 hours |
| **P2** | 7 items | 29 hours |
| **Total** | 16 items | **118.5 hours** |

---

## Recommended Implementation Order

### Sprint 1 (Week 1-2): Foundation
1. P0-1: Complete token reduction (12h)
2. P0-3: Fix edge cases (4h)
3. P0-4: Security hardening (4h)

**Sprint 1 Total:** 20 hours

### Sprint 2 (Week 3-4): Quality
1. P0-2: Integration tests (8.5h)
2. P1-4: Improve error messages (6h)
3. P2-1: Skeleton caching (2h)
4. P2-2: Consolidate constraints (3h)

**Sprint 2 Total:** 19.5 hours

### Sprint 3 (Week 5-6): Features
1. P1-1: Add 2 SOTA models (Z-Image, SD3.5) (7h)
2. P1-5: Add 3 workflow templates (9h)
3. P1-2: Progress reporting (7h)

**Sprint 3 Total:** 23 hours

### Sprint 4+ (Future): Expansion
1. P1-1: Remaining SOTA models (10h)
2. P1-3: ControlNet/IP-Adapter (16h)
3. P2-3 through P2-7: Polish items (20h)

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Token count | ~23,462 | <10,000 | Claude Code context warning |
| Test coverage | ~40% | >80% | pytest --cov |
| Supported models | 4 | 8+ | list_supported_workflows() |
| Error actionability | Low | High | User feedback |
| Template count | 35 | 50+ | list_workflow_templates() |

---

## Dependencies & Blockers

| Item | Dependency | Blocker Risk |
|------|------------|--------------|
| Integration tests | Running ComfyUI | Medium |
| SOTA model support | Model availability | Low |
| WebSocket progress | ComfyUI WS API stability | Medium |
| ControlNet templates | Model downloads | Low |
| Semantic search | Embedding model | High |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Token reduction breaks compatibility | Medium | High | Maintain deprecated aliases |
| New model constraints wrong | Medium | Medium | Validate against ComfyUI |
| Security fixes introduce regressions | Low | High | Comprehensive test coverage |
| WebSocket connection instability | Medium | Low | Fallback to polling |
