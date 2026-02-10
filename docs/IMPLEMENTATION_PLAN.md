# Implementation Plan: Workflow Drift Prevention

**Date:** January 2026
**Goal:** Prevent Claude from generating broken ComfyUI workflows by providing queryable exact patterns

---

## Problem Statement

When Claude constructs ComfyUI workflows, it drifts from working patterns because:
1. It uses SD/SDXL-era patterns (KSampler, CheckpointLoaderSimple)
2. It doesn't know model-specific requirements (CFG 3.0 for LTX, not 7.0)
3. It misses required intermediate nodes (LTXVConditioning, FluxGuidance)
4. It guesses slot indices incorrectly

**Result:** Generated workflows fail at runtime.

---

## Solution: Queryable Pattern Tools

Instead of documentation Claude can't access, we add MCP tools that return exact working patterns.

### New MCP Tools (4)

| Tool | Purpose | Returns |
|------|---------|---------|
| `get_workflow_skeleton(model, task)` | Exact working workflow | Complete JSON with placeholders |
| `get_model_constraints(model)` | Hard constraints | CFG, resolution, frame rules |
| `get_node_chain(model, task)` | Node order + connections | Ordered list with slot indices |
| `validate_against_pattern(workflow, model)` | Drift detection | Errors + corrected workflow |

---

## Task List

### Phase 1: Core Pattern Tools (P0)

| Task | Description | Files |
|------|-------------|-------|
| #10 | `get_workflow_skeleton()` | server.py, new patterns.py |
| #11 | `get_model_constraints()` | server.py, patterns.py |

### Phase 2: Validation Tools (P1)

| Task | Description | Files |
|------|-------------|-------|
| #12 | `validate_against_pattern()` | server.py, validation.py |
| #13 | `get_node_chain()` | server.py, patterns.py |

### Phase 3: Template Expansion (P1)

| Task | Description | Files |
|------|-------------|-------|
| #2 | Template validation on load | templates/__init__.py |
| #3 | ltx2_i2v_distilled template | templates/ |
| #4 | wan21_txt2vid template | templates/ |
| #5 | Document HunyuanVideo patterns | docs/ |
| #6 | hunyuan15_txt2vid template | templates/ |
| #7 | sdxl_txt2img template | templates/ |

### Phase 4: Discovery Tools (P2)

| Task | Description | Files |
|------|-------------|-------|
| #8 | Workflow builder helpers | workflow_builder.py |
| #9 | Template discovery tools | server.py |

---

## Implementation Details

### get_workflow_skeleton()

```python
WORKFLOW_SKELETONS = {
    ("ltx2", "txt2vid"): {
        "1": {"class_type": "LTXVLoader", ...},
        "2": {"class_type": "CLIPTextEncode", ...},
        # ... complete working workflow
    },
    ("flux2", "txt2img"): {...},
    ("wan21", "img2vid"): {...},
}

@mcp.tool()
def get_workflow_skeleton(model: str, task: str) -> dict:
    key = (model.lower(), task.lower())
    if key not in WORKFLOW_SKELETONS:
        return {"error": f"No skeleton for {model}/{task}"}
    return WORKFLOW_SKELETONS[key]
```

### get_model_constraints()

```python
MODEL_CONSTRAINTS = {
    "ltx2": {
        "cfg": {"min": 2.5, "max": 4.0, "default": 3.0},
        "resolution": {"divisible_by": 8, "native": [768, 512]},
        "frames": {"formula": "8n+1", "valid": [9,17,25,33,41,49,57,65,73,81,89,97]},
        "sampler": "SamplerCustom",
        "scheduler": "LTXVScheduler",
        "conditioning_wrapper": "LTXVConditioning",
        "loader": "LTXVLoader",
        "latent_node": "EmptyLTXVLatentVideo",
        "output_node": "VHS_VideoCombine"
    },
    "flux2": {
        "cfg": {"via": "FluxGuidance", "default": 3.5},
        "resolution": {"divisible_by": 16, "native": [1024, 1024]},
        "sampler": "SamplerCustomAdvanced",
        "scheduler": "BasicScheduler",
        "conditioning_wrapper": "FluxGuidance",
        "loader": ["UNETLoader", "DualCLIPLoader", "VAELoader"],
        "latent_node": "EmptySD3LatentImage",
        "output_node": "SaveImage"
    },
    # ...
}
```

### validate_against_pattern()

```python
def validate_against_pattern(workflow: dict, model: str) -> dict:
    errors = []
    constraints = MODEL_CONSTRAINTS.get(model, {})

    # Check for wrong sampler
    for node in workflow.values():
        if node.get("class_type") == "KSampler":
            if constraints.get("sampler") != "KSampler":
                errors.append(f"Using KSampler - should use {constraints['sampler']}")

        # Check CFG value
        if "cfg" in node.get("inputs", {}):
            cfg = node["inputs"]["cfg"]
            cfg_range = constraints.get("cfg", {})
            if cfg_range.get("min") and cfg < cfg_range["min"]:
                errors.append(f"CFG {cfg} too low - min is {cfg_range['min']}")

    # Check for missing wrapper nodes
    required_wrapper = constraints.get("conditioning_wrapper")
    if required_wrapper:
        has_wrapper = any(n.get("class_type") == required_wrapper for n in workflow.values())
        if not has_wrapper:
            errors.append(f"Missing required {required_wrapper} node")

    return {"valid": len(errors) == 0, "errors": errors}
```

---

## Success Criteria

1. Claude can query `get_workflow_skeleton("ltx2", "txt2vid")` and get exact working workflow
2. Claude can query `get_model_constraints("flux2")` and know CFG must be via FluxGuidance
3. Generated workflows pass `validate_against_pattern()` before execution
4. Zero drift from working patterns

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/.../patterns.py` | CREATE | Workflow skeletons + constraints |
| `src/.../server.py` | MODIFY | Add 4 new MCP tools |
| `src/.../validation.py` | MODIFY | Add pattern validation |
| `src/.../templates/__init__.py` | MODIFY | Add validation on load |
| `src/.../templates/*.json` | CREATE | New templates |

---

## Estimated Effort

| Phase | Tasks | Complexity |
|-------|-------|------------|
| Phase 1 | 2 | High - core pattern infrastructure |
| Phase 2 | 2 | Medium - validation logic |
| Phase 3 | 6 | Medium - templates + docs |
| Phase 4 | 2 | Low - discovery helpers |

**Total: 12 tasks**
