# MCP Token Reduction Implementation Plan

**Date:** January 2026
**Status:** In Progress
**Goal:** Reduce MCP tool context from ~23,462 tokens to <10,000 tokens (~60% reduction)

---

## Problem Statement

The ComfyUI MassMediaFactory MCP server exposes 100 tools consuming ~23,462 tokens of context space. This is 89% of the total MCP context overhead when the server is registered:

```
Context Usage Warnings
└ ⚠ Large MCP tools context (~26,377 tokens > 25,000)
└ MCP servers:
└ comfyui-massmediafactory: 100 tools (~23,462 tokens)
└ sota-tracker: 10 tools (~2,008 tokens)
└ context7: 2 tools (~907 tokens)
```

This excessive context usage:
1. Reduces available context for actual work
2. Slows down response times
3. Increases costs
4. Makes tool discovery harder for the LLM

---

## Current Architecture Analysis

### Tool Distribution by Category (100 tools)

| Category | Count | Tools |
|----------|-------|-------|
| Discovery | 8 | `list_checkpoints`, `list_unets`, `list_loras`, `list_vaes`, `list_controlnets`, `get_node_info`, `search_nodes`, `get_all_models` |
| Execution | 8 | `execute_workflow`, `get_workflow_status`, `wait_for_completion`, `get_system_stats`, `free_memory`, `interrupt_execution`, `get_queue_status`, `regenerate` |
| Assets | 6 | `list_assets`, `get_asset_metadata`, `view_output`, `cleanup_expired_assets`, `upload_image`, `download_output` |
| Publishing | 3 | `publish_asset`, `get_publish_info`, `set_publish_dir` |
| Workflow Library | 7 | `save_workflow`, `load_workflow`, `list_saved_workflows`, `delete_workflow`, `duplicate_workflow`, `export_workflow`, `import_workflow` |
| VRAM | 2 | `estimate_vram`, `check_model_fits` |
| Validation | 3 | `validate_workflow`, `validate_and_fix_workflow`, `check_connection_compatibility` |
| SOTA | 5 | `get_sota_models`, `recommend_model`, `check_model_freshness`, `get_model_settings`, `check_installed_sota` |
| Templates | 4 | `list_workflow_templates`, `get_template`, `create_workflow_from_template`, `get_workflow_skeleton` |
| Patterns | 6 | `get_model_constraints`, `get_node_chain`, `validate_against_pattern`, `list_available_patterns`, `explain_workflow`, `get_required_nodes` |
| Builder | 4 | `get_connection_pattern`, `compare_workflows`, `find_template_for_task`, `get_templates_by_type` |
| Template Discovery | 4 | `get_templates_by_model`, `validate_all_templates`, `get_template` |
| Batch | 3 | `execute_batch_workflows`, `execute_parameter_sweep`, `generate_seed_variations` |
| Pipelines | 3 | `execute_pipeline_stages`, `run_image_to_video_pipeline`, `run_upscale_pipeline` |
| Models | 4 | `search_civitai`, `download_model`, `get_model_info`, `list_installed_models` |
| Analysis | 5 | `get_image_dimensions`, `detect_objects`, `get_video_info`, `qa_output`, `check_vlm_available` |
| Style Learning | 10 | `record_generation`, `rate_generation`, `suggest_prompt_enhancement`, `find_similar_prompts`, `get_best_seeds_for_style`, `save_style_preset`, `get_style_preset`, `list_style_presets`, `get_style_learning_stats` |
| Schemas/Meta | 5 | `get_tool_schema`, `list_tool_schemas`, `get_tool_output_schema`, `get_tool_annotation`, `list_user_facing_tools` |
| Reference Docs | 9 | `get_node_spec`, `list_node_specs`, `get_model_pattern`, `get_workflow_skeleton_json`, `get_parameter_rules`, `search_patterns`, `get_llm_system_prompt`, `validate_topology`, `auto_correct_workflow` |
| Workflow Gen | 3 | `generate_workflow`, `list_supported_workflows`, `list_workflow_skeletons` |

### Token Breakdown (Estimated)

| Component | Tokens | Calculation |
|-----------|--------|-------------|
| Tool names | ~500 | 100 tools × 5 tokens avg |
| Tool descriptions | ~18,000 | 100 tools × 180 tokens avg (verbose docstrings) |
| Parameter schemas | ~5,000 | 100 tools × 50 tokens avg |
| **Total** | ~23,500 | |

### Sample Verbose Docstring (Current)

```python
def regenerate(asset_id: str, prompt: str = None, ...) -> dict:
    """
    Regenerate an asset with parameter overrides.

    Enables quick iteration: tweak CFG, change prompt, try new seed.

    Args:
        asset_id: The asset to regenerate from.
        prompt: New prompt (optional).
        negative_prompt: New negative prompt (optional).
        seed: New seed. Use -1 to keep original, None/omit for random.
        steps: New step count (optional).
        cfg: New CFG scale (optional).

    Returns:
        New prompt_id for the regenerated workflow.

    Example:
        # Generate initial image
        result = execute_workflow(workflow)
        output = wait_for_completion(result["prompt_id"])
        asset_id = output["outputs"][0]["asset_id"]

        # Iterate with higher CFG
        result = regenerate(asset_id, cfg=4.5)
        output = wait_for_completion(result["prompt_id"])
    """
```

**Token count:** ~180 tokens per tool

---

## Alternatives Considered

### Alternative A: MCP Tool Deferral (Claude Code Native)

**Approach:** Claude Code already supports deferred tools via `ToolSearch`. Tools are listed in the system prompt but not fully loaded until selected.

**Pros:**
- No code changes needed
- Already implemented in Claude Code

**Cons:**
- Still need tool names + short descriptions in system prompt
- Only reduces schema tokens, not description tokens
- 100 deferred tools still consume ~10,000+ tokens in system prompt

**Verdict:** Insufficient alone. Tool list still too large.

### Alternative B: Split Into Multiple MCP Servers

**Approach:** Split into 3-4 specialized servers:
- `comfyui-core` - Essential 20 tools
- `comfyui-batch` - Batch/pipeline tools
- `comfyui-style` - Style learning tools
- `comfyui-reference` - Documentation tools

**Pros:**
- User can register only what they need
- Clear separation of concerns

**Cons:**
- Management overhead (multiple servers to update)
- Breaks atomic functionality (e.g., style learning depends on execution)
- More complex installation

**Verdict:** Good fallback if consolidation isn't enough.

### Alternative C: Convert Reference Tools to MCP Resources (RECOMMENDED)

**Approach:**
- Remove ~20 reference/documentation tools from the tool list
- Expose them as MCP Resources instead (static content)
- Resources don't consume tool context tokens

**Pros:**
- ~20% immediate token reduction
- Resources are the correct abstraction for static docs
- Clean separation: tools = actions, resources = data

**Cons:**
- Requires updating how Claude accesses reference docs
- Resources are less discoverable than tools

**Verdict:** Yes - this is semantically correct.

### Alternative D: Tool Consolidation (RECOMMENDED)

**Approach:** Merge similar tools into parameterized versions:
- `list_checkpoints/unets/loras/vaes/controlnets` → `list_models(type)`
- `save/get/list_style_preset` → `manage_presets(action, name)`

**Pros:**
- ~50% reduction in tool count
- Cleaner API surface
- Easier to discover and use

**Cons:**
- Changes API (breaking for existing users)
- More complex parameter validation

**Verdict:** Yes - this is the primary strategy.

### Alternative E: Docstring Minimization (RECOMMENDED)

**Approach:** Reduce all docstrings to one-liners:

```python
# Before (~180 tokens)
def regenerate(...):
    """
    Regenerate an asset with parameter overrides.
    Enables quick iteration... [180 tokens of docs]
    """

# After (~30 tokens)
def regenerate(...):
    """Re-run workflow with modified params. Returns new prompt_id."""
```

**Pros:**
- ~60% reduction in description tokens
- Forces clarity in naming
- Parameter types provide enough context

**Cons:**
- Less helpful for humans reading code
- May reduce LLM accuracy slightly

**Verdict:** Yes - move detailed docs to separate reference files.

---

## Chosen Strategy: Hybrid Approach

Combine **C + D + E**:

1. **Convert reference tools to MCP Resources** (~20 tools removed)
2. **Consolidate similar tools** (~30 tools merged into ~10)
3. **Minimize docstrings** (~60% description token reduction)

### Resulting Architecture

#### Tier 1: Core Tools (Always Loaded) - 20 tools

| Category | New Tool | Replaces | Description |
|----------|----------|----------|-------------|
| Discovery | `list_models(type)` | list_checkpoints, list_unets, list_loras, list_vaes, list_controlnets | `type`: checkpoint\|unet\|lora\|vae\|controlnet\|all |
| Discovery | `get_node_info(node_type)` | unchanged | Get node schema |
| Discovery | `search_nodes(query)` | unchanged | Search by name/category |
| Execution | `execute_workflow(workflow)` | unchanged | Queue workflow |
| Execution | `wait_for_completion(prompt_id)` | unchanged | Wait and get outputs |
| Execution | `get_queue_status()` | get_workflow_status, get_queue_status | Combined status |
| Execution | `interrupt()` | interrupt_execution | Stop current job |
| System | `get_system_stats()` | unchanged | VRAM/GPU info |
| System | `free_memory(unload_models)` | unchanged | Free GPU memory |
| Assets | `list_assets(...)` | unchanged | Browse outputs |
| Assets | `view_output(asset_id)` | unchanged | View asset URL |
| Assets | `regenerate(...)` | unchanged | Iterate on asset |
| Assets | `upload_image(...)` | unchanged | For ControlNet/I2V |
| Assets | `download_output(...)` | unchanged | Save to disk |
| Templates | `list_templates()` | list_workflow_templates | List available |
| Templates | `get_template(name)` | unchanged | Get template JSON |
| Templates | `create_from_template(name, params)` | create_workflow_from_template | Create workflow |
| Validation | `validate_workflow(workflow)` | unchanged | Check for errors |
| Validation | `validate_and_fix(workflow)` | validate_and_fix_workflow | Auto-correct |
| SOTA | `recommend_model(task)` | unchanged | Get best model |

#### Tier 2: Extended Tools - 12 tools

| Category | Tool | Description |
|----------|------|-------------|
| Batch | `execute_batch(workflow, params_list)` | Run multiple variations |
| Batch | `parameter_sweep(workflow, sweeps)` | Grid search parameters |
| Pipelines | `run_pipeline(stages)` | Multi-stage workflow |
| Publishing | `publish_asset(asset_id, ...)` | Export to web dir |
| QA | `qa_output(asset_id, prompt)` | VLM quality check |
| Models | `search_civitai(query)` | Find models |
| Models | `download_model(url, type)` | Install model |
| Style | `record_generation(...)` | Log for learning |
| Style | `rate_generation(id, rating)` | User feedback |
| Style | `style_suggest(mode)` | prompt\|seeds\|similar |
| Style | `manage_presets(action, name)` | list\|get\|save\|delete |
| VRAM | `check_model_fits(model, task)` | Pre-check VRAM |

#### MCP Resources (Replace ~25 tools)

| Resource URI | Content | Replaces |
|--------------|---------|----------|
| `patterns://flux` | FLUX workflow pattern | get_model_pattern("flux") |
| `patterns://ltx` | LTX workflow pattern | get_model_pattern("ltx") |
| `patterns://wan` | Wan workflow pattern | get_model_pattern("wan") |
| `nodes://spec/{name}` | Node specification | get_node_spec() |
| `rules://parameters` | Parameter validation rules | get_parameter_rules() |
| `skeletons://{model}` | Workflow skeleton | get_workflow_skeleton_json() |
| `docs://system-prompt` | LLM system prompt | get_llm_system_prompt() |

---

## Implementation Phases

### Phase 1: Consolidate Discovery Tools
**Target:** 5 tools → 1 tool

**Changes to `server.py`:**
```python
# Remove these individual tools:
# - list_checkpoints()
# - list_unets()
# - list_loras()
# - list_vaes()
# - list_controlnets()

# Add consolidated tool:
@mcp.tool()
def list_models(model_type: str = "all") -> dict:
    """List models. type: checkpoint|unet|lora|vae|controlnet|all"""
    if model_type == "all":
        return discovery.get_all_models()
    type_map = {
        "checkpoint": discovery.list_checkpoints,
        "unet": discovery.list_unets,
        "lora": discovery.list_loras,
        "vae": discovery.list_vaes,
        "controlnet": discovery.list_controlnets,
    }
    if model_type not in type_map:
        return mcp_error(f"Unknown type: {model_type}")
    return type_map[model_type]()
```

**Files:**
- `src/comfyui_massmediafactory_mcp/server.py`

**Verification:**
```python
list_models("checkpoint")  # Should return checkpoints
list_models("lora")        # Should return loras
list_models()              # Should return all models
```

---

### Phase 2: Remove Reference/Meta Tools
**Target:** Remove ~25 tools, convert to MCP Resources

**Tools to Remove from `server.py`:**
```python
# Schema/introspection tools (MCP handles this natively)
get_tool_schema
list_tool_schemas
get_tool_output_schema
get_tool_annotation
list_user_facing_tools
list_tools_by_category

# Reference documentation tools (convert to resources)
get_node_spec
list_node_specs
get_model_pattern
get_workflow_skeleton_json
get_parameter_rules
search_patterns
get_llm_system_prompt
list_workflow_skeletons

# Rarely used builder tools
explain_workflow
get_required_nodes
get_connection_pattern
compare_workflows
find_template_for_task
get_templates_by_type
get_templates_by_model
validate_all_templates
```

**Add MCP Resources:**
```python
@mcp.resource("patterns://{model}")
def get_pattern_resource(model: str) -> str:
    """Get workflow pattern for a model."""
    return patterns.get_model_pattern(model)

@mcp.resource("rules://parameters")
def get_rules_resource() -> str:
    """Get parameter validation rules."""
    return reference_docs.get_parameter_rules()
```

**Files:**
- `src/comfyui_massmediafactory_mcp/server.py`

---

### Phase 3: Consolidate Style Learning
**Target:** 10 tools → 4 tools

**Changes to `server.py`:**
```python
# Keep these unchanged:
record_generation()
rate_generation()

# Replace these:
# - suggest_prompt_enhancement
# - find_similar_prompts
# - get_best_seeds_for_style
# With:
@mcp.tool()
def style_suggest(mode: str, prompt: str = None, style: str = None) -> dict:
    """Get style suggestions. mode: prompt|seeds|similar"""
    if mode == "prompt":
        return style_learning.suggest_prompt_enhancement(prompt)
    elif mode == "seeds":
        return style_learning.get_best_seeds_for_style(style)
    elif mode == "similar":
        return style_learning.find_similar_prompts(prompt)

# Replace these:
# - save_style_preset
# - get_style_preset
# - list_style_presets
# With:
@mcp.tool()
def manage_presets(action: str, name: str = None, data: dict = None) -> dict:
    """Manage style presets. action: list|get|save|delete"""
    if action == "list":
        return style_learning.list_style_presets()
    elif action == "get":
        return style_learning.get_style_preset(name)
    elif action == "save":
        return style_learning.save_style_preset(name, data)
    elif action == "delete":
        return style_learning.delete_style_preset(name)
```

**Files:**
- `src/comfyui_massmediafactory_mcp/server.py`
- `src/comfyui_massmediafactory_mcp/style_learning.py` (add delete_style_preset if missing)

---

### Phase 4: Minimize All Docstrings
**Target:** ~60% reduction in description tokens

**Pattern to Apply:**

```python
# Before:
def execute_workflow(workflow: dict, client_id: str = "massmediafactory") -> dict:
    """
    Execute a ComfyUI workflow and return the prompt_id for tracking.

    Args:
        workflow: The workflow JSON with node definitions.
                  Each key is a node ID, value is {"class_type": "...", "inputs": {...}}
        client_id: Optional identifier for tracking.

    Returns:
        prompt_id for polling results.

    Example:
        {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a dragon", "clip": ["1", 1]}}
        }
    """

# After:
def execute_workflow(workflow: dict, client_id: str = "massmediafactory") -> dict:
    """Queue workflow for execution. Returns prompt_id for tracking."""
```

**Apply to all ~32 remaining tools.**

**Files:**
- `src/comfyui_massmediafactory_mcp/server.py`

---

### Phase 5: Update Annotations

Update `annotations.py` to reflect new consolidated tool names.

**Files:**
- `src/comfyui_massmediafactory_mcp/annotations.py`

---

### Phase 6: Consolidate Workflow Library (Optional)
**Target:** 7 tools → 3 tools

```python
# Combine:
# - save_workflow, load_workflow, list_saved_workflows
# - delete_workflow, duplicate_workflow
# - export_workflow, import_workflow

@mcp.tool()
def manage_workflow(action: str, name: str = None, workflow: dict = None) -> dict:
    """Manage saved workflows. action: list|load|save|delete|duplicate"""

@mcp.tool()
def transfer_workflow(action: str, path: str = None, name: str = None) -> dict:
    """Import/export workflows. action: export|import"""
```

---

## Token Reduction Summary

| Phase | Tools Before | Tools After | Token Reduction |
|-------|--------------|-------------|-----------------|
| Phase 1: Discovery | 5 | 1 | ~720 tokens |
| Phase 2: Reference | 25 | 0 (resources) | ~4,500 tokens |
| Phase 3: Style Learning | 10 | 4 | ~1,080 tokens |
| Phase 4: Docstrings | - | - | ~10,800 tokens |
| Phase 5: Annotations | - | - | 0 (internal) |
| Phase 6: Workflow Library | 7 | 2 | ~900 tokens |
| **Total** | 100 | ~32 | **~18,000 tokens (~77%)** |

**Projected Final:** ~5,500 tokens (down from 23,462)

---

## Migration Guide

For users with existing scripts/integrations:

### Deprecated Functions → Replacements

| Old | New | Notes |
|-----|-----|-------|
| `list_checkpoints()` | `list_models("checkpoint")` | |
| `list_unets()` | `list_models("unet")` | |
| `list_loras()` | `list_models("lora")` | |
| `list_vaes()` | `list_models("vae")` | |
| `list_controlnets()` | `list_models("controlnet")` | |
| `get_workflow_status(id)` | `get_queue_status()` | Returns all jobs |
| `interrupt_execution()` | `interrupt()` | Renamed |
| `list_workflow_templates()` | `list_templates()` | Renamed |
| `create_workflow_from_template()` | `create_from_template()` | Renamed |
| `validate_and_fix_workflow()` | `validate_and_fix()` | Renamed |
| `suggest_prompt_enhancement()` | `style_suggest("prompt", prompt)` | |
| `find_similar_prompts()` | `style_suggest("similar", prompt)` | |
| `get_best_seeds_for_style()` | `style_suggest("seeds", style=style)` | |
| `save_style_preset()` | `manage_presets("save", name, data)` | |
| `get_style_preset()` | `manage_presets("get", name)` | |
| `list_style_presets()` | `manage_presets("list")` | |

---

## Verification Checklist

- [ ] Token count < 10,000 (check via Claude Code context warnings)
- [ ] All existing tests pass: `pytest tests/`
- [ ] Core workflow works: `execute_workflow` → `wait_for_completion`
- [ ] Discovery works: `list_models("checkpoint")` returns models
- [ ] Templates work: `create_from_template("flux2_txt2img", {...})`
- [ ] MCP Resources accessible: `patterns://flux`
- [ ] No regressions in existing functionality

---

## Rollback Plan

If issues arise:
1. Git revert to pre-change commit
2. Reinstall original package: `pip install -e .`
3. Restart Claude Code

Keep old function implementations (just remove `@mcp.tool()` decorator) so internals still work.

---

## Progress Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Discovery | ⬜ Not Started | |
| Phase 2: Reference | ⬜ Not Started | |
| Phase 3: Style Learning | ⬜ Not Started | |
| Phase 4: Docstrings | ⬜ Not Started | |
| Phase 5: Annotations | ⬜ Not Started | |
| Phase 6: Workflow Library | ⬜ Not Started | Optional |
