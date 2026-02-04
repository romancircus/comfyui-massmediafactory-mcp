# Implementation Plan - First Feature

**Date:** February 2026
**Author:** Claude Opus 4.5
**Repository:** comfyui-massmediafactory-mcp

---

## Selected Task: P0-1 Token Reduction (Phase 1-4)

### Why This First?

1. **Blocking Issue:** 23,462 tokens severely limits conversation context
2. **Quick Win:** Phase 1 alone removes 4 tools in ~2 hours
3. **No Dependencies:** Doesn't require running ComfyUI
4. **Low Risk:** Backward-compatible with aliases
5. **High Visibility:** Users will immediately notice faster responses

---

## Implementation Plan

### Phase 1: Consolidate Discovery Tools (2 hours)

**Current State:** 5 separate tools for listing models
```
list_checkpoints()
list_unets()
list_loras()
list_vaes()
list_controlnets()
```

**Target State:** 1 consolidated tool
```
list_models(model_type: str = "all")
# type: checkpoint|unet|lora|vae|controlnet|all
```

#### Step 1.1: Verify Current Implementation (15 min)

1. Read `src/comfyui_massmediafactory_mcp/server.py`
2. Locate all `list_*` tool registrations
3. Verify the consolidated `list_models()` already exists
4. Check if old individual tools still exist

**Expected:** list_models() exists but old tools may still be registered

#### Step 1.2: Remove Deprecated Tools (30 min)

**File:** `src/comfyui_massmediafactory_mcp/server.py`

Remove these tool registrations if present:
```python
# REMOVE these @mcp.tool() decorated functions:
# - list_checkpoints()
# - list_unets()
# - list_loras()
# - list_vaes()
# - list_controlnets()
```

Keep the underlying functions in `discovery.py` (they're called by list_models).

#### Step 1.3: Update Tests (30 min)

**File:** `tests/test_discovery.py` (create if not exists)

```python
import pytest
from comfyui_massmediafactory_mcp import discovery

class TestListModels:
    def test_list_all_models(self):
        result = discovery.list_models("all")
        assert "checkpoints" in result
        assert "loras" in result

    def test_list_checkpoints(self):
        result = discovery.list_models("checkpoint")
        assert isinstance(result, dict)

    def test_invalid_type_error(self):
        result = discovery.list_models("invalid")
        assert "error" in result
```

#### Step 1.4: Update Annotations (15 min)

**File:** `src/comfyui_massmediafactory_mcp/annotations.py`

Remove annotations for deprecated tools, update category for list_models.

#### Step 1.5: Verify Token Reduction (15 min)

1. Run `pytest tests/ -v` to ensure tests pass
2. Install package: `pip install -e .`
3. Start MCP server and check Claude Code context warnings
4. Expected reduction: ~720 tokens

---

### Phase 2: Convert Reference Tools to MCP Resources (4 hours)

**Current State:** ~25 reference/documentation tools consuming context
**Target State:** 13 MCP Resources (static content)

#### Step 2.1: Identify Tools to Remove (30 min)

**Tools to convert to Resources:**
```python
# Schema introspection (MCP handles natively)
get_tool_schema
list_tool_schemas
get_tool_output_schema
get_tool_annotation
list_user_facing_tools
list_tools_by_category

# Reference documentation
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

#### Step 2.2: Create MCP Resource Definitions (2 hours)

**File:** `src/comfyui_massmediafactory_mcp/server.py`

Add resource registrations:
```python
# =============================================================================
# MCP Resources (Static Documentation)
# =============================================================================

@mcp.resource("comfyui://docs/patterns/{model}")
def get_pattern_resource(model: str) -> str:
    """Get workflow pattern for a model."""
    from . import patterns
    result = patterns.get_model_pattern(model)
    if "error" in result:
        return f"Pattern not found for model: {model}"
    return json.dumps(result, indent=2)

@mcp.resource("comfyui://docs/rules")
def get_rules_resource() -> str:
    """Get parameter validation rules."""
    from . import reference_docs
    return json.dumps(reference_docs.get_parameter_rules(), indent=2)

@mcp.resource("comfyui://docs/system-prompt")
def get_system_prompt_resource() -> str:
    """Get LLM system prompt for workflow generation."""
    from . import reference_docs
    return reference_docs.get_llm_system_prompt()

@mcp.resource("comfyui://docs/skeletons/{model}-{type}")
def get_skeleton_resource(model: str, type: str) -> str:
    """Get workflow skeleton for model and type."""
    from . import workflow_generator
    skeleton, name = workflow_generator.load_skeleton(model, type)
    if skeleton is None:
        return f"Skeleton not found: {model}-{type}"
    return json.dumps(skeleton, indent=2)

@mcp.resource("comfyui://patterns/available")
def get_available_patterns() -> str:
    """List all available workflow patterns."""
    from . import patterns
    return json.dumps(patterns.list_available_patterns(), indent=2)

@mcp.resource("comfyui://workflows/supported")
def get_supported_workflows() -> str:
    """List all supported model+type combinations."""
    from . import workflow_generator
    return json.dumps(workflow_generator.list_supported_workflows(), indent=2)
```

#### Step 2.3: Remove Tool Registrations (1 hour)

Remove `@mcp.tool()` decorators from the 25 tools identified.
Keep the underlying functions for internal use.

#### Step 2.4: Update CLAUDE.md (30 min)

Document new resource URIs:
```markdown
## MCP Resources (13)

| URI | Content |
|-----|---------|
| `comfyui://docs/patterns/{model}` | Workflow patterns |
| `comfyui://docs/rules` | Parameter validation rules |
| `comfyui://docs/system-prompt` | LLM generation guide |
| `comfyui://docs/skeletons/{model}-{type}` | Workflow skeletons |
| `comfyui://patterns/available` | All available patterns |
| `comfyui://workflows/supported` | Supported combinations |
```

---

### Phase 3: Consolidate Style Learning (2 hours)

**Current State:** 10 tools for style learning
**Target State:** 4 consolidated tools

#### Step 3.1: Consolidate Suggestion Tools (1 hour)

**Current:**
```python
suggest_prompt_enhancement(prompt)
find_similar_prompts(prompt)
get_best_seeds_for_style(style)
```

**Target:**
```python
@mcp.tool()
def style_suggest(
    mode: str,  # "prompt" | "seeds" | "similar"
    prompt: str = None,
    style: str = None,
    tags: list = None,
    limit: int = 5,
    min_rating: float = 0.7,
    model: str = None
) -> dict:
    """Style suggestions. mode: prompt|seeds|similar"""
    if mode == "prompt":
        return style_learning.suggest_prompt_enhancement(prompt)
    elif mode == "seeds":
        return style_learning.get_best_seeds_for_style(style or prompt)
    elif mode == "similar":
        return style_learning.find_similar_prompts(prompt)
    else:
        return mcp_error(f"Invalid mode: {mode}. Use: prompt|seeds|similar")
```

#### Step 3.2: Consolidate Preset Tools (45 min)

**Current:**
```python
save_style_preset(name, data)
get_style_preset(name)
list_style_presets()
delete_style_preset(name)  # May not exist
```

**Target:**
```python
@mcp.tool()
def manage_presets(
    action: str,  # "list" | "get" | "save" | "delete"
    name: str = None,
    description: str = None,
    prompt_additions: str = None,
    negative_additions: str = "",
    recommended_model: str = None,
    recommended_params: dict = None
) -> dict:
    """Manage style presets. action: list|get|save|delete"""
    if action == "list":
        return style_learning.list_style_presets()
    elif action == "get":
        return style_learning.get_style_preset(name)
    elif action == "save":
        return style_learning.save_style_preset(name, {...})
    elif action == "delete":
        return style_learning.delete_style_preset(name)
```

#### Step 3.3: Remove Old Tool Registrations (15 min)

Remove individual tool registrations, keep underlying functions.

---

### Phase 4: Minimize Docstrings (3 hours)

**Current State:** ~180 tokens per docstring (with Args, Returns, Examples)
**Target State:** ~30 tokens per docstring (one-liner)

#### Step 4.1: Create Docstring Template (15 min)

**Pattern:**
```python
# Before (180 tokens):
def execute_workflow(workflow: dict, client_id: str = "massmediafactory") -> dict:
    """
    Execute a ComfyUI workflow and return the prompt_id for tracking.

    Args:
        workflow: The workflow JSON with node definitions.
        client_id: Optional identifier for tracking.

    Returns:
        prompt_id for polling results.

    Example:
        result = execute_workflow({"1": {...}})
    """

# After (30 tokens):
def execute_workflow(workflow: dict, client_id: str = "massmediafactory") -> dict:
    """Queue workflow for execution. Returns prompt_id."""
```

#### Step 4.2: Apply to All Tools (2.5 hours)

**File:** `src/comfyui_massmediafactory_mcp/server.py`

Apply minimized docstrings to all ~32 remaining tools:

| Tool | New Docstring |
|------|---------------|
| `list_models(type)` | `"List models by type. type: checkpoint\|unet\|lora\|vae\|controlnet\|all"` |
| `get_node_info(node_type)` | `"Get ComfyUI node schema by class name."` |
| `search_nodes(query)` | `"Search ComfyUI nodes by name/category."` |
| `execute_workflow(workflow)` | `"Queue workflow for execution. Returns prompt_id."` |
| `get_workflow_status(prompt_id)` | `"Check workflow/queue status. With prompt_id: single job. Without: all jobs."` |
| `wait_for_completion(prompt_id)` | `"Wait for workflow completion. Returns outputs."` |
| `get_system_stats()` | `"Get GPU VRAM and system stats."` |
| `free_memory(unload_models)` | `"Free GPU memory. unload_models=True to clear all."` |
| `interrupt()` | `"Stop currently running workflow."` |
| `regenerate(asset_id, ...)` | `"Re-run workflow with modified params. Returns new prompt_id."` |
| `list_assets(type, limit)` | `"List generated assets. type: images\|video\|audio"` |
| `get_asset_metadata(asset_id)` | `"Get full asset metadata including workflow."` |
| `view_output(asset_id, mode)` | `"View asset. mode: thumb\|metadata"` |
| `cleanup_assets()` | `"Remove expired assets (default 24h TTL)."` |
| `upload_image(path, ...)` | `"Upload image for ControlNet/I2V workflows."` |
| `download_output(asset_id, path)` | `"Download asset to local file."` |
| `publish_asset(asset_id, ...)` | `"Publish asset to web dir. Use target_filename or manifest_key."` |
| `get_publish_info()` | `"Get current publish directory config."` |
| `set_publish_dir(path)` | `"Set publish directory path."` |
| `workflow_library(action, ...)` | `"Workflow library. action: save\|load\|list\|delete\|duplicate\|export\|import"` |
| `estimate_vram(workflow)` | `"Estimate VRAM usage for workflow."` |
| `check_model_fits(model, precision)` | `"Check if model fits in VRAM. precision: fp32\|fp16\|bf16\|fp8\|default"` |
| `validate_workflow(workflow, ...)` | `"Validate workflow. auto_fix=True to correct params. check_pattern=True for drift detection."` |
| `sota_query(mode, ...)` | `"SOTA queries. mode: category\|recommend\|check\|settings\|installed"` |
| `list_workflow_templates()` | `"List available workflow templates. Paginated."` |
| `get_template(name)` | `"Get workflow template by name."` |
| `create_workflow_from_template(name, params)` | `"Create workflow from template. Injects {{PLACEHOLDER}} values."` |
| `get_workflow_skeleton(model, task)` | `"Get tested workflow structure for model+task."` |
| `get_model_constraints(model)` | `"Get model constraints (CFG, resolution, frames, required nodes)."` |
| `get_node_chain(model, task)` | `"Get ordered nodes with exact connection slots."` |
| `batch_execute(workflow, mode, ...)` | `"Batch execution. mode: batch\|sweep\|seeds"` |
| `execute_pipeline_stages(stages, params)` | `"Run multi-stage pipeline (e.g., image→upscale→video)."` |
| `run_image_to_video_pipeline(...)` | `"Generate image then animate to video."` |
| `run_upscale_pipeline(...)` | `"Generate image then upscale."` |
| `search_civitai(query, type)` | `"Search Civitai for models. type: checkpoint\|lora\|embedding\|controlnet\|upscaler"` |
| `download_model(url, type)` | `"Download model from Civitai/HF. type: checkpoint\|unet\|lora\|vae\|controlnet\|clip"` |
| `get_model_info(path)` | `"Get info about installed model."` |
| `list_installed_models(type)` | `"List installed models by type."` |
| `get_image_dimensions(asset_id)` | `"Get image dimensions and recommended video size."` |
| `detect_objects(asset_id, objects)` | `"Detect objects in image via VLM."` |
| `get_video_info(asset_id)` | `"Get video duration, fps, frame count."` |
| `qa_output(asset_id, prompt, checks)` | `"QA check via VLM. checks: prompt_match\|artifacts\|faces\|text\|composition"` |
| `check_vlm_available(model)` | `"Check if VLM (Ollama) is available for QA."` |
| `record_generation(prompt, model, seed)` | `"Record a generation for style learning. Returns record_id."` |
| `rate_generation(record_id, rating)` | `"Rate a generation 0.0-1.0. Returns success status."` |
| `style_suggest(mode, ...)` | `"Style suggestions. mode: prompt\|seeds\|similar"` |
| `manage_presets(action, ...)` | `"Manage style presets. action: list\|get\|save\|delete"` |
| `generate_workflow(model, type, prompt, ...)` | `"Generate validated workflow. model: ltx\|flux\|wan\|qwen. type: t2v\|i2v\|t2i"` |

#### Step 4.3: Run Tests (15 min)

```bash
pytest tests/ -v
```

Ensure all tests still pass after docstring changes.

---

## Verification Checklist

### After Phase 1
- [ ] `list_models("checkpoint")` returns checkpoints
- [ ] `list_models("all")` returns all model types
- [ ] `list_models("invalid")` returns error
- [ ] Old tools (list_checkpoints, etc.) no longer exist as MCP tools
- [ ] Token count decreased by ~720

### After Phase 2
- [ ] Resources accessible: `comfyui://docs/patterns/flux`
- [ ] 25 reference tools removed
- [ ] Token count decreased by ~4,500
- [ ] CLAUDE.md updated with resource URIs

### After Phase 3
- [ ] `style_suggest("prompt", prompt="...")` works
- [ ] `manage_presets("list")` returns presets
- [ ] 10 style tools reduced to 4
- [ ] Token count decreased by ~1,080

### After Phase 4
- [ ] All tools have one-liner docstrings
- [ ] Tests still pass
- [ ] Token count decreased by ~10,800

### Final Verification
- [ ] Total tool count: ~32
- [ ] Total token count: <10,000
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Package installs: `pip install -e .`

---

## Rollback Plan

If issues arise at any phase:

1. **Git revert:** `git revert HEAD~N` to undo commits
2. **Reinstall:** `pip install -e .`
3. **Restart MCP server**

Keep old function implementations (just remove decorators) so internals still work.

---

## Time Estimate Summary

| Phase | Task | Time |
|-------|------|------|
| Phase 1 | Consolidate discovery | 2h |
| Phase 2 | Reference → Resources | 4h |
| Phase 3 | Consolidate style learning | 2h |
| Phase 4 | Minimize docstrings | 3h |
| **Total** | | **11h** |

---

## Success Criteria

1. **Token count:** <10,000 (from ~23,462)
2. **Tool count:** ~32 (from ~100)
3. **All tests pass:** `pytest tests/ -v`
4. **Backward compatible:** Old tool names work via aliases or clear error message
5. **Documentation updated:** CLAUDE.md, README.md accurate
