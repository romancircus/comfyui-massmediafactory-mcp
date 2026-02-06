# ComfyUI MassMediaFactory MCP - Agent Guide

---

## Cyrus Execution Tasks

**For multi-phase overnight work, use the centralized orchestration pattern.**

### Quick Start

1. **Copy template:**
   ```bash
   cp ~/.cyrus/templates/execution_script_template.py scripts/<task_name>_execute.py
   ```

2. **Implement phases:**
   ```python
   class MyExecutor(OrchestrationScript):
       def get_phases(self):
           return {
               "1": self.phase_1_setup,
               "2": self.phase_2_process,
               "3": self.phase_3_verify,
           }
   ```

3. **Create Linear issue from template:**
   ```bash
   cat ~/.cyrus/templates/linear_execution_issue.md
   # Replace <REPO_NAME>, <SCRIPT_NAME>, <ISSUE_ID>
   # Paste into Linear issue description
   ```

4. **Validate before delegating:**
   ```bash
   python ~/.cyrus/scripts/validate_execution_issue.py ROM-XXX
   ```

5. **Delegate to Cyrus** - execution happens automatically

### Resources

- **Template:** `~/.cyrus/templates/execution_script_template.py` (640 lines - complete base class)
- **Docs:** `~/.cyrus/docs/EXECUTION_PATTERN.md` (usage guide)
- **Issue template:** `~/.cyrus/templates/linear_execution_issue.md`
- **Validation:** `~/.cyrus/scripts/validate_execution_issue.py`

### Why Use This Pattern

- Worktree-aware (auto-setup symlinks)
- Crash recovery (checkpoint system)
- Progress tracking (Linear comments)
- Overnight execution (background-safe)
- Proven pattern (ROM-121 reference)

**DON'T:** Create multiple Linear issues with `blockedBy` (doesn't auto-trigger)
**DO:** Single issue, single orchestration script, all phases sequential

---

## Linear Integration

**Check Linear before starting work:**
```python
mcp__linear__list_issues(project="Infra: ComfyUI MCP")
```

**Active issues:**
- ROM-10: Template System Improvements

**Update status when working:**
```python
mcp__linear__update_issue("ROM-10", state="in_progress")
mcp__linear__update_issue("ROM-10", state="done", comment="Added: [feature]")
```

---

## Quick Start

This MCP server lets you generate images and videos via ComfyUI. The fastest path:

```python
# 1. Generate a workflow
workflow = generate_workflow(model="flux", workflow_type="t2i", prompt="a dragon in the clouds")

# 2. Execute it
result = execute_workflow(workflow["workflow"])

# 3. Wait for output
output = wait_for_completion(result["prompt_id"])
# output["outputs"][0]["asset_id"] contains your generated image
```

## Core Workflow

### Step 1: Choose Your Approach (Dual Generation Path)

There are two complementary generation paths. Choose based on your needs:

**Path A: `generate_workflow()` - Auto-Generated Workflows**
- 12 model+type combos (flux/t2i, ltx/t2v, ltx/i2v, wan/t2v, qwen/t2i, etc.)
- Auto-corrects resolution, frames, CFG to model constraints
- Best for: prompt-to-output with standard settings

**Path B: `create_workflow_from_template()` - Template-Based Workflows**
- 30+ templates including advanced workflows (ControlNet, LoRA stacking, TTS, upscaling, inpainting)
- Exact configurations preserved, no auto-correction
- Best for: specialized workflows, batch production, Cyrus overnight

| Need | Use Path A | Use Path B |
|------|-----------|-----------|
| Quick image/video from prompt | `generate_workflow(model="flux", ...)` | |
| ControlNet/LoRA/face ID | | `create_workflow_from_template("flux2_union_controlnet", ...)` |
| Background replacement | | `create_workflow_from_template("qwen_edit_background", ...)` |
| Standard t2v/i2v/t2i | Either works | Either works |
| Batch production (>10) | | Templates + direct API (see Token Optimization) |
| TTS/audio | | `create_workflow_from_template("chatterbox_tts", ...)` |
| Custom node combos | Build from scratch | |

### Step 2: Execute & Monitor

```python
result = execute_workflow(workflow)
# result["prompt_id"] is your tracking ID

# ALWAYS use blocking wait (uses WebSocket internally, falls back to polling)
output = wait_for_completion(result["prompt_id"], timeout_seconds=600)

# For real-time progress (percent, ETA, nodes completed):
progress = get_progress(result["prompt_id"])
```

**WARNING:** Do NOT poll `get_workflow_status()` in a loop. Use `wait_for_completion()` which blocks
internally via WebSocket. Polling burns 100x more API calls/tokens. See global CLAUDE.md for details.

### Step 3: Iterate

```python
# Tweak and regenerate
new_result = regenerate(
    asset_id=output["outputs"][0]["asset_id"],
    cfg=4.5,      # Adjust CFG
    seed=None     # New random seed (or specific number)
)
```

## Tool Reference (58 Tools)

### Discovery (3 tools)

| Tool | Usage |
|------|-------|
| `list_models(type)` | `type`: checkpoint\|unet\|lora\|vae\|controlnet\|all |
| `get_node_info(node_type)` | Get node schema by class name |
| `search_nodes(query)` | Search nodes by name/category |

### Execution (7 tools)

| Tool | Usage |
|------|-------|
| `execute_workflow(workflow)` | Queue workflow, returns `prompt_id` |
| `get_workflow_status(prompt_id)` | With ID: single job. Without: queue status |
| `wait_for_completion(prompt_id)` | **Block until done** (WebSocket + polling fallback), returns outputs |
| `get_progress(prompt_id)` | Real-time progress: stage, percent, ETA, nodes completed/total |
| `get_system_stats()` | GPU VRAM and system info |
| `free_memory(unload_models)` | Free GPU memory |
| `interrupt()` | Stop current workflow |

### Assets (7 tools)

| Tool | Usage |
|------|-------|
| `regenerate(asset_id, ...)` | Re-run with tweaks (prompt, cfg, seed, steps) |
| `list_assets(type, limit)` | Browse outputs. `type`: images\|video\|audio |
| `get_asset_metadata(asset_id)` | Full metadata including workflow |
| `view_output(asset_id, mode)` | `mode`: thumb\|metadata |
| `cleanup_assets()` | Remove expired assets (24h TTL) |
| `upload_image(path)` | Upload for ControlNet/I2V |
| `download_output(asset_id, path)` | Save to local disk |

### Publishing (3 tools)

| Tool | Usage |
|------|-------|
| `publish_asset(asset_id, ...)` | Export to web directory (path-traversal protected) |
| `get_publish_info()` | Current publish config |
| `set_publish_dir(path)` | Set publish directory (validated against blocklist) |

### Workflow Library (1 tool)

| Tool | Usage |
|------|-------|
| `workflow_library(action, ...)` | `action`: save\|load\|list\|delete\|duplicate\|export\|import |

### Validation & VRAM (3 tools)

| Tool | Usage |
|------|-------|
| `validate_workflow(workflow, auto_fix, check_pattern)` | Validate with optional auto-fix |
| `estimate_vram(workflow)` | Estimate VRAM usage |
| `check_model_fits(model, precision)` | Check if model fits in VRAM |

### Workflow Generation (4 tools)

| Tool | Usage |
|------|-------|
| `generate_workflow(model, type, prompt)` | Generate validated workflow |
| `get_workflow_skeleton(model, task)` | Get base structure |
| `get_model_constraints(model)` | CFG, resolution, frames limits |
| `get_node_chain(model, task)` | Node order with connections |

### Templates (3 tools)

| Tool | Usage |
|------|-------|
| `list_workflow_templates()` | List available templates |
| `get_template(name)` | Get template JSON |
| `create_workflow_from_template(name, params)` | Create from template |

### Batch & Pipelines (4 tools)

| Tool | Usage |
|------|-------|
| `batch_execute(workflow, mode, ...)` | `mode`: batch\|sweep\|seeds |
| `execute_pipeline_stages(stages, params)` | Multi-stage pipelines |
| `run_image_to_video_pipeline(...)` | Image → Video pipeline |
| `run_upscale_pipeline(...)` | Generate → Upscale pipeline |

### Models (4 tools)

| Tool | Usage |
|------|-------|
| `search_civitai(query)` | Search Civitai (domain-validated) |
| `download_model(url, type)` | Download from Civitai/HF (URL-validated) |
| `get_model_info(path)` | Installed model info |
| `list_installed_models(type)` | List installed models |

### Analysis & QA (5 tools)

| Tool | Usage |
|------|-------|
| `get_image_dimensions(asset_id)` | Dimensions + recommended video size |
| `get_video_info(asset_id)` | Duration, fps, frames |
| `detect_objects(asset_id, objects)` | VLM object detection |
| `qa_output(asset_id, prompt)` | VLM quality check |
| `check_vlm_available()` | Check Ollama VLM status |

### SOTA & Style (5 tools)

| Tool | Usage |
|------|-------|
| `sota_query(mode, ...)` | `mode`: category\|recommend\|check\|settings\|installed |
| `record_generation(prompt, model, seed)` | Log for style learning |
| `rate_generation(record_id, rating)` | Rate 0.0-1.0 |
| `style_suggest(mode, ...)` | `mode`: prompt\|seeds\|similar |
| `manage_presets(action, ...)` | `action`: list\|get\|save\|delete |

### Visualization (2 tools)

| Tool | Usage |
|------|-------|
| `visualize_workflow(workflow)` | Generate Mermaid diagram from workflow |
| `get_workflow_summary(workflow)` | Text summary of workflow structure (nodes, params) |

### Rate Limiting (3 tools)

| Tool | Usage |
|------|-------|
| `get_rate_limit_status(tool_name)` | Per-tool rate limit status (remaining, reset time) |
| `get_all_tools_rate_status()` | Rate status for all tools at once |
| `get_rate_limit_summary()` | Brief dashboard summary |

### Prompt & Performance (4 tools) - Round 2

| Tool | Usage |
|------|-------|
| `enhance_prompt(prompt, model, style)` | LLM-powered prompt enhancement with model-specific quality tokens. Uses local Ollama, falls back to token injection |
| `get_execution_profile(prompt_id)` | Per-node execution timing for completed workflows. Identifies slowest nodes for optimization |
| `diff_workflows(workflow_a, workflow_b)` | Compare two workflows showing added/removed/modified nodes and params |
| `get_compatibility_matrix()` | Model compatibility matrix: installed models, VRAM fit, supported workflow types |

## MCP Resources (13 resources)

Access via `ReadMcpResourceTool`:

| URI | Content |
|-----|---------|
| `comfyui://docs/patterns/flux` | FLUX workflow pattern |
| `comfyui://docs/patterns/ltx` | LTX-Video pattern |
| `comfyui://docs/patterns/wan` | Wan 2.1 pattern |
| `comfyui://docs/patterns/qwen` | Qwen pattern |
| `comfyui://docs/rules` | Parameter validation rules |
| `comfyui://docs/system-prompt` | LLM workflow generation guide |
| `comfyui://docs/skeletons/flux-t2i` | FLUX text-to-image skeleton |
| `comfyui://docs/skeletons/ltx-t2v` | LTX text-to-video skeleton |
| `comfyui://docs/skeletons/ltx-i2v` | LTX image-to-video skeleton |
| `comfyui://docs/skeletons/wan-t2v` | Wan text-to-video skeleton |
| `comfyui://docs/skeletons/qwen-t2i` | Qwen text-to-image skeleton |
| `comfyui://patterns/available` | All available patterns |
| `comfyui://workflows/supported` | Supported model+type combos |

## Supported Models & Workflows

| Model | Supported Types |
|-------|-----------------|
| `flux` | t2i (text-to-image) |
| `ltx` | t2v (text-to-video), i2v (image-to-video) |
| `wan` | t2v (text-to-video) |
| `qwen` | t2i (text-to-image) |
| `qwen_edit` | edit (image editing, background replacement) |

## Common Patterns

### Text-to-Image (FLUX)

```python
wf = generate_workflow(model="flux", workflow_type="t2i",
    prompt="cyberpunk city at night, neon lights",
    width=1024, height=1024, steps=20, cfg=3.5)
result = execute_workflow(wf["workflow"])
output = wait_for_completion(result["prompt_id"])
```

### Text-to-Video (LTX)

```python
wf = generate_workflow(model="ltx", workflow_type="t2v",
    prompt="a cat walking through a garden",
    width=768, height=512, frames=97, steps=30)
result = execute_workflow(wf["workflow"])
output = wait_for_completion(result["prompt_id"], timeout_seconds=900)
```

### Image-to-Video (LTX)

```python
# Upload source image first
upload = upload_image("/path/to/image.png")

wf = generate_workflow(model="ltx", workflow_type="i2v",
    prompt="the scene comes to life with gentle motion")
# Inject the uploaded image into the workflow
# ... then execute
```

### Batch Seed Variations

```python
results = batch_execute(
    workflow=wf["workflow"],
    mode="seeds",
    num_variations=4,
    start_seed=42
)
# Returns 4 variations with seeds 42, 43, 44, 45
```

### Parameter Sweep

```python
results = batch_execute(
    workflow=wf["workflow"],
    mode="sweep",
    sweep_params={"cfg": [2.5, 3.5, 4.5], "steps": [20, 30]}
)
# Tests all combinations: 6 total runs
```

### Background Replacement (Qwen Edit)

```python
# Use the template for Qwen Edit background replacement
template = get_template("qwen_edit_background")
wf = create_workflow_from_template("qwen_edit_background", {
    "IMAGE_PATH": "uploaded_image.png",
    "EDIT_PROMPT": "Change the background to an outdoor playground with green grass and blue sky. Keep the child exactly the same.",
    "SEED": 42,
    "CFG": 2.0,  # CRITICAL: Keep low (2.0-2.5). Higher causes color distortion.
    "STEPS": 20
})
result = execute_workflow(wf)
output = wait_for_completion(result["prompt_id"])
```

**Qwen Edit Critical Settings:**
| Setting | Value | Why |
|---------|-------|-----|
| CFG | **2.0-2.5** | PRIMARY color control. >4 causes oversaturation |
| Denoise | **1.0** | Must be 1.0 for background changes. Lower preserves original including background |
| Latent | EmptyQwenImageLayeredLatentImage | NOT VAEEncode. VAEEncode fails for background replacement |

## Security Model

All external I/O is validated:

| Protection | Implementation |
|------------|----------------|
| **Path validation** | `Path.resolve()` + `is_relative_to()` prevents traversal (not `startswith()`) |
| **URL validation** | `urlparse().hostname` + exact domain/subdomain matching. Allowed: civitai.com, huggingface.co, github.com, raw.githubusercontent.com |
| **Publish sandboxing** | Path traversal check on filenames, blocklist on `set_publish_dir` (rejects `/etc`, `~/.ssh`, etc.) |
| **Conditioning safety** | `_find_conditioning_nodes()` traces sampler connections to identify positive vs negative prompt nodes (prevents clobbering) |
| **Rate limiting** | All 58 tools wrapped with `@mcp_tool_wrapper` providing per-tool rate limits + structured logging |

## Error Handling

All errors include `isError: true` and a `code`:

| Code | Meaning |
|------|---------|
| `INVALID_PARAMS` | Bad parameter values |
| `NOT_FOUND` | Asset/workflow not found |
| `TIMEOUT` | Execution timed out |
| `COMFYUI_ERROR` | ComfyUI returned error |
| `VALIDATION_ERROR` | Workflow validation failed |
| `HISTORY_UNAVAILABLE` | ComfyUI history API unreachable (profiling) |

## Tips

1. **Always use `wait_for_completion()`** - it uses WebSocket internally, falls back to polling. NEVER poll in a loop.
2. **Always use `generate_workflow()`** - it handles model-specific constraints
3. **Check VRAM first** with `check_model_fits()` for large models
4. **Use `validate_workflow(auto_fix=True)`** to auto-correct common issues
5. **Iterate with `regenerate()`** - faster than rebuilding workflows
6. **Enhance prompts** with `enhance_prompt()` before generation for better results
7. **Profile slow workflows** with `get_execution_profile()` to identify bottleneck nodes
8. **Check compatibility** with `get_compatibility_matrix()` to see what's ready to use
9. **Read resources** for model-specific patterns when building custom workflows
10. **For batch (>10 workflows):** Use templates + direct API (`urllib`) instead of MCP tools to save tokens
