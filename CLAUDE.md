# ComfyUI MassMediaFactory MCP - Agent Guide

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

### Step 1: Choose Your Approach

| Approach | When to Use |
|----------|-------------|
| `generate_workflow()` | Best for most cases - generates validated workflows |
| `get_template()` + `create_workflow_from_template()` | When you need specific template variations |
| Build from scratch | Only when you need custom node combinations |

### Step 2: Execute & Monitor

```python
result = execute_workflow(workflow)
# result["prompt_id"] is your tracking ID

# Option A: Wait (blocking)
output = wait_for_completion(result["prompt_id"], timeout_seconds=600)

# Option B: Poll status (non-blocking)
status = get_workflow_status(result["prompt_id"])
# status["status"]: "queued" | "running" | "completed" | "error"
```

### Step 3: Iterate

```python
# Tweak and regenerate
new_result = regenerate(
    asset_id=output["outputs"][0]["asset_id"],
    cfg=4.5,      # Adjust CFG
    seed=None     # New random seed (or specific number)
)
```

## Tool Reference (48 Tools)

### Discovery (3 tools)

| Tool | Usage |
|------|-------|
| `list_models(type)` | `type`: checkpoint\|unet\|lora\|vae\|controlnet\|all |
| `get_node_info(node_type)` | Get node schema by class name |
| `search_nodes(query)` | Search nodes by name/category |

### Execution (6 tools)

| Tool | Usage |
|------|-------|
| `execute_workflow(workflow)` | Queue workflow, returns `prompt_id` |
| `get_workflow_status(prompt_id)` | With ID: single job. Without: queue status |
| `wait_for_completion(prompt_id)` | Block until done, returns outputs |
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
| `publish_asset(asset_id, ...)` | Export to web directory |
| `get_publish_info()` | Current publish config |
| `set_publish_dir(path)` | Set publish directory |

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
| `search_civitai(query)` | Search Civitai |
| `download_model(url, type)` | Download from Civitai/HF |
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

## Error Handling

All errors include `isError: true` and a `code`:

| Code | Meaning |
|------|---------|
| `INVALID_PARAMS` | Bad parameter values |
| `NOT_FOUND` | Asset/workflow not found |
| `TIMEOUT` | Execution timed out |
| `COMFYUI_ERROR` | ComfyUI returned error |
| `VALIDATION_ERROR` | Workflow validation failed |

## Tips

1. **Always use `generate_workflow()`** - it handles model-specific constraints
2. **Check VRAM first** with `check_model_fits()` for large models
3. **Use `validate_workflow(auto_fix=True)`** to auto-correct common issues
4. **Iterate with `regenerate()`** - faster than rebuilding workflows
5. **Read resources** for model-specific patterns when building custom workflows
