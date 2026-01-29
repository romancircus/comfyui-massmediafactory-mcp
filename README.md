# ComfyUI MassMediaFactory MCP

A Model Context Protocol (MCP) server for ComfyUI workflow orchestration. Enables Claude and other AI assistants to **create, iterate, and maintain** image and video generation pipelines.

## Features

- **Workflow Generation**: Generate validated workflows for FLUX, LTX-Video, Wan, Qwen
- **Execution**: Run workflows, monitor status, retrieve outputs
- **Iteration**: Regenerate with parameter tweaks (prompt, CFG, seed)
- **Batch Processing**: Seed variations, parameter sweeps, pipelines
- **Memory Management**: Check VRAM, free memory, interrupt jobs
- **Quality Assurance**: VLM-based output validation via Ollama

## Installation

```bash
pip install comfyui-massmediafactory-mcp
```

Or install from source:

```bash
git clone https://github.com/romancircus/comfyui-massmediafactory-mcp
cd comfyui-massmediafactory-mcp
pip install -e .
```

## Configuration

Set your ComfyUI URL via environment variable:

```bash
export COMFYUI_URL="http://localhost:8188"
```

## Usage with Claude Code

```bash
claude mcp add --transport stdio --scope user comfyui-massmediafactory \
    -- comfyui-massmediafactory-mcp
```

Or from source:

```bash
claude mcp add --transport stdio --scope user comfyui-massmediafactory \
    -- python -m comfyui_massmediafactory_mcp.server
```

## Quick Start

```python
# Generate an image with FLUX
workflow = generate_workflow(model="flux", workflow_type="t2i",
    prompt="a dragon in the clouds", width=1024, height=1024)
result = execute_workflow(workflow["workflow"])
output = wait_for_completion(result["prompt_id"])

# Iterate with higher CFG
new_result = regenerate(output["outputs"][0]["asset_id"], cfg=4.5)
```

## Available Tools (48)

### Discovery

| Tool | Description |
|------|-------------|
| `list_models(type)` | List models. type: checkpoint\|unet\|lora\|vae\|controlnet\|all |
| `get_node_info(node_type)` | Get node schema by class name |
| `search_nodes(query)` | Search nodes by name/category |

### Execution

| Tool | Description |
|------|-------------|
| `execute_workflow(workflow)` | Queue workflow, returns prompt_id |
| `get_workflow_status(prompt_id)` | With ID: single job status. Without: queue status |
| `wait_for_completion(prompt_id)` | Wait and return outputs |
| `get_system_stats()` | GPU VRAM and system info |
| `free_memory(unload_models)` | Free GPU memory |
| `interrupt()` | Stop current workflow |

### Assets

| Tool | Description |
|------|-------------|
| `regenerate(asset_id, ...)` | Re-run with parameter tweaks (prompt, cfg, seed) |
| `list_assets(type, limit)` | Browse outputs. type: images\|video\|audio |
| `get_asset_metadata(asset_id)` | Full metadata including workflow |
| `view_output(asset_id, mode)` | View asset. mode: thumb\|metadata |
| `cleanup_assets()` | Remove expired assets (24h TTL) |
| `upload_image(path, ...)` | Upload for ControlNet/I2V workflows |
| `download_output(asset_id, path)` | Save to local disk |

### Workflow Generation

| Tool | Description |
|------|-------------|
| `generate_workflow(model, type, prompt, ...)` | Generate validated workflow |
| `validate_workflow(workflow, auto_fix)` | Validate with optional auto-fix |
| `get_workflow_skeleton(model, task)` | Get base workflow structure |
| `get_model_constraints(model)` | CFG, resolution, frame limits |
| `get_node_chain(model, task)` | Node order with connections |

### Workflow Library

| Tool | Description |
|------|-------------|
| `workflow_library(action, ...)` | action: save\|load\|list\|delete\|duplicate\|export\|import |

### Templates

| Tool | Description |
|------|-------------|
| `list_workflow_templates()` | List available templates |
| `get_template(name)` | Get template JSON |
| `create_workflow_from_template(name, params)` | Create from template |

### Batch & Pipelines

| Tool | Description |
|------|-------------|
| `batch_execute(workflow, mode, ...)` | mode: batch\|sweep\|seeds |
| `execute_pipeline_stages(stages, params)` | Multi-stage pipelines |
| `run_image_to_video_pipeline(...)` | Image-to-video pipeline |
| `run_upscale_pipeline(...)` | Generate + upscale pipeline |

### Quality Assurance

| Tool | Description |
|------|-------------|
| `qa_output(asset_id, prompt, checks)` | VLM-based quality check |
| `check_vlm_available(model)` | Check Ollama VLM status |

**QA Checks:** prompt_match, artifacts, faces, text, composition

### VRAM & Models

| Tool | Description |
|------|-------------|
| `estimate_vram(workflow)` | Estimate VRAM usage |
| `check_model_fits(model, precision)` | Check if model fits |
| `search_civitai(query, type)` | Search Civitai |
| `download_model(url, type)` | Download from Civitai/HF |
| `list_installed_models(type)` | List installed models |
| `get_model_info(path)` | Installed model info |

### Style Learning

| Tool | Description |
|------|-------------|
| `record_generation(prompt, model, seed)` | Log generation |
| `rate_generation(record_id, rating)` | Rate 0.0-1.0 |
| `style_suggest(mode, ...)` | mode: prompt\|seeds\|similar |
| `manage_presets(action, ...)` | action: list\|get\|save\|delete |

## MCP Resources (13)

Static documentation accessible via MCP resource protocol:

| URI | Content |
|-----|---------|
| `comfyui://docs/patterns/{model}` | Workflow patterns (flux, ltx, wan, qwen) |
| `comfyui://docs/rules` | Parameter validation rules |
| `comfyui://docs/system-prompt` | LLM generation guide |
| `comfyui://docs/skeletons/{model}-{type}` | Workflow skeletons |
| `comfyui://patterns/available` | All available patterns |
| `comfyui://workflows/supported` | Supported model+type combinations |

## Supported Models

| Model | Types | Notes |
|-------|-------|-------|
| FLUX | t2i | Text-to-image, 1024x1024 default |
| LTX-Video | t2v, i2v | Text/image-to-video, 768x512 default |
| Wan 2.1 | t2v | Text-to-video |
| Qwen | t2i | Text-to-image |

## Example: Batch Seed Variations

```python
# Generate 4 variations with different seeds
results = batch_execute(
    workflow=workflow,
    mode="seeds",
    num_variations=4,
    start_seed=42
)
```

## Example: Parameter Sweep

```python
# Test CFG and steps combinations
results = batch_execute(
    workflow=workflow,
    mode="sweep",
    sweep_params={"cfg": [2.5, 3.5, 4.5], "steps": [20, 30]}
)
# Runs 6 combinations (3 CFG x 2 steps)
```

## Requirements

- ComfyUI running and accessible
- Python 3.10+
- For QA: Ollama with VLM model (`ollama pull qwen2.5-vl:7b`)

## Documentation

- **[CLAUDE.md](./CLAUDE.md)** - Comprehensive agent guide
- **[docs/](./docs/)** - Additional documentation

## License

MIT
