# ComfyUI MassMediaFactory MCP

A Model Context Protocol (MCP) server for ComfyUI workflow orchestration. Enables Claude and other AI assistants to **create, iterate, and maintain** image and video generation pipelines.

## Features

- **Discovery**: Find installed models (checkpoints, LoRAs, VAEs, ControlNets)
- **Execution**: Run workflows, monitor status, retrieve outputs
- **Memory Management**: Check VRAM, free memory, interrupt jobs
- **Templates**: Pre-built workflow templates for SOTA models

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

Or for remote ComfyUI:

```bash
export COMFYUI_URL="http://your-server:8188"
```

## Usage with Claude Code

Add to Claude Code:

```bash
claude mcp add --transport stdio --scope user comfyui-massmediafactory \
    -- comfyui-massmediafactory-mcp
```

Or if running from source:

```bash
claude mcp add --transport stdio --scope user comfyui-massmediafactory \
    -- python -m comfyui_massmediafactory_mcp.server
```

## Available Tools

### Discovery

| Tool | Description |
|------|-------------|
| `list_checkpoints()` | List available checkpoint models |
| `list_unets()` | List UNET models (Flux, SD3) |
| `list_loras()` | List LoRA models |
| `list_vaes()` | List VAE models |
| `list_controlnets()` | List ControlNet models |
| `get_node_info(node_type)` | Get node schema (inputs/outputs) |
| `search_nodes(query)` | Search nodes by name/category |
| `get_all_models()` | Summary of all installed models |

### Execution

| Tool | Description |
|------|-------------|
| `execute_workflow(workflow)` | Queue workflow, get prompt_id |
| `get_workflow_status(prompt_id)` | Check status (queued/running/completed) |
| `wait_for_completion(prompt_id)` | Wait and return outputs |
| `get_queue_status()` | Get running/pending job counts |

### Management

| Tool | Description |
|------|-------------|
| `get_system_stats()` | GPU VRAM usage and system info |
| `free_memory(unload_models)` | Free GPU memory |
| `interrupt_execution()` | Stop current workflow |

### Asset Iteration (NEW)

| Tool | Description |
|------|-------------|
| `regenerate(asset_id, ...)` | Re-run with parameter tweaks (prompt, cfg, seed) |
| `list_assets(session_id, limit)` | Browse recent generations |
| `get_asset_metadata(asset_id)` | Full provenance including workflow |
| `view_output(asset_id)` | Get asset URL and preview info |
| `cleanup_expired_assets()` | Remove expired assets (24h TTL) |

## Example: Generate Image with Flux

```python
# Claude discovers available models
models = list_checkpoints()
# → {"checkpoints": ["flux1-dev.safetensors", ...]}

# Claude builds and executes workflow
result = execute_workflow({
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a dragon", "clip": ["1", 1]}},
    # ... rest of workflow
})
# → {"prompt_id": "abc123", "status": "queued"}

# Claude waits for completion
output = wait_for_completion("abc123")
# → {"status": "completed", "outputs": [{"filename": "flux_00001.png", "asset_id": "abc-123-def", ...}]}
```

## Example: Iterate on Generation

```python
# Generate initial image
result = execute_workflow(workflow)
output = wait_for_completion(result["prompt_id"])
asset_id = output["outputs"][0]["asset_id"]

# View the result
info = view_output(asset_id)
# → {"url": "http://localhost:8188/view?filename=...", "prompt_preview": "a dragon..."}

# Not quite right? Iterate with higher CFG
result = regenerate(asset_id, cfg=4.5, seed=None)  # None = new random seed
output = wait_for_completion(result["prompt_id"])

# Browse all generations this session
assets = list_assets(limit=10)
# → {"assets": [{"asset_id": "...", "prompt_preview": "...", ...}], "count": 5}
```

## Roadmap

- [x] Workflow persistence (save/load)
- [x] VRAM estimation before execution
- [x] Workflow validation
- [x] SOTA tracker integration
- [x] Batch execution
- [x] Multi-stage pipelines
- [x] Asset registry with iteration support
- [ ] Inline image preview (base64)
- [ ] Asset publishing to web directories

## License

MIT
