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

### Asset Iteration

| Tool | Description |
|------|-------------|
| `regenerate(asset_id, ...)` | Re-run with parameter tweaks (prompt, cfg, seed) |
| `list_assets(session_id, limit)` | Browse recent generations |
| `get_asset_metadata(asset_id)` | Full provenance including workflow |
| `view_output(asset_id)` | Get asset URL and preview info |
| `cleanup_expired_assets()` | Remove expired assets (24h TTL) |
| `upload_image(image_path, ...)` | Upload reference images for ControlNet/IP-Adapter |
| `download_output(asset_id, path)` | Download generated files to local disk |

### Publishing

| Tool | Description |
|------|-------------|
| `publish_asset(asset_id, ...)` | Export to web directory with compression |
| `get_publish_info()` | Show publish configuration |
| `set_publish_dir(path)` | Configure publish directory |

### Quality Assurance

| Tool | Description |
|------|-------------|
| `qa_output(asset_id, prompt, checks)` | VLM-based quality check on generated images |
| `check_vlm_available(model)` | Check if Ollama VLM is available |

**QA Checks:**
- `prompt_match` - Does image match the original prompt?
- `artifacts` - Visual artifacts, distortions, blur?
- `faces` - Face/hand issues (extra fingers, asymmetry)?
- `text` - Text rendering issues?
- `composition` - Overall composition quality?

**Requirements:** Ollama with a VLM model (default: `qwen2.5-vl:7b`)
```bash
ollama pull qwen2.5-vl:7b
```

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

# Publish final image to web directory
result = publish_asset(asset_id, target_filename="hero_image.png")
# → {"success": True, "url": "/gen/hero_image.png", "bytes": 245000}
```

## Roadmap

- [x] Workflow persistence (save/load)
- [x] VRAM estimation before execution
- [x] Workflow validation (with cycle detection, resolution warnings)
- [x] SOTA tracker integration
- [x] Batch execution
- [x] Multi-stage pipelines
- [x] Asset registry with iteration support
- [x] Asset publishing to web directories
- [x] Automated VLM QA for generated outputs
- [x] Image upload API (ControlNet, IP-Adapter, I2V support)
- [x] Output download API
- [x] Workflow format conversion (UI ↔ API)
- [x] Connection type wildcards (*, COMBO, union types)
- [x] Structured error codes with retry logic
- [ ] Inline image preview (base64)
- [ ] Video QA support
- [ ] WebSocket real-time progress

## License

MIT
