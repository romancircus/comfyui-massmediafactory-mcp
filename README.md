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
# → {"status": "completed", "outputs": [{"filename": "flux_00001.png", ...}]}
```

## Roadmap

- [ ] Workflow persistence (save/load)
- [ ] VRAM estimation before execution
- [ ] Workflow validation
- [ ] SOTA tracker integration
- [ ] Batch execution
- [ ] Multi-stage pipelines

## License

MIT
