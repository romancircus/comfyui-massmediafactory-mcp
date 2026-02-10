# ComfyUI MassMediaFactory MCP

A Model Context Protocol (MCP) server for ComfyUI workflow orchestration. Enables Claude and other AI assistants to **create, iterate, and maintain** image and video generation pipelines.

## Architecture

**CLI-first design**: The `mmf` CLI handles all execution (generate, batch, pipeline). MCP tools (18) handle discovery and planning only. This gives 0 schema overhead for execution and pre-tested parameters that produce correct results.

```
mmf CLI (execution)  ──►  ComfyUI Server (:8188)
MCP tools (discovery) ──►  Node/model/template info
```

## Installation

```bash
pip install comfyui-massmediafactory-mcp
```

Or from source:

```bash
git clone https://github.com/romancircus/comfyui-massmediafactory-mcp
cd comfyui-massmediafactory-mcp
pip install -e .
```

## Configuration

```bash
export COMFYUI_URL="http://localhost:8188"
```

## Quick Start

```bash
# Text-to-image (FLUX)
mmf run --model flux --type t2i --prompt "a dragon in the clouds" -o dragon.png

# Image-to-video (WAN)
mmf run --model wan --type i2v --image photo.png --prompt "gentle motion" -o video.mp4

# Template-based (49 templates available)
mmf run --template wan26_img2vid --params '{"IMAGE_PATH":"img.png","PROMPT":"motion"}'

# Batch seed variations
mmf batch seeds workflow.json --count 8 --start-seed 42

# Pre-tested pipeline
mmf pipeline viral-short --prompt "dancing character" --style-image style.png -o video.mp4
```

## MCP Setup (Claude Code)

```bash
claude mcp add --transport stdio --scope user comfyui-massmediafactory \
    -- comfyui-massmediafactory-mcp
```

## MCP Tools (18 - Discovery Only)

| Category | Tools |
|----------|-------|
| Discovery | `list_models`, `get_node_info`, `search_nodes` |
| System | `get_system_stats`, `free_memory`, `interrupt` |
| I/O | `upload_image`, `download_output` |
| Publishing | `publish_asset`, `get_publish_info`, `set_publish_dir` |
| Validation | `validate_workflow` |
| Patterns | `get_workflow_skeleton`, `get_model_constraints`, `get_node_chain` |
| Templates | `list_workflow_templates`, `get_template` |
| Prompt | `enhance_prompt` |

## Supported Models (9 models, 49 templates)

| Model | Types |
|-------|-------|
| FLUX | t2i, controlnet, lora, face_id, inpaint, edit |
| LTX-2 | t2v, i2v, v2v, audio_reactive |
| Wan 2.2/2.6 | t2v, i2v, s2v, flf2v, camera_i2v, animate |
| Qwen | t2i, controlnet, poster, edit |
| HunyuanVideo | t2v, i2v |
| Z-Image-Turbo | t2i |
| SDXL | t2i |
| Audio | tts (chatterbox, f5, qwen3), v2a (mmaudio) |
| Utility | telestyle, video_inpaint, video_stitch, upscale |

## Requirements

- ComfyUI running and accessible
- Python 3.10+

## Documentation

Full agent guide with all CLI commands, template reference, and architecture details: **[CLAUDE.md](./CLAUDE.md)**

## License

MIT
