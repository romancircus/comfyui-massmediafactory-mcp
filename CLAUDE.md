# ComfyUI MassMediaFactory MCP - Agent Guide

---

## jinyang Execution Tasks

**For multi-phase overnight work, use the centralized orchestration pattern.**

### Quick Start

1. **Copy template:**
   ```bash
   cp ~/.jinyang/templates/execution_script_template.py scripts/<task_name>_execute.py
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
   cat ~/.jinyang/templates/linear_execution_issue.md
   # Replace <REPO_NAME>, <SCRIPT_NAME>, <ISSUE_ID>
   # Paste into Linear issue description
   ```

4. **Validate before delegating:**
   ```bash
   python ~/.jinyang/scripts/validate_execution_issue.py ROM-XXX
   ```

5. **Delegate to jinyang** - execution happens automatically

### Resources

- **Template:** `~/.jinyang/templates/execution_script_template.py` (640 lines - complete base class)
- **Docs:** `~/.jinyang/docs/EXECUTION_PATTERN.md` (usage guide)
- **Issue template:** `~/.jinyang/templates/linear_execution_issue.md`
- **Validation:** `~/.jinyang/scripts/validate_execution_issue.py`

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
mcp__linear__linear_searchIssues(teamId="<team_id>", states=["In Progress", "Todo"])
```

**Update status when working:**
```python
mcp__linear__linear_updateIssue(id="ROM-XXX", stateId="<state_id>")
```

---

## Quick Start — `mmf` CLI (Preferred)

The `mmf` CLI is the fastest and most reliable way to generate images/videos. One command does everything:

```bash
# Text-to-image (FLUX)
mmf run --model flux --type t2i --prompt "a dragon in the clouds" --pretty

# Image-to-video (WAN)
mmf run --model wan --type i2v --image photo.png --prompt "gentle motion" -o output.mp4

# Template-based run
mmf run --template qwen_txt2img --params '{"PROMPT":"test","SEED":42}'

# Pre-tested pipeline (3 stages in 1 command)
mmf pipeline viral-short --prompt "dancing character" --style-image style.png -o video.mp4
mmf pipeline i2v --image keyframe.png --prompt "gentle breathing" -o video.mp4
mmf pipeline upscale --image input.png --factor 2 -o upscaled.png

# Batch
mmf batch seeds workflow.json --count 8 --start-seed 42
mmf batch dir --input keyframes/ --template wan26_img2vid --prompt "motion" --output videos/

# System
mmf stats --pretty                    # GPU info
mmf models constraints wan --pretty   # Model limits
mmf enhance --prompt "a cat" --model wan  # LLM prompt enhancement
```

**Why CLI over MCP**: 0 schema overhead (vs ~15K tokens), 1 command instead of 3 MCP calls, pre-tested parameters that produce correct results. JSON to stdout (machine-readable). Exit codes: 0=ok, 1=error, 2=timeout, 3=validation.

**Install**: `pip install -e .` from repo root, then `mmf` is available globally.

**For agents and scripts:** Always call `mmf` via Bash tool instead of MCP tools. MCP tools remain available but carry ~15K token schema overhead per invocation.

### MCP Tools (Discovery Only)

The MCP server (18 tools) is for **discovery and planning only** — finding nodes, checking constraints, browsing templates. All **execution** goes through the `mmf` CLI:

```bash
# Discovery via MCP (still useful)
list_models(), search_nodes(), get_model_constraints(), list_workflow_templates(), get_template()

# Execution via CLI (always)
mmf run --model flux --type t2i --prompt "a dragon in the clouds"
```

## Core Workflow

### Step 1: Choose Your Approach (Dual Generation Path)

There are two complementary generation paths. Choose based on your needs:

**Path A: `generate_workflow()` - Auto-Generated Workflows**
- 12 model+type combos (flux/t2i, ltx/t2v, ltx/i2v, wan/t2v, qwen/t2i, etc.)
- Auto-corrects resolution, frames, CFG to model constraints
- Best for: prompt-to-output with standard settings

**Path B: `mmf run --template` - Template-Based Workflows**
- 43 templates including advanced workflows (ControlNet, LoRA stacking, TTS, upscaling, inpainting)
- Exact configurations preserved, no auto-correction
- Best for: specialized workflows, batch production, jinyang overnight

| Need | Use Path A | Use Path B |
|------|-----------|-----------|
| Quick image/video from prompt | `mmf run --model flux --type t2i` | |
| ControlNet/LoRA/face ID | | `mmf run --template flux2_union_controlnet` |
| Background replacement | | `mmf run --template qwen_edit_background` |
| Standard t2v/i2v/t2i | Either works | Either works |
| Batch production (>10) | | `mmf batch seeds` or `mmf batch dir` |
| TTS/audio | | `mmf run --template chatterbox_tts` |
| Custom node combos | Build from scratch | |

### Step 2: Execute & Monitor

```bash
# One command does generate + execute + wait
mmf run --model flux --type t2i --prompt "cyberpunk city" -o output.png --pretty

# Or with more control:
mmf run --template wan26_img2vid --params '{"IMAGE_PATH":"img.png","PROMPT":"motion"}' --timeout 600

# Check progress on a running job:
mmf progress <prompt_id>
```

**WARNING:** Do NOT poll in a loop. `mmf run` blocks until completion via WebSocket internally.

### Step 3: Iterate

```bash
# Regenerate with new seed/params
mmf regenerate <asset_id> --seed 42 --cfg 4.5
```

## Tool Reference (18 MCP Tools + mmf CLI)

> **Note:** 41 tools moved to `mmf` CLI in ROM-548/ROM-562. MCP tools are kept only for
> interactive workflow building. Everything else uses `mmf` commands.

### Discovery (3 tools)

| Tool | Usage |
|------|-------|
| `list_models(type)` | `type`: checkpoint\|unet\|lora\|vae\|controlnet\|all |
| `get_node_info(node_type)` | Get node schema by class name |
| `search_nodes(query)` | Search nodes by name/category |

### System (5 tools)

| Tool | Usage |
|------|-------|
| `get_system_stats()` | GPU VRAM and system info |
| `free_memory(unload_models)` | Free GPU memory |
| `interrupt()` | Stop current workflow |
| `upload_image(path)` | Upload for ControlNet/I2V |
| `download_output(asset_id, path)` | Save to local disk |

### Publishing (3 tools)

| Tool | Usage |
|------|-------|
| `publish_asset(asset_id, ...)` | Export to web directory (path-traversal protected) |
| `get_publish_info()` | Current publish config |
| `set_publish_dir(path)` | Set publish directory (validated against blocklist) |

### Validation (1 tool)

| Tool | Usage |
|------|-------|
| `validate_workflow(workflow, auto_fix, check_pattern)` | Validate with optional auto-fix |

### Workflow Patterns (3 tools)

| Tool | Usage |
|------|-------|
| `get_workflow_skeleton(model, task)` | Get base structure |
| `get_model_constraints(model)` | CFG, resolution, frames limits |
| `get_node_chain(model, task)` | Node order with connections |

### Templates (2 tools)

| Tool | Usage |
|------|-------|
| `list_workflow_templates()` | List available templates |
| `get_template(name)` | Get template JSON |

### Prompt (1 tool)

| Tool | Usage |
|------|-------|
| `enhance_prompt(prompt, model, style)` | LLM-powered prompt enhancement with model-specific quality tokens |

> **Moved to CLI (ROM-548):** `generate_workflow` → `mmf run`, `create_workflow_from_template` → `mmf run --template`, `batch_execute` → `mmf batch`, pipelines → `mmf pipeline`, QA → `mmf qa`, style learning → `mmf style`, visualization → `mmf visualize`
>
> **Moved to CLI (ROM-562):** `cleanup_assets` → `mmf assets cleanup`, `workflow_library` → `mmf workflow-lib`, `estimate_vram` → `mmf models estimate-vram`, `check_model_fits` → `mmf models check-fit`, `sota_query` → `mmf sota`, `search_civitai` → `mmf search-model`, `download_model` → `mmf install-model`, `get_model_info` → `mmf models info`, `list_installed_models` → `mmf models list`, `get_optimal_workflow_params` → `mmf models optimize`, `get_execution_profile` → `mmf profile`, `diff_workflows` → `mmf diff`, `get_compatibility_matrix` → `mmf models compatibility`, rate limiting tools → internal only

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

## Supported Models & Workflows (9 models, 43 templates)

| Model | Supported Types | Templates |
|-------|-----------------|-----------|
| `flux` | t2i, controlnet, lora, face_id, inpaint, edit | 8 templates |
| `ltx` | t2v, i2v, v2v, audio_reactive | 6 templates |
| `wan` | t2v, i2v, s2v, flf2v, camera_i2v, animate | 10 templates |
| `qwen` | t2i, controlnet, poster, edit (background) | 4 templates |
| `hunyuan` | t2v, i2v | 2 templates |
| `z_turbo` | t2i (4-step fast) | 1 template |
| `sdxl` | t2i | 1 template |
| `cogvideox` | t2v | (model registered, no template yet) |
| `audio` | tts (chatterbox, f5, qwen3, voice clone), v2a (mmaudio) | 7 templates |
| `utility` | telestyle, video_inpaint, video_stitch, upscale | 4 templates |

## Common Patterns (mmf CLI)

### Text-to-Image (FLUX)

```bash
mmf run --model flux --type t2i --prompt "cyberpunk city at night, neon lights" -o city.png --pretty
```

### Text-to-Video (LTX)

```bash
mmf run --model ltx --type t2v --prompt "a cat walking through a garden" --timeout 900 -o cat.mp4
```

### Image-to-Video (WAN)

```bash
mmf run --model wan --type i2v --image photo.png --prompt "gentle motion" -o video.mp4
```

### Batch Seed Variations

```bash
mmf batch seeds workflow.json --count 4 --start-seed 42
# Returns 4 variations with seeds 42, 43, 44, 45
```

### Batch from Directory

```bash
mmf batch dir --input keyframes/ --template wan26_img2vid --prompt "motion" --output videos/
```

### Background Replacement (Qwen Edit)

```bash
mmf run --template qwen_edit_background --params '{
  "IMAGE_PATH": "uploaded_image.png",
  "EDIT_PROMPT": "Change the background to an outdoor playground with green grass and blue sky.",
  "SEED": 42, "CFG": 2.0, "STEPS": 20
}' -o edited.png
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
| **Rate limiting** | All tools wrapped with `@mcp_tool_wrapper` providing per-tool rate limits + structured logging |

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

1. **Use `mmf run`** for all execution - it blocks via WebSocket internally. NEVER poll in a loop.
2. **Use `mmf run --model`** for auto-corrected workflows, `mmf run --template` for exact configs
3. **Check VRAM first** with `mmf models check-fit <model>` for large models
4. **Validate** with `mmf validate workflow.json --auto-fix` to auto-correct common issues
5. **Enhance prompts** with `mmf enhance --prompt "..." --model wan` before generation
6. **Profile slow workflows** with `mmf profile <prompt_id>` to identify bottleneck nodes
7. **Check compatibility** with `mmf models compatibility --pretty` to see what's ready to use
8. **MCP for discovery only** - `search_nodes()`, `get_model_constraints()`, `list_workflow_templates()`
9. **Read MCP resources** for model-specific patterns when building custom workflows
10. **For batch** use `mmf batch seeds`, `mmf batch dir`, or `mmf batch sweep`
