# Template Migration Guide

Migrating from direct ComfyUI client calls to ComfyUI MassMediaFactory MCP templates.

---

## Quick Reference: Direct Methods vs MCP Templates

| Direct Method | MCP Template | Model | Notes |
|--------------|--------------|-------|-------|
| `generate_qwen_bio()` | Not available | Qwen | Use `qwen_txt2img` + ControlNet |
| `generate_ltx_video()` | `ltx2_txt2vid`, `ltx2_i2v` | LTX-2 | Direct replacement |
| `generate_flux_image()` | `flux2_txt2img` | FLUX.2-dev | Direct replacement |
| `generate_audio()` | `audio_tts_*` | F5-TTS/Chatterbox/Qwen3 | Multiple options |

---

## Repository-Specific Mappings

### pokedex-generator (Python: `src/adapters/comfyui_client.py`)

#### Legacy Method → MCP Template Mapping

| Legacy Method | Current Use | MCP Equivalent | Migration Path |
|---------------|-------------|----------------|----------------|
| `generate_qwen_bio_txt2img()` | Bio images (Canny ControlNet) | `qwen_txt2img` + `flux2_union_controlnet` | Use ControlNet directly |
| `generate_qwen_bio()` | Bio images (IP-Adapter) | `flux2_txt2img` | Direct replacement (simpler) |
| `generate_ltx_i2v_video()` | Creature videos | `ltx2_img2vid` | Direct replacement |
| `generate_wan_t2v()` | T2V videos | `wan21_txt2vid` | Direct replacement |
| `generate_sdxl_image()` | SDXL images | `sdxl_txt2img` | Direct replacement |

#### Migration Example: Python

**Before (direct client):**
```python
from src.adapters.comfyui_client import ComfyUIClient

client = ComfyUIClient()
client.generate_qwen_bio_txt2img(
    reference_image=ref_img,
    prompt="biological dragon",
    output_path=output,
    seed=42
)
```

**After (MCP template):**
```python
# In your MCP client setup
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("pokedex-generator")

# Use template
@mcp.tool()
async def generate_bio_image(
    prompt: str,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
    shift: float = 7.0
) -> dict:
    """Generate bio image using Qwen-Image-2512"""
    # Call ComfyUI MCP template
    result = await comfyui_generate_workflow(
        template="qwen_txt2img",
        params={
            "PROMPT": prompt,
            "SEED": seed,
            "WIDTH": width,
            "HEIGHT": height,
            "SHIFT": shift
        }
    )
    return result
```

---

### KDH-Automation (JavaScript/Node: `src/core/ComfyUIClient.js`)

#### Legacy Method → MCP Template Mapping

| Legacy Method | Current Use | MCP Equivalent | Migration Path |
|---------------|-------------|----------------|----------------|
| `queuePrompt()` | Manual workflow submission | Any template via `create_workflow_from_template` | Use template system |
| `executeWorkflow()` | Full execution | `execute_workflow()` + templates | Direct replacement |
| `getImage()` | Output download | `wait_for_completion()` | Built-in to MCP |
| `uploadImage()` | Input upload | `upload_image()` | Direct replacement |

#### Migration Example: JavaScript

**Before (direct client):**
```javascript
import { ComfyUIClient } from '../core/ComfyUIClient.js';

const client = new ComfyUIClient('http://127.0.0.1:8188');

// Load and modify workflow
const workflow = await client.loadWorkflowTemplate('flux2_text2img.json');
const modified = client.modifyWorkflow(workflow, {
  '7': { inputs: { text: prompt } }
});

// Execute
const { prompt_id } = await client.queuePrompt(modified);
const { images } = await client.executeWorkflow(modified, outputDir);
```

**After (MCP template):**
```javascript
// Use MCP tools from ComfyUI MassMediaFactory
async function generateImage(prompt, seed = 42, width = 1024, height = 1024) {
  const workflow = await mcp_create_workflow_from_template("flux2_txt2img", {
    PROMPT: prompt,
    SEED: seed,
    WIDTH: width,
    HEIGHT: height,
    GUIDANCE: 3.5
  });

  const result = await mcp_execute_workflow(workflow);
  const output = await mcp_wait_for_completion(result.prompt_id);

  return output.outputs[0].asset_id;
}
```

---

### Goat (JavaScript/Node: `src/clients/ComfyUIClient.js`)

#### Legacy Method → MCP Template Mapping

| Legacy Method | Current Use | MCP Equivalent | Migration Path |
|---------------|-------------|----------------|----------------|
| `isRunning()` | Connection check | `get_system_stats()` | Built-in validation |
| `getSystemStats()` | VRAM/queue status | `get_system_stats()` | Direct replacement |
| `executeWorkflow()` | Image generation | Any template | Use template system |
| `waitForCompletion()` | Polling | `wait_for_completion()` | Direct replacement |

#### Migration Example: JavaScript

**Before (direct client):**
```javascript
import ComfyUIClient from '../clients/ComfyUIClient.js';

const client = new ComfyUIClient();

async function generateMovieFrame(prompt) {
  const workflow = await client.modifyWorkflow(baseWorkflow, {
    '7': { inputs: { text: prompt } }
  });

  const result = await client.executeWorkflow(workflow, outputDir, 'frame');
  return result.images[0];
}
```

**After (MCP template):**
```javascript
async function generateMovieFrame(prompt) {
  const workflow = await mcp_create_workflow_from_template("flux2_txt2img", {
    PROMPT: prompt,
    SEED: 42,
    WIDTH: 1920,
    HEIGHT: 1080,
    STEPS: 28,
    GUIDANCE: 3.5
  });

  const result = await mcp_execute_workflow(workflow);
  const output = await mcp_wait_for_completion(result.prompt_id, 600);

  return output.outputs[0].asset_id;
}
```

---

## Parameter Mapping Tables

### Image Generation Parameters

| Parameter Name (Client) | MCP Template Param | Description | Default Value |
|------------------------|---------------------|-------------|---------------|
| `prompt` | `PROMPT` | Text prompt | - |
| `negative_prompt` | `NEGATIVE` | Negative prompt | "" |
| `seed` | `SEED` | Random seed | 42 |
| `width` | `WIDTH` | Image width | 1024 |
| `height` | `HEIGHT` | Image height | 1024 |
| `steps` | `STEPS` | Sampling steps | 28 |
| `cfg` / `guidance_scale` | `GUIDANCE` / `CFG` | Guidance scale | 3.5 |
| `shift` | `SHIFT` | Qwen shift value | 7.0 |
| `denoise` | `DENOISE` | Strength (0-1) | 1.0 |
| `guidance` | `GUIDANCE` | FLUX guidance | 3.5 |

### Video Generation Parameters

| Parameter Name (Client) | MCP Template Param | Description | Default Value |
|------------------------|---------------------|-------------|---------------|
| `frames` | `FRAMES` | Number of frames | 25 |
| `fps` | `FPS` | Frames per second | 24 |
| `strength` | `STRENGTH` | Motion strength | 0.5 |
| `steps` | `STEPS` | Sampling steps | 30 |
| `cfg` | `CFG` | Classifier-free guidance | 2.5 |

### Audio/TTS Parameters

| Parameter Name (Client) | MCP Template Param | Description | Default Value |
|------------------------|---------------------|-------------|---------------|
| `text` | `TEXT` | Text to synthesize | - |
| `reference_audio` | `REFERENCE_AUDIO` | Voice sample path | - |
| `reference_text` | `REFERENCE_TEXT` | Transcript of reference | - |
| `speed` | `SPEED` | Speaking speed | 1.0 |
| `speaker` | `SPEAKER` | Qwen3 voice preset | - |

---

## Template Categories & Use Cases

### Text-to-Image Templates

| Template | Model | Best For | When to Use |
|----------|-------|----------|-------------|
| `flux2_txt2img` | FLUX.2-dev | General purpose, character consistency | Default choice for images |
| `qwen_txt2img` | Qwen-Image-2512 | Sharp portraits, high detail | When detail matters more than speed |
| `sdxl_txt2img` | SDXL 1.0 | Fast, general purpose | When resources limited |

### ControlNet Templates

| Template | Model | Best For | When to Use |
|----------|-------|----------|-------------|
| `flux2_union_controlnet` | FLUX.2-dev | Structure preservation | Need structure from reference |
| `flux2_grounding_dino_inpaint` | FLUX.2-dev | Precise object editing | Inpainting specific objects |
| `wan21_img2vid` | Wan 2.1 | Image-to-video | Converting stills to motion |

### Text-to-Video Templates

| Template | Model | Best For | When to Use |
|----------|-------|----------|-------------|
| `ltx2_txt2vid` | LTX-2 19B | Fast T2V | Speed matters (3-4x faster) |
| `ltx2_txt2vid_distilled` | LTX-2 Distilled | Fastest T2V | Maximum speed needed |
| `wan21_txt2vid` | Wan 2.1 14B | Human motion quality | Character movement quality |
| `hunyuan15_txt2vid` | HunyuanVideo 1.5 | Complex scenes | Detailed, complex backgrounds |

### Special Purpose Templates

| Template | Model | Best For | When to Use |
|----------|-------|----------|-------------|
| `flux2_face_id` | FLUX.2-dev | Face identity preservation | Need specific face |
| `flux2_edit_by_text` | FLUX.2-dev | Text-based inpainting | Edit images by description |
| `flux2_lora_stack` | FLUX.2-dev | Style LoRA stacking | Need multiple LoRA styles |
| `audio_tts_f5` | F5-TTS | Voice cloning | Have voice sample |
| `qwen3_tts_voice_clone` | Qwen3-TTS | Quick voice cloning | ~3 seconds of audio |

---

## Code Examples by Language

### Python (pokedex-generator pattern)

```python
import json
from pathlib import Path

async def migrate_to_mcp():
    # Legacy pattern
    client = ComfyUIClient()
    result = client.generate_qwen_bio(ref_img, prompt, output)

    # MCP pattern
    workflow = comfyui_mcp.create_workflow_from_template(
        "flux2_txt2img",
        {
            "PROMPT": prompt,
            "SEED": 42,
            "WIDTH": 1024,
            "HEIGHT": 1024,
            "GUIDANCE": 3.5
        }
    )
    result = comfyui_mcp.execute_workflow(workflow)
    output = comfyui_mcp.wait_for_completion(result["prompt_id"])
    asset_id = output["outputs"][0]["asset_id"]
```

### JavaScript (KDH-Automation pattern)

```javascript
async function migrateToMCP() {
  // Legacy pattern
  const client = new ComfyUIClient();
  const workflow = await client.loadWorkflowTemplate("template.json");
  const result = await client.executeWorkflow(workflow, outputDir);

  // MCP pattern
  const workflow = await mcpCreateWorkflowFromTemplate("flux2_txt2img", {
    PROMPT: prompt,
    SEED: 42,
    WIDTH: 1024,
    HEIGHT: 1024,
    GUIDANCE: 3.5
  });
  const result = await mcpExecuteWorkflow(workflow);
  const output = await mcpWaitForCompletion(result.prompt_id);
  return output.outputs[0].asset_id;
}
```

### JavaScript (Goat pattern)

```javascript
async function migrateToMCP() {
  // Legacy pattern
  const client = new ComfyUIClient();
  const { prompt_id } = await client.queuePrompt(workflow);
  const completion = await client.waitForCompletion(prompt_id);

  // MCP pattern
  const result = await mcpExecuteWorkflow(workflow);
  const output = await mcpWaitForCompletion(result.prompt_id, 600);
  return output.outputs[0].asset_id;
}
```

---

## Troubleshooting Common Issues

### Issue 1: Template Not Found

**Error:** `Template 'flux2_txt2img' not found`

**Solution:**
```bash
# List available templates
list_workflow_templates(limit=50)

# Or check docs
ls ~/Applications/comfyui-massmediafactory-mcp/docs/patterns/
```

### Issue 2: Parameter Value Type Mismatch

**Error:** `Invalid parameter type for SEED: expected int, got string`

**Solution:**
```python
# WRONG
workflow = create_workflow_from_template("flux2_txt2img", {
    "SEED": "42"  # String! Bad.
})

# CORRECT
workflow = create_workflow_from_template("flux2_txt2img", {
    "SEED": 42  # Integer. Good.
})
```

### Issue 3: Image Paths Not Found

**Error:** `Image path not found: /path/to/image.png`

**Solution:**
```python
# Upload first, then use filename
asset_id = comfyui_mcp_upload_image("/path/to/image.png")
workflow = create_workflow_from_template("wan21_img2vid", {
    "IMAGE_PATH": asset_id  # Use asset ID from upload
})
```

### Issue 4: Timeout on Long Workflows

**Error:** `Timeout waiting for completion after 300s`

**Solution:**
```python
# Increase timeout
output = comfyui_mcp_wait_for_completion(
    prompt_id,
    timeout_seconds=900  # 15 minutes for video
)
```

### Issue 5: VRAM Exhaustion

**Error:** `CUDA out of memory`

**Solution:**
```python
# 1. Check VRAM before execution
stats = comfyui_mcp_get_system_stats()
print(f"Available VRAM: {stats['vram_used']} / {stats['vram_total']} GB")

# 2. Use distilled models for lower VRAM
workflow = create_workflow_from_template("ltx2_txt2vid_distilled", ...)  # 3-4x less VRAM

# 3. Free memory first
comfyui_mcp_free_memory(unload_models=True)

# 4. Use smaller batch/steps
workflow = create_workflow_from_template("flux2_txt2img", {
    "WIDTH": 512,
    "HEIGHT": 512,
    "STEPS": 20
})
```

### Issue 6: Workflow Validation Failed

**Error:** `Validation failed: missing required parameter WIDTH`

**Solution:**
```python
# 1. Validate before executing
result = comfyui_mcp_validate_workflow(workflow, auto_fix=True)

# 2. Check parameter availability
template = comfyui_mcp_get_template("flux2_txt2img")
print("Required params:", template["parameters"])

# 3. Auto-fix validation errors
workflow = comfyui_mcp_validate_workflow(workflow, auto_fix=True)["workflow"]
```

### Issue 7: Static Video Output (I2V)

**Error:** Video has no motion, frames all same

**Solution:**
```python
# LTX-2 I2V needs active verb prompts
# WRONG
"dragon breathing fire"  # Only 1 active verb

# CORRECT (Use I2VPromptBuilder)
from src.prompts.ltx_i2v_prompts import I2VPromptBuilder

builder = I2VPromptBuilder()
prompt = builder.build_creature_prompt(
    creature="dragon",
    action="breathing fire",
    audio_description="crackling flames, roars"
)

workflow = create_workflow_from_template("ltx2_i2v", {
    "IMAGE_PATH": ref_image,
    "PROMPT": prompt,
    "STEPS": 30
})
```

### Issue 8: Prompt ID Not Returned Correctly

**Error:** `prompt_id is None or undefined`

**Solution:**
```python
# 1. Check execution response
result = comfyui_mcp_execute_workflow(workflow)
if result.get("prompt_id"):
    prompt_id = result["prompt_id"]
else:
    # Check for errors
    if result.get("isError"):
        print("Error:", result["error"])
```

### Issue 9: Output File Not Downloaded

**Error:** `No output images found in history`

**Solution:**
```python
# 1. Wait for completion first
output = comfyui_mcp_wait_for_completion(prompt_id)

# 2. Check output structure
print("Output structure:", output.keys())
print("Outputs:", output.get("outputs", []))

# 3. Download if asset_id exists
if output["outputs"] and output["outputs"][0].get("asset_id"):
    asset_id = output["outputs"][0]["asset_id"]
    comfyui_mcp_download_output(asset_id, "/path/to/save.png")
```

### Issue 10: Batch Execution Timeout

**Error:** Individual jobs timing out when batched

**Solution:**
```python
# Use batch_execute for proper parallel execution
workflow = create_workflow_from_template("flux2_txt2img", {...})

results = comfyui_mcp_batch_execute(
    workflow=workflow,
    mode="seeds",
    num_variations=4,
    parallel=2,  # Max 2 concurrent jobs
    timeout_per_job=300
)
```

---

## Migration Checklist

### Phase 1: Discovery

- [ ] List all ComfyUI client method calls in your repo
- [ ] Identify which templates map to each method
- [ ] Check for custom workflow JSON files being loaded
- [ ] Note any custom parameter injection patterns

### Phase 2: Code Changes

- [ ] Replace `queuePrompt()` with `create_workflow_from_template()`
- [ ] Replace `waitForCompletion()` with `wait_for_completion()`
- [ ] Replace `uploadImage()` with `upload_image()`
- [ ] Replace `downloadImage()` with `download_output()`

### Phase 3: Testing

- [ ] Test each migration point with a simple workflow
- [ ] Verify output quality matches or improves
- [ ] Check VRAM usage and performance
- [ ] Validate error handling

### Phase 4: Documentation

- [ ] Update inline code comments
- [ ] Update README with new workflow
- [ ] Document any custom templates created
- [ ] Add migration notes to CHANGELOG

---

## Advanced Patterns

### Multi-Stage Pipeline Execution

```python
# Image → Upscale → Video
stages = [
    {
        "template": "flux2_txt2img",
        "params": {"PROMPT": prompt, "SEED": 42},
        "output_key": "base_image"
    },
    {
        "template": "flux2_ultimate_upscale",
        "params": {"SCALE_FACTOR": 2},
        "input_from": "base_image"
    },
    {
        "template": "ltx2_img2vid",
        "params": {"PROMPT": prompt, "FRAMES": 49},
        "input_from": "upscaled_image"
    }
]

await comfyui_mcp_execute_pipeline_stages(
    stages=stages,
    initial_params={"prompt": prompt, "seed": 42},
    timeout_per_stage=600
)
```

### Parameter Sweep for Optimization

```python
# Test multiple CFG values
results = comfyui_mcp_batch_execute(
    workflow=base_workflow,
    mode="sweep",
    sweep_params={
        "CFG": [2.5, 3.5, 4.5],
        "STEPS": [20, 30]
    },
    timeout_per_job=300
)
```

### Seed Variations for Iteration

```python
# Generate 4 variations of same prompt
results = comfyui_mcp_batch_execute(
    workflow=workflow,
    mode="seeds",
    num_variations=4,
    start_seed=42,
    parallel=2
)
```

---

## References

### MCP Tool Documentation

- `create_workflow_from_template(template_name, params)` - Create workflow from template
- `execute_workflow(workflow)` - Queue and execute workflow
- `wait_for_completion(prompt_id, timeout_seconds)` - Block until done
- `list_workflow_templates()` - List all available templates
- `get_template(template_name)` - Get template structure
- `validate_workflow(workflow, auto_fix)` - Validate and fix workflow
- `get_system_stats()` - Check VRAM and queue status
- `upload_image(image_path)` - Upload image for ControlNet/I2V
- `download_output(asset_id, output_path)` - Save generated asset
- `list_models(type)` - Check installed models

### Model Constraints

| Model | Key Constraints | Recommended Settings |
|-------|----------------|---------------------|
| Qwen | shift=7.0 (critical) | 3.1 = blurry, 7.0 = sharp |
| LTX-2 | frames=8n+1 (9, 17, 25, 33...) | 8-frame groups |
| Wan 2.1 | shift=8.0, cfg=5.0 | Human motion quality |
| FLUX.2 | guidance=3.5-4.0 | Character consistency |

---

## Getting Help

1. **Check templates first:**
   ```bash
   comfyui_mcp_list_workflow_templates(limit=50)
   ```

2. **Validate workflows:**
   ```python
   comfyui_mcp_validate_workflow(workflow, auto_fix=True)
   ```

3. **Check system stats:**
   ```python
   comfyui_mcp_get_system_stats()
   ```

4. **Error recovery:**
   ```bash
   ~/Applications/comfyui-massmediafactory-mcp/docs/ERROR_RECOVERY.md
   ```
