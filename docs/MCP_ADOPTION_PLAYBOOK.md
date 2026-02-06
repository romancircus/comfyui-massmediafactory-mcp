# MCP Adoption Playbook for New Repos

**Created**: 2026-02-04
**Purpose**: Standardized guide for integrating ComfyUI MCP into new repositories

---

## 1. Template System: 32 Categorized Templates

### 1.1 Template Categories

| Category | Templates | Use Cases | Model Type |
|----------|----------|-----------|------------|
| **Text-to-Image** (5) | `flux2_txt2img`, `flux2_ultimate_upscale`, `qwen_txt2img`, `sdxl_txt2img`, `qwen_poster_design` | Character generation, posters, upscaling | FLUX.2, Qwen, SDXL |
| **Text-to-Video** (7) | `ltx2_txt2vid`, `ltx2_txt2vid_distilled`, `ltx2_audio_reactive`, `wan26_txt2vid`, `hunyuan15_txt2vid`, `telestyle_video` | Narrative video, fast render, audio-synced | LTX-2, Wan 2.6, Hunyuan |
| **Image-to-Video** (3) | `ltx2_img2vid`, `ltx2_i2v_distilled`, `hunyuan15_img2vid` | Image animation, character motion | LTX-2, Hunyuan |
| **Image Editing** (4) | `flux2_edit_by_text`, `qwen_edit_background`, `flux2_face_id`, `video_inpaint` | Background replacement, face preservation, video cleanup | FLUX.2, Qwen Edit, Generic |
| **Advanced Control** (3) | `flux2_union_controlnet`, `flux2_grounding_dino_inpaint`, `flux2_lora_stack` | ControlNet, inpainting, style mixing | FLUX.2 |
| **Audio/Text-to-Speech** (6) | `chatterbox_tts`, `audio_tts_f5`, `audio_tts_voice_clone`, `qwen3_tts_voice_design`, `qwen3_tts_voice_clone`, `qwen3_tts_custom_voice` | Expressive TTS, voice cloning, character voices | Chatterbox, F5, Qwen |
| **Video Utilities** (2) | `video_stitch`, `telestyle_image` | Video concatenation, style transfer | Utility |
| **Specialized** (2) | `turnaround_sheet`, `expression_sheet` | Character sprite sheets | Pony/3D |

### 1.2 Must-Use Templates for Common Tasks

| Task | Primary Template | Alternative |
|------|------------------|-------------|
| **Character generation** | `flux2_txt2img` | `qwen_txt2img` (portraits) |
| **Background replacement** | `qwen_edit_background` | N/A |
| **Video generation (fast)** | `ltx2_txt2vid_distilled` | `ltx2_txt2vid` (quality) |
| **Expressive TTS** | `chatterbox_tts` | `audio_tts_voice_clone` |
| **Face preservation** | `flux2_face_id` | `qwen_edit_background` |
| **Upscaling** | `flux2_ultimate_upscale` | N/A |

### 1.3 Template Metadata Structure

```json
{
  "_meta": {
    "description": "Human-readable description",
    "model": "Canonical model name",
    "type": "txt2img / txt2vid / edit / audio_tts",
    "parameters": ["PROMPT", "SEED", "WIDTH", ...],
    "defaults": {"PROMPT": "", "SEED": 42, ...},
    "vram_min": 12,
    "tags": ["model:flux2", "priority:recommended"],
    "agent_notes": {"cfg": "...", "denoise": "..."}
  }
}
```

---

## 2. Architecture Patterns

### 2.1 Adapter Layer Structure

```
new-repo/
├── comfyui_mcp/
│   ├── __init__.py          # MCP client wrapper
│   ├── client.py            # Direct API calls
│   ├── templates.py         # Template injection
│   └── workflow_utils.py    # Model-specific helpers
├── workflows/
│   └── generated/           # Auto-saved workflows
└── CLAUDE.md                # Integration guide (MUST exist)
```

### 2.2 Model Registry Usage

**Single source of truth** for model constraints:

```python
from comfyui_massmediafactory_mcp.model_registry import (
    get_model_constraints,
    get_model_defaults,
    list_supported_models,
)

# Get constraints before building
constraints = get_model_constraints("flux2")
print(constraints.cfg.max)      # 5.0
print(constraints.resolution.native)  # [1024, 1024]
```

### 2.3 Template Lookup Pattern

```python
from comfyui_massmediafactory_mcp.templates import (
    list_templates,
    load_template,
    inject_parameters,
)

# 1. Find template
templates = list_templates(model_type="flux2", tags=["priority:recommended"])
template_name = templates["templates"][0]["name"]

# 2. Load with validation
template = load_template(template_name, validate=True)

# 3. Inject parameters
workflow = inject_parameters(template, {
    "PROMPT": "a dragon",
    "SEED": 42,
    "WIDTH": 1024,
})
```

### 2.4 Error Handling Standardization

```python
from comfyui_massmediafactory_mcp.mcp_utils import (
    mcp_error,
    not_found_error,
    validation_error,
    timeout_error
)

# Use standardized error format
return mcp_error(
    "Workflow execution failed",
    "COMFYUI_ERROR",
    {"prompt_id": prompt_id, "details": exception}
)

# Errors include: isError: true, code, error message, details
```

---

## 3. Best Practices

### 3.1 MCP vs. Direct API Decision Matrix

| Use Case | Approach | Why |
|----------|----------|-----|
| **Interactive debugging** | MCP Tools | Discovery, validation, quick tests |
| **Production scripts** | Direct API (`urllib`) | Token optimization, 100x cost reduction |
| **Batch operations** | Direct API | Looping costs ~$20-40 vs $0.20-0.40 |
| **Cyrus overnight** | Direct API | No token overhead in production |
| **Single experiments** | MCP Tools | Faster iteration, built-in validation |

**Reference**: `~/.claude/MCP_TOKEN_OPTIMIZATION.md`

### 3.2 Token Optimization Strategies

```python
# ❌ BAD (polling - burns API calls)
while True:
    status = get_workflow_status(prompt_id)
    if status == "complete": break
    time.sleep(5)  # 100 iterations = 100 API calls

# ✅ GOOD (blocking wait)
output = wait_for_completion(prompt_id, timeout=600)  # 1 API call
```

**Rules**:
- Use `wait_for_completion()` for ComfyUI jobs (blocks internally)
- Never poll `get_workflow_status()` more than 3 times
- Capture MCP output to avoid re-fetching
- Batch MCP calls in single turn (multi-file reads)

### 3.3 Blocking Wait Patterns

```python
# ComfyUI execution
prompt_id = execute_workflow(workflow)["prompt_id"]
output = wait_for_completion(prompt_id, timeout=600)

# Direct API pattern
response = client.queue_prompt(workflow)
prompt_id = response["prompt_id"]

# Poll with exponential backoff (fallback)
output = wait_for_completion_with_retry(prompt_id, max_wait=600)
```

### 3.4 Cost Awareness

| Operation | MCP Cost | Direct API Cost | Savings |
|-----------|----------|-----------------|---------|
| Single generation | ~500 tokens | ~500 tokens | - |
| 151 generation batch | ~75K tokens | ~500 tokens | **150x** |
| Workflow discovery | ~10K tokens | ~2K tokens | 5x |

**Actionable guidance**:
- Interactive: MCP is fine (< 10 generations)
- Batch > 20: Use Direct API
- Cyrus: Always Direct API
- Loops: Always Direct API

---

## 4. Onboarding Checklist

### 4.1 Files That Must Exist

```bash
new-repo/
├── CLAUDE.md                              # ✅ MANDATORY
├── comfyui_mcp/
│   ├── __init__.py                        # MCP import wrapper
│   └── integration.py                     # Repo-specific helpers
├── scripts/
│   └── generate_<content_type>.py         # Generation scripts
├── outputs/                              # Generated content
├── references/                            # Source assets (if I2V)
└── .gitignore                             # Ignore large outputs
```

### 4.2 CLAUDE.md Required Sections

```markdown
# [<REPO_NAME>](CLAUDE.md)

## MCP Integration

### ComfyUI Quick Start

```python
from comfyui_mcp import generate_image, generate_video

# Image
image = generate_image(prompt="a dragon", model="flux2")

# Video
video = generate_video(
    prompt="a cat walking",
    model="ltx2",
    frames=97,
    width=768
)
```

### Cyrus Execution Pattern

For overnight work:

1. **Create orchestration script**:
   ```bash
   cp ~/.cyrus/templates/execution_script_template.py scripts/<task>_execute.py
   ```

2. **Implement phases** (see template for examples)

3. **Validate before delegating**:
   ```bash
   python ~/.cyrus/scripts/validate_execution_issue.py ROM-XXX
   ```

4. **Delegate to Cyrus** via `mcp__linear__update_issue(id, delegate="Cyrus")`

### Supported Models

| Model | Type | Templates |
|-------|------|-----------|
| FLUX.2 | txt2img | flux2_txt2img, flux2_ultimate_upscale |
| LTX-2 | txt2vid/i2v | ltx2_txt2vid, ltx2_img2vid |
| Wan 2.6 | txt2vid/i2v | wan26_txt2vid, wan26_img2vid |
| Qwen Edit | edit | qwen_edit_background |

### Critical Settings

| Setting | FLUX.2 | LTX-2 | Wan | Qwen Edit |
|---------|--------|-------|-----|-----------|
| CFG | 3.5 | 2.5-4.0 | 5.0 | 2.0-2.5 |
| Steps | 20 | 30 | 30 | 20 |
| Frame Constraint | N/A | 8n+1 | 81-121 | N/A |
| VRAM | 16GB | 12GB | 24GB | 24GB |

### MCP vs Direct API

- ✅ **Use MCP**: Interactive debugging, < 10 generations, discovery
- ✅ **Use Direct API**: Batch operations, Cyrus overnight, loops
- ❌ **Never Poll**: Use `wait_for_completion()` instead of status polls
```

### 4.3 Mandatory Tests

```python
# tests/test_mcp_integration.pyimport pytest
from comfyui_mcp import generate_image, generate_video

def test_flux2_txt2img():
    """Test FLUX.2 image generation."""
    result = generate_image(
        prompt="test image",
        model="flux2",
        width=512,  # Small for test speed
        height=512
    )
    assert "asset_id" in result
    assert result["status"] == "completed"

def test_ltx2_txt2vid():
    """Test LTX-2 video generation."""
    result = generate_video(
        prompt="cat walking",
        model="ltx2",
        frames=17,  # Small for test speed
        width=512,
        height=384
    )
    assert "asset_id" in result
    assert result["duration_seconds"] > 0
```

### 4.4 Required Documentation

- [ ] CLAUDE.md with MCP Integration section
- [ ] README.md with ComfyUI Quick Start
- [ ] `comfyui_mcp/__init__.py` with imports
- [ ] 1+ template references per supported model
- [ ] Cyrus execution pattern section
- [ ] MCP vs Direct API decision matrix

---

## 5. Architecture Decision Points

### 5.1 Decision: Template vs. Generate Workflow

**Use Template When**:
- Task matches a predefined workflow pattern
- Need model-specific validation
- Want to inject parameters quickly

```python
# Template approach
template = load_template("flux2_txt2img")
workflow = inject_parameters(template, {"PROMPT": "dragon", "SEED": 42})
```

**Use Generate Workflow When**:
- Building custom node combinations
- Testing novel workflows
- Need programmatic flexibility

```python
# Generate workflow approach
workflow = generate_workflow(
    model="flux2",
    workflow_type="t2i",
    prompt="dragon",
    seed=42
)
```

**Decision Point**: Start with templates, validate, then generate if needed.

### 5.2 Decision: MCP Tools vs. Direct Client

| Criteria | MCP Tools | Direct Client (`client.py`) |
|----------|-----------|---------------------------|
| Token cost | High per call | Near-zero |
| Validation | Built-in | Manual |
| Discovery | Easy | Manual |
| Batch overhead | Prohibitive | Minimal |
| Production ready | No | Yes |

**Rule of Thumb**:
- Interactive development: MCP Tools
- Production scripts: Direct Client (`ComfyUIClient`)

### 5.3 Decision: Blocking Wait vs. Polling

**Blocking Wait** (default):
```python
output = wait_for_completion(prompt_id, timeout=600)
# Single API call, returns when done
```

**Polling** (avoid unless necessary):
```python
while True:
    status = get_workflow_status(prompt_id)
    if status["status"] == "completed": break
    time.sleep(5)  # Burns 100 API calls
```

**Decision**: Always use blocking waits unless waiting with retry logic.

### 5.4 Decision: Workflow Storage

**Persist Workflows**:
- Store executed workflows in `workflows/generated/`
- Include metadata: `{"prompt": "...", "seed": 42, "asset_id": "..."}`
- Enables regeneration and debugging

**Don't Persist**:
- Source images/video (use `references/` symlink)
- Model weights (use model registry)
- Temporary cache files

---

## 6. Common Integration Patterns

### 6.1 Template-Based Generation

```python
from comfyui_massmediafactory_mcp.templates import (
    load_template,
    inject_parameters,
)
from comfyui_mcp.client import ComfyUIClient

client = ComfyUIClient()

# Load and inject
template = load_template("flux2_txt2img", validate=True)
workflow = inject_parameters(template, {
    "PROMPT": "a dragon in clouds",
    "SEED": 42,
    "WIDTH": 1024,
    "HEIGHT": 1024,
    "STEPS": 20,
    "GUIDANCE": 3.5,
})

# Execute
result = client.queue_prompt(workflow)
prompt_id = result["prompt_id"]

# Wait (blocking)
output = client.wait_for_completion(prompt_id)
asset_id = output["outputs"][0]["images"][0]["filename"]
```

### 6.2 Batch Generation Pattern

```python
from pathlib import Path
from comfyui_mcp.client import ComfyUIClient

client = ComfyUIClient()

prompts = [
    "a dragon in sky",
    "a phoenix rising",
    "a griffin soaring",
]

for i, prompt in enumerate(prompts):
    workflow = load_template("flux2_txt2img")
    workflow = inject_parameters(workflow, {
        "PROMPT": prompt,
        "SEED": i,  # Different seeds
    })

    result = client.queue_prompt(workflow)
    print(f"Queued {i}: {result['prompt_id']}")
    # Don't wait - let them queue

# Wait for all at once
for i in range(len(prompts)):
    output = wait_for_completion(prompt_ids[i])
    print(f"Completed {i}")
```

### 6.3 Cyrus Overnight Pattern

```python
#!/usr/bin/env python3
"""Generate 151 Pokemon images overnight."""

from pathlib import Path
from comfyui_mcp.client import ComfyUIClient
from templates.cyrus_execution import OrchestrationScript

class PokemonGenerator(OrchestrationScript):
    def __init__(self):
        super().__init__()
        self.client = ComfyUIClient()

    def phase_1_generate(self):
        """Generate all 151 Pokemon."""
        for i in range(1, 152):
            workflow = load_template("flux2_txt2img")
            workflow = inject_parameters(workflow, {
                "PROMPT": f"Pokemon #{i}: {get_pokemon_name(i)}",
                "SEED": i,
            })

            result = self.client.queue_prompt(workflow)
            prompt_id = result["prompt_id"]

            # Checkpoint
            if i % 10 == 0:
                self.checkpoint({"last_generated": i})

    def phase_2_verify(self):
        """Verify all assets exist."""
        missing = []
        for i in range(1, 152):
            asset_path = f"outputs/pokemon_{i}.png"
            if not Path(asset_path).exists():
                missing.append(i)

        if missing:
            print(f"Missing: {missing}")
            return False
        return True

    def get_phases(self):
        return {
            "1_phase_1_generate": self.phase_1_generate,
            "2_phase_2_verify": self.phase_2_verify,
        }
```

### 6.4 Error Recovery Pattern

```python
from comfyui_massmediafactory_mcp.mcp_utils import (
    timeout_error,
    connection_error,
)

def generate_with_retry(prompt, max_attempts=3):
    client = ComfyUIClient()

    for attempt in range(max_attempts):
        try:
            workflow = load_template("flux2_txt2img")
            workflow = inject_parameters(workflow, {"PROMPT": prompt})
            result = client.queue_prompt(workflow)

            if "error" in result:
                if "connection" in result["error"].lower():
                    time.sleep(5 ** attempt)
                    continue
                raise RuntimeError(result["error"])

            return result

        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError("Failed after max attempts")
```

---

## 7. Quick Reference

### 7.1 Essential Imports

```python
from comfyui_mcp import ComfyUIClient, generate_image, generate_video
from comfyui_massmediafactory_mcp.templates import (
    load_template,
    inject_parameters,
    list_templates,
)
from comfyui_massmediafactory_mcp.models import (
    get_model_constraints,
    list_installed_models,
)
from comfyui_massmediafactory_mcp.mcp_utils import (
    mcp_error,
    timeout_error,
    wait_for_completion,
)
```

### 7.2 Common Workflows

```python
# Image generation
generate_image(prompt="...", model="flux2")

# Video generation
generate_video(prompt="...", model="ltx2", frames=97)

# Template lookup
templates = list_templates(model_type="flux2")

# Batch operations
for i in range(10):
    generate_image(prompt=f"... {i}", seed=i)

# Background replacement
template = load_template("qwen_edit_background")
workflow = inject_parameters(template, {
    "IMAGE_PATH": "source.png",
    "EDIT_PROMPT": "change background to sky",
    "CFG": 2.0,
    "STEPS": 20,
})
```

### 7.3 Critical Gotchas

| Issue | Why It Happens | Fix |
|-------|----------------|-----|
| **Oversaturation** | Qwen Edit CFG > 4 | Keep CFG 2.0-2.5 |
| **Jittery video** | Frames not 8n+1 (LTX) | Use 17, 25, 33, 41, 49... |
| **High token cost** | Polling in loops | Use `wait_for_completion()` |
| **Memory leaks** | Not freeing VRAM | Call `free_memory()` between jobs |
| **Wrong node order** | Manual workflow builds | Use templates or skeletons |

---

## Appendix: Template Quick Reference

### FLUX.2 Templates (5)
- `flux2_txt2img` - Standard image generation
- `flux2_ultimate_upscale` - Image upscaling
- `flux2_edit_by_text` - Text-guided editing
- `flux2_face_id` - Face-preserved editing
- `flux2_lora_stack` - Multi-LoRA mixing
- `flux2_union_controlnet` - ControlNet integration
- `flux2_grounding_dino_inpaint` - Object-aware inpainting

### LTX-2 Templates (5)
- `ltx2_txt2vid` - Full quality video
- `ltx2_txt2vid_distilled` - Fast render (3-4x)
- `ltx2_img2vid` - Image-to-video
- `ltx2_audio_reactive` - Audio-synced video
- `ltx2_i2v_distilled` - Fast I2V

### Wan 2.6 Templates (2)
- `wan26_txt2vid` - Text-to-video
- `wan26_img2vid` - Image-to-video

### Qwen Templates (4)
- `qwen_txt2img` - Portrait generation
- `qwen_edit_background` - Background replacement
- `qwen_poster_design` - Poster layout

### Audio/TTS Templates (6)
- `chatterbox_tts` - Expressive TTS
- `audio_tts_f5` - Fast TTS
- `audio_tts_voice_clone` - Voice cloning
- `qwen3_tts_voice_design` - Voice design
- `qwen3_tts_voice_clone` - Qwen voice cloning
- `qwen3_tts_custom_voice` - Custom voice

### Utility Templates (3)
- `telestyle_image` - Style transfer
- `telestyle_video` - Video style transfer
- `video_stitch` - Video concatenation
- `video_inpaint` - Video inpainting

### Hunyuan Templates (2)
- `hunyuan15_txt2vid` - Text-to-video
- `hunyuan15_img2vid` - Image-to-video

### SDXL Templates (1)
- `sdxl_txt2img` - SDXL image generation

### Pony Templates (2)
- `turnaround_sheet` - Character turnaround
- `expression_sheet` - Character expressions
