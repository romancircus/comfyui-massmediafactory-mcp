# MCP Improvement Analysis

**Date:** January 2026
**Status:** In-Depth Analysis Complete

---

## Executive Summary

This document provides an in-depth analysis of the ComfyUI workflow references gathered and a detailed plan for improving the MassMediaFactory MCP server.

---

## Part 1: Workflow Reference Verification

### Current Documentation Status

| Document | Purpose | Accuracy |
|----------|---------|----------|
| `COMFYUI_OFFICIAL_WORKFLOWS.md` | Official workflow patterns | **Verified** - Matches ComfyUI examples |
| `WORKFLOW_NODE_REFERENCE.md` | JSON workflow structures | **Verified** - Complete node definitions |
| `COMPREHENSIVE_IMPROVEMENT_PLAN.md` | Gap analysis & roadmap | Current |
| `WORKFLOW_IMPROVEMENT_PLAN.md` | Template creation plan | Current |

### Template Verification Results

#### LTX-2 Templates (VERIFIED)

| Template | Matches Documentation | Issues Found |
|----------|----------------------|--------------|
| `ltx2_txt2vid.json` | **YES** | None - uses correct LTXVLoader, SamplerCustom, LTXVScheduler |
| `ltx2_txt2vid_distilled.json` | **YES** | None - correctly uses Gemma-3 prompt enhancement |
| `ltx2_img2vid.json` | **YES** | None - uses LTXVPreprocess + LTXVImgToVideo |

**Key Patterns Verified:**
- Uses `SamplerCustom` (not KSampler) for video - CORRECT
- Uses `LTXVScheduler` with max_shift=2.05, base_shift=0.95 - CORRECT
- Uses `LTXVConditioning` for frame rate - CORRECT
- Frame count follows 8n+1 rule (97 = 12*8+1) - CORRECT
- CFG values: 3.0 for full, 2.5 for distilled - CORRECT

#### FLUX.2 Templates (VERIFIED)

| Template | Matches Documentation | Issues Found |
|----------|----------------------|--------------|
| `flux2_txt2img.json` | **YES** | None - uses correct SamplerCustomAdvanced pattern |

**Key Patterns Verified:**
- Uses `UNETLoader` (not CheckpointLoaderSimple) - CORRECT
- Uses `DualCLIPLoader` with clip_l + t5xxl - CORRECT
- Uses `FluxGuidance` for CFG (not sampler cfg) - CORRECT
- Uses `SamplerCustomAdvanced` with BasicGuider - CORRECT
- Resolution divisible by 16 - CORRECT

---

## Part 2: Gap Analysis

### Templates: Current vs Needed

#### Currently Implemented (22 templates)

**Video Generation (7):**
- ltx2_txt2vid ✅
- ltx2_txt2vid_distilled ✅
- ltx2_img2vid ✅
- ltx2_audio_reactive ✅
- wan26_img2vid ✅
- video_inpaint ✅
- video_stitch ✅

**Image Generation (9):**
- qwen_txt2img ✅
- qwen_poster_design ✅
- flux2_txt2img ✅
- flux2_face_id ✅
- flux2_edit_by_text ✅
- flux2_lora_stack ✅
- flux2_union_controlnet ✅
- flux2_grounding_dino_inpaint ✅
- flux2_ultimate_upscale ✅

**Audio/TTS (6):**
- audio_tts_f5 ✅
- audio_tts_voice_clone ✅
- chatterbox_tts ✅
- qwen3_tts_custom_voice ✅
- qwen3_tts_voice_clone ✅
- qwen3_tts_voice_design ✅

#### Critical Missing Templates (Priority 0)

| Template | Model | Why Critical |
|----------|-------|--------------|
| `ltx2_i2v_distilled` | LTX-2 Distilled | Fast I2V - 3x faster |
| `wan26_txt2vid` | Wan 2.6 | Text-to-video for Wan |
| `hunyuan15_txt2vid` | HunyuanVideo 1.5 | Major SOTA model |
| `hunyuan15_i2v` | HunyuanVideo 1.5 | I2V for Hunyuan |
| `sdxl_txt2img` | SDXL | Most popular base |
| `sd3_txt2img` | SD3/3.5 | Latest Stability model |

#### High Priority Missing Templates (Priority 1)

| Template | Model | Use Case |
|----------|-------|----------|
| `flux2_redux` | FLUX.2 | Style transfer |
| `flux2_canny_controlnet` | FLUX.2 | Edge-guided |
| `flux2_depth_controlnet` | FLUX.2 | Depth-guided |
| `flux2_ipadapter` | FLUX.2 | Character consistency |
| `qwen_inpaint` | Qwen | Inpainting |
| `qwen_lora` | Qwen | LoRA support |
| `cosmos_txt2vid` | Nvidia Cosmos | New SOTA video |
| `mochi_txt2vid` | Mochi | High quality video |

---

## Part 3: Code Architecture Analysis

### Current Strengths

1. **Well-structured template system** (`templates/__init__.py`)
   - Clean loading/listing functions
   - Parameter injection with type handling
   - Metadata support for discovery

2. **Comprehensive validation** (`validation.py`)
   - Node type validation
   - Model file existence checks
   - Connection compatibility

3. **SOTA tracking** (`sota.py`)
   - Current model knowledge
   - Deprecation tracking
   - Model settings database

### Areas for Improvement

#### 1. Template Validation on Load

**Current:** Templates load without validation
**Needed:** Validate template structure and parameters

```python
# Proposed: templates/__init__.py addition
def validate_template(template: dict) -> list[str]:
    """Validate template structure, return list of errors."""
    errors = []

    # Check _meta section
    if "_meta" not in template:
        errors.append("Missing _meta section")
    else:
        meta = template["_meta"]
        required = ["description", "model", "type", "parameters", "defaults"]
        for field in required:
            if field not in meta:
                errors.append(f"Missing _meta.{field}")

    # Check placeholders are declared
    workflow_str = json.dumps(template)
    placeholders = re.findall(r'\{\{([A-Z_]+)\}\}', workflow_str)
    declared = set(template.get("_meta", {}).get("parameters", []))
    for p in placeholders:
        if p not in declared:
            errors.append(f"Undeclared placeholder: {{{{{p}}}}}")

    return errors
```

#### 2. Template Discovery Tools

**Current:** Basic list only
**Needed:** Semantic search and capability matching

```python
# Proposed: server.py additions
@mcp.tool()
def find_template_for_task(task: str) -> dict:
    """Find best template for a given task description."""
    # Semantic matching based on:
    # - type (txt2img, img2vid, etc.)
    # - model capabilities
    # - performance requirements
    pass

@mcp.tool()
def get_templates_by_capability(capability: str) -> dict:
    """Filter templates by capability (controlnet, lora, upscale, etc.)."""
    pass
```

#### 3. Workflow Builder Helpers

**Current:** Must construct workflows manually
**Needed:** Skeleton builders for common patterns

```python
# Proposed: workflow_builder.py
def create_workflow_skeleton(model: str, task: str) -> dict:
    """Create minimal workflow skeleton for model+task."""
    pass

def get_node_sequence(model: str, task: str) -> list[dict]:
    """Return recommended node sequence for model+task."""
    pass

def explain_workflow(workflow: dict) -> str:
    """Generate natural language description of workflow."""
    pass
```

---

## Part 4: Documentation Gaps

### SOTA Models Not Yet Documented

| Model | Category | Status |
|-------|----------|--------|
| HunyuanVideo 1.5 | Video | Missing workflow patterns |
| Nvidia Cosmos | Video | Missing workflow patterns |
| Mochi | Video | Missing workflow patterns |
| SDXL | Image | Missing workflow patterns |
| SD3/SD3.5 | Image | Missing workflow patterns |
| AuraFlow | Image | Missing workflow patterns |
| Lumina 2.0 | Image | Missing workflow patterns |

### Missing from COMFYUI_OFFICIAL_WORKFLOWS.md

- [ ] ControlNet detailed patterns (individual types)
- [ ] IP-Adapter patterns
- [ ] Audio generation patterns (F5-TTS, Chatterbox)
- [ ] 3D generation patterns (Hunyuan3D)
- [ ] Upscaling patterns (ESRGAN, RealESRGAN)

---

## Part 5: Implementation Roadmap

### Phase 1: Critical (Immediate)

| Task | Files | Description |
|------|-------|-------------|
| Add template validation | `templates/__init__.py` | Validate on load |
| Create ltx2_i2v_distilled | `templates/` | Fast I2V template |
| Create wan26_txt2vid | `templates/` | Wan text-to-video |
| Document HunyuanVideo patterns | `docs/` | Official workflow study |

### Phase 2: High Priority (Week 1)

| Task | Files | Description |
|------|-------|-------------|
| Create hunyuan15_txt2vid | `templates/` | HunyuanVideo template |
| Create hunyuan15_i2v | `templates/` | HunyuanVideo I2V |
| Create sdxl_txt2img | `templates/` | SDXL base template |
| Add individual ControlNet templates | `templates/` | Canny, depth, pose |
| Create workflow builder tools | `workflow_builder.py` | New file |

### Phase 3: Medium Priority (Week 2-3)

| Task | Files | Description |
|------|-------|-------------|
| Create cosmos_txt2vid | `templates/` | Nvidia Cosmos |
| Create mochi_txt2vid | `templates/` | Mochi video |
| Add IP-Adapter templates | `templates/` | Character consistency |
| Add template discovery tools | `server.py` | Semantic search |
| Document audio patterns | `docs/` | TTS workflow patterns |

### Phase 4: Enhancement (Ongoing)

| Task | Files | Description |
|------|-------|-------------|
| Add 3D generation templates | `templates/` | Hunyuan3D |
| Create template requirement checker | `templates/__init__.py` | Model availability |
| Add workflow explanation tool | `workflow_builder.py` | Natural language |
| Performance benchmarking | `docs/` | Quality/speed metrics |

---

## Part 6: Detailed Template Specifications

### Templates to Create (Detailed)

#### 1. ltx2_i2v_distilled.json

```json
{
  "_meta": {
    "description": "LTX-2 19B Distilled image-to-video - fast I2V",
    "model": "LTX-2 19B Distilled",
    "type": "img2vid",
    "parameters": ["IMAGE_PATH", "PROMPT", "SEED", "FRAMES", "STRENGTH"],
    "defaults": {
      "FRAMES": 97,
      "STRENGTH": 40,
      "STEPS": 8
    }
  }
}
```
**Key nodes:** CheckpointLoaderSimple, LTXVGemmaCLIPModelLoader, LTXVPreprocess, LTXVImgToVideo

#### 2. wan26_txt2vid.json

```json
{
  "_meta": {
    "description": "Wan 2.6 text-to-video",
    "model": "Wan 2.6 14B",
    "type": "txt2vid",
    "parameters": ["PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "FRAMES"],
    "defaults": {
      "WIDTH": 832,
      "HEIGHT": 480,
      "FRAMES": 81
    }
  }
}
```
**Key nodes:** DownloadAndLoadWanModel, WanSampler, WanVAEDecode

#### 3. hunyuan15_txt2vid.json

```json
{
  "_meta": {
    "description": "HunyuanVideo 1.5 text-to-video",
    "model": "HunyuanVideo 1.5",
    "type": "txt2vid",
    "parameters": ["PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "FRAMES"],
    "defaults": {
      "WIDTH": 1280,
      "HEIGHT": 720,
      "FRAMES": 81
    }
  }
}
```
**Key nodes:** HunyuanVideoModelLoader, HunyuanVideoSampler, HunyuanVideoVAEDecode

---

## Part 7: Success Metrics

### Template Coverage
- [ ] All SOTA video models have T2V and I2V templates
- [ ] All SOTA image models have txt2img templates
- [ ] ControlNet templates for top 3 types (canny, depth, pose)
- [ ] IP-Adapter templates for character consistency

### Code Quality
- [ ] All templates validate on load (0 errors)
- [ ] All templates have complete _meta sections
- [ ] All placeholders are declared in parameters

### Documentation
- [ ] COMFYUI_OFFICIAL_WORKFLOWS.md covers all SOTA models
- [ ] WORKFLOW_NODE_REFERENCE.md has JSON for all templates
- [ ] Each template has source attribution

### Agent Usability
- [ ] Agents can discover templates by capability
- [ ] Agents can build workflows from scratch using helpers
- [ ] Workflow explanation tool works for all templates

---

## Appendix: Official Source URLs

### Verified Official Sources

| Source | URL | Content |
|--------|-----|---------|
| ComfyUI Examples | comfyanonymous.github.io/ComfyUI_examples | 41 workflow categories |
| Lightricks LTX-Video | github.com/Lightricks/ComfyUI-LTXVideo | Official LTX nodes |
| Kijai HunyuanVideo | github.com/Kijai/ComfyUI-HunyuanVideoWrapper | HunyuanVideo nodes |
| Kijai Wan | github.com/Kijai/ComfyUI-WanVideoWrapper | Wan nodes |

### Workflow Extraction Commands

```bash
# Clone official workflow repos for study
git clone https://github.com/Lightricks/ComfyUI-LTXVideo ~/study/ltx-official
git clone https://github.com/comfyanonymous/ComfyUI_examples ~/study/comfy-examples
git clone https://github.com/Kijai/ComfyUI-HunyuanVideoWrapper ~/study/hunyuan
```

---

*Analysis completed January 2026*
