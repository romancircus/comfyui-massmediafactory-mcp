# Workflow Improvement Plan

**Date:** January 2026
**Goal:** Maximize performance and expand workflow coverage for SOTA models

---

## Part 1: LTX-2 Distilled Integration

### Current State

| Model | File | VRAM | Steps | Speed |
|-------|------|------|-------|-------|
| LTX-2 19B Dev FP8 | `ltx-2-19b-dev-fp8.safetensors` | ~18GB | 30 | Baseline |
| LTX-2 19B Distilled | `ltx-2-19b-distilled.safetensors` | ~16GB | 8-12 | **3-4x faster** |

### Distilled vs Full: Analysis

**Distilled Model Advantages:**
- **Steps**: 8-12 steps vs 30 steps (3-4x fewer)
- **Speed**: ~3-4x faster generation
- **VRAM**: Slightly lower (~16GB vs ~18GB)
- **Quality**: 90-95% of full model quality

**When to Use Distilled:**
- Rapid prototyping and iteration
- Real-time preview workflows
- Batch generation (high volume)
- Time-sensitive production

**When to Use Full (Dev):**
- Final production renders
- Maximum quality required
- Complex motion sequences
- Fine detail preservation

### Templates to Create

| Template | Model | Use Case |
|----------|-------|----------|
| `ltx2_txt2vid_distilled` | Distilled | Fast text-to-video |
| `ltx2_i2v_distilled` | Distilled | Fast image-to-video |
| `ltx2_depth2vid` | Dev FP8 | Depth-guided video |
| `ltx2_canny2vid` | Dev FP8 | Edge-guided video |
| `ltx2_extend` | Dev FP8 | Video extension |

### Optimal Settings for RTX 5090 (32GB)

```yaml
# Distilled (Fast)
model: ltx-2-19b-distilled.safetensors
steps: 8-12
cfg: 2.5-3.0
resolution: 768x512 (480p) or 1280x720 (720p)
frames: 97 (4 seconds @ 24fps)
scheduler: LTXVScheduler with stretch=true

# Dev FP8 (Quality)
model: ltx-2-19b-dev-fp8.safetensors
steps: 25-30
cfg: 3.0-3.5
resolution: 768x512 or 1280x720
frames: 97-193
scheduler: LTXVScheduler
```

### Performance Optimization Checklist

- [ ] Use `--highvram` flag (already enabled)
- [ ] Use bfloat16 dtype for LTX models
- [ ] Enable VAE tiling for 720p+ (LTXVTiledVAEDecode)
- [ ] Use euler sampler (fastest for LTX)
- [ ] Set stretch=true in LTXVScheduler

---

## Part 2: Top ComfyUI Workflows to Study

### Priority Workflow Sources

| Source | URL | Content |
|--------|-----|---------|
| **Lightricks Official** | github.com/Lightricks/ComfyUI-LTXVideo | Official LTX nodes + examples |
| **ComfyUI Examples** | github.com/comfyanonymous/ComfyUI_examples | Core workflow patterns |
| **CivitAI Workflows** | civitai.com/models (filter: workflow) | Community best practices |
| **OpenArt Workflows** | openart.ai/workflows | Curated quality workflows |
| **ComfyWorkflows** | comfyworkflows.com | Searchable workflow database |

### Target Workflows to Extract (Top 20)

#### LTX-2 Family (8 workflows)
1. `ltx2_txt2vid` - Text-to-video (basic) ✅ Have
2. `ltx2_txt2vid_distilled` - Text-to-video (fast) ❌ Need
3. `ltx2_i2v` - Image-to-video (full) ❌ Need
4. `ltx2_i2v_distilled` - Image-to-video (fast) ❌ Need
5. `ltx2_depth2vid` - Depth ControlNet ❌ Need
6. `ltx2_canny2vid` - Canny ControlNet ❌ Need
7. `ltx2_extend` - Video extension ❌ Need
8. `ltx2_audio_reactive` - Audio sync ✅ Have

#### Qwen Image Family (4 workflows)
9. `qwen_txt2img` - Basic generation ✅ Have
10. `qwen_poster_design` - Text-heavy designs ✅ Have
11. `qwen_inpaint` - Inpainting ❌ Need
12. `qwen_outpaint` - Outpainting/extension ❌ Need

#### FLUX.2 Family (4 workflows)
13. `flux2_txt2img` - Basic generation ✅ Have
14. `flux2_face_id` - Face consistency ✅ Have
15. `flux2_ultimate_upscale` - 4K upscaling ✅ Have
16. `flux2_redux` - Style transfer ❌ Need

#### Wan 2.1 Family (2 workflows)
17. `wan21_img2vid` - Image-to-video ✅ Have
18. `wan21_txt2vid` - Text-to-video ❌ Need

#### HunyuanVideo 1.5 (2 workflows)
19. `hunyuan15_txt2vid` - Text-to-video ❌ Need
20. `hunyuan15_i2v` - Image-to-video ❌ Need

### Workflow Study Framework

For each workflow, extract:

```yaml
workflow_name: ltx2_i2v_distilled
source: Lightricks Official
nodes_used:
  - LTXVLoader
  - LTXVImgToVideo
  - LTXVScheduler
  - SamplerCustom
  - VAEDecode
connection_pattern: |
  Image → LTXVPreprocess → LTXVImgToVideo → conditioning
  Model → SamplerCustom → VAEDecode → output
key_parameters:
  steps: 8
  cfg: 2.5
  denoise: 0.85
performance_notes: |
  - Uses img2vid conditioning node
  - Denoise <1.0 preserves input structure
  - 3x faster than full model
```

---

## Part 3: Extraction Plan

### Phase 1: Official Sources (Highest Quality)

#### Lightricks Official LTX-2 Workflows
Source: https://github.com/Lightricks/ComfyUI-LTXVideo/tree/master/example_workflows

| Workflow | URL | Description |
|----------|-----|-------------|
| **LTX-2_I2V_Distilled_wLora.json** | [Download](https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/example_workflows/LTX-2_I2V_Distilled_wLora.json) | Image-to-video, distilled model, with LoRA |
| **LTX-2_I2V_Full_wLora.json** | [Download](https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/example_workflows/LTX-2_I2V_Full_wLora.json) | Image-to-video, full model, with LoRA |
| **LTX-2_T2V_Distilled_wLora.json** | [Download](https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/example_workflows/LTX-2_T2V_Distilled_wLora.json) | Text-to-video, distilled model |
| **LTX-2_T2V_Full_wLora.json** | [Download](https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/example_workflows/LTX-2_T2V_Full_wLora.json) | Text-to-video, full model |
| **LTX-2_ICLoRA_All_Distilled.json** | [Download](https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/example_workflows/LTX-2_ICLoRA_All_Distilled.json) | In-context LoRA, all distilled |
| **LTX-2_V2V_Detailer.json** | [Download](https://raw.githubusercontent.com/Lightricks/ComfyUI-LTXVideo/master/example_workflows/LTX-2_V2V_Detailer.json) | Video-to-video enhancement |

#### ComfyUI Official Examples
Source: https://comfyanonymous.github.io/ComfyUI_examples/ltxv/

| Workflow | Type | Description |
|----------|------|-------------|
| Image to Video (Simple) | WebP/JSON | Single start image |
| Image to Video (Complex) | WebP/JSON | Multiple guiding images |
| Text to Video | WebP/JSON | Text prompt based |

```bash
# Clone official repos
git clone https://github.com/Lightricks/ComfyUI-LTXVideo ~/workflows/ltx-official
git clone https://github.com/comfyanonymous/ComfyUI_examples ~/workflows/comfy-examples
git clone https://github.com/Kijai/ComfyUI-HunyuanVideoWrapper ~/workflows/hunyuan
```

### Phase 2: API-Based Extraction

```python
# CivitAI API - search for workflows
GET https://civitai.com/api/v1/models?types=Workflows&sort=Most%20Downloaded

# OpenArt API - curated workflows
GET https://openart.ai/api/workflows?category=video&sort=popular
```

### Phase 3: Manual Curation

1. Visit ComfyWorkflows.com
2. Search by model: "LTX-2", "Qwen", "FLUX.2"
3. Filter by rating: 4.5+ stars
4. Download JSON workflows
5. Convert to API format if needed

---

## Part 4: Implementation Tasks

### Task 1: Create Distilled Templates
- [ ] `ltx2_txt2vid_distilled` - 8 steps, cfg 2.5
- [ ] `ltx2_i2v_distilled` - 8 steps, denoise 0.85

### Task 2: Create ControlNet Templates
- [ ] `ltx2_depth2vid` - Depth-guided generation
- [ ] `ltx2_canny2vid` - Edge-guided generation

### Task 3: Expand Model Coverage
- [ ] `wan21_txt2vid` - Wan text-to-video
- [ ] `hunyuan15_txt2vid` - HunyuanVideo 1.5
- [ ] `qwen_inpaint` - Qwen inpainting

### Task 4: Performance Benchmarking
- [ ] Benchmark distilled vs full (same prompt)
- [ ] Document quality/speed tradeoffs
- [ ] Create recommendation matrix

---

## Part 5: Workflow Template Structure

### Standard Template Format

```json
{
  "_meta": {
    "name": "ltx2_i2v_distilled",
    "description": "Fast LTX-2 image-to-video with distilled model",
    "model": "LTX-2 19B Distilled",
    "type": "img2vid",
    "parameters": ["IMAGE_PATH", "PROMPT", "SEED", "WIDTH", "HEIGHT", "FRAMES"],
    "defaults": {
      "WIDTH": 768,
      "HEIGHT": 512,
      "FRAMES": 97,
      "SEED": 42
    },
    "performance": {
      "vram_gb": 16,
      "time_seconds": 45,
      "quality_rating": 0.92
    }
  },
  "nodes": { ... }
}
```

### Quality Tiers

| Tier | Model | Steps | Use Case |
|------|-------|-------|----------|
| **Draft** | Distilled | 4-6 | Quick preview |
| **Standard** | Distilled | 8-12 | Production fast |
| **Quality** | Dev FP8 | 20-25 | Production quality |
| **Premium** | Dev FP8 | 30+ | Maximum quality |

---

## Next Actions

1. **Immediate**: Create `ltx2_i2v_distilled` template using existing nodes
2. **Today**: Fetch Lightricks official workflow examples
3. **This Week**: Extract and convert top 10 community workflows
4. **Ongoing**: Benchmark and document performance differences

---

*Plan created January 2026 - Ready for implementation*
