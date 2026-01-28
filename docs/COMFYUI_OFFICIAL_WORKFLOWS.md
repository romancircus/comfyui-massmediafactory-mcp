# Official ComfyUI Workflow Patterns

**Reference Documentation for MassMediaFactory MCP**
**Last Updated:** January 2026
**Source:** comfyanonymous.github.io/ComfyUI_examples, Lightricks Official

---

## Purpose

This document captures official ComfyUI workflow patterns that should be used as reference when creating or updating templates. These patterns represent best practices endorsed by ComfyUI maintainers and model creators.

---

## Table of Contents

1. [LTX-Video Workflows](#ltx-video-workflows)
2. [FLUX Workflows](#flux-workflows)
3. [Wan 2.1 Workflows](#wan-21-workflows)
4. [Qwen Image Workflows](#qwen-image-workflows)
5. [Common Patterns](#common-patterns)
6. [Node Reference](#node-reference)

---

## LTX-Video Workflows

### Official Sources

| Source | URL |
|--------|-----|
| ComfyUI Examples | https://comfyanonymous.github.io/ComfyUI_examples/ltxv/ |
| Lightricks GitHub | https://github.com/Lightricks/ComfyUI-LTXVideo/tree/master/example_workflows |

### Text-to-Video (Full Model)

**Node Flow:**
```
CheckpointLoaderSimple (model)
    ↓
CLIPLoader (separate!) ─────────────────────┐
    ↓                                       │
CLIPTextEncode (positive) ←─────────────────┤
CLIPTextEncode (negative) ←─────────────────┘
    ↓
LTXVConditioning (strength: 25)
    ↓
EmptyLTXVLatentVideo (768×512, 97 frames)
    ↓
LTXVScheduler (steps: 30, scale: 2.05, threshold: 0.95)
    ↓
KSamplerSelect (sampler: "res_multistep")
    ↓
SamplerCustom (seed, cfg: 3) ← NOT KSampler!
    ↓
VAEDecode
    ↓
SaveAnimatedWEBP / SaveWEBM (24fps)
```

**Key Parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 768×512 | Native for LTX |
| Frames | 97 | ~4 seconds @ 24fps (must be 8n+1) |
| Steps | 30 | Full model |
| CFG | 3.0 | Low CFG for video |
| Scheduler max_shift | 2.05 | LTXVScheduler specific |
| Scheduler base_shift | 0.95 | LTXVScheduler specific |
| Sampler | euler or res_multistep | Via KSamplerSelect |

### Text-to-Video (Distilled Model) - Lightricks Official

**Enhanced Node Flow with Gemma-3:**
```
CheckpointLoaderSimple (ltx-2-19b-distilled)
    ↓
LTXVGemmaCLIPModelLoader (gemma_3_12B_it_fp8)
    ↓
LTXVGemmaEnhancePrompt (AI prompt enhancement)
    ↓
CLIPTextEncode (encode enhanced prompt)
    ↓
LTXVConditioning (frame_rate: 24)
    ↓
LoraLoaderModelOnly (camera LoRA, optional)
    ↓
EmptyLTXVLatentVideo (1920×1088, 121 frames)
    ↓
LTXVScheduler (steps: 8-12)
    ↓
KSamplerSelect → SamplerCustom (cfg: 2.5)
    ↓
LatentUpscaleModelLoader (2x spatial upscaler)
    ↓
VAEDecode
    ↓
LTXVAudioVAELoader → Audio generation (optional)
    ↓
CreateVideo (30fps) → SaveVideo
```

**Distilled Parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 1920×1088 | Divisible by 64 |
| Frames | 121 | Divisible by 8+1 |
| Steps | 8-12 | 3-4x fewer than full |
| CFG | 2.5 | Lower than full model |
| Output FPS | 30 | Upconverted from 24 |

### Image-to-Video

**Additional Nodes for I2V:**
```
LoadImage (input image)
    ↓
LTXVPreprocess (strength: 40)
    ↓
LTXVImgToVideo (768×512, 97 frames) ← Replaces EmptyLTXVLatentVideo
```

**I2V Parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Preprocess strength | 40 | Higher = more input preservation |
| Denoise | 0.85-0.95 | Lower = more input preservation |

---

## FLUX Workflows

### Official Source

| Source | URL |
|--------|-----|
| ComfyUI Examples | https://comfyanonymous.github.io/ComfyUI_examples/flux/ |

### Available Variants

| Workflow | Use Case | Key Features |
|----------|----------|--------------|
| flux_schnell | Fast drafts | 4 steps, distilled |
| flux_dev | Quality generation | 20-50 steps |
| flux_lora | Custom styles | LoRA loading |
| flux_redux | Style transfer | Reference image |
| flux_controlnet | Guided generation | Depth/canny/pose |
| flux_fill_inpaint | Region filling | Mask-based |
| flux_fill_outpaint | Canvas extension | Expand image |
| flux_regional | Area prompting | Multiple regions |
| flux_tools | Multi-tool | Combined pipeline |

### FLUX Node Pattern

**Key difference from SD/SDXL:**
```
DualCLIPLoader (clip_l + t5xxl) ─────────────────┐
    ↓                                            │
UNETLoader (flux1-dev.safetensors)               │
    ↓                                            │
CLIPTextEncode ←─────────────────────────────────┘
    ↓
FluxGuidance (guidance: 3.5)
    ↓
ModelSamplingFlux (resolution targeting)
    ↓
EmptySD3LatentImage (1024×1024)
    ↓
BasicScheduler (steps: 20, denoise: 1.0)
    ↓
KSamplerSelect → SamplerCustom
    ↓
VAEDecode
    ↓
SaveImage
```

**FLUX Parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 1024×1024 | Native, divisible by 16 |
| Steps (schnell) | 4 | Fast, distilled |
| Steps (dev) | 20-50 | Quality |
| Guidance | 3.5 | FluxGuidance node |
| Sampler | euler | Via KSamplerSelect |

### FLUX vs SD/SDXL Differences

| Aspect | SD/SDXL | FLUX |
|--------|---------|------|
| CLIP Loading | Single CLIPLoader | DualCLIPLoader (L + T5) |
| Model Loading | CheckpointLoaderSimple | UNETLoader + separate CLIP |
| CFG | cfg parameter | FluxGuidance node |
| Resolution | 512/1024 | 1024 (native) |
| Divisibility | 8 | 16 |

---

## Wan 2.1 Workflows

### Official Source

| Source | URL |
|--------|-----|
| ComfyUI Examples | https://comfyanonymous.github.io/ComfyUI_examples/wan/ |

### Available Variants

| Workflow | Model Size | Resolution |
|----------|------------|------------|
| wan_t2v_1.3b_480p | 1.3B | 480p |
| wan_t2v_14b_480p | 14B | 480p |
| wan_t2v_14b_720p | 14B | 720p HD |
| wan_i2v_480p | 14B | 480p |
| wan_i2v_720p | 14B | 720p HD |

### Wan Node Pattern

```
WanVideoModelLoader (wan_2.1_vae_bf16.safetensors)
    ↓
CLIPLoader (umt5_xxl_fp8.safetensors)
    ↓
CLIPTextEncode (positive/negative)
    ↓
WanImageEncode (for I2V only)
    ↓
EmptyWanLatentVideo (resolution, frames)
    ↓
BasicScheduler
    ↓
SamplerCustom
    ↓
WanVAEDecode
    ↓
VHS_VideoCombine
```

**Wan Parameters:**

| Parameter | 480p | 720p |
|-----------|------|------|
| Width | 832 | 1280 |
| Height | 480 | 720 |
| Frames | 81 | 81 |
| Steps | 30 | 30 |
| CFG | 5.0 | 5.0 |

---

## HunyuanVideo 1.5 Workflows

### Official Source

| Source | URL |
|--------|-----|
| Kijai Wrapper | https://github.com/Kijai/ComfyUI-HunyuanVideoWrapper |
| Tencent HunyuanVideo | https://github.com/Tencent/HunyuanVideo |

### Text-to-Video

**Node Flow:**
```
HunyuanVideoModelLoader (hunyuanvideo_t2v_720p)
    ↓
CLIPLoader (llava_llama3_fp8) ─────────────────┐
    ↓                                          │
CLIPTextEncode (positive) ←────────────────────┤
CLIPTextEncode (negative) ←────────────────────┘
    ↓
EmptyHunyuanLatentVideo (1280×720, 81 frames)
    ↓
HunyuanVideoSampler (cfg: 6.0, steps: 30)
    ↓
HunyuanVideoVAEDecode
    ↓
VHS_VideoCombine (24fps)
```

**Key Parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 1280×720 | 720p native, divisible by 16 |
| Frames | 81-129 | ~3-5 seconds @ 24fps |
| Steps | 30 | Quality generation |
| CFG | 6.0 | Higher than LTX/Wan |
| Sampler | euler | Standard |

### Image-to-Video

**Additional Nodes for I2V:**
```
LoadImage (input image)
    ↓
HunyuanVideoImageEncode (encodes reference)
    ↓
HunyuanVideoSampler (with image_embeds input)
```

**I2V Parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Image strength | 0.7-1.0 | Higher = more input preservation |
| Steps | 30-40 | Slightly more for I2V |

### HunyuanVideo Node Reference

| Node | Purpose | Outputs |
|------|---------|---------|
| HunyuanVideoModelLoader | Load HunyuanVideo model | MODEL, VAE |
| EmptyHunyuanLatentVideo | Create blank video latent | LATENT |
| HunyuanVideoSampler | Sample video frames | LATENT |
| HunyuanVideoVAEDecode | Decode to video frames | IMAGE |
| HunyuanVideoImageEncode | Encode reference image | IMAGE_EMBEDS |

---

## Qwen Image Workflows

### Requirements

| Component | Model |
|-----------|-------|
| Vision Model | Qwen-VL |
| VRAM | 16GB minimum (FP16) |
| Resolution | 1296×1296 native |

### Qwen Node Pattern

```
QwenVLLoader (model loading)
    ↓
QwenVLTextEncode (prompt encoding)
    ↓
EmptyLatentImage (1296×1296)
    ↓
KSampler (steps: 25-30)
    ↓
VAEDecode
    ↓
SaveImage
```

---

## Common Patterns

### Pattern 1: SamplerCustom vs KSampler

**Old Pattern (Avoid for video):**
```
KSampler (all-in-one)
```

**Modern Pattern (Preferred):**
```
LTXVScheduler / BasicScheduler → sigmas
    ↓
KSamplerSelect → sampler
    ↓
SamplerCustom (model, positive, negative, sampler, sigmas, latent)
```

**Why SamplerCustom?**
- Separates scheduler from sampler
- Model-specific scheduler support
- Better control over sigma values
- Required for advanced video models

### Pattern 2: Model-Specific Conditioning

**Generic (SD/SDXL):**
```
CLIPTextEncode → directly to sampler
```

**Video Models (LTX, Wan):**
```
CLIPTextEncode → LTXVConditioning/WanConditioning → sampler
```

**FLUX:**
```
CLIPTextEncode → FluxGuidance → sampler
```

### Pattern 3: Separate CLIP Loading

**Embedded (Simple):**
```
CheckpointLoaderSimple → outputs MODEL, CLIP, VAE together
```

**Separate (Advanced):**
```
UNETLoader → MODEL only
CLIPLoader → CLIP only (or DualCLIPLoader)
VAELoader → VAE only
```

**When to use separate:**
- Different precision for different components
- Custom text encoders (Gemma-3 for LTX)
- Memory optimization (fp8 model, fp16 CLIP)

---

## Node Reference

### Key Nodes by Model

| Model | Essential Nodes |
|-------|-----------------|
| LTX-2 | LTXVLoader, LTXVScheduler, LTXVConditioning, EmptyLTXVLatentVideo |
| LTX-2 Enhanced | LTXVGemmaCLIPModelLoader, LTXVGemmaEnhancePrompt |
| FLUX | DualCLIPLoader, UNETLoader, FluxGuidance, ModelSamplingFlux |
| Wan 2.1 | WanVideoModelLoader, EmptyWanLatentVideo, WanVAEDecode |
| Qwen | QwenVLLoader, QwenVLTextEncode |

### Sampler Compatibility

| Sampler | Image | Video | Notes |
|---------|-------|-------|-------|
| euler | Yes | Yes | Fast, good quality |
| euler_ancestral | Yes | No | Can cause flickering |
| dpmpp_2m | Yes | Limited | Good for images |
| res_multistep | Limited | Yes | Designed for video |
| uni_pc | Yes | Limited | Good for images |

### Resolution Rules

| Model | Native | Divisible By | Min | Max |
|-------|--------|--------------|-----|-----|
| SD 1.5 | 512 | 8 | 256 | 1024 |
| SDXL | 1024 | 8 | 512 | 2048 |
| FLUX | 1024 | 16 | 512 | 2048 |
| LTX | 768×512 | 8 | 256 | 2048 |
| Wan | 832×480 | 8 | 256 | 1280 |
| Qwen | 1296 | 8 | 512 | 2048 |

### Frame Count Rules (Video)

| Model | Formula | Examples |
|-------|---------|----------|
| LTX | 8n + 1 | 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121 |
| Wan | Any | 81 default |
| SVD | 14 or 25 | Fixed options |

---

## Implementation Checklist

When creating new templates, verify:

- [ ] Using SamplerCustom (not KSampler) for video
- [ ] Using model-specific scheduler (LTXVScheduler, etc.)
- [ ] Using model-specific conditioning wrapper
- [ ] Resolution divisible by model requirement
- [ ] Frame count follows model formula
- [ ] CFG appropriate for model type
- [ ] Separate CLIP loading if enhanced features needed

---

## Changelog

- **January 2026**: Initial documentation based on official ComfyUI examples and Lightricks workflows
