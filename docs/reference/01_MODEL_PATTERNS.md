# ComfyUI Model Patterns Reference

> **Purpose**: High-level architecture reference for LLM agents constructing workflow JSONs.
> **Usage**: Consult this before generating any workflow. Find your model, follow its pattern.

---

## Quick Pattern Lookup

| Model | Sampler Type | Conditioning | Key Nodes | Resolution |
|-------|--------------|--------------|-----------|------------|
| LTX-Video | SamplerCustom | LTXVConditioning | LTXVScheduler, EmptyLTXVLatentVideo | 768x512 |
| LTX-2 Distilled | SamplerCustom | LTXVConditioning | LTXVScheduler + LoRA | 1920x1088 |
| FLUX | SamplerCustom | FluxGuidance | DualCLIPLoader, ModelSamplingFlux | 1024x1024 |
| Wan 2.1 | SamplerCustom | Direct | WanVideoModelLoader, EmptyWanLatentVideo | 832x480 |
| Qwen | KSampler | Direct | UNETLoader, ModelSamplingAuraFlow | 1296x1296 |

---

## Pattern: LTX-Video (Text-to-Video)

**Topology Strategy**: `Separate Scheduler/Sampler` - Do NOT use KSampler

### Node Configuration

| Node Class | Input Name | Required Value | Reason |
|:-----------|:-----------|:---------------|:-------|
| `CheckpointLoaderSimple` | `ckpt_name` | `ltx-video-2b*.safetensors` | Full model |
| `CLIPLoader` | `clip_name` | Separate from checkpoint | Enhanced prompt support |
| `LTXVScheduler` | `max_shift` | `2.05` | Model requirement |
| `LTXVScheduler` | `base_shift` | `0.95` | Model requirement |
| `LTXVScheduler` | `steps` | `30` | Full model quality |
| `EmptyLTXVLatentVideo` | `width` | `768` | Native resolution |
| `EmptyLTXVLatentVideo` | `height` | `512` | Native resolution |
| `EmptyLTXVLatentVideo` | `length` | `97` | Must be `8n+1` |
| `LTXVConditioning` | `strength` | `25` | Default strength |
| `KSamplerSelect` | `sampler_name` | `euler` or `res_multistep` | Video-safe samplers |
| `SamplerCustom` | `cfg` | `3.0` | Low CFG for video coherence |

### Connection Logic

```
CheckpointLoaderSimple.MODEL ──────────────────────→ SamplerCustom.model
CLIPLoader.CLIP ───→ CLIPTextEncode.clip (x2)
CLIPTextEncode ───→ LTXVConditioning ───→ SamplerCustom.positive/negative
LTXVScheduler.SIGMAS ──────────────────────────────→ SamplerCustom.sigmas
KSamplerSelect.SAMPLER ────────────────────────────→ SamplerCustom.sampler
EmptyLTXVLatentVideo.LATENT ───────────────────────→ SamplerCustom.latent_image
SamplerCustom.output ───→ VAEDecode.samples
```

### Critical Warning

> **Do NOT connect `CLIPTextEncode` directly to the Sampler.**
> You MUST pass it through `LTXVConditioning` first to apply frame-rate encoding.

---

## Pattern: LTX-2 Distilled (Text-to-Video)

**Topology Strategy**: `Distilled + LoRA + Upscaling`

### Key Differences from Full Model

| Aspect | Full Model | Distilled |
|--------|------------|-----------|
| Steps | 30 | 8-12 |
| CFG | 3.0 | 2.5 |
| Resolution | 768x512 | 1920x1088 |
| Prompt Enhancement | Manual | LTXVGemmaEnhancePrompt |
| Output FPS | 24 | 30 (upscaled) |

### Additional Nodes

| Node Class | Purpose |
|:-----------|:--------|
| `LTXVGemmaCLIPModelLoader` | Load Gemma-3 for prompt enhancement |
| `LTXVGemmaEnhancePrompt` | AI-enhance user prompts |
| `LoraLoaderModelOnly` | Camera movement LoRAs |
| `LatentUpscaleModelLoader` | 2x spatial upscaling |
| `LTXVAudioVAELoader` | Audio generation (optional) |

---

## Pattern: FLUX (Text-to-Image)

**Topology Strategy**: `Dual CLIP + FluxGuidance` - Do NOT use standard CFG

### Node Configuration

| Node Class | Input Name | Required Value | Reason |
|:-----------|:-----------|:---------------|:-------|
| `DualCLIPLoader` | `clip_name1` | `clip_l*.safetensors` | CLIP L encoder |
| `DualCLIPLoader` | `clip_name2` | `t5xxl*.safetensors` | T5 XXL encoder |
| `UNETLoader` | `unet_name` | `flux1-dev.safetensors` | Model separately |
| `FluxGuidance` | `guidance` | `3.5` | Replaces CFG |
| `ModelSamplingFlux` | - | Required | Resolution targeting |
| `EmptySD3LatentImage` | `width` | `1024` | Must be divisible by 16 |
| `EmptySD3LatentImage` | `height` | `1024` | Must be divisible by 16 |
| `BasicScheduler` | `steps` | `20-50` (dev), `4` (schnell) | Variant-specific |
| `KSamplerSelect` | `sampler_name` | `euler` | Standard choice |
| `SamplerCustom` | - | No CFG param | Guidance via FluxGuidance |

### FLUX vs SD/SDXL Differences

| Aspect | SD/SDXL | FLUX |
|:-------|:--------|:-----|
| CLIP Loading | `CLIPLoader` | `DualCLIPLoader` (L + T5) |
| Model Loading | `CheckpointLoaderSimple` | `UNETLoader` + separate CLIP |
| CFG Control | `cfg` parameter on sampler | `FluxGuidance` node |
| Resolution Divisibility | 8 | 16 |

### Critical Warning

> **FLUX does NOT use the `cfg` parameter on SamplerCustom.**
> You MUST use the `FluxGuidance` node before conditioning.

---

## Pattern: Wan 2.1 (Video)

**Topology Strategy**: `Separate Model + VAE Loading`

### Node Configuration

| Node Class | Input Name | Required Value | Reason |
|:-----------|:-----------|:---------------|:-------|
| `WanVideoModelLoader` | `model_name` | `wan_2.1_*.safetensors` | Video model |
| `CLIPLoader` | `clip_name` | `umt5_xxl_fp8.safetensors` | Text encoder |
| `EmptyWanLatentVideo` | `width` | `832` (480p) or `1280` (720p) | Resolution tier |
| `EmptyWanLatentVideo` | `height` | `480` or `720` | Resolution tier |
| `EmptyWanLatentVideo` | `frames` | `81` | Default frame count |
| `BasicScheduler` | `steps` | `30` | Standard steps |
| `SamplerCustom` | `cfg` | `5.0` | Higher than LTX |

### Wan Resolution Tiers

| Tier | Width | Height | Model Size |
|:-----|:------|:-------|:-----------|
| 480p | 832 | 480 | 1.3B or 14B |
| 720p | 1280 | 720 | 14B only |

---

## Pattern: Qwen Image

**Topology Strategy**: `Standard KSampler + AuraFlow Sampling`

### Node Configuration

| Node Class | Input Name | Required Value | Reason |
|:-----------|:-----------|:---------------|:-------|
| `UNETLoader` | `unet_name` | `qwen_image_2512*.safetensors` | Image model |
| `CLIPLoader` | `clip_name` | `qwen_2.5_vl_7b*.safetensors` | Vision-language encoder |
| `CLIPLoader` | `type` | `qwen_image` | Specific type |
| `ModelSamplingAuraFlow` | `shift` | `3.1` | Required sampling |
| `EmptySD3LatentImage` | `width` | `1296` | Native resolution |
| `EmptySD3LatentImage` | `height` | `1296` | Native resolution |
| `KSampler` | `steps` | `25-30` | Quality setting |
| `KSampler` | `cfg` | `3.0` | Low CFG |
| `KSampler` | `sampler_name` | `euler` | Standard sampler |
| `KSampler` | `scheduler` | `simple` | AuraFlow compatible |

### Qwen Prompt Tips

- Be explicit about text placement: "Text at top center"
- Specify font style: "bold font, red color"
- Describe layout sections: "Header area, main content, footer"
- Use design terminology: "minimalist", "gradient background"

---

## Sampler Selection Guide

### When to Use KSampler

- Simple image generation (SD 1.5, SDXL, Qwen)
- No model-specific scheduler required
- Standard CFG control needed

### When to Use SamplerCustom

- Video generation (LTX, Wan)
- Model-specific scheduler (LTXVScheduler)
- FluxGuidance for FLUX models
- Advanced sigma control needed

### Video-Safe Samplers

| Sampler | Image | Video | Notes |
|:--------|:------|:------|:------|
| `euler` | Yes | Yes | Universal, fast |
| `euler_ancestral` | Yes | **No** | Causes flickering |
| `dpmpp_2m` | Yes | Limited | Better for images |
| `res_multistep` | Limited | Yes | Designed for video |
| `uni_pc` | Yes | Limited | Better for images |

---

## Connection Types Reference

### Standard Outputs

| Node Type | Slot 0 | Slot 1 | Slot 2 |
|:----------|:-------|:-------|:-------|
| CheckpointLoaderSimple | MODEL | CLIP | VAE |
| UNETLoader | MODEL | - | - |
| CLIPLoader | CLIP | - | - |
| VAELoader | VAE | - | - |
| CLIPTextEncode | CONDITIONING | - | - |
| SamplerCustom | LATENT | LATENT (denoised) | - |

### Type Safety

| Output Type | Can Connect To |
|:------------|:---------------|
| MODEL | model inputs only |
| CLIP | clip inputs only |
| VAE | vae inputs only |
| CONDITIONING | positive/negative conditioning |
| LATENT | latent_image, samples |
| IMAGE | images input |
| SIGMAS | sigmas input |
| SAMPLER | sampler input |

---

## Changelog

- **January 2026**: Initial LLM reference documentation based on Gemini architecture recommendations
