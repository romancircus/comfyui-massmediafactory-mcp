# ComfyUI Parameter Rules

> **Purpose**: Validation constraints for LLM-generated workflows.
> **Usage**: Check all parameters against these rules before outputting JSON.

---

## Resolution Rules

### Divisibility Requirements

| Model | Divisible By | Native Resolution | Min | Max |
|:------|:-------------|:------------------|:----|:----|
| SD 1.5 | 8 | 512x512 | 256 | 1024 |
| SDXL | 8 | 1024x1024 | 512 | 2048 |
| **FLUX** | **16** | 1024x1024 | 512 | 2048 |
| LTX | 8 | 768x512 | 256 | 2048 |
| LTX-2 Distilled | 64 | 1920x1088 | 512 | 1920 |
| Wan 2.1 (480p) | 8 | 832x480 | 256 | 1280 |
| Wan 2.1 (720p) | 8 | 1280x720 | 256 | 1280 |
| Qwen | 8 | 1296x1296 | 512 | 2048 |

### Validation Logic

```python
def validate_resolution(width, height, model):
    rules = {
        "flux": {"div": 16, "min": 512, "max": 2048},
        "ltx": {"div": 8, "min": 256, "max": 2048},
        "ltx_distilled": {"div": 64, "min": 512, "max": 1920},
        "wan": {"div": 8, "min": 256, "max": 1280},
        "qwen": {"div": 8, "min": 512, "max": 2048},
        "sdxl": {"div": 8, "min": 512, "max": 2048},
        "sd15": {"div": 8, "min": 256, "max": 1024},
    }
    r = rules.get(model)
    if not r:
        return False, "Unknown model"

    if width % r["div"] != 0:
        return False, f"Width {width} not divisible by {r['div']}"
    if height % r["div"] != 0:
        return False, f"Height {height} not divisible by {r['div']}"
    if width < r["min"] or width > r["max"]:
        return False, f"Width {width} outside range [{r['min']}, {r['max']}]"
    if height < r["min"] or height > r["max"]:
        return False, f"Height {height} outside range [{r['min']}, {r['max']}]"

    return True, "Valid"
```

---

## Frame Count Rules (Video)

### LTX Video: 8n + 1 Rule

**Formula**: `frames = 8 * n + 1` where n >= 1

**Valid Frame Counts**:
```
9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161
```

**Common Choices**:
| Frames | Duration @ 24fps | Notes |
|:-------|:-----------------|:------|
| 25 | ~1 second | Quick test |
| 49 | ~2 seconds | Short clip |
| 73 | ~3 seconds | Standard |
| 97 | ~4 seconds | Default |
| 121 | ~5 seconds | Extended |

### Validation Logic

```python
def validate_ltx_frames(frames):
    if frames < 9:
        return False, "Minimum 9 frames"
    if (frames - 1) % 8 != 0:
        valid = ((frames // 8) * 8) + 1
        if frames > valid:
            valid += 8
        return False, f"Use {valid} instead of {frames}"
    return True, "Valid"
```

### Wan 2.1 Frame Counts

| Tier | Default | Range |
|:-----|:--------|:------|
| 480p | 81 | 1-120 |
| 720p | 81 | 1-120 |

---

## CFG (Classifier-Free Guidance) Rules

### Model-Specific CFG

| Model | CFG Range | Default | Notes |
|:------|:----------|:--------|:------|
| SD 1.5 | 5-15 | 7.5 | Standard |
| SDXL | 5-12 | 7.0 | Slightly lower |
| **FLUX** | N/A | N/A | Uses FluxGuidance node |
| LTX | 2-5 | 3.0 | Low for video coherence |
| LTX-2 Distilled | 2-4 | 2.5 | Even lower |
| Wan 2.1 | 4-7 | 5.0 | Medium |
| Qwen | 2-5 | 3.0 | Low |

### FLUX Guidance (Replaces CFG)

```json
{
  "class_type": "FluxGuidance",
  "inputs": {
    "conditioning": ["CLIPTextEncode", 0],
    "guidance": 3.5
  }
}
```

**Valid Range**: 0.0 - 100.0 (practical: 1.0 - 10.0)

---

## Steps Rules

### Model-Specific Steps

| Model | Min | Default | Max | Notes |
|:------|:----|:--------|:----|:------|
| FLUX Schnell | 1 | 4 | 8 | Distilled, fast |
| FLUX Dev | 15 | 20-30 | 50 | Quality |
| LTX Full | 20 | 30 | 50 | Quality |
| LTX Distilled | 6 | 8-12 | 15 | Fast |
| Wan 2.1 | 20 | 30 | 50 | Standard |
| Qwen | 20 | 25-30 | 50 | Standard |
| SDXL | 20 | 25-35 | 100 | Standard |
| SD 1.5 | 15 | 20-30 | 100 | Standard |

---

## Scheduler Rules

### Model-Scheduler Compatibility

| Model | Recommended Scheduler | Alternative |
|:------|:---------------------|:------------|
| FLUX | simple, sgm_uniform | normal |
| LTX | LTXVScheduler only | - |
| Wan 2.1 | normal, karras | sgm_uniform |
| Qwen | simple | normal |
| SDXL | karras | exponential |
| SD 1.5 | karras | normal |

### Scheduler Nodes by Model

| Model | Scheduler Node |
|:------|:---------------|
| FLUX | BasicScheduler |
| LTX | LTXVScheduler |
| Wan 2.1 | BasicScheduler |
| Qwen | (built into KSampler) |
| SDXL | BasicScheduler or KSampler |
| SD 1.5 | BasicScheduler or KSampler |

---

## Sampler Rules

### Video-Safe Samplers

| Sampler | Video Safe | Image | Notes |
|:--------|:-----------|:------|:------|
| euler | Yes | Yes | Universal |
| euler_ancestral | **No** | Yes | Causes flickering |
| dpmpp_2m | Limited | Yes | May cause artifacts |
| res_multistep | Yes | Limited | Designed for video |
| uni_pc | Limited | Yes | Better for images |
| ddpm | Yes | Yes | High quality, slow |

### LTX-Specific Samplers

Always use via `KSamplerSelect`:
- `euler` (recommended)
- `res_multistep` (alternative)

---

## LTX-Specific Parameters

### LTXVScheduler Parameters

| Parameter | Default | Range | Notes |
|:----------|:--------|:------|:------|
| max_shift | 2.05 | 0.0-10.0 | Model requirement |
| base_shift | 0.95 | 0.0-10.0 | Model requirement |
| stretch | true | boolean | Enable scaling |
| terminal | 0.1 | 0.0-1.0 | Ending sigma |

### LTXVConditioning Parameters

| Parameter | Default | Range | Notes |
|:----------|:--------|:------|:------|
| frame_rate | 24 | 1-60 | Target FPS |
| strength | 25 | 0-100 | Conditioning strength |

---

## Type Compatibility Matrix

### What Can Connect to What

| Output Type | Valid Inputs |
|:------------|:-------------|
| MODEL | model, unet |
| CLIP | clip |
| VAE | vae |
| CONDITIONING | positive, negative, conditioning |
| LATENT | latent_image, samples, latent |
| IMAGE | image, images, pixels |
| SIGMAS | sigmas |
| SAMPLER | sampler |
| MASK | mask |

### Common Mistakes

| Mistake | Error | Fix |
|:--------|:------|:----|
| MODEL -> LATENT | Type mismatch | MODEL goes to sampler |
| CLIP -> MODEL | Type mismatch | CLIP goes to text encode |
| CONDITIONING -> LATENT | Type mismatch | CONDITIONING goes to sampler |
| IMAGE -> LATENT | Type mismatch | Use VAEEncode |

---

## Quick Validation Checklist

Before outputting workflow JSON, verify:

- [ ] Resolution divisible by model requirement (16 for FLUX)
- [ ] Frame count is 8n+1 for LTX
- [ ] CFG appropriate for model (or FluxGuidance for FLUX)
- [ ] Using SamplerCustom for video models
- [ ] Using model-specific scheduler
- [ ] Using model-specific conditioning wrapper
- [ ] Sampler is video-safe (no euler_ancestral for video)
- [ ] All type connections are valid

---

## Changelog

- **January 2026**: Initial parameter rules for LLM validation
