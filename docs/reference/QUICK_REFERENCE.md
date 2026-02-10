# ComfyUI Workflow Quick Reference

One-page reference for building valid ComfyUI workflows.

---

## Model -> Sampler Type

| Model | Sampler | Notes |
|-------|---------|-------|
| LTX-2 | SamplerCustom | Use LTXVScheduler for sigmas |
| FLUX.2 | SamplerCustomAdvanced | Use BasicGuider + RandomNoise |
| Wan 2.1 | WanSampler | Built-in sampler, no separate scheduler |
| HunyuanVideo | HunyuanVideoSampler | Built-in CFG, no FluxGuidance |
| Qwen | KSampler | Standard sampler OK |
| SDXL | KSampler | Standard sampler OK |

---

## Resolution Rules

| Model | Divisible By | Native | Max |
|-------|--------------|--------|-----|
| FLUX.2 | 16 | 1024x1024 | 2048x2048 |
| LTX-2 | 8 | 768x512 | 1920x1088 |
| Wan 2.1 | 8 | 832x480 | 1280x720 |
| HunyuanVideo | 16 | 1280x720 | 1920x1080 |
| Qwen | 8 | 1296x1296 | 2048x2048 |
| SDXL | 8 | 1024x1024 | 2048x2048 |

---

## Frame Count Rules

| Model | Rule | Valid Examples |
|-------|------|----------------|
| LTX-2 | 8n+1 | 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121 |
| Wan 2.1 | Any | 81 default (~3.4s at 24fps) |
| HunyuanVideo | Any | 81-129 typical |

---

## CFG/Guidance Values

| Model | Range | Default | Notes |
|-------|-------|---------|-------|
| LTX-2 | 2.5-4.0 | 3.0 | Low CFG works best |
| FLUX.2 | 2.5-5.0 | 3.5 | Via FluxGuidance node (not cfg param) |
| Wan 2.1 | 4.0-7.0 | 5.0 | Standard CFG in sampler |
| HunyuanVideo | 4.0-8.0 | 6.0 | Standard CFG in sampler |
| Qwen | 3.0-8.0 | 7.0 | Standard CFG |
| SDXL | 5.0-10.0 | 7.0 | Standard CFG |

---

## Required Nodes by Model

### LTX-2 Video
```
LTXVLoader -> CLIPTextEncode (x2) -> LTXVConditioning -> EmptyLTXVLatentVideo
           -> LTXVScheduler -> KSamplerSelect -> SamplerCustom -> VAEDecode -> VHS_VideoCombine
```

### FLUX.2 Image
```
UNETLoader + DualCLIPLoader + VAELoader
-> CLIPTextEncode -> FluxGuidance -> BasicGuider
-> EmptySD3LatentImage + BasicScheduler + RandomNoise + KSamplerSelect
-> SamplerCustomAdvanced -> VAEDecode -> SaveImage
```

### Wan 2.1 Video
```
DownloadAndLoadWanModel -> WanSampler -> WanVAEDecode -> VHS_VideoCombine
```

### HunyuanVideo
```
HunyuanVideoModelLoader + CLIPLoader -> CLIPTextEncode (x2)
-> EmptyHunyuanLatentVideo -> HunyuanVideoSampler -> HunyuanVideoVAEDecode -> VHS_VideoCombine
```

### Qwen/SDXL Image
```
CheckpointLoaderSimple -> CLIPTextEncode (x2) -> EmptyLatentImage
-> KSampler -> VAEDecode -> SaveImage
```

---

## Forbidden Nodes

| Model | Forbidden | Use Instead |
|-------|-----------|-------------|
| LTX-2 | KSampler | SamplerCustom |
| LTX-2 | EmptyLatentImage | EmptyLTXVLatentVideo |
| FLUX.2 | KSampler | SamplerCustomAdvanced |
| FLUX.2 | CheckpointLoaderSimple | UNETLoader + DualCLIPLoader |
| FLUX.2 | EmptyLatentImage | EmptySD3LatentImage |
| Wan 2.1 | KSampler | WanSampler |
| Wan 2.1 | VAEEncode | WanImageEncode |
| HunyuanVideo | KSampler | HunyuanVideoSampler |
| HunyuanVideo | SamplerCustom | HunyuanVideoSampler |

---

## Connection Slot Reference

### Common Output Slots
| Node | Slot 0 | Slot 1 | Slot 2 |
|------|--------|--------|--------|
| CheckpointLoaderSimple | MODEL | CLIP | VAE |
| LTXVLoader | MODEL | CLIP | VAE |
| HunyuanVideoModelLoader | MODEL | VAE | - |
| CLIPTextEncode | CONDITIONING | - | - |
| LTXVConditioning | CONDITIONING+ | CONDITIONING- | - |
| SamplerCustom | LATENT | LATENT_DENOISED | - |
| LoadImage | IMAGE | MASK | - |

---

## Common Mistakes to Avoid

1. **Wrong sampler for video models** - Use SamplerCustom/WanSampler/HunyuanVideoSampler, not KSampler
2. **CFG in FLUX** - Use FluxGuidance node, not cfg parameter in sampler
3. **Wrong resolution divisor** - FLUX/Hunyuan need divisible by 16
4. **Wrong frame count for LTX** - Must be 8n+1 (9, 17, 25, ...)
5. **Missing conditioning wrapper** - LTX needs LTXVConditioning, FLUX needs FluxGuidance
6. **Wrong VAE decode** - Use model-specific decoder (WanVAEDecode, HunyuanVideoVAEDecode)
7. **Wrong latent node** - Use model-specific latent (EmptyLTXVLatentVideo, EmptySD3LatentImage)

---

## Quick API Format

```json
{
  "1": {
    "class_type": "NodeClassName",
    "inputs": {
      "text_input": "string value",
      "number_input": 42,
      "connection_input": ["source_node_id", slot_index]
    }
  }
}
```

Connections are `["node_id", slot_index]` where slot_index is 0-based.
