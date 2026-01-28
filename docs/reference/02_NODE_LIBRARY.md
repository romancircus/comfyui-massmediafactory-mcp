# ComfyUI Node Library

> **Purpose**: Compact API reference for LLM workflow generation.
> **Usage**: Look up exact input/output types before creating connections.

---

## Core Loading Nodes

### CheckpointLoaderSimple

```json
{
  "class_type": "CheckpointLoaderSimple",
  "inputs": {
    "ckpt_name": "STRING (filename)"
  },
  "outputs": ["MODEL", "CLIP", "VAE"]
}
```

### UNETLoader

```json
{
  "class_type": "UNETLoader",
  "inputs": {
    "unet_name": "STRING (filename)",
    "weight_dtype": "STRING (default|fp8_e4m3fn|fp8_e5m2|fp16)"
  },
  "outputs": ["MODEL"]
}
```

### CLIPLoader

```json
{
  "class_type": "CLIPLoader",
  "inputs": {
    "clip_name": "STRING (filename)",
    "type": "STRING (sd1|sd2|sdxl|flux|qwen_image)",
    "device": "STRING (default|cpu)"
  },
  "outputs": ["CLIP"]
}
```

### DualCLIPLoader

```json
{
  "class_type": "DualCLIPLoader",
  "inputs": {
    "clip_name1": "STRING (CLIP L)",
    "clip_name2": "STRING (T5 XXL)",
    "type": "STRING (flux|sd3)"
  },
  "outputs": ["CLIP"]
}
```

### VAELoader

```json
{
  "class_type": "VAELoader",
  "inputs": {
    "vae_name": "STRING (filename)"
  },
  "outputs": ["VAE"]
}
```

### LoraLoader

```json
{
  "class_type": "LoraLoader",
  "inputs": {
    "model": "MODEL",
    "clip": "CLIP",
    "lora_name": "STRING (filename)",
    "strength_model": "FLOAT (0.0-2.0, default: 1.0)",
    "strength_clip": "FLOAT (0.0-2.0, default: 1.0)"
  },
  "outputs": ["MODEL", "CLIP"]
}
```

### LoraLoaderModelOnly

```json
{
  "class_type": "LoraLoaderModelOnly",
  "inputs": {
    "model": "MODEL",
    "lora_name": "STRING (filename)",
    "strength_model": "FLOAT (0.0-2.0, default: 1.0)"
  },
  "outputs": ["MODEL"]
}
```

---

## Text Encoding Nodes

### CLIPTextEncode

```json
{
  "class_type": "CLIPTextEncode",
  "inputs": {
    "clip": "CLIP",
    "text": "STRING (prompt)"
  },
  "outputs": ["CONDITIONING"]
}
```

### FluxGuidance

```json
{
  "class_type": "FluxGuidance",
  "inputs": {
    "conditioning": "CONDITIONING",
    "guidance": "FLOAT (0.0-100.0, default: 3.5)"
  },
  "outputs": ["CONDITIONING"]
}
```

---

## LTX-Video Specific Nodes

### LTXVConditioning

```json
{
  "class_type": "LTXVConditioning",
  "inputs": {
    "positive": "CONDITIONING",
    "negative": "CONDITIONING",
    "frame_rate": "FLOAT (default: 24)"
  },
  "outputs": ["CONDITIONING (positive)", "CONDITIONING (negative)"]
}
```

### LTXVScheduler

```json
{
  "class_type": "LTXVScheduler",
  "inputs": {
    "steps": "INT (1-100, default: 30)",
    "max_shift": "FLOAT (default: 2.05)",
    "base_shift": "FLOAT (default: 0.95)",
    "stretch": "BOOLEAN (default: true)",
    "terminal": "FLOAT (default: 0.1)"
  },
  "outputs": ["SIGMAS"]
}
```

### EmptyLTXVLatentVideo

```json
{
  "class_type": "EmptyLTXVLatentVideo",
  "inputs": {
    "width": "INT (divisible by 8, default: 768)",
    "height": "INT (divisible by 8, default: 512)",
    "length": "INT (must be 8n+1: 9,17,25,...97,...121)",
    "batch_size": "INT (default: 1)"
  },
  "outputs": ["LATENT"]
}
```

### LTXVImgToVideo

```json
{
  "class_type": "LTXVImgToVideo",
  "inputs": {
    "image": "IMAGE",
    "vae": "VAE",
    "width": "INT",
    "height": "INT",
    "length": "INT (8n+1)"
  },
  "outputs": ["LATENT"]
}
```

### LTXVPreprocess

```json
{
  "class_type": "LTXVPreprocess",
  "inputs": {
    "image": "IMAGE",
    "strength": "INT (0-100, default: 40)"
  },
  "outputs": ["IMAGE"]
}
```

### LTXVGemmaCLIPModelLoader

```json
{
  "class_type": "LTXVGemmaCLIPModelLoader",
  "inputs": {
    "model_name": "STRING (gemma_3_12B_it_fp8)"
  },
  "outputs": ["GEMMA_MODEL"]
}
```

### LTXVGemmaEnhancePrompt

```json
{
  "class_type": "LTXVGemmaEnhancePrompt",
  "inputs": {
    "gemma_model": "GEMMA_MODEL",
    "prompt": "STRING"
  },
  "outputs": ["STRING (enhanced prompt)"]
}
```

---

## Wan 2.1 Specific Nodes

### WanVideoModelLoader

```json
{
  "class_type": "WanVideoModelLoader",
  "inputs": {
    "model_name": "STRING (wan_2.1_*.safetensors)"
  },
  "outputs": ["MODEL", "VAE"]
}
```

### EmptyWanLatentVideo

```json
{
  "class_type": "EmptyWanLatentVideo",
  "inputs": {
    "width": "INT (480p: 832, 720p: 1280)",
    "height": "INT (480p: 480, 720p: 720)",
    "frames": "INT (default: 81)",
    "batch_size": "INT (default: 1)"
  },
  "outputs": ["LATENT"]
}
```

### WanImageEncode

```json
{
  "class_type": "WanImageEncode",
  "inputs": {
    "image": "IMAGE",
    "vae": "VAE"
  },
  "outputs": ["LATENT"]
}
```

### WanVAEDecode

```json
{
  "class_type": "WanVAEDecode",
  "inputs": {
    "samples": "LATENT",
    "vae": "VAE"
  },
  "outputs": ["IMAGE"]
}
```

---

## FLUX Specific Nodes

### ModelSamplingFlux

```json
{
  "class_type": "ModelSamplingFlux",
  "inputs": {
    "model": "MODEL",
    "width": "INT",
    "height": "INT"
  },
  "outputs": ["MODEL"]
}
```

---

## Qwen Specific Nodes

### ModelSamplingAuraFlow

```json
{
  "class_type": "ModelSamplingAuraFlow",
  "inputs": {
    "model": "MODEL",
    "shift": "FLOAT (default: 3.1)"
  },
  "outputs": ["MODEL"]
}
```

---

## Latent Generation Nodes

### EmptyLatentImage

```json
{
  "class_type": "EmptyLatentImage",
  "inputs": {
    "width": "INT (divisible by 8)",
    "height": "INT (divisible by 8)",
    "batch_size": "INT (default: 1)"
  },
  "outputs": ["LATENT"]
}
```

### EmptySD3LatentImage

```json
{
  "class_type": "EmptySD3LatentImage",
  "inputs": {
    "width": "INT (divisible by 16 for FLUX)",
    "height": "INT (divisible by 16 for FLUX)",
    "batch_size": "INT (default: 1)"
  },
  "outputs": ["LATENT"]
}
```

---

## Sampler Nodes

### KSampler

```json
{
  "class_type": "KSampler",
  "inputs": {
    "model": "MODEL",
    "positive": "CONDITIONING",
    "negative": "CONDITIONING",
    "latent_image": "LATENT",
    "seed": "INT",
    "steps": "INT (1-100)",
    "cfg": "FLOAT (1.0-30.0)",
    "sampler_name": "STRING (euler|euler_ancestral|dpmpp_2m|...)",
    "scheduler": "STRING (normal|karras|exponential|simple|...)",
    "denoise": "FLOAT (0.0-1.0, default: 1.0)"
  },
  "outputs": ["LATENT"]
}
```

### KSamplerSelect

```json
{
  "class_type": "KSamplerSelect",
  "inputs": {
    "sampler_name": "STRING (euler|dpmpp_2m|res_multistep|...)"
  },
  "outputs": ["SAMPLER"]
}
```

### SamplerCustom

```json
{
  "class_type": "SamplerCustom",
  "inputs": {
    "model": "MODEL",
    "positive": "CONDITIONING",
    "negative": "CONDITIONING",
    "sampler": "SAMPLER",
    "sigmas": "SIGMAS",
    "latent_image": "LATENT",
    "add_noise": "BOOLEAN (default: true)",
    "noise_seed": "INT"
  },
  "outputs": ["LATENT", "LATENT (denoised)"]
}
```

### BasicScheduler

```json
{
  "class_type": "BasicScheduler",
  "inputs": {
    "model": "MODEL",
    "scheduler": "STRING (normal|karras|exponential|simple|sgm_uniform)",
    "steps": "INT (1-100)",
    "denoise": "FLOAT (0.0-1.0, default: 1.0)"
  },
  "outputs": ["SIGMAS"]
}
```

---

## Decode/Encode Nodes

### VAEDecode

```json
{
  "class_type": "VAEDecode",
  "inputs": {
    "samples": "LATENT",
    "vae": "VAE"
  },
  "outputs": ["IMAGE"]
}
```

### VAEEncode

```json
{
  "class_type": "VAEEncode",
  "inputs": {
    "pixels": "IMAGE",
    "vae": "VAE"
  },
  "outputs": ["LATENT"]
}
```

---

## Output Nodes

### SaveImage

```json
{
  "class_type": "SaveImage",
  "inputs": {
    "images": "IMAGE",
    "filename_prefix": "STRING (default: ComfyUI)"
  },
  "outputs": []
}
```

### SaveAnimatedWEBP

```json
{
  "class_type": "SaveAnimatedWEBP",
  "inputs": {
    "images": "IMAGE",
    "filename_prefix": "STRING",
    "fps": "INT (default: 24)",
    "lossless": "BOOLEAN (default: false)",
    "quality": "INT (0-100, default: 80)"
  },
  "outputs": []
}
```

### VHS_VideoCombine

```json
{
  "class_type": "VHS_VideoCombine",
  "inputs": {
    "images": "IMAGE",
    "frame_rate": "FLOAT (default: 24)",
    "format": "STRING (webm_video|mp4_video|...)"
  },
  "outputs": ["VHS_FILENAMES"]
}
```

---

## Image Input Nodes

### LoadImage

```json
{
  "class_type": "LoadImage",
  "inputs": {
    "image": "STRING (filename)"
  },
  "outputs": ["IMAGE", "MASK"]
}
```

### ImageScale

```json
{
  "class_type": "ImageScale",
  "inputs": {
    "image": "IMAGE",
    "upscale_method": "STRING (nearest|bilinear|bicubic|lanczos)",
    "width": "INT",
    "height": "INT",
    "crop": "STRING (disabled|center)"
  },
  "outputs": ["IMAGE"]
}
```

---

## Changelog

- **January 2026**: Initial node library for LLM reference
