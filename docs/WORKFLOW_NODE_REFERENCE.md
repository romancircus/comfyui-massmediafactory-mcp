# Workflow Node Reference

**Purpose:** Enable LLM agents to construct ComfyUI workflows programmatically
**Format:** Each workflow documented with exact node structure for JSON recreation

---

## JSON Workflow Format

ComfyUI workflows use this structure:

```json
{
  "_meta": {
    "description": "...",
    "model": "...",
    "type": "txt2img|img2vid|txt2vid|tts",
    "parameters": ["PARAM1", "PARAM2"],
    "defaults": {"PARAM1": "value"}
  },
  "1": {
    "class_type": "NodeClassName",
    "_meta": {"title": "Human readable title"},
    "inputs": {
      "input_name": "literal_value",
      "connected_input": ["source_node_id", output_slot_index]
    }
  }
}
```

**Connection format:** `["node_id", slot_index]` where slot_index is 0-based.

---

## LTX-Video 2.0 Workflows

### ltx2_txt2vid (Text-to-Video, Full Quality)

**Node Chain:**
```
LTXVLoader → CLIPTextEncode(+) → LTXVConditioning
                                        ↓
           CLIPTextEncode(-) → LTXVConditioning
                                        ↓
           EmptyLTXVLatentVideo ────────┘
                                        ↓
           LTXVScheduler ──────────→ SamplerCustom
                                        ↓
           KSamplerSelect ─────────→ SamplerCustom
                                        ↓
                                   VAEDecode
                                        ↓
                                   VHS_VideoCombine
```

**Complete Node Definitions:**

```json
{
  "1": {
    "class_type": "LTXVLoader",
    "inputs": {
      "ckpt_name": "ltx-2-19b-dev-fp8.safetensors",
      "dtype": "bfloat16"
    },
    "outputs": ["MODEL(0)", "CLIP(1)", "VAE(2)"]
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "clip": ["1", 1],
      "text": "{{PROMPT}}"
    },
    "outputs": ["CONDITIONING(0)"]
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "clip": ["1", 1],
      "text": "{{NEGATIVE}}"
    },
    "outputs": ["CONDITIONING(0)"]
  },
  "4": {
    "class_type": "LTXVConditioning",
    "inputs": {
      "positive": ["2", 0],
      "negative": ["3", 0],
      "frame_rate": 24.0
    },
    "outputs": ["CONDITIONING(0)", "CONDITIONING(1)"]
  },
  "5": {
    "class_type": "EmptyLTXVLatentVideo",
    "inputs": {
      "width": 768,
      "height": 512,
      "length": 97,
      "batch_size": 1
    },
    "outputs": ["LATENT(0)"]
  },
  "6": {
    "class_type": "LTXVScheduler",
    "inputs": {
      "steps": 30,
      "max_shift": 2.05,
      "base_shift": 0.95,
      "stretch": true
    },
    "outputs": ["SIGMAS(0)"]
  },
  "7": {
    "class_type": "KSamplerSelect",
    "inputs": {
      "sampler_name": "euler"
    },
    "outputs": ["SAMPLER(0)"]
  },
  "8": {
    "class_type": "SamplerCustom",
    "inputs": {
      "model": ["1", 0],
      "add_noise": true,
      "noise_seed": 42,
      "cfg": 3.0,
      "positive": ["4", 0],
      "negative": ["4", 1],
      "sampler": ["7", 0],
      "sigmas": ["6", 0],
      "latent_image": ["5", 0]
    },
    "outputs": ["LATENT(0)", "LATENT(1)"]
  },
  "9": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["8", 0],
      "vae": ["1", 2]
    },
    "outputs": ["IMAGE(0)"]
  },
  "10": {
    "class_type": "VHS_VideoCombine",
    "inputs": {
      "images": ["9", 0],
      "frame_rate": 24,
      "loop_count": 0,
      "filename_prefix": "ltx2_output",
      "format": "video/h264-mp4",
      "pingpong": false,
      "save_output": true
    },
    "outputs": ["VHS_FILENAMES(0)"]
  }
}
```

**Parameter Rules:**
- `length` (frames): Must be `8n + 1` (9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121)
- `width`/`height`: Divisible by 8, native 768×512
- `cfg`: 2.5-4.0, typically 3.0
- `steps`: 25-35 for full model

---

### ltx2_txt2vid_distilled (Fast Generation)

**Additional/Modified Nodes:**

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "ltx-2-19b-distilled-fp8.safetensors"
    },
    "outputs": ["MODEL(0)", "CLIP(1)", "VAE(2)"]
  },
  "2": {
    "class_type": "LTXVGemmaCLIPModelLoader",
    "inputs": {
      "gemma_path": "gemma_3_12B_it_fp8_e4m3fn.safetensors",
      "ltxv_path": "ltx-2-19b-distilled-fp8.safetensors",
      "max_length": 1024
    },
    "outputs": ["CLIP(0)"]
  },
  "3": {
    "class_type": "LTXVGemmaEnhancePrompt",
    "inputs": {
      "clip": ["2", 0],
      "prompt": "{{PROMPT}}",
      "max_tokens": 512,
      "bypass_i2v": false,
      "seed": 42
    },
    "outputs": ["STRING(0)"]
  }
}
```

**Distilled Parameters:**
- `steps`: 8-12 (not 30)
- `cfg`: 2.5 (not 3.0)
- Can use higher resolution: 1920×1088

---

### ltx2_img2vid (Image-to-Video)

**Additional Nodes for I2V:**

```json
{
  "load_image": {
    "class_type": "LoadImage",
    "inputs": {
      "image": "input_filename.png"
    },
    "outputs": ["IMAGE(0)", "MASK(1)"]
  },
  "preprocess": {
    "class_type": "LTXVPreprocess",
    "inputs": {
      "image": ["load_image", 0],
      "strength": 40
    },
    "outputs": ["IMAGE(0)"]
  },
  "img_to_video": {
    "class_type": "LTXVImgToVideo",
    "inputs": {
      "positive": ["conditioning", 0],
      "negative": ["conditioning", 1],
      "vae": ["model_loader", 2],
      "image": ["preprocess", 0],
      "width": 768,
      "height": 512,
      "length": 97,
      "batch_size": 1
    },
    "outputs": ["CONDITIONING(0)", "CONDITIONING(1)", "LATENT(2)"]
  }
}
```

**I2V replaces:** `EmptyLTXVLatentVideo` with `LTXVImgToVideo`

**Strength parameter:**
- 20-40: More motion, less input preservation
- 50-80: Balanced
- 80-100: Less motion, more input preservation

---

## FLUX Workflows

### flux2_txt2img (Standard Quality)

**Node Chain:**
```
UNETLoader ─────────────────────────→ BasicGuider → SamplerCustomAdvanced
                                              ↑
DualCLIPLoader → CLIPTextEncode → FluxGuidance ┘

VAELoader ──────────────────────────→ VAEDecode

EmptySD3LatentImage ────────────────→ SamplerCustomAdvanced

BasicScheduler ─────────────────────→ SamplerCustomAdvanced

KSamplerSelect ─────────────────────→ SamplerCustomAdvanced

RandomNoise ────────────────────────→ SamplerCustomAdvanced
```

**Complete Node Definitions:**

```json
{
  "1": {
    "class_type": "UNETLoader",
    "inputs": {
      "unet_name": "flux2-dev-fp8.safetensors",
      "weight_dtype": "fp8_e4m3fn"
    },
    "outputs": ["MODEL(0)"]
  },
  "2": {
    "class_type": "DualCLIPLoader",
    "inputs": {
      "clip_name1": "clip_l.safetensors",
      "clip_name2": "t5xxl_fp16.safetensors",
      "type": "flux"
    },
    "outputs": ["CLIP(0)"]
  },
  "3": {
    "class_type": "VAELoader",
    "inputs": {
      "vae_name": "ae.safetensors"
    },
    "outputs": ["VAE(0)"]
  },
  "4": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "clip": ["2", 0],
      "text": "{{PROMPT}}"
    },
    "outputs": ["CONDITIONING(0)"]
  },
  "5": {
    "class_type": "FluxGuidance",
    "inputs": {
      "conditioning": ["4", 0],
      "guidance": 3.5
    },
    "outputs": ["CONDITIONING(0)"]
  },
  "6": {
    "class_type": "EmptySD3LatentImage",
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    },
    "outputs": ["LATENT(0)"]
  },
  "7": {
    "class_type": "KSamplerSelect",
    "inputs": {
      "sampler_name": "euler"
    },
    "outputs": ["SAMPLER(0)"]
  },
  "8": {
    "class_type": "BasicScheduler",
    "inputs": {
      "model": ["1", 0],
      "scheduler": "simple",
      "steps": 20,
      "denoise": 1.0
    },
    "outputs": ["SIGMAS(0)"]
  },
  "9": {
    "class_type": "RandomNoise",
    "inputs": {
      "noise_seed": 42
    },
    "outputs": ["NOISE(0)"]
  },
  "10": {
    "class_type": "BasicGuider",
    "inputs": {
      "model": ["1", 0],
      "conditioning": ["5", 0]
    },
    "outputs": ["GUIDER(0)"]
  },
  "11": {
    "class_type": "SamplerCustomAdvanced",
    "inputs": {
      "noise": ["9", 0],
      "guider": ["10", 0],
      "sampler": ["7", 0],
      "sigmas": ["8", 0],
      "latent_image": ["6", 0]
    },
    "outputs": ["LATENT(0)", "LATENT(1)"]
  },
  "12": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["11", 0],
      "vae": ["3", 0]
    },
    "outputs": ["IMAGE(0)"]
  },
  "13": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["12", 0],
      "filename_prefix": "flux2_output"
    },
    "outputs": []
  }
}
```

**FLUX Parameter Rules:**
- `width`/`height`: Divisible by 16, native 1024×1024
- `guidance`: 3.0-4.0 typical (via FluxGuidance, not cfg)
- `steps`: 20-50 for dev, 4 for schnell

---

## Qwen Image Workflows

### qwen_txt2img (Text-to-Image)

**Node Chain:**
```
CheckpointLoaderSimple → CLIPTextEncode(+) → KSampler
                                                 ↑
                       → CLIPTextEncode(-) → KSampler
                                                 ↓
                       → EmptyLatentImage ───────┘
                                                 ↓
                                            VAEDecode
                                                 ↓
                                            SaveImage
```

**Complete Node Definitions:**

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "qwen_image_2512.safetensors"
    },
    "outputs": ["MODEL(0)", "CLIP(1)", "VAE(2)"]
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "clip": ["1", 1],
      "text": "{{PROMPT}}"
    },
    "outputs": ["CONDITIONING(0)"]
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "clip": ["1", 1],
      "text": "{{NEGATIVE}}"
    },
    "outputs": ["CONDITIONING(0)"]
  },
  "4": {
    "class_type": "EmptyLatentImage",
    "inputs": {
      "width": 1296,
      "height": 1296,
      "batch_size": 1
    },
    "outputs": ["LATENT(0)"]
  },
  "5": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["1", 0],
      "positive": ["2", 0],
      "negative": ["3", 0],
      "latent_image": ["4", 0],
      "seed": 42,
      "steps": 25,
      "cfg": 7.0,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1.0
    },
    "outputs": ["LATENT(0)"]
  },
  "6": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["5", 0],
      "vae": ["1", 2]
    },
    "outputs": ["IMAGE(0)"]
  },
  "7": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["6", 0],
      "filename_prefix": "qwen_output"
    },
    "outputs": []
  }
}
```

**Qwen Parameters:**
- `width`/`height`: Native 1296×1296, divisible by 8
- `cfg`: 5.0-8.0
- `steps`: 20-30
- Best for: Text rendering, posters, logos

---

## Wan 2.6 Workflows

### wan26_img2vid (Image-to-Video)

**Key Nodes:**

```json
{
  "model_loader": {
    "class_type": "DownloadAndLoadWanModel",
    "inputs": {
      "model": "Wan-AI/Wan2.6-I2V-14B-480P",
      "base_precision": "bf16",
      "quantization": "disabled"
    },
    "outputs": ["WANMODEL(0)"]
  },
  "image_encoder": {
    "class_type": "WanImageEncode",
    "inputs": {
      "wan_model": ["model_loader", 0],
      "image": ["load_image", 0],
      "image_strength": 1.0,
      "enable_tiling": false
    },
    "outputs": ["WANMODEL(0)", "WANIMAGEEMBEDS(1)"]
  },
  "sampler": {
    "class_type": "WanSampler",
    "inputs": {
      "wan_model": ["image_encoder", 0],
      "positive": "{{PROMPT}}",
      "negative": "{{NEGATIVE}}",
      "image_embeds": ["image_encoder", 1],
      "width": 832,
      "height": 480,
      "num_frames": 81,
      "steps": 30,
      "cfg": 5.0,
      "seed": 42,
      "shift": 8.0,
      "scheduler": "unipc"
    },
    "outputs": ["LATENT(0)"]
  }
}
```

**Wan Parameters:**
- 480p: 832×480, 81 frames
- 720p: 1280×720, 81 frames
- `cfg`: 5.0
- `shift`: 8.0
- `scheduler`: "unipc"

---

## Common Node Patterns

### Pattern 1: Model Loading

**Simple (SD/SDXL/Qwen):**
```json
{"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}}
```

**Separate (FLUX):**
```json
{"class_type": "UNETLoader", "inputs": {"unet_name": "flux.safetensors"}}
{"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl.safetensors"}}
{"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}}
```

### Pattern 2: Sampling

**Image (KSampler):**
```json
{"class_type": "KSampler", "inputs": {"model": [...], "positive": [...], "negative": [...], "latent_image": [...], "seed": 42, "steps": 25, "cfg": 7.0}}
```

**Video/Advanced (SamplerCustom):**
```json
{"class_type": "SamplerCustom", "inputs": {"model": [...], "positive": [...], "negative": [...], "sampler": [...], "sigmas": [...], "latent_image": [...], "cfg": 3.0}}
```

### Pattern 3: Output Saving

**Image:**
```json
{"class_type": "SaveImage", "inputs": {"images": [...], "filename_prefix": "output"}}
```

**Video:**
```json
{"class_type": "VHS_VideoCombine", "inputs": {"images": [...], "frame_rate": 24, "format": "video/h264-mp4"}}
```

---

## Node Type Quick Reference

| Node | Inputs | Outputs | Use |
|------|--------|---------|-----|
| CheckpointLoaderSimple | ckpt_name | MODEL, CLIP, VAE | Load all-in-one |
| UNETLoader | unet_name | MODEL | Load model only |
| DualCLIPLoader | clip_name1, clip_name2 | CLIP | FLUX text encoders |
| CLIPTextEncode | clip, text | CONDITIONING | Encode prompts |
| FluxGuidance | conditioning, guidance | CONDITIONING | FLUX CFG |
| LTXVConditioning | positive, negative, frame_rate | CONDITIONING×2 | LTX frame rate |
| EmptyLatentImage | width, height | LATENT | Image generation |
| EmptyLTXVLatentVideo | width, height, length | LATENT | Video generation |
| KSampler | model, positive, negative, latent, seed, steps, cfg | LATENT | Simple sampling |
| SamplerCustom | model, positive, negative, sampler, sigmas, latent | LATENT | Advanced sampling |
| VAEDecode | samples, vae | IMAGE | Latent to image |
| SaveImage | images, filename_prefix | - | Save to disk |
| VHS_VideoCombine | images, frame_rate, format | FILENAMES | Save video |
