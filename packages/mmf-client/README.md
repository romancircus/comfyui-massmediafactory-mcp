# @romancircus/mmf-client

Thin wrapper around the `mmf` CLI for ComfyUI image/video generation. Replaces thousands of lines of ComfyUI integration code with ~200 lines calling the mmf CLI.

## Installation

```bash
npm install @romancircus/mmf-client
```

**Prerequisite:** Install the `mmf` CLI globally:

```bash
pip install mmf
# or if developing locally:
pip install -e /path/to/comfyui-massmediafactory-mcp
```

## Quick Start

```javascript
import { qwenTxt2Img, wanI2V, viralShort, hasError } from '@romancircus/mmf-client';

// Generate an image
const result = await qwenTxt2Img({
  prompt: 'A cyberpunk city at sunset',
  seed: 42,
  output: '/path/to/output.png'
});

if (hasError(result)) {
  console.error('Generation failed:', result.error);
} else {
  console.log('Generated:', result.asset_id);
}

// Animate an image into video
const videoResult = await wanI2V({
  image: '/path/to/keyframe.png',
  prompt: 'Gentle motion with cinematic lighting',
  frames: 81,
  seed: 42,
  output: '/path/to/video.mp4'
});

// Run a full pipeline
const pipelineResult = await viralShort({
  prompt: 'A dancing anime character',
  styleImage: '/path/to/style.jpg',
  seed: 42,
  output: '/path/to/viral.mp4'
});
```

## Function Reference (22 Functions)

### Image Generation

#### `qwenTxt2Img(options)`
Qwen-Image-2512 text-to-image generation.

**Options:**
- `prompt` (string, required) - Main prompt describing the image
- `negative` (string, optional) - Negative prompt (default: 'blurry, low quality, distorted, watermark')
- `seed` (number, optional) - Random seed for reproducibility
- `width` (number, optional) - Image width (default: 1664)
- `height` (number, optional) - Image height (default: 928)
- `shift` (number, optional) - Shift parameter (default: 7.0, **CRITICAL**: keep at 7.0 for sharpness)
- `cfg` (number, optional) - CFG scale (default: 3.5)
- `steps` (number, optional) - Inference steps (default: 50)
- `output` (string, optional) - Local path to save the output

**Returns:** `Promise<{asset_id: string, ...} | {error: string}>`

---

#### `fluxTxt2Img(options)`
FLUX.2-dev text-to-image generation.

**Options:**
- `prompt` (string, required) - Main prompt describing the image
- `seed` (number, optional) - Random seed
- `width` (number, optional) - Image width (default: 1024)
- `height` (number, optional) - Image height (default: 1024)
- `output` (string, optional) - Local path to save the output

---

#### `faceIdTxt2Img(options)`
FLUX.2 Face ID text-to-image with face reference using IP-Adapter.

**Options:**
- `prompt` (string, required) - Main prompt
- `faceImage` (string, required) - Path to reference face image
- `seed` (number, optional) - Random seed
- `faceStrength` (number, optional) - Face identity strength 0.0-1.0 (default: 0.85)
- `width` (number, optional) - Image width (default: 1024)
- `height` (number, optional) - Image height (default: 1024)
- `output` (string, optional) - Local path to save the output

---

#### `teleStyleImage(options)`
TeleStyle image style transfer.

**Options:**
- `content` (string, required) - Path to content image
- `style` (string, required) - Path to style reference image
- `seed` (number, optional) - Random seed (default: 42)
- `cfg` (number, optional) - CFG scale (default: 2.0, **CRITICAL**: keep 2.0-2.5 to avoid distortion)
- `steps` (number, optional) - Inference steps (default: 20)
- `output` (string, optional) - Local path to save the output

---

#### `kontextEdit(options)`
FLUX Kontext Dev - character-consistent image editing.

**Options:**
- `image` (string, required) - Path to source image
- `editPrompt` (string, required) - Description of desired edit
- `denoise` (number, optional) - Denoising strength (default: 0.65)
- `seed` (number, optional) - Random seed (default: 42)
- `width` (number, optional) - Output width (default: 1024)
- `height` (number, optional) - Output height (default: 1024)
- `steps` (number, optional) - Inference steps (default: 20)
- `guidance` (number, optional) - Guidance scale (default: 3.5)
- `output` (string, optional) - Local path to save the output

---

### Video Generation

#### `wanI2V(options)`
Wan 2.2 Image-to-Video animation.

**Options:**
- `image` (string, required) - Path to keyframe image
- `prompt` (string, required) - Motion description
- `negative` (string, optional) - Negative prompt
- `frames` (number, optional) - Number of frames (default: 81)
- `fps` (number, optional) - Frames per second (default: 16)
- `seed` (number, optional) - Random seed
- `cfg` (number, optional) - CFG scale (default: 5.0)
- `shift` (number, optional) - Shift parameter (default: 5.0)
- `steps` (number, optional) - Inference steps (default: 30)
- `noiseAug` (number, optional) - Noise augmentation (default: 0.03)
- `width` (number, optional) - Video width (default: 832)
- `height` (number, optional) - Video height (default: 480)
- `output` (string, optional) - Local path to save the output

---

#### `wanS2V(options)`
Wan 2.2 Sound-to-Video with audio synchronization.

**Options:**
- `image` (string, required) - Path to keyframe image
- `prompt` (string, required) - Motion description
- `frames` (number, optional) - Number of frames (default: 77)
- `fps` (number, optional) - Frames per second (default: 16)
- `seed` (number, optional) - Random seed
- `output` (string, optional) - Local path to save the output

---

#### `wanAnimate(options)`
Wan 2.2 Animate - character animation with identity preservation.

**Options:**
- `image` (string, required) - Path to character reference image
- `prompt` (string, required) - Motion description
- `negative` (string, optional) - Negative prompt
- `frames` (number, optional) - Number of frames (default: 81)
- `fps` (number, optional) - Frames per second (default: 16)
- `seed` (number, optional) - Random seed
- `cfg` (number, optional) - CFG scale (default: 5.0)
- `shift` (number, optional) - Shift parameter (default: 5.0)
- `steps` (number, optional) - Inference steps (default: 30)
- `faceStrength` (number, optional) - Face identity strength (default: 1.0)
- `poseStrength` (number, optional) - Pose strength (default: 1.0)
- `width` (number, optional) - Video width (default: 832)
- `height` (number, optional) - Video height (default: 480)
- `output` (string, optional) - Local path to save the output

---

#### `phantomS2V(options)`
Phantom multi-subject S2V - 1-4 character references simultaneously.

**Options:**
- `images` (string[], required) - Array of 1-4 character reference image paths
- `prompt` (string, required) - Motion description
- `negative` (string, optional) - Negative prompt
- `frames` (number, optional) - Number of frames (default: 81)
- `fps` (number, optional) - Frames per second (default: 16)
- `seed` (number, optional) - Random seed
- `cfg` (number, optional) - CFG scale (default: 5.0)
- `width` (number, optional) - Video width (default: 832)
- `height` (number, optional) - Video height (default: 480)
- `output` (string, optional) - Local path to save the output

---

#### `ltxT2V(options)`
LTX-2 text-to-video generation.

**Options:**
- `prompt` (string, required) - Motion description
- `negative` (string, optional) - Negative prompt
- `frames` (number, optional) - Number of frames (default: 97, **CRITICAL**: must follow 8n+1 rule)
- `fps` (number, optional) - Frames per second (default: 24)
- `seed` (number, optional) - Random seed
- `width` (number, optional) - Video width (default: 832)
- `height` (number, optional) - Video height (default: 480)
- `steps` (number, optional) - Inference steps (default: 30)
- `cfg` (number, optional) - CFG scale (default: 3.0)
- `prefix` (string, optional) - Output filename prefix (default: 'ltx2_output')
- `output` (string, optional) - Local path to save the output

---

#### `ltxI2V(options)`
LTX-2 image-to-video generation.

**Options:**
- `image` (string, required) - Path to keyframe image
- `prompt` (string, required) - Motion description
- `negative` (string, optional) - Negative prompt
- `frames` (number, optional) - Number of frames (default: 97)
- `fps` (number, optional) - Frames per second (default: 24)
- `seed` (number, optional) - Random seed
- `width` (number, optional) - Video width (default: 768)
- `height` (number, optional) - Video height (default: 512)
- `strength` (number, optional) - Animation strength (default: 0.85)
- `steps` (number, optional) - Inference steps (default: 30)
- `cfg` (number, optional) - CFG scale (default: 3.0)
- `crf` (number, optional) - Video compression quality (default: 38)
- `blurRadius` (number, optional) - Blur radius for preprocessing (default: 1)
- `prefix` (string, optional) - Output filename prefix (default: 'ltx2_i2v')
- `output` (string, optional) - Local path to save the output

---

#### `audioReactiveI2V(options)`
LTX-2 audio-reactive image-to-video.

**Options:**
- `image` (string, required) - Path to keyframe image
- `audioPath` (string, required) - Path to audio file
- `prompt` (string, required) - Motion description
- `frames` (number, optional) - Number of frames (default: 121)
- `fps` (number, optional) - Frames per second (default: 24)
- `seed` (number, optional) - Random seed
- `output` (string, optional) - Local path to save the output

---

#### `teleStyleVideo(options)`
TeleStyle video style transfer with temporal consistency.

**Options:**
- `video` (string, required) - Path to input video
- `style` (string, required) - Path to style reference image
- `seed` (number, optional) - Random seed (default: 42)
- `cfg` (number, optional) - CFG scale (default: 1.0)
- `steps` (number, optional) - Inference steps (default: 12)
- `fps` (number, optional) - Target FPS (default: 24)
- `output` (string, optional) - Local path to save the output

---

#### `videoInpaint(options)`
Video inpainting via CLIPSeg text-based mask + FLUX inpaint.

**Options:**
- `video` (string, required) - Path to input video
- `selectText` (string, required) - Text description of what to select/mask
- `replacePrompt` (string, required) - Replacement content description
- `denoise` (number, optional) - Denoising strength (default: 0.7)
- `seed` (number, optional) - Random seed
- `output` (string, optional) - Local path to save the output

---

### Pipelines

#### `viralShort(options)`
Full viral short pipeline (image -> style -> animate -> speedup).

**Options:**
- `prompt` (string, required) - Image generation prompt
- `styleImage` (string, required) - Path to style reference image
- `seed` (number, optional) - Random seed
- `output` (string, optional) - Local path to save the final video

---

### System Operations

#### `freeMemory(unload?)`
Free GPU memory.

**Parameters:**
- `unload` (boolean, optional) - Unload all models from VRAM (default: true)

---

#### `interrupt()`
Interrupt the currently running workflow.

---

#### `stats()`
Get GPU VRAM and system stats.

---

#### `upload(path)`
Upload an image to ComfyUI for use in workflows.

**Parameters:**
- `path` (string, required) - Local path to the image

---

#### `download(assetId, path)`
Download an asset from ComfyUI to a local path.

**Parameters:**
- `assetId` (string, required) - Asset ID from a generation result
- `path` (string, required) - Local path to save the file

---

### Utilities

#### `execute(workflowJson, options?)`
Execute a raw ComfyUI workflow JSON via mmf CLI.

**Parameters:**
- `workflowJson` (Object, required) - ComfyUI API workflow JSON
- `options` (Object, optional)
  - `timeout` (number) - Timeout in ms (default: 11 min)
  - `output` (string) - Download result to this local path

---

#### `resizeImage(path, width?, height?)`
Resize image using sharp (CPU, no GPU needed).

**Parameters:**
- `path` (string, required) - Source image path
- `width` (number, optional) - Target width (default: 832)
- `height` (number, optional) - Target height (default: 480)

**Returns:** `Promise<{path: string}>`

---

## Error Handling

All functions return either a success result or an error object:

```javascript
import { hasError, getErrorMessage, isRetryableError } from '@romancircus/mmf-client';

const result = await fluxTxt2Img({ prompt: 'Test', seed: 42 });

if (hasError(result)) {
  console.error('Error:', getErrorMessage(result));

  if (isRetryableError(result)) {
    // VRAM, timeout, or connection error - safe to retry
    console.log('Can retry this error');
  }
}
```

**Error codes:**
- `0` - Success
- `1` - General error
- `2` - Timeout
- `3` - Validation error
- `4` - VRAM/memory error
- `5` - Connection error

---

## Timeout Configuration

Timeouts are pre-configured based on operation type:

| Operation | Timeout | Description |
|-----------|---------|-------------|
| Images | 11 min | 10 min generation + 1 min buffer |
| Videos | 15 min | Complex video generation |
| Pipelines | 30 min | Multi-stage operations |
| System | 30 sec | Stats, interrupt, upload |

Override per-call:

```javascript
import { mmf } from '@romancircus/mmf-client/utils';

const result = mmf('run --model flux ...', { timeout: 120000 }); // 2 min
```

---

## Template Reference

Import template names as constants:

```javascript
import {
  TEMPLATE_QWEN_TXT2IMG,
  TEMPLATE_WAN26_IMG2VID,
  TEMPLATE_LTX2_IMG2VID,
  PIPELINE_VIRAL_SHORT,
} from '@romancircus/mmf-client';
```

**Available templates:**

| Template | Description |
|----------|-------------|
| `TEMPLATE_QWEN_TXT2IMG` | Qwen-Image-2512 text-to-image |
| `TEMPLATE_FLUX2_FACE_ID` | FLUX.2 Face ID with IP-Adapter |
| `TEMPLATE_FLUX_KONTEXT_EDIT` | FLUX Kontext character editing |
| `TEMPLATE_WAN26_IMG2VID` | Wan 2.6 Image-to-Video |
| `TEMPLATE_WAN22_S2V` | Wan 2.2 Sound-to-Video |
| `TEMPLATE_WAN22_ANIMATE` | Wan 2.2 Character animation |
| `TEMPLATE_WAN22_PHANTOM` | Phantom multi-subject S2V |
| `TEMPLATE_LTX2_TXT2VID` | LTX-2 text-to-video |
| `TEMPLATE_LTX2_IMG2VID` | LTX-2 image-to-video |
| `TEMPLATE_LTX2_AUDIO_REACTIVE` | LTX-2 audio-reactive video |
| `TEMPLATE_TELESTYLE_IMAGE` | TeleStyle image style transfer |
| `TEMPLATE_TELESTYLE_VIDEO` | TeleStyle video style transfer |
| `TEMPLATE_VIDEO_INPAINT` | Video inpainting |
| `PIPELINE_VIRAL_SHORT` | Viral short pipeline |

---

## Examples

### Basic Image Generation

```javascript
import { qwenTxt2Img, hasError } from '@romancircus/mmf-client';

const result = await qwenTxt2Img({
  prompt: 'A majestic dragon flying over mountains at sunset, highly detailed',
  negative: 'blurry, low quality, distorted, watermark, text',
  seed: 42,
  width: 1664,
  height: 928,
  output: './dragon.png'
});

if (hasError(result)) {
  console.error('Failed:', result.error);
} else {
  console.log('Generated:', result.asset_id);
}
```

### Character Animation Pipeline

```javascript
import { teleStyleImage, wanAnimate, hasError } from '@romancircus/mmf-client';

// Step 1: Generate styled character
const styleResult = await teleStyleImage({
  content: './character_photo.jpg',
  style: './anime_style.jpg',
  cfg: 2.0,
  seed: 42,
  output: './styled_character.png'
});

if (hasError(styleResult)) {
  throw new Error(styleResult.error);
}

// Step 2: Animate
const animResult = await wanAnimate({
  image: styleResult.path || './styled_character.png',
  prompt: 'Character walks forward confidently, natural movement',
  faceStrength: 1.2,
  seed: 42,
  output: './character_walk.mp4'
});

if (hasError(animResult)) {
  throw new Error(animResult.error);
}

console.log('Animation complete:', animResult.asset_id);
```

### Batch Generation

```javascript
import { fluxTxt2Img, hasError } from '@romancircus/mmf-client';

const seeds = [1, 2, 3, 4, 5];
const results = [];

for (const seed of seeds) {
  const result = await fluxTxt2Img({
    prompt: 'A futuristic cityscape with neon lights',
    seed,
    output: `./city_${seed}.png`
  });

  if (hasError(result)) {
    console.error(`Seed ${seed} failed:`, result.error);
  } else {
    results.push(result);
  }
}

console.log(`Generated ${results.length}/${seeds.length} images`);
```

### Video Inpainting

```javascript
import { videoInpaint, hasError } from '@romancircus/mmf-client';

const result = await videoInpaint({
  video: './original_video.mp4',
  selectText: 'the red logo on the left side',
  replacePrompt: 'a blue corporate logo with white text',
  denoise: 0.75,
  seed: 42,
  output: './edited_video.mp4'
});

if (hasError(result)) {
  console.error('Inpainting failed:', result.error);
}
```

### Error Recovery

```javascript
import { wanI2V, isRetryableError, getErrorMessage } from '@romancircus/mmf-client';

async function generateWithRetry(image, prompt, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    const result = await wanI2V({
      image,
      prompt,
      seed: Date.now(), // Random seed each attempt
      output: './output.mp4'
    });

    if (!hasError(result)) {
      return result;
    }

    if (isRetryableError(result) && attempt < maxRetries) {
      console.log(`Attempt ${attempt} failed, retrying...`);
      await new Promise(r => setTimeout(r, 5000)); // Wait 5s
    } else {
      throw new Error(getErrorMessage(result));
    }
  }
}
```

---

## TypeScript

Type definitions are included. Use in TypeScript projects:

```typescript
import { qwenTxt2Img, WanI2VOptions, GenerationResult } from '@romancircus/mmf-client';

const options: WanI2VOptions = {
  image: './keyframe.png',
  prompt: 'Gentle motion',
  frames: 81,
  seed: 42
};

const result: GenerationResult = await wanI2V(options);
```

---

## License

MIT
