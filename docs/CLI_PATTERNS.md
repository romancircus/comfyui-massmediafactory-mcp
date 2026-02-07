# CLI Patterns for ComfyUI MassMediaFactory

**Production-hardened patterns from KDH-Automation**

This document captures the proven patterns used in production at KDH-Automation for CLI-based ComfyUI operations. These patterns should be the reference standard for all repositories migrating from HTTP/MCP to CLI execution.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Timeout Hierarchy](#timeout-hierarchy)
3. [Error Handling](#error-handling)
4. [Retry Configuration](#retry-configuration)
5. [Critical Parameters](#critical-parameters)
6. [Template Patterns](#template-patterns)
7. [Shell Escaping](#shell-escaping)
8. [Function Reference](#function-reference)
9. [File Operations](#file-operations)
10. [Migration Guide](#migration-guide)

---

## Architecture Overview

### Why CLI Over MCP for Execution

**Token Cost Analysis:**
| Approach | Per-Call Cost | 100 Calls/Day | 30-Day Cost |
|----------|---------------|---------------|-------------|
| MCP Tools | ~15K tokens | 1.5M tokens | 45M tokens |
| CLI | ~50 tokens | 5K tokens | 150K tokens |
| **Savings** | **99.7%** | | |

At $3-4 per million tokens, this saves **$135-180/month** for production workflows.

### The Hybrid Pattern

```
Discovery Phase (MCP - rare, high information):
  ├─ list_workflow_templates()  - 15K tokens
  ├─ get_model_constraints()    - 15K tokens
  └─ validate_workflow()          - 15K tokens

Execution Phase (CLI - frequent, zero overhead):
  ├─ mmf run --template X ...     - ~10 tokens
  ├─ mmf batch seeds ...          - ~10 tokens
  └─ mmf pipeline ...             - ~10 tokens
```

**Rule of Thumb:**
- Use MCP for discovery, validation, and debugging
- Use CLI for all execution workflows
- Never use MCP for batch operations (>10 calls)

---

## Timeout Hierarchy

From KDH-Automation `src/core/mmf.js` lines 22-25:

```javascript
// Timeout values in milliseconds
const IMAGE_TIMEOUT = 660_000;      // 11 minutes
const VIDEO_TIMEOUT = 900_000;      // 15 minutes
const PIPELINE_TIMEOUT = 1_800_000; // 30 minutes
const SYSTEM_TIMEOUT = 30_000;      // 30 seconds
```

### Timeout Guidelines

| Operation Type | Timeout | Rationale |
|----------------|---------|-----------|
| **Images** (Qwen, FLUX) | 11 min | ~10 min generation + 1 min buffer |
| **Videos** (Wan, LTX, Hunyuan) | 15 min | Frame generation + encoding |
| **Pipelines** (viral-short) | 30 min | Multi-stage: image + style + video |
| **System** (stats, upload) | 30 sec | Fast operations, no retry |

### Code Example

```javascript
// images.js
export function qwenTxt2Img(options) {
  return mmf(buildArgs([/* ... */]), {
    timeout: IMAGE_TIMEOUT  // 660000 ms
  });
}

// video.js
export function wanI2V(options) {
  return mmf(buildArgs([/* ... */]), {
    timeout: VIDEO_TIMEOUT  // 900000 ms
  });
}

// pipeline.js
export function viralShort(options) {
  return mmf(buildArgs([/* ... */]), {
    timeout: PIPELINE_TIMEOUT  // 1800000 ms
  });
}

// system.js - No retry
export function stats() {
  return mmf('stats', {
    timeout: SYSTEM_TIMEOUT,
    noRetry: true  // Don't retry system commands
  });
}
```

---

## Error Handling

### The "Return Error Object" Pattern

**Never throw errors.** Always return error objects that consumers can handle gracefully.

```javascript
// From mmf.js lines 58-76
try {
  const result = execSync(cmd, {
    encoding: 'utf-8',
    timeout: opts.timeout || DEFAULT_TIMEOUT,
    stdio: ['pipe', 'pipe', 'pipe']
  });

  // Parse stdout as JSON
  return JSON.parse(result);
} catch (err) {
  const stderr = err.stderr?.toString().trim() || '';
  const stdout = err.stdout?.toString().trim() || '';

  // Try to parse stdout even on error
  // CLI may return JSON error objects
  if (stdout) {
    try {
      return JSON.parse(stdout);
    } catch {
      // Not JSON, fall through
    }
  }

  // Return error object instead of throwing
  return {
    error: stderr || err.message || `mmf exited with code ${err.status}`,
    code: err.status,
    // Include stdout if available (for debugging)
    ...(stdout && { stdout })
  };
}
```

### Consumer Error Handling

```javascript
// Production pattern from GroupKeyGenerator.js
const result = await mmf.qwenTxt2Img({
  prompt: characterPrompt,
  output: keyframePath
});

if (result.error) {
  console.error(`  FAILED ${characterId}: ${result.error}`);
  failed++;
  continue;  // Continue with next, don't crash batch
}

console.log(`  SUCCESS: ${result.asset_id}`);
```

### Error Classification

| Exit Code | Meaning | Action |
|-----------|---------|--------|
| 0 | Success | Process result |
| 1 | Error | Check result.error message |
| 2 | Timeout | Consider retry with longer timeout |
| 3 | Validation | Fix parameters and retry |
| 4 | Partial | Some succeeded, review individually |
| 5 | Connection | Check ComfyUI server status |
| 6 | Not Found | Asset/workflow missing |
| 7 | VRAM | Clear queue, reduce batch size |

---

## Retry Configuration

### Built-in CLI Retry

The `mmf` CLI has built-in retry logic. Use it instead of implementing custom retry.

```javascript
// From mmf.js line 42
const retryFlags = opts.noRetry
  ? ''
  : ` --retry ${retry} --retry-on vram,timeout,connection`;
```

### Retry Triggers

| Trigger | When It Fires |
|---------|---------------|
| `vram` | VRAM OOM errors |
| `timeout` | Execution timeouts |
| `connection` | ComfyUI server connection issues |

### Code Example

```javascript
// Default: 3 retries on VRAM, timeout, connection
const args = `run --template wan26_img2vid --params '${params}' --retry 3 --retry-on vram,timeout,connection`;

// System commands: no retry
export function freeMemory(unload = true) {
  return mmf(
    `free${unload ? ' --unload' : ''}`,
    {
      timeout: SYSTEM_TIMEOUT,
      noRetry: true  // Don't retry
    }
  );
}

// Custom retry count
export function wanI2V(options) {
  const retry = options.retry ?? 5;  // 5 retries
  return mmf(
    buildArgs([/* ... */]),
    { timeout: VIDEO_TIMEOUT, retry }
  );
}
```

---

## Critical Parameters

### Qwen-Image: Shift Parameter

**CRITICAL:** `shift=7.0` for sharp output

```javascript
// From qwen_txt2img.json template
{
  "_meta": {
    "parameters": ["PROMPT", "NEGATIVE", "SEED", "WIDTH", "HEIGHT", "SHIFT"],
    "defaults": {
      "SHIFT": 7.0,  // CRITICAL
      "WIDTH": 1296,
      "HEIGHT": 1296
    },
    "notes": "CRITICAL: shift=7.0 for sharp output. Default 3.1 causes blurry images. Use 12-13 for maximum sharpness."
  }
}
```

```javascript
// In code
const params = JSON.stringify({
  PROMPT: prompt,
  SHIFT: 7.0,  // Never use default
  // ...
});
```

### TeleStyle: CFG Parameter

**CRITICAL:** `cfg=2.0-2.5` to avoid color distortion

```javascript
// From mmf.js lines 331-337 (inline comment)
export function teleStyleImage(options) {
  const params = JSON.stringify({
    CONTENT_PATH: options.content,
    STYLE_PATH: options.style,
    CFG: options.cfg ?? 2.0,  // CRITICAL: 2.0-2.5
    // ...
  });
  // CRITICAL: Keep CFG 2.0-2.5. Higher causes oversaturation.
  return mmf(buildArgs([/* ... */]), { timeout: IMAGE_TIMEOUT });
}
```

### LTX: Frame Rule

**CRITICAL:** Frames must be `8n+1`: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97

```javascript
// From ltx2_txt2vid.json template
{
  "_meta": {
    "notes": "CRITICAL: frames must be 8n+1 (9, 17, 25...97). Other values cause errors."
  }
}
```

```javascript
// Validation helper
function validateLtxFrames(frames) {
  if ((frames - 1) % 8 !== 0) {
    throw new Error(`LTX requires 8n+1 frames. Got ${frames}. Valid: 9, 17, 25, ..., 97`);
  }
}
```

### Wan I2V: Noise Augmentation

**CRITICAL:** `noiseAug=0.03` for character consistency

```javascript
const params = JSON.stringify({
  IMAGE_PATH: image,
  PROMPT: prompt,
  NOISE_AUG: 0.03,  // Critical for identity preservation
  // Higher values cause identity drift
});
```

---

## Template Patterns

### Template Structure

```json
{
  "_meta": {
    "description": "Template description",
    "parameters": ["PROMPT", "SEED", "WIDTH"],
    "defaults": {
      "WIDTH": 1024,
      "HEIGHT": 1024,
      "SEED": 42
    },
    "notes": "Important usage notes"
  },
  "node_id": {
    "class_type": "NodeClass",
    "inputs": {
      "text": "{{PROMPT}}",  // Double curly braces for substitution
      "seed": "{{SEED}}"
    }
  }
}
```

### Building Parameters

```javascript
// Pattern from mmf.js
function buildTemplateParams(templateName, values) {
  // Get template defaults
  const template = loadTemplate(templateName);
  const defaults = template._meta.defaults || {};

  // Merge with provided values
  const params = {
    ...defaults,
    ...values
  };

  // Validate all required parameters present
  const required = template._meta.parameters || [];
  const missing = required.filter(p => !(p in params));
  if (missing.length > 0) {
    throw new Error(`Missing parameters: ${missing.join(', ')}`);
  }

  return JSON.stringify(params);
}

// Usage
const params = buildTemplateParams('qwen_txt2img', {
  PROMPT: 'a majestic dragon',
  SEED: 42,
  SHIFT: 7.0  // Override default
});

const args = `run --template qwen_txt2img --params '${esc(params)}'`;
```

### Template Categories

| Category | Templates | Use Case |
|----------|-----------|----------|
| **Image Gen** | qwen_txt2img, flux2_txt2img, flux2_face_id | Characters, keyframes |
| **Video Gen** | wan26_img2vid, wan22_animate, ltx2_txt2vid | Video clips |
| **Style Transfer** | telestyle_image, telestyle_video | 3D physics style |
| **Editing** | qwen_edit_background, flux_kontext_edit | Consistency, background |
| **Audio** | ltx2_audio_reactive, mmaudio_v2a, wan22_s2v | Sound sync |

---

## Shell Escaping

### The Escaping Function

**CRITICAL:** Always escape user-provided strings in shell commands.

```javascript
// From mmf.js lines 81-84
function esc(str) {
  // Escape single quotes by ending string, adding escaped quote, resuming
  return str.replace(/'/g, "'\\''");
}
```

### Usage Pattern

```javascript
const prompt = "It's a dragon's world";  // Contains quotes
const output = "/path/to/output/file.png";

const args = [
  `run --template qwen_txt2img`,
  `--params '{"PROMPT":"${esc(prompt)}"}'`,  // Escape user content
  `--output '${esc(output)}'`  // Escape paths too
].join(' ');
```

### Security Considerations

```javascript
// BAD - Command injection possible
const args = `run --prompt "${userPrompt}"`;

// GOOD - Properly escaped
const args = `run --prompt '${esc(userPrompt)}'`;

// ALSO GOOD - Use JSON params (no shell parsing)
const params = JSON.stringify({ PROMPT: userPrompt });  // JSON handles escaping
const args = `run --template X --params '${esc(params)}'`;
```

---

## Function Reference

### Image Generation (5 functions)

#### `qwenTxt2Img(options)`
```javascript
qwenTxt2Img({
  prompt: string,           // Required: generation prompt
  negative?: string,        // Optional: negative prompt
  seed?: number,            // Optional: random seed (default: random)
  width?: number,           // Optional: 256-2048 (default: 1296)
  height?: number,          // Optional: 256-2048 (default: 1296)
  shift?: number,           // Optional: 1-20 (default: 7.0, CRITICAL)
  cfg?: number,             // Optional: guidance scale
  steps?: number,           // Optional: sampling steps
  output?: string           // Optional: download path
}): Promise<{
  asset_id: string,
  images: Array<{asset_id, url}>,
  prompt_id: string
} | {error: string}>
```

**Template:** `qwen_txt2img`
**Timeout:** 11 minutes
**Critical:** `shift=7.0` for sharpness

---

#### `fluxTxt2Img(options)`
```javascript
fluxTxt2Img({
  prompt: string,
  negative?: string,
  seed?: number,
  width?: number,          // 256-2048
  height?: number,
  cfg?: number,            // Default: 3.5
  steps?: number,          // Default: 20
  output?: string
})
```

**Model:** FLUX.2-dev
**Timeout:** 11 minutes

---

#### `faceIdTxt2Img(options)`
```javascript
faceIdTxt2Img({
  prompt: string,
  face_image: string,      // Path to face reference image
  face_weight?: number,    // Default: 1.0
  seed?: number,
  output?: string
})
```

**Template:** `flux2_face_id`
**Features:** IP-Adapter face identity preservation

---

#### `teleStyleImage(options)`
```javascript
teleStyleImage({
  content: string,         // Content image path
  style: string,           // Style image path (or TeleStyle key)
  cfg?: number,           // CRITICAL: 2.0-2.5 (default: 2.0)
  seed?: number,
  output?: string
})
```

**Template:** `telestyle_image`
**Critical:** `cfg=2.0-2.5` to avoid color distortion
**Purpose:** Character consistency across shots

---

#### `kontextEdit(options)`
```javascript
kontextEdit({
  image: string,           // Source image path
  prompt: string,          // Edit instruction
  seed?: number,
  output?: string
})
```

**Template:** `flux_kontext_edit`
**Features:** Character-consistent editing

---

### Video Generation (8 functions)

#### `wanI2V(options)`
```javascript
wanI2V({
  image: string,           // Input image path
  prompt: string,          // Motion description
  negative?: string,
  seed?: number,
  frames?: number,         // Default: 81, must be 8n+1
  cfg?: number,           // Default: 5.0
  shift?: number,         // Default: 5.0
  noiseAug?: number,      // Default: 0.03 (CRITICAL)
  output?: string
})
```

**Template:** `wan26_img2vid`
**Timeout:** 15 minutes
**Critical:** `noiseAug=0.03` for character consistency

---

#### `wanS2V(options)` - Sound to Video
```javascript
wanS2V({
  audio: string,           // Audio file path
  prompt: string,
  seed?: number,
  frames?: number,
  output?: string
})
```

**Template:** `wan22_s2v`
**Features:** wav2vec2 audio encoder

---

#### `ltxT2V(options)` - Text to Video
```javascript
ltxT2V({
  prompt: string,
  negative?: string,
  seed?: number,
  frames?: number,         // CRITICAL: 8n+1 (9,17,25...97)
  width?: number,
  height?: number,
  cfg?: number,
  steps?: number,
  output?: string
})
```

**Template:** `ltx2_txt2vid`
**Timeout:** 15 minutes
**Critical:** Frames must be `8n+1`

---

#### `ltxI2V(options)` - Image to Video
```javascript
ltxI2V({
  image: string,
  prompt: string,
  seed?: number,
  frames?: number,         // 8n+1
  cfg?: number,
  output?: string
})
```

**Template:** `ltx2_img2vid`
**Model:** LTX-2 19B AV

---

#### `wanAnimate(options)` - Character Animation
```javascript
wanAnimate({
  character: string,       // Character reference image
  pose_sequence: string, // Pose reference video
  prompt: string,
  seed?: number,
  output?: string
})
```

**Template:** `wan22_animate`
**Features:** Identity preservation

---

#### `phantomS2V(options)` - Multi-Subject
```javascript
phantomS2V({
  characters: string[],    // 1-4 character reference images
  audio: string,
  prompt: string,
  seed?: number,
  output?: string
})
```

**Template:** `wan22_phantom`
**Features:** Multi-subject sound-to-video

---

#### `audioReactiveI2V(options)`
```javascript
audioReactiveI2V({
  image: string,
  audio: string,
  prompt: string,
  seed?: number,
  output?: string
})
```

**Template:** `ltx2_audio_reactive`
**Features:** Audio encoder sync

---

#### `teleStyleVideo(options)` - Video Style Transfer
```javascript
teleStyleVideo({
  content: string,        // Video path
  style: string,          // Style reference
  cfg?: number,          // 2.0-2.5
  seed?: number,
  output?: string
})
```

**Template:** `telestyle_video`
**Features:** Temporal consistency

---

#### `videoInpaint(options)`
```javascript
videoInpaint({
  video: string,
  mask_prompt: string,    // CLIPSeg mask description
  fill_prompt: string,   // Inpaint content
  seed?: number,
  output?: string
})
```

**Template:** `video_inpaint`
**Features:** CLIPSeg + FLUX inpaint

---

### Pipelines (1 function)

#### `viralShort(options)` - 3-Stage Pipeline
```javascript
viralShort({
  prompt: string,              // Base prompt
  style_image?: string,        // TeleStyle reference
  keyframes?: number,          // Default: 4
  video_length?: number,       // Default: 81 frames
  seed?: number,
  output_dir?: string          // Output directory
}): Promise<{
  keyframe_paths: string[],
  styled_paths: string[],
  video_path: string,
  final_path: string
} | {error: string}>
```

**CLI:** `mmf pipeline viral-short`
**Stages:**
1. Qwen T2I (keyframe generation)
2. TeleStyle Image (character consistency)
3. Wan I2V (animation)
4. FFmpeg speedup (final output)

**Timeout:** 30 minutes

---

### System Operations (5 functions)

#### `freeMemory(unload?)`
```javascript
freeMemory(unload = true): Promise<{status: string}>
```

**CLI:** `mmf free --unload`
**Timeout:** 30 seconds (no retry)

---

#### `interrupt()`
```javascript
interrupt(): Promise<{status: string}>
```

**CLI:** `mmf interrupt`
**Timeout:** 30 seconds (no retry)

---

#### `stats()`
```javascript
stats(): Promise<{
  system: {
    os: string,
    platform: string,
    total_memory_gb: number
  },
  gpu: {
    name: string,
    vram_total_gb: number,
    vram_free_gb: number
  },
  comfyui: {
    url: string,
    status: string
  }
}>
```

**CLI:** `mmf stats`
**Timeout:** 30 seconds (no retry)

---

#### `upload(path)`
```javascript
upload(path: string): Promise<{name: string, asset_id: string}>
```

**CLI:** `mmf upload <path>`
**Timeout:** 30 seconds (no retry)

---

#### `download(assetId, path)`
```javascript
download(assetId: string, path: string): Promise<{status: string, path: string}>
```

**CLI:** `mmf download <asset_id> <path>`
**Timeout:** 30 seconds (no retry)

---

### Utilities (1 function)

#### `execute(workflowJson, options)`
```javascript
execute(
  workflowJson: object,     // Raw ComfyUI workflow
  options?: {
    timeout?: number,
    retry?: number,
    output?: string
  }
): Promise<{
  asset_id: string,
  prompt_id: string
} | {error: string}>
```

**CLI:** `mmf execute -` (via stdin)
**Purpose:** Execute raw workflow JSON

---

## File Operations

### Atomic File Writes

**Pattern for critical state files:**

```javascript
import { writeFileSync, renameSync, unlinkSync } from 'fs';

function atomicWrite(path, data) {
  const tempPath = `${path}.tmp.${process.pid}`;
  try {
    // Write to temp file first
    writeFileSync(tempPath, JSON.stringify(data, null, 2));
    // Atomic rename (POSIX guarantees this)
    renameSync(tempPath, path);
  } catch (err) {
    // Cleanup on error
    try { unlinkSync(tempPath); } catch {}
    throw err;
  }
}
```

**Used in:** ManifestTracker.js for job queue persistence

### File Resolution Strategy

**When mmf downloads fail, try multiple methods:**

```javascript
function resolveOutput(result, localPath, prefix) {
  // Method 1: mmf already downloaded
  if (existsSync(localPath)) return true;

  // Method 2: Use asset_id to download
  const assetId = result?.images?.[0]?.asset_id;
  if (assetId) {
    const dlResult = download(assetId, localPath);
    if (!dlResult.error && existsSync(localPath)) return true;
  }

  // Method 3: Find in ComfyUI output directory
  const comfyOutput = process.env.COMFYUI_OUTPUT || './output';
  const files = readdirSync(comfyOutput)
    .filter(f => f.startsWith(prefix))
    .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);

  if (files.length > 0) {
    copyFileSync(join(comfyOutput, files[0]), localPath);
    return true;
  }

  return false;
}
```

---

## Migration Guide

### HTTP to CLI Migration Checklist

#### Phase 1: Preparation
- [ ] Inventory all HTTP API calls
- [ ] Map each to CLI equivalent
- [ ] Identify custom logic to preserve
- [ ] Document current retry/error handling
- [ ] Create test cases for current behavior

#### Phase 2: CLI Wrapper
- [ ] Create new CLI client file
- [ ] Implement retry logic (`--retry 3`)
- [ ] Add error object pattern (no throwing)
- [ ] Add JSON parsing from stdout
- [ ] Add shell escaping for user input
- [ ] Implement timeout configuration

#### Phase 3: Migration
- [ ] Replace HTTP calls with CLI calls
- [ ] Update all imports
- [ ] Add CLI subprocess execution
- [ ] Handle exit codes properly
- [ ] Test each endpoint

#### Phase 4: Cleanup
- [ ] Mark old HTTP client as deprecated
- [ ] Update documentation
- [ ] Add migration notes to CLAUDE.md
- [ ] Remove old client (after validation period)

### Common Migration Patterns

**HTTP GET /system_stats → CLI:**
```javascript
// Before
const response = await fetch(`${COMFYUI_URL}/system_stats`);
const stats = await response.json();

// After
const result = stats();  // From mmf client
if (result.error) throw new Error(result.error);
return result;
```

**HTTP POST /prompt → CLI:**
```javascript
// Before
const response = await fetch(`${COMFYUI_URL}/prompt`, {
  method: 'POST',
  body: JSON.stringify({ prompt: workflow })
});
const { prompt_id } = await response.json();

// Poll for completion
while (true) {
  const status = await fetch(`${COMFYUI_URL}/history/${prompt_id}`);
  if (status.completed) break;
  await sleep(1000);
}

// After
const result = await qwenTxt2Img({
  prompt: 'a dragon',
  output: './output.png'
});
// CLI blocks until complete, no polling needed
```

**HTTP GET /view → CLI:**
```javascript
// Before
const response = await fetch(`${COMFYUI_URL}/view?filename=${filename}`);
const buffer = await response.buffer();
fs.writeFileSync(outputPath, buffer);

// After
const result = download(assetId, outputPath);
if (result.error) throw new Error(result.error);
```

---

## Additional Resources

### Reference Implementations

| Repository | File | Purpose |
|------------|------|---------|
| **KDH-Automation** | `src/core/mmf.js` | Production reference (687 lines) |
| **Goat** | `src/clients/ComfyUIClientCLI.js` | Migration example |
| **RobloxChristian** | `src/visual_gen_cli.py` | Python implementation |
| **mmf-client** | `packages/mmf-client/src/index.js` | Shared package |

### Template Reference

| Template | Best For | Critical Params |
|----------|----------|-----------------|
| `qwen_txt2img` | Character portraits | shift=7.0 |
| `flux2_face_id` | Face consistency | IP-Adapter weight |
| `wan26_img2vid` | Standard I2V | noiseAug=0.03 |
| `ltx2_txt2vid` | Fast T2V | frames=8n+1 |
| `telestyle_image` | Character consistency | cfg=2.0-2.5 |

### CLI Quick Reference

```bash
# Discovery
mmf stats                                    # System info
mmf templates list                           # List templates
mmf models constraints wan                   # Model limits

# Execution
mmf run --model qwen --type t2i --prompt "..."
mmf run --template wan26_img2vid --params '{"PROMPT":"..."}'
mmf pipeline i2v --image photo.png --prompt "..."
mmf batch seeds workflow.json --count 8

# System
mmf free --unload                            # Clear VRAM
mmf interrupt                                # Stop current
mmf upload image.png                         # Upload for I2V
mmf download <asset_id> output.png         # Get result
```

---

## Summary

**Key Principles:**
1. Use CLI for execution (99.7% cost savings)
2. Return error objects, don't throw
3. Use built-in retry (`--retry 3`)
4. Respect critical parameters (shift=7.0, cfg=2.0-2.5, 8n+1 frames)
5. Escape all user input
6. Use atomic file writes for state
7. Block on CLI calls (no polling)

**Questions?** Reference KDH-Automation `src/core/mmf.js` for production examples.
