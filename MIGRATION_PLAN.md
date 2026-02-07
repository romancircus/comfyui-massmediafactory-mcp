# Full Migration Plan: MCP → mmf CLI [ROM-525 Phase 4+]

## Executive Summary

Migrate ALL ComfyUI generation across 4 repos to a single `mmf` CLI. Additionally migrate Pokedex's native diffusers pipelines to ComfyUI (our server already has all the nodes). This eliminates ~8,700 lines of duplicated ComfyUI client code, native diffusers dependencies, and GPU memory conflicts.

**Scope:** 44 files in KDH + 35 files in Pokedex + CLI improvements + 5 new templates + downstream thin wrappers.

---

## Part 1: CLI Improvements (Before Migration)

Based on research from clig.dev, gh CLI patterns, kubectl, Terraform, and the "CLI is the New MCP" article, these improvements make `mmf` migration-ready.

### 1.1 Output Stream Separation (P0)

**Problem:** Errors sometimes go to stdout, sometimes stderr. Progress and data are mixed.

**Fix:**
- ALL data → stdout (JSON)
- ALL messaging (progress, status, errors) → stderr
- When stdout is NOT a TTY (piped), auto-suppress human formatting

```python
# New helpers in cli.py
def _msg(text: str):
    """Status message to stderr (never pollutes stdout pipe)."""
    print(text, file=sys.stderr)

def _output(data: dict, pretty: bool = False):
    """Data to stdout only."""
    # Existing, but ensure ONLY this writes to stdout
```

**Impact:** Enables clean piping: `mmf run ... | jq .asset_id`

### 1.2 Expanded Exit Codes (P0)

**Current:** 0=ok, 1=error, 2=timeout, 3=validation
**Problem:** Batch partial failures return 0. No code for connection errors.

**New scheme:**
```
0  EXIT_OK           All succeeded
1  EXIT_ERROR        General/unknown error
2  EXIT_TIMEOUT      Generation exceeded --timeout
3  EXIT_VALIDATION   Invalid params, bad workflow
4  EXIT_PARTIAL      Batch: some succeeded, some failed
5  EXIT_CONNECTION    ComfyUI unreachable
6  EXIT_NOT_FOUND    Asset/model/template not found
7  EXIT_VRAM         Out of VRAM
```

**Fix batch commands:** Return EXIT_PARTIAL when some jobs fail.

### 1.3 Retry Logic (P0 - Required for KDH Migration)

KDH has 200+ lines of retry/recovery logic. Must be in `mmf` before KDH can migrate.

```bash
mmf run --model wan --type i2v --image photo.png --prompt "motion" \
  --retry 3 --retry-on vram,timeout,connection
```

**Implementation (add to cli.py):**
```python
TRANSIENT_ERRORS = {"VRAM_EXHAUSTED", "TIMEOUT", "CONNECTION_ERROR"}
PERMANENT_ERRORS = {"VALIDATION_ERROR", "NOT_FOUND", "INVALID_PARAMS"}

def _retry_loop(fn, max_retries, retry_on, backoff=2.0):
    for attempt in range(1, max_retries + 1):
        result = fn()
        if "error" not in result:
            return result
        error_code = result.get("code", "UNKNOWN")
        if error_code in PERMANENT_ERRORS or error_code not in retry_on:
            return result  # Don't retry permanent errors
        if attempt < max_retries:
            if error_code == "VRAM_EXHAUSTED":
                execution.free_memory(unload_models=True)
                time.sleep(5)
            elif error_code == "TIMEOUT":
                execution.interrupt()
                time.sleep(3)
            else:
                time.sleep(backoff ** attempt)
    return result
```

**Flags added to `run`, `pipeline`, `batch` commands:**
- `--retry N` (default: 0, no retry)
- `--retry-on vram,timeout,connection` (default: all transient)

### 1.4 --no-wait Flag (P1)

```bash
# Fire and forget (returns prompt_id immediately)
mmf run --model flux --type t2i --prompt "dragon" --no-wait
# {"prompt_id": "abc123"}

# Wait separately
mmf wait abc123 --timeout 300
```

Currently `mmf run` always blocks. Add `--no-wait` flag.

### 1.5 --dry-run Flag (P1)

```bash
mmf run --model wan --type i2v --image photo.png --prompt "motion" --dry-run
# {
#   "workflow": { ... },     # The workflow that WOULD be submitted
#   "model": "wan",
#   "resolution": "832x480",
#   "frames": 81,
#   "estimated_vram_gb": 14.2,
#   "estimated_time_seconds": 150,
#   "hardware": "RTX 5090 (full_gpu, fp8_e4m3fn_scaled_fast)"
# }
```

Builds workflow + runs hardware optimization but doesn't execute. Critical for debugging.

### 1.6 Input Validation (P1)

Add before execution:
- Verify image paths exist (`--image`)
- Verify image is a real image (check magic bytes, not just extension)
- Validate timeout bounds (5-3600 seconds)
- Validate seed range (0 to 2^32)
- Validate JSON structure in `--params` (check required fields per template)
- Validate output directory exists and is writable

### 1.7 Missing Commands (P1)

Add these 6 commands (~80 lines total):

```bash
mmf regenerate <asset_id> [--prompt] [--seed] [--cfg] [--steps]
mmf status [prompt_id]           # Non-blocking status check
mmf progress <prompt_id>         # Real-time percent/ETA
mmf search-model <query>         # Civitai search
mmf install-model <url> --type   # Download model
mmf workflow-lib <action>        # save|load|list|delete
```

### 1.8 --url Global Flag (P2)

```bash
mmf --url http://remote:8188 run --model flux --type t2i --prompt "test"
```

Overrides COMFYUI_URL env var for per-command targeting.

### 1.9 Config File Support (P2)

```toml
# ~/.config/mmf/config.toml
[server]
url = "http://solapsvs.taila4c432.ts.net:8188"

[defaults]
timeout = 600
pretty = true

[retry]
max_retries = 3
retry_on = ["vram", "timeout", "connection"]
backoff = 2.0
```

Precedence: CLI flags > env vars > project .mmf.toml > user config > defaults.

---

## Part 2: New Templates (5 Templates)

These templates enable migrating ALL native diffusers code to ComfyUI.

### 2.1 `qwen_controlnet_bio` (Pokedex bio generation)

**Purpose:** Replace `BioGenerator(model="comfyui")` which uses Qwen + DiffSynth ControlNet.

**Nodes (all installed on our ComfyUI):**
- `UNETLoader` → qwen_image_2512_fp8_e4m3fn.safetensors
- `CLIPLoader` → qwen_2.5_vl_7b_fp8_scaled.safetensors
- `VAELoader` → qwen_image_vae.safetensors
- `QwenImageDiffsynthControlnet` → Load ControlNet model
- `CannyEdgePreprocessor` → Extract edges from reference image
- `TextEncodeQwenImageEditPlus` → Encode prompt
- `KSampler` → Sample with CFG 3.5, shift 7.0
- `SaveImage` → Output

**Parameters:**
```
IMAGE_PATH, PROMPT, NEGATIVE, SEED, WIDTH, HEIGHT, CFG, SHIFT, STEPS, CONTROLNET_STRENGTH
```

### 2.2 `flux_kontext_edit` (Pokedex shiny transformation)

**Purpose:** Replace `ShinyTransformer(model="kontext")` which uses FLUX Kontext LoRA for color-shifting.

**Nodes:**
- `UNETLoader` → flux1-dev-fp8.safetensors
- `FluxKontextImageScale` → Reference image conditioning
- `CLIPTextEncodeFlux` → Encode edit instruction
- `KSampler` → Sample with Kontext guidance
- `SaveImage` → Output

**Parameters:**
```
IMAGE_PATH, EDIT_PROMPT, SEED, WIDTH, HEIGHT, STEPS, GUIDANCE
```

### 2.3 `z_turbo_txt2img` (Pokedex fast bio generation)

**Purpose:** Replace `BioGenerator(model="z-turbo")` which loads Z-Image-Turbo natively.

**Nodes:**
- Z-Image loader (via `NunchakuZImageTurboLoraStackV4` or standard UNETLoader)
- `CLIPLoader` → Appropriate CLIP for Z-Image
- Standard sampling pipeline
- `SaveImage` → Output

**Parameters:**
```
PROMPT, IMAGE_PATH, SEED, WIDTH, HEIGHT, STEPS, CFG, DENOISE
```

**Note:** Z-Image is an img2img model (takes reference image). Template must include image conditioning.

### 2.4 `wan22_s2v` (KDH subject-to-video)

**Purpose:** Replace `ComfyUIExecutor.executeWan22S2V()`.

**Nodes:**
- WanVideoWrapper nodes (same pattern as wan26_img2vid)
- `WanVideoModelLoader` → wan2.2_s2v_14B_fp8_scaled.safetensors (already installed!)
- Standard Wan I2V pipeline but with S2V model

**Parameters:**
```
IMAGE_PATH, PROMPT, NEGATIVE, SEED, FRAMES, FPS, STEPS, CFG, SHIFT, WIDTH, HEIGHT
```

### 2.5 `flux2_txt2img_controlnet` (Pokedex bio with ControlNet structure)

**Purpose:** Replace `BioGenerator(model="flux1")` which uses FLUX.1 + ControlNet Union for Canny edges.

**Already have `flux2_union_controlnet` template** - may just need parameter adjustments:
- `CONTROL_MODE=0` (Canny mode)
- Default `CONTROL_STRENGTH=0.6` for bio generation

**May not need a new template** - verify flux2_union_controlnet covers this use case.

---

## Part 3: KDH-Automation Migration

### 3.1 What Gets Deleted (3,524+ lines)

| File | Lines | Replacement |
|------|-------|-------------|
| `src/core/ComfyUIClient.js` | 329 | `mmf` handles all HTTP |
| `src/core/ComfyUIHttpAdapter.js` | 457 | `mmf` replaces installGlobals() |
| `src/core/ComfyUIWorkflow.js` | 648 | Deprecated, already unused |
| `src/core/SOTAWorkflowTemplates.js` | 605 | Templates in mmf hub |
| `src/core/ComfyUIExecutor.js` | 2,091 | Thin wrapper → mmf |

**Also delete:** 17 local `mcp_templates/*.json` files (templates live in mmf hub).

### 3.2 New File: `src/core/mmf.js` (~120 lines)

```javascript
import { execSync, execFileSync } from 'child_process';

const MMF_TIMEOUT = 660_000; // 11 min (10 min generation + 1 min buffer)

/**
 * Execute mmf CLI command and return parsed JSON.
 * All ComfyUI operations go through mmf.
 */
function mmf(args, opts = {}) {
  const timeout = opts.timeout || MMF_TIMEOUT;
  const retry = opts.retry || 3;
  const cmd = `mmf ${args} --retry ${retry} --retry-on vram,timeout,connection`;

  const result = execSync(cmd, {
    encoding: 'utf8',
    timeout,
    stdio: ['pipe', 'pipe', 'pipe'], // capture all streams
    env: { ...process.env, COMFYUI_URL: process.env.COMFYUI_URL }
  });

  return JSON.parse(result.trim());
}

// ── Image Generation ──────────────────────────────

export function qwenTxt2Img({ prompt, negative, seed, width = 1920, height = 1080, shift = 7.0, cfg = 3.5, steps = 50, output }) {
  const args = [
    'run --model qwen --type t2i',
    `--prompt '${escapeShell(prompt)}'`,
    negative ? `--negative '${escapeShell(negative)}'` : '',
    `--seed ${seed} --width ${width} --height ${height}`,
    `--cfg ${cfg} --steps ${steps} --guidance ${shift}`,
    output ? `--output '${output}'` : '',
  ].filter(Boolean).join(' ');
  return mmf(args);
}

export function fluxTxt2Img({ prompt, seed, width = 1024, height = 1024, output }) {
  return mmf(`run --model flux --type t2i --prompt '${escapeShell(prompt)}' --seed ${seed} --width ${width} --height ${height}${output ? ` --output '${output}'` : ''}`);
}

export function faceIdTxt2Img({ prompt, faceImage, seed, faceStrength = 0.85, width = 1024, height = 1024, output }) {
  const params = JSON.stringify({ PROMPT: prompt, FACE_IMAGE: faceImage, FACE_STRENGTH: faceStrength, SEED: seed, WIDTH: width, HEIGHT: height });
  return mmf(`run --template flux2_face_id --params '${params}'${output ? ` --output '${output}'` : ''}`);
}

// ── Video Generation ──────────────────────────────

export function wanI2V({ image, prompt, negative, frames = 81, fps = 16, seed, cfg = 5.0, shift = 5.0, steps = 30, noiseAug = 0.03, output }) {
  const args = [
    'run --model wan --type i2v',
    `--image '${image}' --prompt '${escapeShell(prompt)}'`,
    negative ? `--negative '${escapeShell(negative)}'` : '',
    `--seed ${seed} --frames ${frames} --cfg ${cfg} --guidance ${shift} --steps ${steps}`,
    output ? `--output '${output}'` : '',
  ].filter(Boolean).join(' ');
  return mmf(args, { timeout: 900_000 }); // 15 min for video
}

export function ltxT2V({ prompt, negative, frames = 97, fps = 24, seed, width = 1280, height = 720, steps = 30, output }) {
  return mmf(`run --model ltx --type t2v --prompt '${escapeShell(prompt)}' --seed ${seed} --frames ${frames} --width ${width} --height ${height} --steps ${steps}${output ? ` --output '${output}'` : ''}`, { timeout: 900_000 });
}

export function ltxI2V({ image, prompt, negative, frames = 97, fps = 24, seed, output }) {
  return mmf(`run --model ltx --type i2v --image '${image}' --prompt '${escapeShell(prompt)}' --seed ${seed} --frames ${frames}${output ? ` --output '${output}'` : ''}`, { timeout: 900_000 });
}

export function audioReactiveI2V({ image, audioPath, prompt, frames = 121, fps = 24, seed, output }) {
  const params = JSON.stringify({ IMAGE_PATH: image, AUDIO_PATH: audioPath, PROMPT: prompt, FRAMES: frames, FPS: fps, SEED: seed });
  return mmf(`run --template ltx2_audio_reactive --params '${params}'${output ? ` --output '${output}'` : ''}`, { timeout: 900_000 });
}

export function wanS2V({ image, prompt, frames = 77, fps = 16, seed, output }) {
  const params = JSON.stringify({ IMAGE_PATH: image, PROMPT: prompt, FRAMES: frames, FPS: fps, SEED: seed });
  return mmf(`run --template wan22_s2v --params '${params}'${output ? ` --output '${output}'` : ''}`, { timeout: 900_000 });
}

// ── Style Transfer ────────────────────────────────

export function teleStyleImage({ content, style, seed, cfg = 2.0, steps = 20, output }) {
  return mmf(`telestyle image --content '${content}' --style '${style}' --seed ${seed}${output ? ` -o '${output}'` : ''}`);
}

export function teleStyleVideo({ video, style, cfg = 1.0, steps = 12, output }) {
  return mmf(`telestyle video --content '${video}' --style '${style}'${output ? ` -o '${output}'` : ''}`, { timeout: 900_000 });
}

// ── Pipelines ─────────────────────────────────────

export function viralShort({ prompt, styleImage, seed, output }) {
  return mmf(`pipeline viral-short --prompt '${escapeShell(prompt)}' --style-image '${styleImage}' --seed ${seed}${output ? ` -o '${output}'` : ''}`, { timeout: 1_800_000 });
}

export function videoInpaint({ video, selectText, replacePrompt, denoise = 0.7, output }) {
  const params = JSON.stringify({ VIDEO_PATH: video, SELECT_TEXT: selectText, REPLACE_PROMPT: replacePrompt, DENOISE: denoise });
  return mmf(`run --template video_inpaint --params '${params}'${output ? ` --output '${output}'` : ''}`, { timeout: 900_000 });
}

// ── System ────────────────────────────────────────

export function freeMemory(unload = true) {
  return mmf(`free${unload ? ' --unload' : ''}`);
}

export function interrupt() {
  return mmf('interrupt');
}

export function stats() {
  return mmf('stats');
}

export function upload(imagePath) {
  return mmf(`upload '${imagePath}'`);
}

export function download(assetId, outputPath) {
  return mmf(`download ${assetId} '${outputPath}'`);
}

// ── Helpers ───────────────────────────────────────

function escapeShell(str) {
  return str.replace(/'/g, "'\\''");
}
```

### 3.3 Script Migration Map

Each active KDH script gets updated to use `mmf.js`:

| Script | Current Call | New Call |
|--------|-------------|----------|
| `generateViralKeyframesV4.js` | `executor.executeQwenTxt2Img({...})` | `mmf.qwenTxt2Img({...})` |
| `generateViralKeyframesV4.js` | `executor.executeTeleStyleImage({...})` | `mmf.teleStyleImage({...})` |
| `generateViralVideosV4.js` | `executor.executeWan26I2V({...})` | `mmf.wanI2V({...})` |
| `generateViralVideosV4.js` | `executor.executeLtxTxt2Vid({...})` | `mmf.ltxT2V({...})` |
| `generateViralVideosPipelineB.js` | `executor.executeLtxTxt2Vid({...})` | `mmf.ltxT2V({...})` |
| `generateViralVideosPipelineB.js` | `executor.executeTeleStyleVideo({...})` | `mmf.teleStyleVideo({...})` |
| `runWanVideoBatch.js` | Manual fetch() to /prompt | `mmf.wanI2V({...})` in loop |
| `runWanMotionControl.js` | Manual fetch() to /prompt | `mmf run --template wan26_img2vid` |
| `runWanBabyDance.js` | Manual fetch() to /prompt | `mmf.wanI2V({...})` in loop |

**Python scripts (execute_keyframes.py, batch_i2v_choking.py, etc.):** Replace urllib.request calls with `subprocess.run("mmf ...")`.

### 3.4 KDH Retry Logic Migration

KDH's `_executeWithRetry` + `_categorizeError` + `_handleError` maps to:

```bash
mmf run --retry 3 --retry-on vram,timeout,connection ...
```

The CLI's retry logic (Part 1.3) replicates KDH's error categorization:

| KDH Error Code | CLI Category | Recovery |
|----------------|-------------|----------|
| `VRAM_EXHAUSTED` | `vram` | `mmf free --unload` + 5s wait |
| `TIMEOUT` | `timeout` | `mmf interrupt` + 3s wait |
| `COMFYUI_OFFLINE` | `connection` | Exponential backoff |
| `FILE_CORRUPT` | (handled by download) | CLI validates file after download |
| `WORKFLOW_FAILED` | permanent | No retry |
| `VALIDATION_FAILED` | permanent | No retry |

### 3.5 File Size Validation

KDH validates output files: images >= 100KB, videos >= 500KB. Add to `mmf run --output`:

```python
if output_path and output_path.exists():
    size = output_path.stat().st_size
    if output_type == "image" and size < 100_000:
        return _error(f"Output too small ({size} bytes), likely corrupt", "FILE_CORRUPT")
    if output_type == "video" and size < 500_000:
        return _error(f"Output too small ({size} bytes), likely corrupt", "FILE_CORRUPT")
```

### 3.6 KDH Migration Order

1. Create `src/core/mmf.js` (the thin wrapper)
2. Add mmf as dependency: `pip install comfyui-massmediafactory-mcp` in KDH's setup or verify `mmf` is on PATH
3. Migrate ONE script first: `generateViralKeyframesV4.js` (simplest - just Qwen T2I + TeleStyle)
4. Test: run the script, verify output quality matches
5. Migrate remaining scripts one at a time
6. Delete `ComfyUIExecutor.js`, `ComfyUIClient.js`, `ComfyUIHttpAdapter.js`, `ComfyUIWorkflow.js`, `SOTAWorkflowTemplates.js`
7. Delete `mcp_templates/` directory (17 templates)
8. Update KDH's CLAUDE.md to reference `/mmf` skill

---

## Part 4: Pokedex-Generator Migration

### 4.1 The Key Insight: Native Diffusers → ComfyUI

Pokedex loads FLUX.2, Z-turbo, Qwen-Edit, Wan, LTX-2 as native Python diffusers pipelines. This causes:
- **GPU memory conflicts:** Native pipeline loads 12-24GB model into same GPU as ComfyUI
- **No VRAM management:** No coordination between native diffusers and ComfyUI server
- **No hardware optimization:** No fp8, no Nunchaku, no attention mode selection
- **Duplicate model weights:** 24GB FLUX.2 on disk + 11GB flux1-dev-fp8 in ComfyUI = wasted space

**ComfyUI already supports ALL these models** (verified - nodes installed):
- FLUX.2: `EmptyFlux2LatentImage`, `Flux2Scheduler` → `flux2_txt2img` template
- FLUX.1 ControlNet: `ApplyFluxControlNet`, `SetUnionControlNetType` → `flux2_union_controlnet` template
- Qwen-Edit: `TextEncodeQwenImageEdit` → `qwen_edit_background` template
- Z-Image: `NunchakuZImageTurboLoraStackV4`, `ZImageFunControlnet` → new template needed
- Kontext: `FluxKontextImageScale` → new template needed
- Wan I2V: `WanVideoWrapper` suite → `wan26_img2vid` template
- LTX-2: Full suite → `ltx2_img2vid`, `ltx2_txt2vid` templates

### 4.2 What Gets Deleted (~4,800 lines)

| File | Lines | Replacement |
|------|-------|-------------|
| `src/adapters/comfyui_client.py` | 2,347 | `mmf` CLI |
| `src/adapters/mcp_adapter.py` | 466 | `mmf` CLI |
| `src/adapters/job_registry.py` | 165 | `mmf status`/`mmf wait` |
| Native diffusers in `bio_generator.py` | ~450 | `mmf run` (ComfyUI) |
| Native diffusers in `video_generator.py` | ~350 | `mmf run` (ComfyUI) |
| Native diffusers in `shiny_transformer.py` | ~200 | `mmf run --template` (ComfyUI) |
| `workflows/*.json` (local templates) | 5 files | Templates in mmf hub |
| `mcp_templates/*.json` (local copies) | 8 files | Templates in mmf hub |

**Keep (not ComfyUI-related):**
- `src/core/bio_generator.py` - Keep BioGenerator class but replace model loading with mmf calls
- `src/core/video_generator.py` - Keep VideoGenerator class but replace model loading with mmf calls
- `src/core/shiny_transformer.py` - Keep ShinyTransformer class but replace pipeline with mmf calls
- `src/core/tts_generator.py` - ChatterboxTTS can go through ComfyUI (`chatterbox_tts` template exists)
- All batch scripts - Keep orchestration logic, replace generation calls

### 4.3 New File: `src/adapters/mmf_client.py` (~100 lines)

```python
"""Thin wrapper around mmf CLI for all ComfyUI operations."""
import json
import subprocess
from pathlib import Path


MMF_TIMEOUT = 660  # 11 min default
MMF_VIDEO_TIMEOUT = 900  # 15 min for video


def mmf(args: str, timeout: int = MMF_TIMEOUT) -> dict:
    """Execute mmf command, return parsed JSON."""
    result = subprocess.run(
        f"mmf {args} --retry 3 --retry-on vram,timeout,connection",
        shell=True, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0 and not result.stdout.strip():
        return {"error": result.stderr.strip() or f"mmf exited with code {result.returncode}"}
    return json.loads(result.stdout)


# ── Bio Image Generation ─────────────────────────

def generate_bio_flux(prompt: str, seed: int, width: int = 1024, height: int = 1024,
                      output: str = None) -> dict:
    """FLUX.2 text-to-image (was native diffusers, now ComfyUI)."""
    cmd = f"run --model flux --type t2i --prompt '{_esc(prompt)}' --seed {seed} --width {width} --height {height}"
    if output: cmd += f" --output '{output}'"
    return mmf(cmd)


def generate_bio_qwen(prompt: str, seed: int, ref_image: str = None,
                      controlnet_strength: float = 0.6, output: str = None) -> dict:
    """Qwen-Image-2512 with optional ControlNet (was comfyui_client or native diffusers)."""
    if ref_image:
        params = json.dumps({"PROMPT": prompt, "IMAGE_PATH": ref_image, "SEED": seed,
                            "CONTROLNET_STRENGTH": controlnet_strength})
        cmd = f"run --template qwen_controlnet_bio --params '{params}'"
    else:
        cmd = f"run --model qwen --type t2i --prompt '{_esc(prompt)}' --seed {seed}"
    if output: cmd += f" --output '{output}'"
    return mmf(cmd)


def generate_bio_z_turbo(prompt: str, ref_image: str, seed: int, output: str = None) -> dict:
    """Z-Image-Turbo img2img (was native diffusers, now ComfyUI)."""
    params = json.dumps({"PROMPT": prompt, "IMAGE_PATH": ref_image, "SEED": seed})
    cmd = f"run --template z_turbo_txt2img --params '{params}'"
    if output: cmd += f" --output '{output}'"
    return mmf(cmd)


# ── Shiny Transformation ─────────────────────────

def transform_shiny_kontext(image: str, edit_prompt: str, seed: int, output: str = None) -> dict:
    """FLUX Kontext color shift (was native diffusers, now ComfyUI)."""
    params = json.dumps({"IMAGE_PATH": image, "EDIT_PROMPT": edit_prompt, "SEED": seed})
    cmd = f"run --template flux_kontext_edit --params '{params}'"
    if output: cmd += f" --output '{output}'"
    return mmf(cmd)


def transform_shiny_qwen_edit(image: str, edit_prompt: str, seed: int, output: str = None) -> dict:
    """Qwen-Edit color shift (was native diffusers, now ComfyUI)."""
    params = json.dumps({"IMAGE_PATH": image, "EDIT_PROMPT": edit_prompt, "SEED": seed})
    cmd = f"run --template qwen_edit_background --params '{params}'"
    if output: cmd += f" --output '{output}'"
    return mmf(cmd)


# ── Video Generation ──────────────────────────────

def generate_video_wan(image: str, prompt: str, seed: int, frames: int = 81,
                       output: str = None) -> dict:
    """Wan I2V (was comfyui_client or native diffusers, now mmf)."""
    cmd = f"run --model wan --type i2v --image '{image}' --prompt '{_esc(prompt)}' --seed {seed} --frames {frames}"
    if output: cmd += f" --output '{output}'"
    return mmf(cmd, timeout=MMF_VIDEO_TIMEOUT)


def generate_video_ltx(image: str, prompt: str, seed: int, frames: int = 97,
                       output: str = None) -> dict:
    """LTX-2 I2V (was comfyui_client or native diffusers, now mmf)."""
    cmd = f"run --model ltx --type i2v --image '{image}' --prompt '{_esc(prompt)}' --seed {seed} --frames {frames}"
    if output: cmd += f" --output '{output}'"
    return mmf(cmd, timeout=MMF_VIDEO_TIMEOUT)


# ── System ────────────────────────────────────────

def upload_image(path: str) -> dict:
    return mmf(f"upload '{path}'")

def download_asset(asset_id: str, output: str) -> dict:
    return mmf(f"download {asset_id} '{output}'")

def free_memory(unload: bool = True) -> dict:
    return mmf(f"free{'--unload' if unload else ''}")


def _esc(s: str) -> str:
    return s.replace("'", "'\\''")
```

### 4.4 BioGenerator Refactor

**Before (native diffusers):**
```python
class BioGenerator:
    def generate(self, ...):
        if self.model == "flux2":
            pipe = load_pipeline("flux2")  # Loads 24GB model into GPU
            result = pipe(prompt=prompt, ...)
        elif self.model == "comfyui":
            self.comfyui_client.generate_qwen_bio_txt2img(...)
```

**After (all through mmf):**
```python
class BioGenerator:
    def generate(self, ...):
        from src.adapters.mmf_client import generate_bio_flux, generate_bio_qwen, generate_bio_z_turbo

        if self.model in ("flux1", "flux2"):
            result = generate_bio_flux(prompt, seed, width, height, output=str(output_path))
        elif self.model in ("comfyui", "qwen"):
            result = generate_bio_qwen(prompt, seed, ref_image=str(ref_path),
                                       controlnet_strength=controlnet_scale, output=str(output_path))
        elif self.model == "z-turbo":
            result = generate_bio_z_turbo(prompt, str(ref_path), seed, output=str(output_path))
        elif self.model == "qwen-edit":
            result = transform_shiny_qwen_edit(str(ref_path), prompt, seed, output=str(output_path))

        if "error" in result:
            raise GenerationError(result["error"])
        return Image.open(output_path)
```

**What gets removed from bio_generator.py:**
- `load_pipeline()` function (~120 lines) - model loading
- `_load_controlnet()` function (~30 lines) - ControlNet loading
- `MODEL_REGISTRY` dict - model paths/configs
- All `from diffusers import ...` statements
- `torch`, `torchvision` imports
- `enable_model_cpu_offload()` / `enable_sequential_cpu_offload()` calls

### 4.5 VideoGenerator Refactor

**Before:**
```python
class VideoGenerator:
    def generate(self, ...):
        if self.model == "wan_comfyui":
            self.comfyui_client.generate_wan_i2v(...)
        elif self.model == "wan":
            pipe = load_wan_pipeline()  # Loads 14B model
            output = pipe(image=image, prompt=prompt, ...).frames[0]
```

**After:**
```python
class VideoGenerator:
    def generate(self, ...):
        from src.adapters.mmf_client import generate_video_wan, generate_video_ltx

        if self.model in ("wan", "wan_comfyui"):
            result = generate_video_wan(str(image_path), prompt, seed, frames, output=str(output_path))
        elif self.model in ("ltx2", "comfyui"):
            result = generate_video_ltx(str(image_path), prompt, seed, frames, output=str(output_path))

        if "error" in result:
            raise GenerationError(result["error"])
        return output_path
```

**What gets removed from video_generator.py:**
- `load_wan_pipeline()` function (~50 lines)
- `load_ltx2_pipeline()` function (~50 lines)
- `VIDEO_MODEL_REGISTRY` dict
- All `from diffusers import ...` statements
- `torch` imports and CUDA management

### 4.6 Dependency Cleanup

**Remove from requirements.txt:**
```
# These are no longer needed (ComfyUI handles model loading):
torch>=2.4.0          # 2.5GB package
torchvision>=0.19.0   # 500MB
torchaudio>=2.4.0     # 300MB
diffusers>=0.31.0     # 200MB
transformers>=4.44.0  # 400MB
accelerate>=0.33.0    # 50MB
```

**Total saved: ~4GB of Python dependencies.**

The Pokedex Python environment becomes lightweight - just standard libraries + subprocess calls to `mmf`.

### 4.7 Pokedex Migration Order

1. Create `src/adapters/mmf_client.py` (the thin wrapper)
2. Verify `mmf` is installed: `pip install -e ~/Applications/comfyui-massmediafactory-mcp`
3. Create the 3 missing templates (qwen_controlnet_bio, flux_kontext_edit, z_turbo_txt2img)
4. Test each template: `mmf run --template <name> --params '...' --dry-run`
5. Migrate `mcp_adapter.py` calls → `mmf_client.py` calls (easiest, MCP already maps 1:1)
6. Delete `src/adapters/mcp_adapter.py`
7. Migrate `comfyui_client.py` calls → `mmf_client.py` calls in video_generator.py
8. Delete `src/adapters/comfyui_client.py`
9. Migrate native diffusers calls in `bio_generator.py` → `mmf_client.py`
10. Migrate native diffusers calls in `shiny_transformer.py` → `mmf_client.py`
11. Remove `torch`, `diffusers`, `transformers` from requirements.txt
12. Run full test suite + manual QA: generate 5 Pokemon across all stages, compare quality
13. Delete `workflows/` and `mcp_templates/` directories
14. Update Pokedex CLAUDE.md to reference `/mmf` skill

---

## Part 5: jinyang Migration

### 5.1 Current Pattern

jinyang uses direct urllib to ComfyUI API for batch generation (>50 jobs). This is intentional - MCP/CLI overhead doesn't make sense at scale.

### 5.2 New Pattern

```
< 50 jobs:  mmf batch queue --manifest jobs.json --template wan26_img2vid
> 50 jobs:  Keep direct urllib (100x cost savings)
            BUT use: mmf templates get wan26_img2vid > /tmp/template.json
            to fetch templates from the hub (single source of truth)
```

### 5.3 Changes Required

Minimal:
- Update jinyang templates to reference `mmf templates get` for fetching latest hub templates
- No code changes needed for direct urllib path
- Add `mmf` to jinyang's PATH

---

## Part 6: MCP Server Deprecation Path

The MCP server (58 tools) stays but becomes "legacy for interactive discovery only."

### 6.1 Immediate Changes
- Update CLAUDE.md: mark MCP tools as legacy, point to `/mmf` skill
- Add note to server.py docstring: "Prefer mmf CLI for all generation tasks"

### 6.2 Future (Optional)
- Reduce MCP tools to essential discovery tools only (~15 tools)
- Remove execution/wait/batch tools from MCP (handled by CLI)
- Keep: search_nodes, get_node_info, list_models, get_template, sota_query

This reduces schema overhead from ~14,500 tokens to ~4,000 tokens for sessions that still need MCP.

---

## Part 7: Verification & Quality Assurance

### 7.1 Automated Tests

| Test | What It Verifies |
|------|-----------------|
| `test_cli.py` (existing 47 tests) | CLI argument parsing, output format, exit codes |
| New: `test_retry_logic.py` | Retry behavior for transient vs permanent errors |
| New: `test_output_streams.py` | stdout/stderr separation, TTY detection |
| New: `test_validation.py` | Input validation (paths, JSON, bounds) |

### 7.2 Integration Tests (Manual)

Run each command and compare output quality:

```bash
# 1. FLUX T2I
mmf run --model flux --type t2i --prompt "photorealistic dragon" --seed 42 --output /tmp/dragon.png

# 2. Qwen T2I
mmf run --model qwen --type t2i --prompt "cyberpunk city" --seed 42 --output /tmp/city.png

# 3. Wan I2V
mmf run --model wan --type i2v --image /tmp/city.png --prompt "gentle motion" --output /tmp/city.mp4

# 4. TeleStyle
mmf telestyle image --content /tmp/city.png --style style_key.png --output /tmp/styled.png

# 5. Face ID
mmf run --template flux2_face_id --params '{"PROMPT":"portrait","FACE_IMAGE":"face.png","SEED":42}' --output /tmp/face.png

# 6. Audio Reactive
mmf run --template ltx2_audio_reactive --params '{"IMAGE_PATH":"img.png","AUDIO_PATH":"beat.wav","PROMPT":"dance","SEED":42}' --output /tmp/audio.mp4

# 7. Video Inpaint
mmf run --template video_inpaint --params '{"VIDEO_PATH":"clip.mp4","SELECT_TEXT":"car","REPLACE_PROMPT":"bicycle","SEED":42}' --output /tmp/inpaint.mp4

# 8. Pipeline: Viral Short
mmf pipeline viral-short --prompt "dancing character" --style-image style.png --seed 42 -o /tmp/viral.mp4

# 9. Batch seeds
mmf batch seeds /tmp/workflow.json --count 4 --start-seed 42

# 10. New templates
mmf run --template qwen_controlnet_bio --params '{"IMAGE_PATH":"ref.png","PROMPT":"creature","SEED":42}' --output /tmp/bio.png
mmf run --template flux_kontext_edit --params '{"IMAGE_PATH":"bio.png","EDIT_PROMPT":"make it gold","SEED":42}' --output /tmp/shiny.png
mmf run --template wan22_s2v --params '{"IMAGE_PATH":"img.png","PROMPT":"running","SEED":42}' --output /tmp/s2v.mp4
```

### 7.3 A/B Quality Comparison

For each migrated path, generate same prompt+seed with OLD method and NEW method:
- Old: KDH ComfyUIExecutor / Pokedex native diffusers
- New: mmf CLI

Compare using `imv` side-by-side. If quality differs, investigate parameter mismatches.

---

## Part 8: Implementation Schedule

### Wave 1: CLI Improvements (1 session)
- [x] Output stream separation (P0)
- [x] Expanded exit codes (P0) - 7 exit codes: EXIT_OK=0 through EXIT_VRAM=7
- [x] Retry logic with --retry flag (P0) - _retry_loop + _classify_error
- [x] --no-wait flag (P1)
- [x] --dry-run flag (P1)
- [x] Input validation (P1) - image path validation, JSON validation, bounds checking
- [x] 6 missing commands (P1) - regenerate, status, progress, search-model, install-model, workflow-lib
- [x] Tests for all new features - 81 CLI tests
- [x] Update /mmf skill with new flags

### Wave 2: New Templates (1 session)
- [x] qwen_controlnet_bio template
- [x] flux_kontext_edit template
- [x] z_turbo_txt2img template
- [x] wan22_s2v template
- [x] Verify flux2_union_controlnet covers FLUX.1 ControlNet bio use case (CONTROL_MODE=0 for Canny, default CONTROL_STRENGTH=0.85)
- [x] Structural validation: all 34 templates pass (load, inject, class_types, _meta). E2E requires live ComfyUI.
- [x] Sync templates to spoke repos (45 files synced: Pokedex 13, KDH 18, Goat 7, Roblox 7)

### Wave 3: KDH Migration (1-2 sessions)
- [x] Create src/core/mmf.js (480 lines, 17 exports) [35d22ba]
- [x] Migrate generateViralKeyframesV4.js [939251d]
- [x] Migrate generateViralVideosV4.js [939251d]
- [x] Migrate generateViralVideosPipelineB.js [939251d]
- [x] Migrate Python scripts (execute_keyframes.py, batch_i2v_choking.py) [939251d]
- [x] Migrate runWanVideoBatch.js [939251d]
- [x] Migrate runChokingABTest.js [a0927a9]
- [x] Migrate runVideoBatchQueue.js [a0927a9]
- [x] Migrate Viral3DMusicVideoPipeline.js (6 executor calls → mmf.js) [186827a]
- [x] Migrate GroupKeyGenerator.js (2 executor calls → mmf.js) [186827a]
- [x] Migrate runViral3DMusicVideo.js (remove HttpAdapter) [186827a]
- [x] Update generateGroupKeys.js hints to mmf CLI [186827a]
- [x] Update KDH CLAUDE.md [a0927a9]
- [x] Migrate BabyCharacterGenerator.js (ComfyUIClient → mmf.js execute()) [64a1891]
- [x] Delete core ComfyUI files (4 tracked: 3,266 lines deleted) [64a1891]
- [x] Update checkBabyGenSetup.js to use mmf stats [64a1891]
- [x] Add execute() to mmf.js for raw workflow stdin pipe [64a1891]
- Note: HttpAdapter, test scripts, mcp_templates/ were untracked (not in git)

### Wave 4: Pokedex Migration (1-2 sessions)
- [x] Create src/adapters/mmf_client.py (435 lines, 8 functions) [12c0802]
- [x] Migrate bio_generator.py (flux2/z-turbo/comfyui/qwen-edit paths via mmf) [4778b8a]
- [x] Migrate video_generator.py (wan_comfyui/comfyui(ltx) paths via mmf) [4778b8a]
- [x] Migrate shiny_transformer.py (kontext/qwen-edit paths via mmf) [4778b8a]
- [x] Migrate batch_generate_videos.py (ComfyUIClient -> mmf_client) [4778b8a]
- [x] Add TODO comments to audio_generator.py (no mmf audio support yet) [4778b8a]
- [x] Update Pokedex CLAUDE.md [5e0e4b4]
- [ ] Remove torch/diffusers/transformers from requirements.txt (flux1 LoRA still needs diffusers)
- [ ] Run full batch test + QA compare (requires live ComfyUI)
- [x] Delete mcp_adapter.py (465 lines, zero imports) [0ba0fb0]
- Note: comfyui_client.py stays (audio_generator.py MMAudio dependency)
- Note: job_registry.py stays (batch_wan_videos.py dependency)
- [ ] Delete workflows/ (when batch_wan_videos.py migrated)
- [ ] Delete mcp_templates/ (when batch_wan_videos.py migrated)

### Wave 5: Cleanup & Documentation (1 session)
- [x] Update main CLAUDE.md (mark MCP as legacy) [59e87f0]
- [x] Update MEMORY.md with migration learnings
- [x] MCP test suite: 321 tests passing
- [x] Update /mmf skill: 4 new templates, execute/wait commands, z_turbo/wan22_s2v constraints
- [ ] jinyang: update template fetch to use `mmf templates get`
- [ ] Optional: Reduce MCP server to ~15 discovery-only tools
- [x] Sync templates to spoke repos (included in Wave 2 sync)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Quality regression | A/B test every migrated path before deleting old code |
| ComfyUI server down | mmf already has --retry; KDH had local template fallback (keep as escape hatch) |
| Template drift | Hub-and-spoke sync + hash-based drift detection already built |
| Performance regression | Profile with `mmf profile <id>` after migration |
| Missing edge cases | Migrate one script at a time, test each before moving on |
| Pokedex LoRA loading | Create FLUX LoRA template or use `flux2_lora_stack` template |

---

## Metrics

**Before migration:**
- KDH: 3,524 lines of ComfyUI code + 17 local templates
- Pokedex: ~4,800 lines of ComfyUI/diffusers code + 13 local templates/workflows
- Total: ~8,324 lines + 30 template files
- Dependencies: torch (2.5GB), diffusers (200MB), transformers (400MB) in Pokedex

**After migration:**
- KDH: ~120 lines (mmf.js wrapper)
- Pokedex: ~100 lines (mmf_client.py wrapper)
- Total: ~220 lines + 0 local templates (all in hub)
- Dependencies: None (subprocess calls to mmf)
- Reduction: **97% fewer lines, 100% fewer local templates, 4GB fewer Python deps**
