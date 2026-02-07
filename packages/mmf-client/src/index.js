/**
 * @romancircus/mmf-client - Thin wrapper around the `mmf` CLI for ComfyUI operations
 *
 * Replaces thousands of lines of ComfyUI integration code with ~200 lines calling the mmf CLI.
 * All retry logic, VRAM recovery, timeout handling, and error categorization are handled by mmf.
 *
 * @module mmf-client
 */

import { execSync } from 'child_process';
import { mmf, esc, buildArgs } from './utils.js';
import { IMAGE_TIMEOUT, VIDEO_TIMEOUT, PIPELINE_TIMEOUT, SYSTEM_TIMEOUT } from './timeouts.js';

// Re-export utilities for advanced use
export { mmf, esc, buildArgs } from './utils.js';
export * from './timeouts.js';
export * from './templates.js';
export * from './errors.js';

// ── Image Generation ────────────────────────────────────────────────────────

/**
 * Qwen-Image-2512 text-to-image generation.
 *
 * CRITICAL: shift=7.0 for sharp output (default 3.1 is blurry).
 *
 * @param {Object} options
 * @param {string} options.prompt - Main prompt describing the image
 * @param {string} [options.negative='blurry, low quality, distorted, watermark'] - Negative prompt
 * @param {number} [options.seed] - Random seed for reproducibility
 * @param {number} [options.width=1664] - Image width in pixels
 * @param {number} [options.height=928] - Image height in pixels
 * @param {number} [options.shift=7.0] - Shift parameter (CRITICAL: keep at 7.0 for sharpness)
 * @param {number} [options.cfg=3.5] - CFG scale
 * @param {number} [options.steps=50] - Number of inference steps
 * @param {string} [options.output] - Local path to save the output
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function qwenTxt2Img({
  prompt,
  negative = 'blurry, low quality, distorted, watermark',
  seed,
  width = 1664,
  height = 928,
  shift = 7.0,
  cfg = 3.5,
  steps = 50,
  output,
}) {
  const params = JSON.stringify({
    PROMPT: prompt,
    NEGATIVE: negative || '',
    SEED: seed,
    WIDTH: width,
    HEIGHT: height,
    SHIFT: shift,
    CFG: cfg,
    STEPS: steps,
  });
  const args = buildArgs([
    `run --template qwen_txt2img`,
    `--params '${esc(params)}'`,
    output ? `--output '${esc(output)}'` : '',
  ]);
  return mmf(args);
}

/**
 * FLUX.2-dev text-to-image generation.
 *
 * @param {Object} options
 * @param {string} options.prompt - Main prompt describing the image
 * @param {number} [options.seed] - Random seed for reproducibility
 * @param {number} [options.width=1024] - Image width in pixels
 * @param {number} [options.height=1024] - Image height in pixels
 * @param {string} [options.output] - Local path to save the output
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function fluxTxt2Img({
  prompt,
  seed,
  width = 1024,
  height = 1024,
  output,
}) {
  const args = buildArgs([
    'run --model flux --type t2i',
    `--prompt '${esc(prompt)}'`,
    `--seed ${seed} --width ${width} --height ${height}`,
    output ? `--output '${esc(output)}'` : '',
  ]);
  return mmf(args);
}

/**
 * FLUX.2 Face ID text-to-image with face reference.
 * Uses IP-Adapter for face identity injection.
 *
 * @param {Object} options
 * @param {string} options.prompt - Main prompt describing the image
 * @param {string} options.faceImage - Path to reference face image
 * @param {number} [options.seed] - Random seed for reproducibility
 * @param {number} [options.faceStrength=0.85] - Face identity strength (0.0-1.0)
 * @param {number} [options.width=1024] - Image width in pixels
 * @param {number} [options.height=1024] - Image height in pixels
 * @param {string} [options.output] - Local path to save the output
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function faceIdTxt2Img({
  prompt,
  faceImage,
  seed,
  faceStrength = 0.85,
  width = 1024,
  height = 1024,
  output,
}) {
  const params = JSON.stringify({
    PROMPT: prompt,
    FACE_IMAGE: faceImage,
    FACE_STRENGTH: faceStrength,
    SEED: seed,
    WIDTH: width,
    HEIGHT: height,
  });
  const args = buildArgs([
    `run --template flux2_face_id`,
    `--params '${esc(params)}'`,
    output ? `--output '${esc(output)}'` : '',
  ]);
  return mmf(args);
}

/**
 * TeleStyle image style transfer.
 *
 * CRITICAL: cfg 2.0-2.5 to avoid color distortion.
 *
 * @param {Object} options
 * @param {string} options.content - Path to content image
 * @param {string} options.style - Path to style reference image
 * @param {number} [options.seed=42] - Random seed
 * @param {number} [options.cfg=2.0] - CFG scale (keep 2.0-2.5 to avoid distortion)
 * @param {number} [options.steps=20] - Number of inference steps
 * @param {string} [options.output] - Local path to save the output
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function teleStyleImage({
  content,
  style,
  seed = 42,
  cfg = 2.0,
  steps = 20,
  output,
}) {
  const params = JSON.stringify({
    CONTENT_IMAGE: content,
    STYLE_IMAGE: style,
    SEED: seed,
    CFG: cfg,
    STEPS: steps,
  });
  const args = buildArgs([
    `run --template telestyle_image`,
    `--params '${esc(params)}'`,
    output ? `--output '${esc(output)}'` : '',
  ]);
  return mmf(args);
}

/**
 * FLUX Kontext Dev - character-consistent image editing.
 * Preserves character identity across scene/pose changes.
 *
 * @param {Object} options
 * @param {string} options.image - Path to source image
 * @param {string} options.editPrompt - Description of desired edit
 * @param {number} [options.denoise=0.65] - Denoising strength
 * @param {number} [options.seed=42] - Random seed
 * @param {number} [options.width=1024] - Output width
 * @param {number} [options.height=1024] - Output height
 * @param {number} [options.steps=20] - Inference steps
 * @param {number} [options.guidance=3.5] - Guidance scale
 * @param {string} [options.output] - Local path to save the output
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function kontextEdit({
  image,
  editPrompt,
  denoise = 0.65,
  seed = 42,
  width = 1024,
  height = 1024,
  steps = 20,
  guidance = 3.5,
  output,
}) {
  const params = JSON.stringify({
    IMAGE_PATH: image,
    EDIT_PROMPT: editPrompt,
    DENOISE: denoise,
    SEED: seed,
    WIDTH: width,
    HEIGHT: height,
    STEPS: steps,
    GUIDANCE: guidance,
  });
  const args = buildArgs([
    `run --template flux_kontext_edit`,
    `--params '${esc(params)}'`,
    output ? `--output '${esc(output)}'` : '',
  ]);
  return mmf(args);
}

// ── Video Generation ────────────────────────────────────────────────────────

/**
 * Wan 2.2 Image-to-Video animation.
 * Animates a keyframe image into video.
 *
 * @param {Object} options
 * @param {string} options.image - Path to keyframe image
 * @param {string} options.prompt - Motion description
 * @param {string} [options.negative='worst quality, blurry, jittery, static, no motion'] - Negative prompt
 * @param {number} [options.frames=81] - Number of frames to generate
 * @param {number} [options.fps=16] - Frames per second
 * @param {number} [options.seed] - Random seed
 * @param {number} [options.cfg=5.0] - CFG scale
 * @param {number} [options.shift=5.0] - Shift parameter
 * @param {number} [options.steps=30] - Inference steps
 * @param {number} [options.noiseAug=0.03] - Noise augmentation
 * @param {number} [options.width=832] - Video width
 * @param {number} [options.height=480] - Video height
 * @param {string} [options.output] - Local path to save the output
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function wanI2V({
  image,
  prompt,
  negative = 'worst quality, blurry, jittery, static, no motion',
  frames = 81,
  fps = 16,
  seed,
  cfg = 5.0,
  shift = 5.0,
  steps = 30,
  noiseAug = 0.03,
  width = 832,
  height = 480,
  output,
}) {
  const params = JSON.stringify({
    IMAGE_PATH: image,
    PROMPT: prompt,
    NEGATIVE: negative || '',
    SEED: seed,
    FRAMES: frames,
    FPS: fps,
    STEPS: steps,
    CFG: cfg,
    SHIFT: shift,
    WIDTH: width,
    HEIGHT: height,
    NOISE_AUG: noiseAug,
    LOAD_DEVICE: 'main_device',
    QUANTIZATION: 'disabled',
    ATTENTION_MODE: 'sdpa',
    FORCE_OFFLOAD_SAMPLER: true,
  });
  const args = buildArgs([
    `run --template wan26_img2vid`,
    `--params '${esc(params)}'`,
    output ? `--output '${esc(output)}'` : '',
  ]);
  return mmf(args, { timeout: VIDEO_TIMEOUT });
}

/**
 * Wan 2.2 Sound-to-Video.
 * Generates video synchronized to audio using wav2vec2 encoder.
 *
 * @param {Object} options
 * @param {string} options.image - Path to keyframe image
 * @param {string} options.prompt - Motion description
 * @param {number} [options.frames=77] - Number of frames
 * @param {number} [options.fps=16] - Frames per second
 * @param {number} [options.seed] - Random seed
 * @param {string} [options.output] - Local path to save the output
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function wanS2V({
  image,
  prompt,
  frames = 77,
  fps = 16,
  seed,
  output,
}) {
  const params = JSON.stringify({
    IMAGE_PATH: image,
    PROMPT: prompt,
    FRAMES: frames,
    FPS: fps,
    SEED: seed,
  });
  const args = buildArgs([
    `run --template wan22_s2v`,
    `--params '${esc(params)}'`,
    output ? `--output '${esc(output)}'` : '',
  ]);
  return mmf(args, { timeout: VIDEO_TIMEOUT });
}

/**
 * Wan 2.2 Animate - character animation with identity preservation.
 * Better identity preservation than standard I2V for character shots.
 *
 * @param {Object} options
 * @param {string} options.image - Path to character reference image
 * @param {string} options.prompt - Motion description
 * @param {string} [options.negative='worst quality, blurry, jittery, distorted face, identity loss'] - Negative prompt
 * @param {number} [options.frames=81] - Number of frames
 * @param {number} [options.fps=16] - Frames per second
 * @param {number} [options.seed] - Random seed
 * @param {number} [options.cfg=5.0] - CFG scale
 * @param {number} [options.shift=5.0] - Shift parameter
 * @param {number} [options.steps=30] - Inference steps
 * @param {number} [options.faceStrength=1.0] - Face identity strength
 * @param {number} [options.poseStrength=1.0] - Pose strength
 * @param {number} [options.width=832] - Video width
 * @param {number} [options.height=480] - Video height
 * @param {string} [options.output] - Local path to save the output
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function wanAnimate({
  image,
  prompt,
  negative = 'worst quality, blurry, jittery, distorted face, identity loss',
  frames = 81,
  fps = 16,
  seed,
  cfg = 5.0,
  shift = 5.0,
  steps = 30,
  faceStrength = 1.0,
  poseStrength = 1.0,
  width = 832,
  height = 480,
  output,
}) {
  const params = JSON.stringify({
    CHARACTER_IMAGE: image,
    PROMPT: prompt,
    NEGATIVE: negative,
    SEED: seed,
    WIDTH: width,
    HEIGHT: height,
    FRAMES: frames,
    FPS: fps,
    STEPS: steps,
    CFG: cfg,
    SHIFT: shift,
    FACE_STRENGTH: faceStrength,
    POSE_STRENGTH: poseStrength,
  });
  const args = buildArgs([
    `run --template wan22_animate`,
    `--params '${esc(params)}'`,
    output ? `--output '${esc(output)}'` : '',
  ]);
  return mmf(args, { timeout: VIDEO_TIMEOUT });
}

/**
 * Phantom multi-subject S2V - 1-4 character references simultaneously.
 * Requires Phantom-Wan-14B model (download separately).
 *
 * @param {Object} options
 * @param {string[]} options.images - Array of 1-4 character reference image paths
 * @param {string} options.prompt - Motion description
 * @param {string} [options.negative='worst quality, blurry, jittery, identity confusion, wrong character'] - Negative prompt
 * @param {number} [options.frames=81] - Number of frames
 * @param {number} [options.fps=16] - Frames per second
 * @param {number} [options.seed] - Random seed
 * @param {number} [options.cfg=5.0] - CFG scale
 * @param {number} [options.width=832] - Video width
 * @param {number} [options.height=480] - Video height
 * @param {string} [options.output] - Local path to save the output
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function phantomS2V({
  images,
  prompt,
  negative = 'worst quality, blurry, jittery, identity confusion, wrong character',
  frames = 81,
  fps = 16,
  seed,
  cfg = 5.0,
  width = 832,
  height = 480,
  output,
}) {
  const params = {
    PROMPT: prompt,
    NEGATIVE: negative,
    SEED: seed,
    WIDTH: width,
    HEIGHT: height,
    FRAMES: frames,
    FPS: fps,
    PHANTOM_CFG: cfg,
  };
  // Map up to 4 character images to phantom slots
  images.forEach((img, i) => {
    params[`CHARACTER_IMAGE_${i + 1}`] = img;
  });
  const args = buildArgs([
    `run --template wan22_phantom`,
    `--params '${esc(JSON.stringify(params))}'`,
    output ? `--output '${esc(output)}'` : '',
  ]);
  return mmf(args, { timeout: VIDEO_TIMEOUT });
}

/**
 * LTX-2 text-to-video generation.
 *
 * CRITICAL: frames must follow 8n+1 rule (9, 17, 25, ..., 97).
 *
 * @param {Object} options
 * @param {string} options.prompt - Motion description
 * @param {string} [options.negative='worst quality, inconsistent motion, blurry, jittery, distorted'] - Negative prompt
 * @param {number} [options.frames=97] - Number of frames (must be 8n+1)
 * @param {number} [options.fps=24] - Frames per second
 * @param {number} [options.seed] - Random seed
 * @param {number} [options.width=832] - Video width
 * @param {number} [options.height=480] - Video height
 * @param {number} [options.steps=30] - Inference steps
 * @param {number} [options.cfg=3.0] - CFG scale
 * @param {string} [options.prefix='ltx2_output'] - Output filename prefix
 * @param {string} [options.output] - Local path to save the output
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function ltxT2V({
  prompt,
  negative = 'worst quality, inconsistent motion, blurry, jittery, distorted',
  frames = 97,
  fps = 24,
  seed,
  width = 832,
  height = 480,
  steps = 30,
  cfg = 3.0,
  prefix = 'ltx2_output',
  output,
}) {
  const params = JSON.stringify({
    PROMPT: prompt,
    NEGATIVE: negative || '',
    SEED: seed,
    FRAMES: frames,
    FPS: fps,
    WIDTH: width,
    HEIGHT: height,
    STEPS: steps,
    CFG: cfg,
    PREFIX: prefix,
  });
  const args = buildArgs([
    `run --template ltx2_txt2vid`,
    `--params '${esc(params)}'`,
    output ? `--output '${esc(output)}'` : '',
  ]);
  return mmf(args, { timeout: VIDEO_TIMEOUT });
}

/**
 * LTX-2 image-to-video generation.
 * Animates a keyframe using LTX-2 19B AV model.
 *
 * @param {Object} options
 * @param {string} options.image - Path to keyframe image
 * @param {string} options.prompt - Motion description
 * @param {string} [options.negative='worst quality, inconsistent motion, blurry, jittery, distorted'] - Negative prompt
 * @param {number} [options.frames=97] - Number of frames
 * @param {number} [options.fps=24] - Frames per second
 * @param {number} [options.seed] - Random seed
 * @param {number} [options.width=768] - Video width
 * @param {number} [options.height=512] - Video height
 * @param {number} [options.strength=0.85] - Animation strength
 * @param {number} [options.steps=30] - Inference steps
 * @param {number} [options.cfg=3.0] - CFG scale
 * @param {number} [options.crf=38] - Video compression quality
 * @param {number} [options.blurRadius=1] - Blur radius for preprocessing
 * @param {string} [options.prefix='ltx2_i2v'] - Output filename prefix
 * @param {string} [options.output] - Local path to save the output
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function ltxI2V({
  image,
  prompt,
  negative = 'worst quality, inconsistent motion, blurry, jittery, distorted',
  frames = 97,
  fps = 24,
  seed,
  width = 768,
  height = 512,
  strength = 0.85,
  steps = 30,
  cfg = 3.0,
  crf = 38,
  blurRadius = 1,
  prefix = 'ltx2_i2v',
  output,
}) {
  const params = JSON.stringify({
    IMAGE_PATH: image,
    PROMPT: prompt,
    NEGATIVE: negative || '',
    SEED: seed,
    FRAMES: frames,
    FPS: fps,
    WIDTH: width,
    HEIGHT: height,
    STRENGTH: strength,
    STEPS: steps,
    CFG: cfg,
    CRF: crf,
    BLUR_RADIUS: blurRadius,
    PREFIX: prefix,
  });
  const args = buildArgs([
    `run --template ltx2_img2vid`,
    `--params '${esc(params)}'`,
    output ? `--output '${esc(output)}'` : '',
  ]);
  return mmf(args, { timeout: VIDEO_TIMEOUT });
}

/**
 * LTX-2 audio-reactive image-to-video.
 * Video synchronized to audio beats via LTX-2's native audio encoder.
 *
 * @param {Object} options
 * @param {string} options.image - Path to keyframe image
 * @param {string} options.audioPath - Path to audio file
 * @param {string} options.prompt - Motion description
 * @param {number} [options.frames=121] - Number of frames
 * @param {number} [options.fps=24] - Frames per second
 * @param {number} [options.seed] - Random seed
 * @param {string} [options.output] - Local path to save the output
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function audioReactiveI2V({
  image,
  audioPath,
  prompt,
  frames = 121,
  fps = 24,
  seed,
  output,
}) {
  const params = JSON.stringify({
    IMAGE_PATH: image,
    AUDIO_PATH: audioPath,
    PROMPT: prompt,
    FRAMES: frames,
    FPS: fps,
    SEED: seed,
  });
  const args = buildArgs([
    `run --template ltx2_audio_reactive`,
    `--params '${esc(params)}'`,
    output ? `--output '${esc(output)}'` : '',
  ]);
  return mmf(args, { timeout: VIDEO_TIMEOUT });
}

/**
 * TeleStyle video style transfer.
 * Applies style with temporal consistency across all frames.
 * Slower (~10 min) but preserves motion coherence.
 *
 * @param {Object} options
 * @param {string} options.video - Path to input video
 * @param {string} options.style - Path to style reference image
 * @param {number} [options.seed=42] - Random seed
 * @param {number} [options.cfg=1.0] - CFG scale
 * @param {number} [options.steps=12] - Inference steps
 * @param {number} [options.fps=24] - Target FPS
 * @param {string} [options.output] - Local path to save the output
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function teleStyleVideo({
  video,
  style,
  seed = 42,
  cfg = 1.0,
  steps = 12,
  fps = 24,
  output,
}) {
  const params = JSON.stringify({
    VIDEO_PATH: video,
    STYLE_PATH: style,
    SEED: seed,
    CFG: cfg,
    STEPS: steps,
    FPS: fps,
  });
  const args = buildArgs([
    `run --template telestyle_video`,
    `--params '${esc(params)}'`,
    output ? `--output '${esc(output)}'` : '',
  ]);
  return mmf(args, { timeout: VIDEO_TIMEOUT });
}

/**
 * Video inpainting via CLIPSeg text-based mask + FLUX inpaint.
 *
 * @param {Object} options
 * @param {string} options.video - Path to input video
 * @param {string} options.selectText - Text description of what to select/mask
 * @param {string} options.replacePrompt - Replacement content description
 * @param {number} [options.denoise=0.7] - Denoising strength
 * @param {number} [options.seed] - Random seed
 * @param {string} [options.output] - Local path to save the output
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function videoInpaint({
  video,
  selectText,
  replacePrompt,
  denoise = 0.7,
  seed,
  output,
}) {
  const params = JSON.stringify({
    VIDEO_PATH: video,
    SELECT_TEXT: selectText,
    REPLACE_PROMPT: replacePrompt,
    DENOISE: denoise,
    SEED: seed,
  });
  const args = buildArgs([
    `run --template video_inpaint`,
    `--params '${esc(params)}'`,
    output ? `--output '${esc(output)}'` : '',
  ]);
  return mmf(args, { timeout: VIDEO_TIMEOUT });
}

// ── Pipelines ────────────────────────────────────────────────────────────────

/**
 * Full viral short pipeline (image -> style -> animate -> speedup).
 *
 * @param {Object} options
 * @param {string} options.prompt - Image generation prompt
 * @param {string} options.styleImage - Path to style reference image
 * @param {number} [options.seed] - Random seed
 * @param {string} [options.output] - Local path to save the final video
 * @returns {Promise<Object>} Result with asset_id or error object
 */
export function viralShort({
  prompt,
  styleImage,
  seed,
  output,
}) {
  const args = buildArgs([
    `pipeline viral-short`,
    `--prompt '${esc(prompt)}'`,
    `--style-image '${esc(styleImage)}'`,
    `--seed ${seed}`,
    output ? `-o '${esc(output)}'` : '',
  ]);
  return mmf(args, { timeout: PIPELINE_TIMEOUT });
}

// ── System Operations ────────────────────────────────────────────────────────

/**
 * Free GPU memory.
 *
 * @param {boolean} [unload=true] - Unload all models from VRAM
 * @returns {Promise<Object>} Success object or error
 */
export function freeMemory(unload = true) {
  return mmf(`free${unload ? ' --unload' : ''}`, {
    timeout: SYSTEM_TIMEOUT,
    noRetry: true,
  });
}

/**
 * Interrupt the currently running workflow.
 *
 * @returns {Promise<Object>} Success object or error
 */
export function interrupt() {
  return mmf('interrupt', {
    timeout: SYSTEM_TIMEOUT,
    noRetry: true,
  });
}

/**
 * Get GPU VRAM and system stats.
 *
 * @returns {Promise<Object>} System stats or error
 */
export function stats() {
  return mmf('stats', {
    timeout: SYSTEM_TIMEOUT,
    noRetry: true,
  });
}

/**
 * Upload an image to ComfyUI for use in workflows.
 *
 * @param {string} path - Local path to the image
 * @returns {Promise<Object>} Upload result { name, subfolder, type } or error
 */
export function upload(path) {
  return mmf(`upload '${esc(path)}'`, {
    timeout: SYSTEM_TIMEOUT,
    noRetry: true,
  });
}

/**
 * Download an asset from ComfyUI to a local path.
 *
 * @param {string} assetId - Asset ID from a generation result
 * @param {string} path - Local path to save the file
 * @returns {Promise<Object>} { path } or error
 */
export function download(assetId, path) {
  return mmf(`download ${esc(assetId)} '${esc(path)}'`, {
    timeout: SYSTEM_TIMEOUT,
    noRetry: true,
  });
}

// ── Utilities ────────────────────────────────────────────────────────────────

/**
 * Execute a raw ComfyUI workflow JSON via mmf CLI.
 * Used for custom workflows that don't have a pre-built mmf command.
 *
 * @param {Object} workflowJson - ComfyUI API workflow JSON
 * @param {Object} [options] - Options
 * @param {number} [options.timeout] - Timeout in ms (default: IMAGE_TIMEOUT)
 * @param {string} [options.output] - Download result to this local path
 * @returns {Promise<Object>} Execution result or error
 */
export function execute(workflowJson, options = {}) {
  const timeout = options.timeout || IMAGE_TIMEOUT;
  const args = buildArgs([
    'execute -',
    options.output ? `--output '${esc(options.output)}'` : '',
  ]);
  const cmd = `mmf ${args} --retry 3 --retry-on vram,timeout,connection`;

  try {
    const stdout = execSync(cmd, {
      encoding: 'utf8',
      timeout,
      input: JSON.stringify(workflowJson),
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    const trimmed = stdout.trim();
    if (!trimmed) return { success: true };
    return JSON.parse(trimmed);
  } catch (err) {
    const stderr = err.stderr?.toString().trim() || '';
    const stdout = err.stdout?.toString().trim() || '';
    if (stdout) {
      try {
        return JSON.parse(stdout);
      } catch {
        /* not JSON */
      }
    }
    return {
      error: stderr || err.message || `mmf exited with code ${err.status}`,
      code: err.status,
    };
  }
}

/**
 * Resize image using sharp (CPU, no GPU needed).
 * Used in pipeline: Qwen@1664x928 -> TeleStyle@1664x928 -> resize@832x480 -> Wan I2V.
 *
 * @param {string} path - Source image path
 * @param {number} [width=832] - Target width
 * @param {number} [height=480] - Target height
 * @returns {Promise<{path: string}>} Resized image path
 */
export async function resizeImage(path, width = 832, height = 480) {
  const { default: sharp } = await import('sharp');
  const output = path.replace(/\.([^.]+)$/, `_resized.${width}x${height}.$1`);
  await sharp(path).resize(width, height, { fit: 'fill' }).toFile(output);
  return { path: output };
}

// ── Default Export ─────────────────────────────────────────────────────────

export default {
  // Image generation
  qwenTxt2Img,
  fluxTxt2Img,
  faceIdTxt2Img,
  teleStyleImage,
  kontextEdit,
  // Video generation
  wanI2V,
  wanS2V,
  wanAnimate,
  phantomS2V,
  ltxT2V,
  ltxI2V,
  audioReactiveI2V,
  teleStyleVideo,
  videoInpaint,
  // Pipelines
  viralShort,
  // System
  freeMemory,
  interrupt,
  stats,
  upload,
  download,
  // Utilities
  execute,
  resizeImage,
};
