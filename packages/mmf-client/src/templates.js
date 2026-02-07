/**
 * Template name constants for mmf workflows
 *
 * These constants map to the actual template files in the mmf CLI.
 * Use these to ensure you're using the correct template names.
 *
 * @module templates
 */

// ── Image Generation Templates ─────────────────────────────────────────────

/** Qwen-Image-2512 text-to-image template */
export const TEMPLATE_QWEN_TXT2IMG = 'qwen_txt2img';

/** FLUX.2 Face ID with IP-Adapter template */
export const TEMPLATE_FLUX2_FACE_ID = 'flux2_face_id';

/** FLUX Kontext Dev character-consistent editing template */
export const TEMPLATE_FLUX_KONTEXT_EDIT = 'flux_kontext_edit';

// ── Video Generation Templates ────────────────────────────────────────────

/** Wan 2.6 Image-to-Video template */
export const TEMPLATE_WAN26_IMG2VID = 'wan26_img2vid';

/** Wan 2.2 Sound-to-Video (wav2vec2 audio sync) template */
export const TEMPLATE_WAN22_S2V = 'wan22_s2v';

/** Wan 2.2 Character animation with identity preservation */
export const TEMPLATE_WAN22_ANIMATE = 'wan22_animate';

/** Wan 2.2 Phantom multi-subject character animation (1-4 references) */
export const TEMPLATE_WAN22_PHANTOM = 'wan22_phantom';

/** LTX-2 text-to-video template */
export const TEMPLATE_LTX2_TXT2VID = 'ltx2_txt2vid';

/** LTX-2 image-to-video template */
export const TEMPLATE_LTX2_IMG2VID = 'ltx2_img2vid';

/** LTX-2 audio-reactive image-to-video template */
export const TEMPLATE_LTX2_AUDIO_REACTIVE = 'ltx2_audio_reactive';

// ── Style Transfer Templates ───────────────────────────────────────────────

/** TeleStyle image style transfer template */
export const TEMPLATE_TELESTYLE_IMAGE = 'telestyle_image';

/** TeleStyle video style transfer template (slower, temporal consistency) */
export const TEMPLATE_TELESTYLE_VIDEO = 'telestyle_video';

// ── Inpainting & Editing Templates ─────────────────────────────────────────

/** Video inpainting via CLIPSeg mask + FLUX inpaint */
export const TEMPLATE_VIDEO_INPAINT = 'video_inpaint';

// ── Pipeline Templates ─────────────────────────────────────────────────────

/** Viral short pipeline: image -> style -> animate -> speedup */
export const PIPELINE_VIRAL_SHORT = 'viral-short';

// Backward compatibility - also export as default object
export default {
  TEMPLATE_QWEN_TXT2IMG,
  TEMPLATE_FLUX2_FACE_ID,
  TEMPLATE_FLUX_KONTEXT_EDIT,
  TEMPLATE_WAN26_IMG2VID,
  TEMPLATE_WAN22_S2V,
  TEMPLATE_WAN22_ANIMATE,
  TEMPLATE_WAN22_PHANTOM,
  TEMPLATE_LTX2_TXT2VID,
  TEMPLATE_LTX2_IMG2VID,
  TEMPLATE_LTX2_AUDIO_REACTIVE,
  TEMPLATE_TELESTYLE_IMAGE,
  TEMPLATE_TELESTYLE_VIDEO,
  TEMPLATE_VIDEO_INPAINT,
  PIPELINE_VIRAL_SHORT,
};
