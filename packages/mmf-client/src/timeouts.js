/**
 * Timeout constants for mmf operations
 *
 * @module timeouts
 */

/** 11 minutes for image generation (10 min generation + 1 min buffer) */
export const IMAGE_TIMEOUT = 660_000;

/** 15 minutes for video generation operations */
export const VIDEO_TIMEOUT = 900_000;

/** 30 minutes for multi-stage pipelines */
export const PIPELINE_TIMEOUT = 1_800_000;

/** 30 seconds for system commands (stats, interrupt, free memory) */
export const SYSTEM_TIMEOUT = 30_000;

/** Default retry count for transient failures */
export const DEFAULT_RETRY = 3;

/** Retry flags for common failure modes */
export const DEFAULT_RETRY_FLAGS = '--retry 3 --retry-on vram,timeout,connection';

// Backward compatibility - also export as default object
export default {
  IMAGE_TIMEOUT,
  VIDEO_TIMEOUT,
  PIPELINE_TIMEOUT,
  SYSTEM_TIMEOUT,
  DEFAULT_RETRY,
  DEFAULT_RETRY_FLAGS,
};
