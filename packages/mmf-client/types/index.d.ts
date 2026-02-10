/**
 * TypeScript definitions for @romancircus/mmf-client
 */

// ── Timeout Constants ──────────────────────────────────────────────────────

export declare const IMAGE_TIMEOUT: number;
export declare const VIDEO_TIMEOUT: number;
export declare const PIPELINE_TIMEOUT: number;
export declare const SYSTEM_TIMEOUT: number;
export declare const DEFAULT_RETRY: number;
export declare const DEFAULT_RETRY_FLAGS: string;

// ── Template Constants ─────────────────────────────────────────────────────

export declare const TEMPLATE_QWEN_TXT2IMG: string;
export declare const TEMPLATE_FLUX2_FACE_ID: string;
export declare const TEMPLATE_FLUX_KONTEXT_EDIT: string;
export declare const TEMPLATE_WAN21_IMG2VID: string;
export declare const TEMPLATE_WAN22_S2V: string;
export declare const TEMPLATE_WAN22_ANIMATE: string;
export declare const TEMPLATE_WAN22_PHANTOM: string;
export declare const TEMPLATE_LTX2_TXT2VID: string;
export declare const TEMPLATE_LTX2_IMG2VID: string;
export declare const TEMPLATE_LTX2_AUDIO_REACTIVE: string;
export declare const TEMPLATE_TELESTYLE_IMAGE: string;
export declare const TEMPLATE_TELESTYLE_VIDEO: string;
export declare const TEMPLATE_VIDEO_INPAINT: string;
export declare const PIPELINE_VIRAL_SHORT: string;

// ── Error Handling ───────────────────────────────────────────────────────────

export declare const ErrorCodes: {
  SUCCESS: 0;
  ERROR: 1;
  TIMEOUT: 2;
  VALIDATION: 3;
  VRAM: 4;
  CONNECTION: 5;
  UNKNOWN: 99;
};

export declare function hasError(result: unknown): boolean;
export declare function getErrorMessage(result: unknown, defaultMsg?: string): string;
export declare function getErrorCode(result: unknown): number;
export declare function isRetryableError(result: unknown): boolean;
export declare function formatError(result: unknown): string;
export declare function assertSuccess(result: unknown, context?: string): void;

// ── Options Interfaces ─────────────────────────────────────────────────────

export interface QwenTxt2ImgOptions {
  prompt: string;
  negative?: string;
  seed?: number;
  width?: number;
  height?: number;
  shift?: number;
  cfg?: number;
  steps?: number;
  output?: string;
}

export interface FluxTxt2ImgOptions {
  prompt: string;
  seed?: number;
  width?: number;
  height?: number;
  output?: string;
}

export interface FaceIdTxt2ImgOptions {
  prompt: string;
  faceImage: string;
  seed?: number;
  faceStrength?: number;
  width?: number;
  height?: number;
  output?: string;
}

export interface TeleStyleImageOptions {
  content: string;
  style: string;
  seed?: number;
  cfg?: number;
  steps?: number;
  output?: string;
}

export interface KontextEditOptions {
  image: string;
  editPrompt: string;
  denoise?: number;
  seed?: number;
  width?: number;
  height?: number;
  steps?: number;
  guidance?: number;
  output?: string;
}

export interface WanI2VOptions {
  image: string;
  prompt: string;
  negative?: string;
  frames?: number;
  fps?: number;
  seed?: number;
  cfg?: number;
  shift?: number;
  steps?: number;
  noiseAug?: number;
  width?: number;
  height?: number;
  output?: string;
}

export interface WanS2VOptions {
  image: string;
  prompt: string;
  frames?: number;
  fps?: number;
  seed?: number;
  output?: string;
}

export interface WanAnimateOptions {
  image: string;
  prompt: string;
  negative?: string;
  frames?: number;
  fps?: number;
  seed?: number;
  cfg?: number;
  shift?: number;
  steps?: number;
  faceStrength?: number;
  poseStrength?: number;
  width?: number;
  height?: number;
  output?: string;
}

export interface PhantomS2VOptions {
  images: string[];
  prompt: string;
  negative?: string;
  frames?: number;
  fps?: number;
  seed?: number;
  cfg?: number;
  width?: number;
  height?: number;
  output?: string;
}

export interface LtxT2VOptions {
  prompt: string;
  negative?: string;
  frames?: number;
  fps?: number;
  seed?: number;
  width?: number;
  height?: number;
  steps?: number;
  cfg?: number;
  prefix?: string;
  output?: string;
}

export interface LtxI2VOptions {
  image: string;
  prompt: string;
  negative?: string;
  frames?: number;
  fps?: number;
  seed?: number;
  width?: number;
  height?: number;
  strength?: number;
  steps?: number;
  cfg?: number;
  crf?: number;
  blurRadius?: number;
  prefix?: string;
  output?: string;
}

export interface AudioReactiveI2VOptions {
  image: string;
  audioPath: string;
  prompt: string;
  frames?: number;
  fps?: number;
  seed?: number;
  output?: string;
}

export interface TeleStyleVideoOptions {
  video: string;
  style: string;
  seed?: number;
  cfg?: number;
  steps?: number;
  fps?: number;
  output?: string;
}

export interface VideoInpaintOptions {
  video: string;
  selectText: string;
  replacePrompt: string;
  denoise?: number;
  seed?: number;
  output?: string;
}

export interface ViralShortOptions {
  prompt: string;
  styleImage: string;
  seed?: number;
  output?: string;
}

export interface ExecuteOptions {
  timeout?: number;
  output?: string;
}

// ── Result Types ─────────────────────────────────────────────────────────────

export interface SuccessResult {
  asset_id: string;
  [key: string]: unknown;
}

export interface ErrorResult {
  error: string;
  code?: number;
}

export type GenerationResult = SuccessResult | ErrorResult;

// ── Image Generation Functions ───────────────────────────────────────────────

export declare function qwenTxt2Img(options: QwenTxt2ImgOptions): Promise<GenerationResult>;
export declare function fluxTxt2Img(options: FluxTxt2ImgOptions): Promise<GenerationResult>;
export declare function faceIdTxt2Img(options: FaceIdTxt2ImgOptions): Promise<GenerationResult>;
export declare function teleStyleImage(options: TeleStyleImageOptions): Promise<GenerationResult>;
export declare function kontextEdit(options: KontextEditOptions): Promise<GenerationResult>;

// ── Video Generation Functions ───────────────────────────────────────────────

export declare function wanI2V(options: WanI2VOptions): Promise<GenerationResult>;
export declare function wanS2V(options: WanS2VOptions): Promise<GenerationResult>;
export declare function wanAnimate(options: WanAnimateOptions): Promise<GenerationResult>;
export declare function phantomS2V(options: PhantomS2VOptions): Promise<GenerationResult>;
export declare function ltxT2V(options: LtxT2VOptions): Promise<GenerationResult>;
export declare function ltxI2V(options: LtxI2VOptions): Promise<GenerationResult>;
export declare function audioReactiveI2V(options: AudioReactiveI2VOptions): Promise<GenerationResult>;
export declare function teleStyleVideo(options: TeleStyleVideoOptions): Promise<GenerationResult>;
export declare function videoInpaint(options: VideoInpaintOptions): Promise<GenerationResult>;

// ── Pipeline Functions ───────────────────────────────────────────────────────

export declare function viralShort(options: ViralShortOptions): Promise<GenerationResult>;

// ── System Functions ─────────────────────────────────────────────────────────

export declare function freeMemory(unload?: boolean): Promise<GenerationResult>;
export declare function interrupt(): Promise<GenerationResult>;
export declare function stats(): Promise<GenerationResult>;
export declare function upload(path: string): Promise<GenerationResult>;
export declare function download(assetId: string, path: string): Promise<GenerationResult>;

// ── Utility Functions ──────────────────────────────────────────────────────

export declare function execute(workflowJson: object, options?: ExecuteOptions): Promise<GenerationResult>;
export declare function resizeImage(path: string, width?: number, height?: number): Promise<{ path: string }>;

// ── Utility Exports ──────────────────────────────────────────────────────────

export declare function mmf(args: string, opts?: { timeout?: number; retry?: number; noRetry?: boolean; input?: string }): Promise<GenerationResult>;
export declare function esc(str: string | null | undefined): string;
export declare function buildArgs(parts: Array<string | null | undefined>): string;

// ── Default Export ──────────────────────────────────────────────────────────

declare const _default: {
  qwenTxt2Img: typeof qwenTxt2Img;
  fluxTxt2Img: typeof fluxTxt2Img;
  faceIdTxt2Img: typeof faceIdTxt2Img;
  teleStyleImage: typeof teleStyleImage;
  kontextEdit: typeof kontextEdit;
  wanI2V: typeof wanI2V;
  wanS2V: typeof wanS2V;
  wanAnimate: typeof wanAnimate;
  phantomS2V: typeof phantomS2V;
  ltxT2V: typeof ltxT2V;
  ltxI2V: typeof ltxI2V;
  audioReactiveI2V: typeof audioReactiveI2V;
  teleStyleVideo: typeof teleStyleVideo;
  videoInpaint: typeof videoInpaint;
  viralShort: typeof viralShort;
  freeMemory: typeof freeMemory;
  interrupt: typeof interrupt;
  stats: typeof stats;
  upload: typeof upload;
  download: typeof download;
  execute: typeof execute;
  resizeImage: typeof resizeImage;
};

export default _default;
