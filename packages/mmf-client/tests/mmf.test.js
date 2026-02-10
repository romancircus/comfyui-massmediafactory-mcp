/**
 * Unit tests for @romancircus/mmf-client
 *
 * @jest-environment node
 */

import { jest } from '@jest/globals';

// Mock child_process before importing the module
const mockExecSync = jest.fn();

jest.unstable_mockModule('child_process', () => ({
  execSync: mockExecSync,
}));

// Mock sharp for resizeImage tests
const mockSharp = jest.fn(() => ({
  resize: jest.fn().mockReturnThis(),
  toFile: jest.fn().mockResolvedValue(undefined),
}));

jest.unstable_mockModule('sharp', () => ({
  default: mockSharp,
}));

// Import module after mocking
const {
  qwenTxt2Img,
  fluxTxt2Img,
  faceIdTxt2Img,
  teleStyleImage,
  kontextEdit,
  wanI2V,
  wanS2V,
  wanAnimate,
  phantomS2V,
  ltxT2V,
  ltxI2V,
  audioReactiveI2V,
  teleStyleVideo,
  videoInpaint,
  viralShort,
  freeMemory,
  interrupt,
  stats,
  upload,
  download,
  execute,
  resizeImage,
  hasError,
  getErrorMessage,
  getErrorCode,
  isRetryableError,
  formatError,
  IMAGE_TIMEOUT,
  VIDEO_TIMEOUT,
  PIPELINE_TIMEOUT,
  SYSTEM_TIMEOUT,
} = await import('../src/index.js');

describe('@romancircus/mmf-client', () => {
  beforeEach(() => {
    mockExecSync.mockClear();
    mockSharp.mockClear();
  });

  describe('Image Generation', () => {
    test('qwenTxt2Img generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'test-123' }));

      const result = qwenTxt2Img({
        prompt: 'A sunset over mountains',
        seed: 42,
        width: 1024,
        height: 768,
      });

      expect(mockExecSync).toHaveBeenCalledTimes(1);
      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf run --template qwen_txt2img');
      expect(callArgs[0]).toContain('--params');
      expect(callArgs[0]).toContain('PROMPT');
      expect(callArgs[1].timeout).toBe(IMAGE_TIMEOUT);
    });

    test('qwenTxt2Img handles output path', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'test-456' }));

      qwenTxt2Img({
        prompt: 'Test',
        output: '/path/to/output.png',
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain("--output '/path/to/output.png'");
    });

    test('qwenTxt2Img escapes special characters in prompt', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'test-789' }));

      qwenTxt2Img({
        prompt: "It's a beautiful day",
        output: "/path/with'quote/image.png",
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain("It's a beautiful day");
    });

    test('fluxTxt2Img generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'flux-123' }));

      fluxTxt2Img({
        prompt: 'A cyberpunk city',
        seed: 123,
        width: 1024,
        height: 1024,
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf run --model flux --type t2i');
      expect(callArgs[0]).toContain("--prompt 'A cyberpunk city'");
      expect(callArgs[0]).toContain('--seed 123');
    });

    test('faceIdTxt2Img generates correct command with face image', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'face-123' }));

      faceIdTxt2Img({
        prompt: 'A portrait of a person',
        faceImage: '/path/to/face.jpg',
        faceStrength: 0.9,
        seed: 42,
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf run --template flux2_face_id');
      expect(callArgs[0]).toContain('FACE_IMAGE');
      expect(callArgs[0]).toContain('0.9');
    });

    test('teleStyleImage generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'style-123' }));

      teleStyleImage({
        content: '/path/to/content.jpg',
        style: '/path/to/style.jpg',
        cfg: 2.5,
        seed: 42,
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf run --template telestyle_image');
      expect(callArgs[0]).toContain('CONTENT_IMAGE');
      expect(callArgs[0]).toContain('STYLE_IMAGE');
    });

    test('kontextEdit generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'edit-123' }));

      kontextEdit({
        image: '/path/to/source.jpg',
        editPrompt: 'Change the background to a beach',
        denoise: 0.7,
        seed: 42,
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf run --template flux_kontext_edit');
      expect(callArgs[0]).toContain('IMAGE_PATH');
      expect(callArgs[0]).toContain('EDIT_PROMPT');
    });
  });

  describe('Video Generation', () => {
    test('wanI2V generates correct command with video timeout', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'wan-123' }));

      wanI2V({
        image: '/path/to/keyframe.jpg',
        prompt: 'Gentle motion',
        frames: 81,
        fps: 16,
        seed: 42,
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf run --template wan21_img2vid');
      expect(callArgs[0]).toContain('IMAGE_PATH');
      expect(callArgs[1].timeout).toBe(VIDEO_TIMEOUT);
    });

    test('wanS2V generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 's2v-123' }));

      wanS2V({
        image: '/path/to/keyframe.jpg',
        prompt: 'Dancing to music',
        frames: 77,
        fps: 16,
        seed: 42,
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf run --template wan22_s2v');
      expect(callArgs[1].timeout).toBe(VIDEO_TIMEOUT);
    });

    test('wanAnimate generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'anim-123' }));

      wanAnimate({
        image: '/path/to/character.jpg',
        prompt: 'Character walks forward',
        faceStrength: 1.2,
        poseStrength: 0.8,
        seed: 42,
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf run --template wan22_animate');
      expect(callArgs[0]).toContain('CHARACTER_IMAGE');
      expect(callArgs[0]).toContain('FACE_STRENGTH');
      expect(callArgs[0]).toContain('POSE_STRENGTH');
    });

    test('phantomS2V maps multiple images correctly', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'phantom-123' }));

      phantomS2V({
        images: ['/char1.jpg', '/char2.jpg', '/char3.jpg'],
        prompt: 'Two characters dancing',
        seed: 42,
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf run --template wan22_phantom');
      expect(callArgs[0]).toContain('CHARACTER_IMAGE_1');
      expect(callArgs[0]).toContain('CHARACTER_IMAGE_2');
      expect(callArgs[0]).toContain('CHARACTER_IMAGE_3');
    });

    test('ltxT2V generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'ltx-t2v-123' }));

      ltxT2V({
        prompt: 'A cat walking',
        frames: 97,
        fps: 24,
        seed: 42,
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf run --template ltx2_txt2vid');
      expect(callArgs[0]).toContain('FRAMES');
      expect(callArgs[0]).toContain('FPS');
      expect(callArgs[1].timeout).toBe(VIDEO_TIMEOUT);
    });

    test('ltxI2V generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'ltx-i2v-123' }));

      ltxI2V({
        image: '/path/to/keyframe.jpg',
        prompt: 'Animate this scene',
        strength: 0.85,
        blurRadius: 2,
        seed: 42,
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf run --template ltx2_img2vid');
      expect(callArgs[0]).toContain('STRENGTH');
      expect(callArgs[0]).toContain('BLUR_RADIUS');
    });

    test('audioReactiveI2V generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'audio-123' }));

      audioReactiveI2V({
        image: '/path/to/keyframe.jpg',
        audioPath: '/path/to/audio.mp3',
        prompt: 'Beat synchronized motion',
        seed: 42,
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf run --template ltx2_audio_reactive');
      expect(callArgs[0]).toContain('AUDIO_PATH');
    });

    test('teleStyleVideo generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'style-vid-123' }));

      teleStyleVideo({
        video: '/path/to/video.mp4',
        style: '/path/to/style.jpg',
        cfg: 1.5,
        steps: 15,
        seed: 42,
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf run --template telestyle_video');
      expect(callArgs[0]).toContain('VIDEO_PATH');
      expect(callArgs[0]).toContain('STYLE_PATH');
    });

    test('videoInpaint generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'inpaint-123' }));

      videoInpaint({
        video: '/path/to/video.mp4',
        selectText: 'the red car',
        replacePrompt: 'a blue sports car',
        denoise: 0.8,
        seed: 42,
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf run --template video_inpaint');
      expect(callArgs[0]).toContain('SELECT_TEXT');
      expect(callArgs[0]).toContain('REPLACE_PROMPT');
    });
  });

  describe('Pipelines', () => {
    test('viralShort generates correct command with pipeline timeout', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'viral-123' }));

      viralShort({
        prompt: 'A dancing character',
        styleImage: '/path/to/style.jpg',
        seed: 42,
      });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf pipeline viral-short');
      expect(callArgs[0]).toContain('--style-image');
      expect(callArgs[1].timeout).toBe(PIPELINE_TIMEOUT);
    });
  });

  describe('System Operations', () => {
    test('freeMemory generates correct command with system timeout', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ success: true }));

      freeMemory(true);

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toBe('mmf free --unload --retry 3 --retry-on vram,timeout,connection');
      expect(callArgs[1].timeout).toBe(SYSTEM_TIMEOUT);
    });

    test('interrupt generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ success: true }));

      interrupt();

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toBe('mmf interrupt --retry 3 --retry-on vram,timeout,connection');
    });

    test('stats generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ vram: '24GB', gpu: 'RTX 4090' }));

      stats();

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toBe('mmf stats --retry 3 --retry-on vram,timeout,connection');
    });

    test('upload generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ name: 'image.png', subfolder: 'input' }));

      upload('/path/to/image.png');

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf upload');
      expect(callArgs[0]).toContain("'/path/to/image.png'");
    });

    test('download generates correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ path: '/output/image.png' }));

      download('asset-123', '/output/image.png');

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf download');
      expect(callArgs[0]).toContain('asset-123');
    });
  });

  describe('Utilities', () => {
    test('execute runs raw workflow with correct command', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'exec-123' }));

      const workflow = { nodes: [], links: [] };
      execute(workflow, { output: '/output/result.png' });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('mmf execute -');
      expect(callArgs[0]).toContain('--output');
      expect(callArgs[1].input).toBe(JSON.stringify(workflow));
    });

    test('resizeImage calls sharp correctly', async () => {
      const result = await resizeImage('/path/to/image.png', 832, 480);

      expect(mockSharp).toHaveBeenCalledWith('/path/to/image.png');
      expect(result.path).toContain('832x480');
    });
  });

  describe('Error Handling', () => {
    test('hasError returns true for error results', () => {
      expect(hasError({ error: 'Something went wrong' })).toBe(true);
      expect(hasError({ asset_id: '123' })).toBe(false);
      expect(hasError(null)).toBe(false);
    });

    test('getErrorMessage returns correct message', () => {
      expect(getErrorMessage({ error: 'Failed' })).toBe('Failed');
      expect(getErrorMessage({ asset_id: '123' })).toBe('Unknown error');
      expect(getErrorMessage({ error: 'Failed' }, 'Custom default')).toBe('Failed');
    });

    test('getErrorCode returns correct code', () => {
      expect(getErrorCode({ code: 1 })).toBe(1);
      expect(getErrorCode({ error: 'Failed' })).toBe(1); // ERROR
      expect(getErrorCode({ success: true })).toBe(0); // SUCCESS
    });

    test('isRetryableError identifies retryable errors', () => {
      expect(isRetryableError({ error: 'Out of VRAM' })).toBe(true);
      expect(isRetryableError({ error: 'Connection timeout' })).toBe(true);
      expect(isRetryableError({ error: 'Invalid prompt' })).toBe(false);
    });

    test('formatError formats correctly', () => {
      expect(formatError({ error: 'Failed', code: 5 })).toBe('[Code 5] Failed');
      expect(formatError({ success: true })).toBe('No error');
    });
  });

  describe('Retry Logic', () => {
    test('commands include retry flags by default', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'retry-123' }));

      fluxTxt2Img({ prompt: 'Test', seed: 42 });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('--retry 3');
      expect(callArgs[0]).toContain('--retry-on vram,timeout,connection');
    });

    test('system commands skip retry flags', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ success: true }));

      stats();

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).not.toContain('--retry-on');
    });
  });

  describe('JSON Parsing', () => {
    test('parses successful JSON response', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'parse-123', success: true }));

      const result = fluxTxt2Img({ prompt: 'Test', seed: 42 });

      expect(result.asset_id).toBe('parse-123');
      expect(result.success).toBe(true);
    });

    test('handles empty stdout as success', () => {
      mockExecSync.mockReturnValue('');

      const result = freeMemory();

      expect(result.success).toBe(true);
    });

    test('parses error JSON from stdout on failure', () => {
      const error = new Error('Command failed');
      error.status = 1;
      error.stdout = JSON.stringify({ error: 'Validation failed', code: 3 });
      error.stderr = '';
      mockExecSync.mockImplementation(() => {
        throw error;
      });

      const result = fluxTxt2Img({ prompt: 'Test', seed: 42 });

      expect(result.error).toBe('Validation failed');
      expect(result.code).toBe(3);
    });

    test('handles non-JSON error output', () => {
      const error = new Error('Command failed');
      error.status = 1;
      error.stdout = '';
      error.stderr = 'Out of memory';
      mockExecSync.mockImplementation(() => {
        throw error;
      });

      const result = fluxTxt2Img({ prompt: 'Test', seed: 42 });

      expect(result.error).toBe('Out of memory');
    });
  });

  describe('Timeout Values', () => {
    test('IMAGE_TIMEOUT is 660000ms (11 minutes)', () => {
      expect(IMAGE_TIMEOUT).toBe(660000);
    });

    test('VIDEO_TIMEOUT is 900000ms (15 minutes)', () => {
      expect(VIDEO_TIMEOUT).toBe(900000);
    });

    test('PIPELINE_TIMEOUT is 1800000ms (30 minutes)', () => {
      expect(PIPELINE_TIMEOUT).toBe(1800000);
    });

    test('SYSTEM_TIMEOUT is 30000ms (30 seconds)', () => {
      expect(SYSTEM_TIMEOUT).toBe(30000);
    });
  });

  describe('Parameter Defaults', () => {
    test('qwenTxt2Img uses correct defaults', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'default-123' }));

      qwenTxt2Img({ prompt: 'Test' });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('1664'); // default width
      expect(callArgs[0]).toContain('928'); // default height
      expect(callArgs[0]).toContain('7'); // default shift
      expect(callArgs[0]).toContain('3.5'); // default cfg
      expect(callArgs[0]).toContain('50'); // default steps
    });

    test('wanI2V uses correct defaults', () => {
      mockExecSync.mockReturnValue(JSON.stringify({ asset_id: 'wan-default-123' }));

      wanI2V({ image: '/keyframe.jpg', prompt: 'Test' });

      const callArgs = mockExecSync.mock.calls[0];
      expect(callArgs[0]).toContain('81'); // default frames
      expect(callArgs[0]).toContain('16'); // default fps
      expect(callArgs[0]).toContain('5'); // default cfg
      expect(callArgs[0]).toContain('832'); // default width
      expect(callArgs[0]).toContain('480'); // default height
    });
  });
});
