/**
 * Utility functions for mmf CLI operations
 *
 * @module utils
 */

import { execSync } from 'child_process';
import {
  IMAGE_TIMEOUT,
  DEFAULT_RETRY_FLAGS,
} from './timeouts.js';

/**
 * Escape a string for safe use in shell commands.
 * Handles single quotes by escaping them properly.
 * @param {string|null|undefined} str - String to escape
 * @returns {string} Escaped string safe for shell use
 */
export function esc(str) {
  if (str == null) return '';
  return String(str).replace(/'/g, "'\\''");
}

/**
 * Build CLI arguments string from an array of parts.
 * Filters out undefined, null, and empty strings.
 * @param {Array<string|null|undefined>} parts - Argument parts
 * @returns {string} Joined argument string
 */
export function buildArgs(parts) {
  return parts.filter(Boolean).join(' ');
}

/**
 * Execute an mmf CLI command and return parsed JSON output.
 *
 * @param {string} args - CLI arguments (e.g., "run --model flux --type t2i ...")
 * @param {Object} [opts] - Options
 * @param {number} [opts.timeout] - Timeout in ms (default: IMAGE_TIMEOUT)
 * @param {number} [opts.retry] - Retry count (default: 3)
 * @param {boolean} [opts.noRetry] - Skip retry flags (for system commands)
 * @param {string} [opts.input] - Input data to pipe to stdin
 * @returns {Object} Parsed JSON from mmf stdout, or {error: string} on failure
 */
export function mmf(args, opts = {}) {
  const timeout = opts.timeout || IMAGE_TIMEOUT;
  const retry = opts.retry ?? 3;
  const retryFlags = opts.noRetry
    ? ''
    : ` --retry ${retry} --retry-on vram,timeout,connection`;
  const cmd = `mmf ${args}${retryFlags}`;

  try {
    const stdout = execSync(cmd, {
      encoding: 'utf8',
      timeout,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env },
      input: opts.input,
    });

    const trimmed = stdout.trim();
    if (!trimmed) {
      return { success: true };
    }
    return JSON.parse(trimmed);
  } catch (err) {
    // execSync throws on non-zero exit or timeout
    const stderr = err.stderr?.toString().trim() || '';
    const stdout = err.stdout?.toString().trim() || '';

    // Try to parse stdout even on error (mmf may return JSON error objects)
    if (stdout) {
      try {
        return JSON.parse(stdout);
      } catch {
        // Not JSON, fall through
      }
    }

    return {
      error: stderr || err.message || `mmf exited with code ${err.status}`,
      code: err.status,
    };
  }
}

/**
 * Safely parse JSON string with error handling
 * @param {string} str - JSON string to parse
 * @param {*} [defaultValue=null] - Default value if parsing fails
 * @returns {*} Parsed object or default value
 */
export function safeJsonParse(str, defaultValue = null) {
  try {
    return JSON.parse(str);
  } catch {
    return defaultValue;
  }
}

/**
 * Build template params JSON string safely
 * @param {Object} params - Parameters object
 * @returns {string} Escaped JSON string for CLI
 */
export function buildParams(params) {
  return esc(JSON.stringify(params));
}

/**
 * Validate required options are present
 * @param {Object} options - Options object
 * @param {string[]} required - List of required keys
 * @throws {Error} If any required key is missing
 */
export function validateRequired(options, required) {
  const missing = required.filter((key) => !(key in options) || options[key] == null);
  if (missing.length > 0) {
    throw new Error(`Missing required options: ${missing.join(', ')}`);
  }
}

/**
 * Pick specific keys from an object
 * @param {Object} obj - Source object
 * @param {string[]} keys - Keys to pick
 * @returns {Object} New object with only picked keys
 */
export function pick(obj, keys) {
  return keys.reduce((acc, key) => {
    if (key in obj) acc[key] = obj[key];
    return acc;
  }, {});
}

/**
 * Convert kebab-case to camelCase
 * @param {string} str - kebab-case string
 * @returns {string} camelCase string
 */
export function toCamelCase(str) {
  return str.replace(/-([a-z])/g, (g) => g[1].toUpperCase());
}

// Backward compatibility - also export as default object
export default {
  esc,
  buildArgs,
  mmf,
  safeJsonParse,
  buildParams,
  validateRequired,
  pick,
  toCamelCase,
};
