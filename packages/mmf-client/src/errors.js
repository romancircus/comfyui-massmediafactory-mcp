/**
 * Error handling utilities for mmf operations
 *
 * @module errors
 */

/**
 * Error codes returned by mmf CLI
 */
export const ErrorCodes = {
  SUCCESS: 0,
  ERROR: 1,
  TIMEOUT: 2,
  VALIDATION: 3,
  VRAM: 4,
  CONNECTION: 5,
  UNKNOWN: 99,
};

/**
 * Check if a result contains an error
 * @param {Object} result - Result from any mmf function
 * @returns {boolean} True if result has an error
 */
export function hasError(result) {
  return result && typeof result === 'object' && 'error' in result;
}

/**
 * Get error message from result
 * @param {Object} result - Result from any mmf function
 * @param {string} [defaultMsg='Unknown error'] - Default message if no error
 * @returns {string} Error message or default
 */
export function getErrorMessage(result, defaultMsg = 'Unknown error') {
  if (!hasError(result)) return defaultMsg;
  return result.error || defaultMsg;
}

/**
 * Get error code from result
 * @param {Object} result - Result from any mmf function
 * @returns {number} Error code or ErrorCodes.UNKNOWN
 */
export function getErrorCode(result) {
  if (!result || typeof result !== 'object') return ErrorCodes.UNKNOWN;
  if ('code' in result) return result.code;
  if ('error' in result) return ErrorCodes.ERROR;
  return ErrorCodes.SUCCESS;
}

/**
 * Check if error is retryable (VRAM, timeout, connection issues)
 * @param {Object} result - Result from any mmf function
 * @returns {boolean} True if error is retryable
 */
export function isRetryableError(result) {
  if (!hasError(result)) return false;
  const msg = getErrorMessage(result).toLowerCase();
  return (
    msg.includes('vram') ||
    msg.includes('memory') ||
    msg.includes('timeout') ||
    msg.includes('connection') ||
    msg.includes('socket') ||
    msg.includes('econnrefused')
  );
}

/**
 * Format error for logging
 * @param {Object} result - Result from any mmf function
 * @returns {string} Formatted error string
 */
export function formatError(result) {
  if (!hasError(result)) return 'No error';
  const code = getErrorCode(result);
  const msg = getErrorMessage(result);
  return `[Code ${code}] ${msg}`;
}

/**
 * Assert that result is successful, throw if not
 * @param {Object} result - Result from any mmf function
 * @param {string} [context] - Context for the error message
 * @throws {Error} If result contains an error
 */
export function assertSuccess(result, context) {
  if (hasError(result)) {
    const prefix = context ? `[${context}] ` : '';
    throw new Error(`${prefix}${formatError(result)}`);
  }
}

// Backward compatibility - also export as default object
export default {
  ErrorCodes,
  hasError,
  getErrorMessage,
  getErrorCode,
  isRetryableError,
  formatError,
  assertSuccess,
};
