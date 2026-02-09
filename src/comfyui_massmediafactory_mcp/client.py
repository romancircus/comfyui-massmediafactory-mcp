"""
ComfyUI API Client

Low-level HTTP client for ComfyUI API interactions.
"""

import functools
import json
import mimetypes
import os
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Callable, Optional, TypeVar
from uuid import uuid4

# Retry configuration from environment
RETRY_MAX_ATTEMPTS = int(os.environ.get("COMFYUI_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF = float(os.environ.get("COMFYUI_RETRY_BACKOFF", "1.0"))
RETRY_MULTIPLIER = float(os.environ.get("COMFYUI_RETRY_MULTIPLIER", "2.0"))

# Retryable error patterns
RETRYABLE_ERRORS = [
    "timed out",
    "connection refused",
    "connection reset",
    "temporary failure",
    "service unavailable",
    "502",
    "503",
    "504",
]

T = TypeVar("T")


def retry(
    max_attempts: int = RETRY_MAX_ATTEMPTS,
    backoff: float = RETRY_BACKOFF,
    multiplier: float = RETRY_MULTIPLIER,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts.
        backoff: Initial backoff delay in seconds.
        multiplier: Multiplier for backoff between retries.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            delay = backoff

            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)

                    # Check if result is an error dict that might be retryable
                    if isinstance(result, dict) and "error" in result:
                        error_str = str(result["error"]).lower()
                        is_retryable = any(pattern in error_str for pattern in RETRYABLE_ERRORS)

                        if is_retryable and attempt < max_attempts - 1:
                            last_error = result
                            time.sleep(delay)
                            delay *= multiplier
                            continue

                    return result

                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    is_retryable = any(pattern in error_str for pattern in RETRYABLE_ERRORS)

                    if is_retryable and attempt < max_attempts - 1:
                        time.sleep(delay)
                        delay *= multiplier
                        continue

                    raise

            # Return last error if we exhausted retries
            if isinstance(last_error, dict):
                return last_error
            elif last_error:
                return {"error": str(last_error), "retried": max_attempts}
            return {"error": "Max retries exceeded"}

        return wrapper

    return decorator


def get_comfyui_output_dir() -> str:
    """Get ComfyUI output directory from environment or default."""
    return os.environ.get("COMFYUI_OUTPUT_DIR", str(Path.home() / "ComfyUI" / "output"))


class ComfyUIClient:
    """HTTP client for ComfyUI API."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.environ.get("COMFYUI_URL", "http://localhost:8188")

    @retry()
    def request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[dict] = None,
        timeout: int = 30,
    ) -> dict:
        """Make HTTP request to ComfyUI API with automatic retry."""
        url = f"{self.base_url}{endpoint}"
        req = urllib.request.Request(url, method=method)

        if data:
            req.data = json.dumps(data).encode()
            req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as e:
            return {"error": str(e)}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response"}

    def get(self, endpoint: str, timeout: int = 30) -> dict:
        """GET request."""
        return self.request(endpoint, "GET", timeout=timeout)

    def post(self, endpoint: str, data: dict, timeout: int = 30) -> dict:
        """POST request."""
        return self.request(endpoint, "POST", data, timeout=timeout)

    def is_available(self) -> bool:
        """Check if ComfyUI is reachable."""
        result = self.get("/system_stats")
        return "error" not in result

    def get_system_stats(self) -> dict:
        """Get system statistics including VRAM."""
        return self.get("/system_stats")

    def get_object_info(self, node_type: Optional[str] = None) -> dict:
        """Get node information."""
        if node_type:
            return self.get(f"/object_info/{node_type}")
        return self.get("/object_info")

    def queue_prompt(self, workflow: dict, client_id: str = "mcp") -> dict:
        """Queue a workflow for execution."""
        return self.post("/prompt", {"prompt": workflow, "client_id": client_id})

    def get_history(self, prompt_id: str) -> dict:
        """Get execution history for a prompt."""
        return self.get(f"/history/{prompt_id}")

    def get_queue(self) -> dict:
        """Get current queue status."""
        return self.get("/queue")

    def interrupt(self) -> dict:
        """Interrupt current execution."""
        return self.post("/interrupt", {})

    def free_memory(self, unload_models: bool = False) -> dict:
        """Free GPU memory."""
        return self.post("/free", {"free_memory": True, "unload_models": unload_models})

    def upload_image(
        self,
        image_path: str,
        filename: Optional[str] = None,
        subfolder: str = "",
        overwrite: bool = True,
    ) -> dict:
        """
        Upload an image to ComfyUI input folder.

        Args:
            image_path: Local path to the image file.
            filename: Target filename (defaults to original name).
            subfolder: Subfolder within input directory.
            overwrite: Whether to overwrite existing files.

        Returns:
            {"name": "filename.png", "subfolder": "", "type": "input"}
        """
        path = Path(image_path)
        if not path.exists():
            return {"error": f"File not found: {image_path}"}

        if filename is None:
            filename = path.name

        # Detect content type
        content_type, _ = mimetypes.guess_type(str(path))
        if content_type is None:
            content_type = "application/octet-stream"

        # Build multipart/form-data
        boundary = f"----MCPBoundary{uuid4().hex}"

        body_parts = []

        # Add image file
        body_parts.append(f"--{boundary}".encode())
        body_parts.append(f'Content-Disposition: form-data; name="image"; filename="{filename}"'.encode())
        body_parts.append(f"Content-Type: {content_type}".encode())
        body_parts.append(b"")
        with open(path, "rb") as f:
            body_parts.append(f.read())

        # Add overwrite field
        body_parts.append(f"--{boundary}".encode())
        body_parts.append(b'Content-Disposition: form-data; name="overwrite"')
        body_parts.append(b"")
        body_parts.append(b"true" if overwrite else b"false")

        # Add subfolder field if specified
        if subfolder:
            body_parts.append(f"--{boundary}".encode())
            body_parts.append(b'Content-Disposition: form-data; name="subfolder"')
            body_parts.append(b"")
            body_parts.append(subfolder.encode())

        # Close boundary
        body_parts.append(f"--{boundary}--".encode())
        body_parts.append(b"")

        body = b"\r\n".join(body_parts)

        url = f"{self.base_url}/upload/image"
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as e:
            return {"error": str(e)}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response from upload"}

    def download_file(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes | dict:
        """
        Download a file from ComfyUI.

        Args:
            filename: The filename to download.
            subfolder: Subfolder within the directory.
            folder_type: "output", "input", or "temp".

        Returns:
            Raw bytes of the file, or dict with error.
        """
        params = f"filename={urllib.parse.quote(filename)}&type={folder_type}"
        if subfolder:
            params += f"&subfolder={urllib.parse.quote(subfolder)}"

        url = f"{self.base_url}/view?{params}"
        req = urllib.request.Request(url)

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.URLError as e:
            return {"error": str(e)}


# Global client instance
_client: Optional[ComfyUIClient] = None


def get_client() -> ComfyUIClient:
    """Get or create global client instance."""
    global _client
    if _client is None:
        _client = ComfyUIClient()
    return _client
