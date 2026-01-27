"""
ComfyUI API Client

Low-level HTTP client for ComfyUI API interactions.
"""

import json
import os
import urllib.request
import urllib.error
from typing import Any, Optional


class ComfyUIClient:
    """HTTP client for ComfyUI API."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.environ.get(
            "COMFYUI_URL", "http://localhost:8188"
        )

    def request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[dict] = None,
        timeout: int = 30,
    ) -> dict:
        """Make HTTP request to ComfyUI API."""
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
        return self.post("/free", {
            "free_memory": True,
            "unload_models": unload_models
        })


# Global client instance
_client: Optional[ComfyUIClient] = None


def get_client() -> ComfyUIClient:
    """Get or create global client instance."""
    global _client
    if _client is None:
        _client = ComfyUIClient()
    return _client
