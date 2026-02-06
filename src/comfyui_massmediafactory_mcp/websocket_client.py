"""WebSocket Client for ComfyUI Progress Reporting.

Provides real-time progress visibility for long-running video generations.
Also provides synchronous wait_for_prompt() for use in wait_for_completion().
"""

import json
import asyncio
import logging
import os
import threading
import websockets
from typing import Optional, Dict, Any, AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("comfyui-mcp")


@dataclass
class ProgressEvent:
    """Progress event from ComfyUI WebSocket."""

    prompt_id: str
    stage: str  # 'queued', 'running', 'progress', 'completed', 'error'
    percent: float  # 0-100
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    nodes_completed: int = 0
    nodes_total: int = 0
    eta_seconds: Optional[float] = None
    message: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ComfyUIWebSocketClient:
    """WebSocket client for ComfyUI progress monitoring."""

    def __init__(self, host: str = "localhost", port: int = 8188):
        self.host = host
        self.port = port
        self.ws = None
        self._progress_callbacks: Dict[str, Callable] = {}
        self._current_progress: Dict[str, ProgressEvent] = {}
        self._connected = False
        self._client_id = "massmediafactory"

    async def connect(self) -> bool:
        """Connect to ComfyUI WebSocket.

        Returns:
            True if connected successfully
        """
        try:
            uri = f"ws://{self.host}:{self.port}/ws?clientId={self._client_id}"
            self.ws = await websockets.connect(uri)
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from WebSocket."""
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        self._connected = False

    async def subscribe_progress(self, prompt_id: str) -> AsyncIterator[ProgressEvent]:
        """Subscribe to progress updates for a prompt_id.

        Yields:
            ProgressEvent objects as they arrive

        Example:
            async for event in client.subscribe_progress("prompt-123"):
                print(f"{event.percent}% - {event.stage}")
        """
        if not self._connected:
            await self.connect()

        if not self._connected:
            raise ConnectionError("Cannot connect to ComfyUI WebSocket")

        try:
            async for message in self.ws:
                event = self._parse_message(message, prompt_id)
                if event:
                    self._current_progress[prompt_id] = event
                    yield event

                    # Stop if execution completed or errored
                    if event.stage in ("completed", "error"):
                        break
        except websockets.exceptions.ConnectionClosed:
            self._connected = False

    def get_current_progress(self, prompt_id: str) -> Optional[ProgressEvent]:
        """Get current progress for a prompt_id.

        Returns:
            Latest ProgressEvent or None if no progress recorded
        """
        return self._current_progress.get(prompt_id)

    def _parse_message(self, message, target_prompt_id: str) -> Optional[ProgressEvent]:
        """Parse WebSocket message into ProgressEvent."""
        try:
            if isinstance(message, str):
                data = json.loads(message)
            else:
                return None

            msg_type = data.get("type", "")

            # Handle execution_start
            if msg_type == "execution_start":
                prompt_id = data.get("data", {}).get("prompt_id", "")
                if prompt_id == target_prompt_id:
                    return ProgressEvent(
                        prompt_id=prompt_id,
                        stage="running",
                        percent=0.0,
                        nodes_completed=0,
                        nodes_total=0,
                        message="Execution started",
                    )

            # Handle executing (node execution started)
            elif msg_type == "executing":
                prompt_id = data.get("data", {}).get("prompt_id", "")
                node_id = data.get("data", {}).get("node")

                if prompt_id == target_prompt_id:
                    # node_id being None means execution is done (ComfyUI sends this after all nodes)
                    if node_id is None:
                        return ProgressEvent(
                            prompt_id=prompt_id,
                            stage="completed",
                            percent=100.0,
                            message="All nodes executed",
                        )
                    return ProgressEvent(
                        prompt_id=prompt_id,
                        stage="running",
                        percent=0.0,
                        node_id=node_id,
                        message=f"Executing node {node_id}",
                    )

            # Handle progress (sampling progress)
            elif msg_type == "progress":
                prompt_id = data.get("data", {}).get("prompt_id", "")
                value = data.get("data", {}).get("value", 0)
                max_val = data.get("data", {}).get("max", 100)

                if prompt_id == target_prompt_id:
                    percent = (value / max_val * 100) if max_val > 0 else 0
                    return ProgressEvent(
                        prompt_id=prompt_id,
                        stage="progress",
                        percent=percent,
                        message=f"Sampling {value}/{max_val}",
                    )

            # Handle execution_cached
            elif msg_type == "execution_cached":
                prompt_id = data.get("data", {}).get("prompt_id", "")
                if prompt_id == target_prompt_id:
                    return ProgressEvent(
                        prompt_id=prompt_id,
                        stage="running",
                        percent=50.0,
                        message="Using cached result",
                    )

            # Handle execution_complete (some ComfyUI versions use this)
            elif msg_type == "execution_complete":
                prompt_id = data.get("data", {}).get("prompt_id", "")
                if prompt_id == target_prompt_id:
                    return ProgressEvent(
                        prompt_id=prompt_id,
                        stage="completed",
                        percent=100.0,
                        nodes_completed=1,
                        nodes_total=1,
                        message="Execution completed",
                    )

            # Handle execution_error
            elif msg_type == "execution_error":
                prompt_id = data.get("data", {}).get("prompt_id", "")
                if prompt_id == target_prompt_id:
                    error_msg = data.get("data", {}).get(
                        "exception_message",
                        data.get("data", {}).get("error", "Unknown error"),
                    )
                    return ProgressEvent(
                        prompt_id=prompt_id,
                        stage="error",
                        percent=0.0,
                        message=f"Error: {error_msg}",
                    )

            # Handle execution_interrupted
            elif msg_type == "execution_interrupted":
                prompt_id = data.get("data", {}).get("prompt_id", "")
                if prompt_id == target_prompt_id:
                    return ProgressEvent(
                        prompt_id=prompt_id,
                        stage="error",
                        percent=0.0,
                        message="Execution interrupted",
                    )

            return None

        except Exception:
            return None


class ProgressTracker:
    """Synchronous wrapper for progress tracking."""

    def __init__(self, host: str = "localhost", port: int = 8188):
        self.client = ComfyUIWebSocketClient(host, port)
        self._progress_cache: Dict[str, ProgressEvent] = {}

    def get_progress(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress as dictionary.

        Returns:
            {
                'stage': str,
                'percent': float,
                'eta_seconds': float or None,
                'nodes_completed': int,
                'nodes_total': int,
                'message': str or None,
                'timestamp': str
            }
        """
        event = self.client.get_current_progress(prompt_id)
        if not event:
            return None

        return {
            "stage": event.stage,
            "percent": round(event.percent, 1),
            "eta_seconds": event.eta_seconds,
            "nodes_completed": event.nodes_completed,
            "nodes_total": event.nodes_total,
            "message": event.message,
            "timestamp": event.timestamp.isoformat() if event.timestamp else None,
        }

    async def watch_progress(self, prompt_id: str, callback: Callable[[ProgressEvent], None]):
        """Watch progress and call callback for each update."""
        async for event in self.client.subscribe_progress(prompt_id):
            callback(event)


def wait_for_prompt_ws(
    prompt_id: str,
    timeout_seconds: int = 600,
    host: str = None,
    port: int = None,
) -> Optional[str]:
    """
    Wait for a prompt to complete via WebSocket (synchronous, blocking).

    Opens a WebSocket, listens for execution_complete/error/executing(node=None)
    for the given prompt_id, then returns the terminal stage.

    Returns:
        "completed" or "error" on success, None on connection failure or timeout.
    """
    if host is None:
        base_url = os.environ.get("COMFYUI_URL", "http://localhost:8188")
        # Extract host and port from URL
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8188
    elif port is None:
        port = 8188

    result = {"stage": None}

    async def _ws_wait():
        client = ComfyUIWebSocketClient(host, port)
        try:
            connected = await client.connect()
            if not connected:
                return

            async for event in client.subscribe_progress(prompt_id):
                if event.stage in ("completed", "error"):
                    result["stage"] = event.stage
                    break
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.debug(f"WebSocket wait error: {e}")
        finally:
            await client.disconnect()

    def _run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(asyncio.wait_for(_ws_wait(), timeout=timeout_seconds))
        except asyncio.TimeoutError:
            pass
        finally:
            loop.close()

    # Run in a separate thread to avoid blocking the MCP server's event loop
    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds + 5)

    return result.get("stage")


# Global tracker instance
_progress_tracker: Optional[ProgressTracker] = None


def get_progress_tracker(host: str = "localhost", port: int = 8188) -> ProgressTracker:
    """Get or create progress tracker instance."""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = ProgressTracker(host, port)
    return _progress_tracker


def get_progress_sync(prompt_id: str, host: str = "localhost", port: int = 8188) -> Optional[Dict[str, Any]]:
    """Synchronous function to get current progress.

    Returns:
        Progress dict or None if not found
    """
    tracker = get_progress_tracker(host, port)
    return tracker.get_progress(prompt_id)
