"""WebSocket Client for ComfyUI Progress Reporting.

Provides real-time progress visibility for long-running video generations.
"""

import json
import asyncio
import websockets
from typing import Optional, Dict, Any, AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime


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
        except Exception as e:
            self._connected = False
            return False
    
    def disconnect(self):
        """Disconnect from WebSocket."""
        if self.ws:
            asyncio.create_task(self.ws.close())
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
                    if event.stage in ('completed', 'error'):
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
            
            msg_type = data.get('type', '')
            
            # Handle execution_start
            if msg_type == 'execution_start':
                prompt_id = data.get('data', {}).get('prompt_id', '')
                if prompt_id == target_prompt_id:
                    return ProgressEvent(
                        prompt_id=prompt_id,
                        stage='running',
                        percent=0.0,
                        nodes_completed=0,
                        nodes_total=0,
                        message='Execution started'
                    )
            
            # Handle executing (node execution started)
            elif msg_type == 'executing':
                prompt_id = data.get('data', {}).get('prompt_id', '')
                node_id = data.get('data', {}).get('node')
                
                if prompt_id == target_prompt_id:
                    return ProgressEvent(
                        prompt_id=prompt_id,
                        stage='running',
                        percent=0.0,
                        node_id=node_id,
                        message=f'Executing node {node_id}'
                    )
            
            # Handle progress (sampling progress)
            elif msg_type == 'progress':
                prompt_id = data.get('data', {}).get('prompt_id', '')
                value = data.get('data', {}).get('value', 0)
                max_val = data.get('data', {}).get('max', 100)
                
                if prompt_id == target_prompt_id:
                    percent = (value / max_val * 100) if max_val > 0 else 0
                    return ProgressEvent(
                        prompt_id=prompt_id,
                        stage='progress',
                        percent=percent,
                        message=f'Sampling {value}/{max_val}'
                    )
            
            # Handle execution_cached
            elif msg_type == 'execution_cached':
                prompt_id = data.get('data', {}).get('prompt_id', '')
                if prompt_id == target_prompt_id:
                    return ProgressEvent(
                        prompt_id=prompt_id,
                        stage='running',
                        percent=50.0,
                        message='Using cached result'
                    )
            
            # Handle execution_complete
            elif msg_type == 'execution_complete':
                prompt_id = data.get('data', {}).get('prompt_id', '')
                if prompt_id == target_prompt_id:
                    return ProgressEvent(
                        prompt_id=prompt_id,
                        stage='completed',
                        percent=100.0,
                        nodes_completed=1,
                        nodes_total=1,
                        message='Execution completed'
                    )
            
            # Handle execution_error
            elif msg_type == 'execution_error':
                prompt_id = data.get('data', {}).get('prompt_id', '')
                if prompt_id == target_prompt_id:
                    error_msg = data.get('data', {}).get('error', 'Unknown error')
                    return ProgressEvent(
                        prompt_id=prompt_id,
                        stage='error',
                        percent=0.0,
                        message=f'Error: {error_msg}'
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
            'stage': event.stage,
            'percent': round(event.percent, 1),
            'eta_seconds': event.eta_seconds,
            'nodes_completed': event.nodes_completed,
            'nodes_total': event.nodes_total,
            'message': event.message,
            'timestamp': event.timestamp.isoformat() if event.timestamp else None
        }
    
    async def watch_progress(self, prompt_id: str, callback: Callable[[ProgressEvent], None]):
        """Watch progress and call callback for each update."""
        async for event in self.client.subscribe_progress(prompt_id):
            callback(event)


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
