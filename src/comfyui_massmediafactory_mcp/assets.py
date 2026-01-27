"""
Asset Registry for MassMediaFactory MCP

Enables iteration on generated assets without re-prompting.
Based on patterns from joenorton/comfyui-mcp-server.
"""

import os
import time
import uuid
import copy
import base64
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path

# Default TTL: 24 hours
DEFAULT_TTL_HOURS = int(os.environ.get("COMFY_MCP_ASSET_TTL_HOURS", "24"))


@dataclass
class AssetRecord:
    """Represents a generated asset with full provenance."""

    asset_id: str
    filename: str
    subfolder: str
    asset_type: str  # "images", "video", "audio"
    mime_type: str
    workflow: dict
    parameters: dict
    session_id: str
    created_at: float
    expires_at: float
    width: Optional[int] = None
    height: Optional[int] = None
    node_id: Optional[str] = None
    prompt_preview: Optional[str] = None

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "filename": self.filename,
            "subfolder": self.subfolder,
            "type": self.asset_type,
            "mime_type": self.mime_type,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "width": self.width,
            "height": self.height,
            "prompt_preview": self.prompt_preview,
        }

    def to_full_dict(self) -> dict:
        """Full representation including workflow for debugging."""
        result = self.to_dict()
        result["workflow"] = self.workflow
        result["parameters"] = self.parameters
        result["node_id"] = self.node_id
        return result


class AssetRegistry:
    """
    Thread-safe registry for generated assets with TTL expiration.

    Identity is based on (filename, subfolder, type) tuple for stability
    across hostname changes.
    """

    def __init__(self, ttl_hours: int = DEFAULT_TTL_HOURS, comfyui_base_url: str = None):
        self._assets: Dict[str, AssetRecord] = {}
        self._identity_to_id: Dict[tuple, str] = {}
        self._lock = threading.RLock()
        self._ttl_seconds = ttl_hours * 3600
        self._comfyui_base_url = comfyui_base_url or os.environ.get(
            "COMFYUI_URL", "http://localhost:8188"
        )
        self._default_session_id = str(uuid.uuid4())[:8]

    def _make_identity_key(self, filename: str, subfolder: str, asset_type: str) -> tuple:
        """Create stable identity key for deduplication."""
        return (filename, subfolder or "", asset_type)

    def register_asset(
        self,
        filename: str,
        subfolder: str,
        asset_type: str,
        workflow: dict,
        parameters: dict = None,
        session_id: str = None,
        width: int = None,
        height: int = None,
        node_id: str = None,
        mime_type: str = None,
    ) -> AssetRecord:
        """
        Register a generated asset.

        Args:
            filename: Output filename from ComfyUI
            subfolder: Subfolder within ComfyUI output
            asset_type: Type of asset ("images", "video", "audio")
            workflow: The workflow JSON used to generate this asset
            parameters: Template parameters used (for regeneration)
            session_id: Session identifier for grouping related generations
            width: Image/video width if known
            height: Image/video height if known
            node_id: The node ID that produced this output
            mime_type: MIME type (auto-detected if not provided)

        Returns:
            AssetRecord (new or existing if deduplicated)
        """
        with self._lock:
            identity_key = self._make_identity_key(filename, subfolder, asset_type)

            # Check for existing asset (deduplication)
            if identity_key in self._identity_to_id:
                existing_id = self._identity_to_id[identity_key]
                if existing_id in self._assets:
                    existing = self._assets[existing_id]
                    if not existing.is_expired():
                        return existing
                    # Expired, remove and create new
                    del self._assets[existing_id]
                del self._identity_to_id[identity_key]

            # Auto-detect MIME type
            if mime_type is None:
                mime_type = self._detect_mime_type(filename, asset_type)

            # Extract prompt preview from workflow
            prompt_preview = self._extract_prompt_preview(workflow, parameters)

            # Create new asset record
            now = time.time()
            asset = AssetRecord(
                asset_id=str(uuid.uuid4()),
                filename=filename,
                subfolder=subfolder or "",
                asset_type=asset_type,
                mime_type=mime_type,
                workflow=copy.deepcopy(workflow),
                parameters=parameters or {},
                session_id=session_id or self._default_session_id,
                created_at=now,
                expires_at=now + self._ttl_seconds,
                width=width,
                height=height,
                node_id=node_id,
                prompt_preview=prompt_preview,
            )

            self._assets[asset.asset_id] = asset
            self._identity_to_id[identity_key] = asset.asset_id

            return asset

    def get_asset(self, asset_id: str) -> Optional[AssetRecord]:
        """
        Retrieve an asset by ID.

        Returns None if not found or expired.
        """
        with self._lock:
            asset = self._assets.get(asset_id)
            if asset is None:
                return None

            if asset.is_expired():
                self._remove_asset(asset_id)
                return None

            return asset

    def get_asset_by_identity(
        self, filename: str, subfolder: str, asset_type: str
    ) -> Optional[AssetRecord]:
        """Retrieve asset by stable identity tuple."""
        with self._lock:
            identity_key = self._make_identity_key(filename, subfolder, asset_type)
            asset_id = self._identity_to_id.get(identity_key)
            if asset_id:
                return self.get_asset(asset_id)
            return None

    def list_assets(
        self,
        session_id: str = None,
        asset_type: str = None,
        limit: int = 20,
        include_expired: bool = False,
    ) -> List[AssetRecord]:
        """
        List recent assets with optional filtering.

        Args:
            session_id: Filter by session
            asset_type: Filter by type ("images", "video", "audio")
            limit: Maximum results
            include_expired: Include expired assets

        Returns:
            List of AssetRecords, newest first
        """
        with self._lock:
            results = []
            for asset in self._assets.values():
                if not include_expired and asset.is_expired():
                    continue
                if session_id and asset.session_id != session_id:
                    continue
                if asset_type and asset.asset_type != asset_type:
                    continue
                results.append(asset)

            # Sort by creation time, newest first
            results.sort(key=lambda a: a.created_at, reverse=True)
            return results[:limit]

    def cleanup_expired(self) -> int:
        """
        Remove all expired assets.

        Returns:
            Number of assets removed
        """
        with self._lock:
            expired_ids = [
                asset_id
                for asset_id, asset in self._assets.items()
                if asset.is_expired()
            ]
            for asset_id in expired_ids:
                self._remove_asset(asset_id)
            return len(expired_ids)

    def _remove_asset(self, asset_id: str) -> None:
        """Remove asset from both storage dictionaries."""
        asset = self._assets.get(asset_id)
        if asset:
            identity_key = self._make_identity_key(
                asset.filename, asset.subfolder, asset.asset_type
            )
            self._identity_to_id.pop(identity_key, None)
            del self._assets[asset_id]

    def _detect_mime_type(self, filename: str, asset_type: str) -> str:
        """Auto-detect MIME type from filename and asset type."""
        ext = Path(filename).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
        }
        return mime_map.get(ext, f"application/octet-stream")

    def _extract_prompt_preview(
        self, workflow: dict, parameters: dict = None
    ) -> Optional[str]:
        """Extract prompt preview from workflow or parameters."""
        # Try parameters first
        if parameters:
            for key in ["PROMPT", "prompt", "text"]:
                if key in parameters:
                    prompt = str(parameters[key])
                    return prompt[:100] + "..." if len(prompt) > 100 else prompt

        # Try to find in workflow nodes
        if workflow:
            for node_id, node in workflow.items():
                if not isinstance(node, dict):
                    continue
                inputs = node.get("inputs", {})
                for key in ["text", "prompt", "positive"]:
                    if key in inputs and isinstance(inputs[key], str):
                        prompt = inputs[key]
                        return prompt[:100] + "..." if len(prompt) > 100 else prompt

        return None

    def get_asset_url(self, asset: AssetRecord) -> str:
        """Build ComfyUI URL for viewing/downloading asset."""
        base = self._comfyui_base_url.rstrip("/")
        if asset.subfolder:
            return f"{base}/view?filename={asset.filename}&subfolder={asset.subfolder}&type=output"
        return f"{base}/view?filename={asset.filename}&type=output"

    def get_asset_path(self, asset: AssetRecord, comfyui_output_dir: str = None) -> Path:
        """Build local file path for an asset."""
        output_dir = comfyui_output_dir or os.environ.get(
            "COMFYUI_OUTPUT_DIR", "/home/romancircus/ComfyUI/output"
        )
        if asset.subfolder:
            return Path(output_dir) / asset.subfolder / asset.filename
        return Path(output_dir) / asset.filename


# Global registry instance
_registry: Optional[AssetRegistry] = None


def get_registry() -> AssetRegistry:
    """Get or create the global asset registry."""
    global _registry
    if _registry is None:
        _registry = AssetRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _registry
    _registry = None
