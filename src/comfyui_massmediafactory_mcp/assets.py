"""
Asset Registry for MassMediaFactory MCP

Enables iteration on generated assets without re-prompting.
SQLite-backed for persistence across server restarts.
In-memory dict acts as hot cache.
"""

import os
import time
import uuid
import copy
import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path

# Default TTL: 24 hours
DEFAULT_TTL_HOURS = int(os.environ.get("COMFY_MCP_ASSET_TTL_HOURS", "24"))

# Default DB path
DEFAULT_DB_DIR = Path.home() / ".massmediafactory"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "assets.db"


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


class _SQLiteStore:
    """SQLite persistence layer for asset records."""

    def __init__(self, db_path: Path = None):
        self._db_path = db_path or DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self._db_path), timeout=5)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                subfolder TEXT DEFAULT '',
                asset_type TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                workflow TEXT NOT NULL,
                parameters TEXT NOT NULL,
                session_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                width INTEGER,
                height INTEGER,
                node_id TEXT,
                prompt_preview TEXT
            )
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_assets_identity
            ON assets (filename, subfolder, asset_type)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_assets_session
            ON assets (session_id)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_assets_expires
            ON assets (expires_at)
        """
        )
        conn.commit()

    def save(self, asset: AssetRecord):
        """Insert or replace an asset record."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO assets
            (asset_id, filename, subfolder, asset_type, mime_type,
             workflow, parameters, session_id, created_at, expires_at,
             width, height, node_id, prompt_preview)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                asset.asset_id,
                asset.filename,
                asset.subfolder,
                asset.asset_type,
                asset.mime_type,
                json.dumps(asset.workflow),
                json.dumps(asset.parameters),
                asset.session_id,
                asset.created_at,
                asset.expires_at,
                asset.width,
                asset.height,
                asset.node_id,
                asset.prompt_preview,
            ),
        )
        conn.commit()

    def load_all_active(self) -> List[AssetRecord]:
        """Load all non-expired assets from DB."""
        conn = self._get_conn()
        now = time.time()
        cursor = conn.execute("SELECT * FROM assets WHERE expires_at > ?", (now,))
        rows = cursor.fetchall()
        assets = []
        for row in rows:
            assets.append(self._row_to_asset(row))
        return assets

    def delete(self, asset_id: str):
        """Delete an asset record."""
        conn = self._get_conn()
        conn.execute("DELETE FROM assets WHERE asset_id = ?", (asset_id,))
        conn.commit()

    def cleanup_expired(self) -> int:
        """Delete expired records. Returns count removed."""
        conn = self._get_conn()
        now = time.time()
        cursor = conn.execute("DELETE FROM assets WHERE expires_at <= ?", (now,))
        conn.commit()
        return cursor.rowcount

    def _row_to_asset(self, row) -> AssetRecord:
        """Convert DB row to AssetRecord."""
        return AssetRecord(
            asset_id=row[0],
            filename=row[1],
            subfolder=row[2],
            asset_type=row[3],
            mime_type=row[4],
            workflow=json.loads(row[5]),
            parameters=json.loads(row[6]),
            session_id=row[7],
            created_at=row[8],
            expires_at=row[9],
            width=row[10],
            height=row[11],
            node_id=row[12],
            prompt_preview=row[13],
        )


class AssetRegistry:
    """
    Thread-safe registry for generated assets with TTL expiration.

    Uses in-memory dict as hot cache, SQLite as persistent store.
    On startup, loads non-expired assets from SQLite.
    On register, writes to both cache and SQLite.
    """

    def __init__(
        self,
        ttl_hours: int = DEFAULT_TTL_HOURS,
        comfyui_base_url: str = None,
        db_path: Path = None,
    ):
        self._assets: Dict[str, AssetRecord] = {}
        self._identity_to_id: Dict[tuple, str] = {}
        self._lock = threading.RLock()
        self._ttl_seconds = ttl_hours * 3600
        self._comfyui_base_url = comfyui_base_url or os.environ.get("COMFYUI_URL", "http://localhost:8188")
        self._default_session_id = str(uuid.uuid4())[:8]

        # Initialize SQLite store
        self._store = _SQLiteStore(db_path)

        # Load existing assets from SQLite into cache
        self._load_from_store()

    def _load_from_store(self):
        """Load non-expired assets from SQLite into memory cache."""
        try:
            assets = self._store.load_all_active()
            for asset in assets:
                self._assets[asset.asset_id] = asset
                identity_key = self._make_identity_key(asset.filename, asset.subfolder, asset.asset_type)
                self._identity_to_id[identity_key] = asset.asset_id
        except Exception as e:
            # If DB is corrupted, start fresh but log it
            import logging

            logging.getLogger("comfyui-mcp").warning(f"Failed to load assets from DB: {e}")

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

        Writes to both in-memory cache and SQLite for persistence.
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
                    self._store.delete(existing_id)
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

            # Write to both cache and SQLite
            self._assets[asset.asset_id] = asset
            self._identity_to_id[identity_key] = asset.asset_id
            self._store.save(asset)

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

    def get_asset_by_identity(self, filename: str, subfolder: str, asset_type: str) -> Optional[AssetRecord]:
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
        Remove all expired assets from both cache and SQLite.

        Returns:
            Number of assets removed
        """
        with self._lock:
            expired_ids = [asset_id for asset_id, asset in self._assets.items() if asset.is_expired()]
            for asset_id in expired_ids:
                self._remove_asset(asset_id)

            # Also clean SQLite (catches any that were only in DB)
            db_cleaned = self._store.cleanup_expired()
            return max(len(expired_ids), db_cleaned)

    def _remove_asset(self, asset_id: str) -> None:
        """Remove asset from both cache and SQLite."""
        asset = self._assets.get(asset_id)
        if asset:
            identity_key = self._make_identity_key(asset.filename, asset.subfolder, asset.asset_type)
            self._identity_to_id.pop(identity_key, None)
            del self._assets[asset_id]
        self._store.delete(asset_id)

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
        return mime_map.get(ext, "application/octet-stream")

    def _extract_prompt_preview(self, workflow: dict, parameters: dict = None) -> Optional[str]:
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
        from urllib.parse import quote

        base = self._comfyui_base_url.rstrip("/")
        filename = quote(asset.filename, safe="")
        if asset.subfolder:
            subfolder = quote(asset.subfolder, safe="")
            return f"{base}/view?filename={filename}&subfolder={subfolder}&type=output"
        return f"{base}/view?filename={filename}&type=output"

    def get_asset_path(self, asset: AssetRecord, comfyui_output_dir: str = None) -> Path:
        """Build local file path for an asset."""
        from .client import get_comfyui_output_dir

        output_dir = comfyui_output_dir or get_comfyui_output_dir()
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
