"""
Publishing Tools for MassMediaFactory MCP

Export generated assets to web directories with compression.
Based on patterns from joenorton/comfyui-mcp-server.
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from .assets import get_registry

# Filename validation regex (security)
SAFE_FILENAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}\.(webp|png|jpg|jpeg|mp4|webm|gif)$")

# Default publish directories to auto-detect
AUTO_DETECT_DIRS = [
    "public/gen",
    "static/gen",
    "assets/gen",
    "public/images",
    "static/images",
]


class PublishManager:
    """Manages asset publishing to web directories."""

    def __init__(
        self,
        comfyui_output_dir: str = None,
        default_publish_dir: str = None,
        max_bytes: int = 600_000,  # 600KB default
    ):
        self._comfyui_output_dir = comfyui_output_dir or os.environ.get(
            "COMFYUI_OUTPUT_DIR", "/home/romancircus/ComfyUI/output"
        )
        self._default_publish_dir = default_publish_dir
        self._max_bytes = max_bytes

    def get_publish_dir(self, project_root: str = None) -> Optional[Path]:
        """Auto-detect or return configured publish directory."""
        if self._default_publish_dir:
            return Path(self._default_publish_dir)

        # Try to auto-detect from project root
        root = Path(project_root or os.getcwd())
        for subdir in AUTO_DETECT_DIRS:
            candidate = root / subdir
            if candidate.exists() and candidate.is_dir():
                return candidate

        return None

    def validate_filename(self, filename: str) -> tuple[bool, str]:
        """Validate filename for security."""
        if not filename:
            return False, "Filename required"

        filename_lower = filename.lower()
        if not SAFE_FILENAME_PATTERN.match(filename_lower):
            return (
                False,
                f"Invalid filename format: {filename}. Must match pattern: lowercase alphanumeric, dots, dashes, underscores, with valid extension.",
            )

        return True, ""

    def get_source_path(self, asset) -> Optional[Path]:
        """Get validated source path for an asset."""
        if asset.subfolder:
            source = Path(self._comfyui_output_dir) / asset.subfolder / asset.filename
        else:
            source = Path(self._comfyui_output_dir) / asset.filename

        # Security: validate path doesn't escape output dir
        try:
            source_real = source.resolve()
            output_real = Path(self._comfyui_output_dir).resolve()
            if not source_real.is_relative_to(output_real):
                return None
        except Exception:
            return None

        if not source.exists():
            return None

        return source

    def copy_asset(
        self,
        source: Path,
        dest: Path,
        web_optimize: bool = True,
    ) -> dict:
        """
        Copy asset to destination with optional optimization.

        For now, simple copy. WebP conversion would require PIL.
        """
        try:
            # Ensure parent directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(source, dest)

            # Get file size
            size = dest.stat().st_size

            return {
                "success": True,
                "path": str(dest),
                "bytes": size,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def update_manifest(
        self,
        manifest_path: Path,
        manifest_key: str,
        asset_info: dict,
    ) -> bool:
        """Update manifest.json with new asset entry."""
        try:
            manifest = {}
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)

            manifest[manifest_key] = {
                "filename": asset_info.get("filename"),
                "url": asset_info.get("url"),
                "mime_type": asset_info.get("mime_type"),
                "bytes": asset_info.get("bytes"),
                "published_at": datetime.utcnow().isoformat() + "Z",
            }

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            return True

        except Exception:
            return False


# Global instance
_publish_manager: Optional[PublishManager] = None


def get_publish_manager() -> PublishManager:
    """Get or create global publish manager."""
    global _publish_manager
    if _publish_manager is None:
        _publish_manager = PublishManager()
    return _publish_manager


def publish_asset(
    asset_id: str,
    target_filename: str = None,
    manifest_key: str = None,
    publish_dir: str = None,
    web_optimize: bool = True,
) -> dict:
    """
    Publish a generated asset to a web directory.

    Args:
        asset_id: The asset to publish.
        target_filename: Explicit filename (demo mode).
        manifest_key: Key for manifest.json (library mode, auto-generates filename).
        publish_dir: Target directory (auto-detected if not provided).
        web_optimize: Apply compression (default True).

    Returns:
        Published asset info with URL and path.
    """
    registry = get_registry()
    manager = get_publish_manager()

    # Get asset
    asset = registry.get_asset(asset_id)
    if asset is None:
        return {"error": "ASSET_NOT_FOUND_OR_EXPIRED", "asset_id": asset_id}

    # Determine filename
    if target_filename:
        # Demo mode: use explicit filename
        valid, error = manager.validate_filename(target_filename)
        if not valid:
            return {"error": "INVALID_FILENAME", "message": error}
        filename = target_filename
    elif manifest_key:
        # Library mode: auto-generate filename
        ext = Path(asset.filename).suffix.lower()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{manifest_key}_{timestamp}{ext}"
    else:
        # Use original filename
        filename = asset.filename

    # Get publish directory
    if publish_dir:
        dest_dir = Path(publish_dir)
    else:
        dest_dir = manager.get_publish_dir()
        if dest_dir is None:
            return {
                "error": "NO_PUBLISH_DIR",
                "message": "No publish directory configured or auto-detected. Set publish_dir or create public/gen/",
            }

    # Get source path
    source = manager.get_source_path(asset)
    if source is None:
        return {
            "error": "SOURCE_NOT_FOUND",
            "message": "Asset file not found at expected location",
            "asset_id": asset_id,
        }

    # Copy to destination
    dest = dest_dir / filename

    # Security: validate destination doesn't escape publish dir (filename with ../ attack)
    try:
        dest_resolved = dest.resolve()
        dir_resolved = dest_dir.resolve()
        if not dest_resolved.is_relative_to(dir_resolved):
            return {
                "error": "PATH_TRAVERSAL",
                "message": "Target filename escapes publish directory",
            }
    except Exception as e:
        return {"error": "INVALID_PATH", "message": str(e)}

    result = manager.copy_asset(source, dest, web_optimize)

    if not result.get("success"):
        return {"error": "COPY_FAILED", "message": result.get("error")}

    # Build response
    response = {
        "success": True,
        "path": str(dest),
        "filename": filename,
        "bytes": result.get("bytes"),
        "mime_type": asset.mime_type,
        "asset_id": asset_id,
    }

    # Determine relative URL
    # Try to infer URL from common patterns
    path_str = str(dest)
    for pattern in ["public/", "static/"]:
        if pattern in path_str:
            idx = path_str.index(pattern) + len(pattern)
            response["url"] = "/" + path_str[idx:]
            break

    # Update manifest if requested
    if manifest_key:
        manifest_path = dest_dir / "manifest.json"
        manifest_updated = manager.update_manifest(manifest_path, manifest_key, response)
        response["manifest_updated"] = manifest_updated

    return response


def get_publish_info() -> dict:
    """Get current publish configuration."""
    manager = get_publish_manager()
    publish_dir = manager.get_publish_dir()

    return {
        "comfyui_output_dir": manager._comfyui_output_dir,
        "publish_dir": str(publish_dir) if publish_dir else None,
        "auto_detected": publish_dir is not None and manager._default_publish_dir is None,
        "max_bytes": manager._max_bytes,
    }


def set_publish_dir(publish_dir: str) -> dict:
    """Set the publish directory."""
    BLOCKED_PUBLISH_PATHS = [
        "/etc",
        "/usr",
        "/bin",
        "/sbin",
        "/boot",
        "/proc",
        "/sys",
        "/dev",
        "/root",
    ]

    manager = get_publish_manager()
    path = Path(publish_dir)

    # Security: validate against sensitive system directories
    try:
        resolved = path.resolve()
        resolved_str = str(resolved)

        # Check if path is under any blocked directory
        for blocked in BLOCKED_PUBLISH_PATHS:
            if resolved_str == blocked or resolved_str.startswith(blocked + "/"):
                return {
                    "error": "FORBIDDEN_PATH",
                    "message": f"Cannot set publish directory under system path: {blocked}",
                }

        # Ensure it's a directory (or will be)
        if resolved.exists() and not resolved.is_dir():
            return {
                "error": "NOT_A_DIRECTORY",
                "message": f"Path '{publish_dir}' exists but is not a directory",
            }

    except Exception as e:
        return {"error": "INVALID_PATH", "message": str(e)}

    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return {"error": "CANNOT_CREATE_DIR", "message": str(e)}

    manager._default_publish_dir = str(path)

    return {
        "success": True,
        "publish_dir": str(path),
    }
