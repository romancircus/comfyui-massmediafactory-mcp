"""
Integration tests for asset lifecycle - Task ROM-202
Tests asset operations, TTL expiration, metadata retrieval
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

# Set up mocks before importing modules
if "comfyui_massmediafactory_mcp.client" not in sys.modules:
    client_mock = types.ModuleType("comfyui_massmediafactory_mcp.client")
    client_mock.get_client = lambda: None
    client_mock.ComfyUIClient = MagicMock
    sys.modules["comfyui_massmediafactory_mcp.client"] = client_mock

if "comfyui_massmediafactory_mcp" not in sys.modules:
    pkg = types.ModuleType("comfyui_massmediafactory_mcp")
    pkg.__path__ = [str(Path(__file__).parent.parent / "src" / "comfyui_massmediafactory_mcp")]
    sys.modules["comfyui_massmediafactory_mcp"] = pkg

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comfyui_massmediafactory_mcp.server import list_assets, get_asset_metadata, view_output, cleanup_assets


class TestListAssets:
    """Tests for list_assets()"""

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_list_all_assets(self, mock_execution):
        """Test listing all assets"""
        mock_execution.list_assets.return_value = {
            "assets": [{"asset_id": "img-1", "type": "image"}, {"asset_id": "vid-1", "type": "video"}]
        }

        result = list_assets()

        assert "assets" in result
        assert len(result["assets"]) == 2

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_list_by_type(self, mock_execution):
        """Test filtering assets by type"""
        mock_execution.list_assets.return_value = {
            "assets": [{"asset_id": "img-1", "type": "image"}, {"asset_id": "img-2", "type": "image"}]
        }

        result = list_assets(asset_type="images")

        assert "assets" in result
        for asset in result["assets"]:
            assert asset["type"] == "image"

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_list_with_pagination(self, mock_execution):
        """Test asset listing with pagination"""
        # Mock should return all assets, pagination happens in server layer
        mock_execution.list_assets.return_value = {"assets": [{"asset_id": f"asset-{i}"} for i in range(100)]}

        result = list_assets(limit=20)

        assert "assets" in result
        assert len(result["assets"]) <= 20
        assert "nextCursor" in result

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_list_by_session(self, mock_execution):
        """Test filtering by session_id"""
        mock_execution.list_assets.return_value = {
            "assets": [{"asset_id": "session-asset-1", "session_id": "sess-123"}]
        }

        result = list_assets(session_id="sess-123")

        assert "assets" in result


class TestGetAssetMetadata:
    """Tests for get_asset_metadata()"""

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_get_metadata_success(self, mock_execution):
        """Test successful metadata retrieval"""
        mock_execution.get_asset_metadata.return_value = {
            "asset_id": "asset-123",
            "type": "image",
            "workflow": {"nodes": []},
            "parameters": {"prompt": "test prompt"},
            "created_at": "2026-02-04T17:00:00",
        }

        result = get_asset_metadata("asset-123")

        assert result.get("asset_id") == "asset-123"
        assert "workflow" in result
        assert "parameters" in result

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_get_metadata_not_found(self, mock_execution):
        """Test metadata for non-existent asset"""
        mock_execution.get_asset_metadata.return_value = {"error": "Asset not found", "isError": True}

        result = get_asset_metadata("nonexistent-id")

        assert "error" in result
        assert result.get("isError") is True

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_get_metadata_workflow_included(self, mock_execution):
        """Test that workflow data is included in metadata"""
        mock_execution.get_asset_metadata.return_value = {
            "asset_id": "asset-123",
            "workflow": {"1": {"class_type": "TestNode", "inputs": {}}, "2": {"class_type": "SaveImage", "inputs": {}}},
        }

        result = get_asset_metadata("asset-123")
        workflow = result.get("workflow", {})

        assert len(workflow) > 0


class TestViewOutput:
    """Tests for view_output()"""

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_view_thumbnail(self, mock_execution):
        """Test viewing asset thumbnail"""
        mock_execution.view_output.return_value = {"url": "/view/asset-123/thumb.png", "mode": "thumb"}

        result = view_output("asset-123", mode="thumb")

        assert "url" in result

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_view_metadata(self, mock_execution):
        """Test viewing asset metadata mode"""
        mock_execution.view_output.return_value = {
            "asset_id": "asset-123",
            "type": "image",
            "dimensions": {"width": 1024, "height": 1024},
        }

        result = view_output("asset-123", mode="metadata")

        assert result.get("asset_id") == "asset-123"


class TestCleanupAssets:
    """Tests for cleanup_assets()"""

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_cleanup_expired_assets(self, mock_execution):
        """Test cleaning up expired assets"""
        mock_execution.cleanup_expired_assets.return_value = {"cleaned": 5, "remaining": 10, "ttl_hours": 24}

        result = cleanup_assets()

        assert "cleaned" in result
        assert "remaining" in result

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_cleanup_no_expired(self, mock_execution):
        """Test cleanup when no assets are expired"""
        mock_execution.cleanup_expired_assets.return_value = {"cleaned": 0, "remaining": 10}

        result = cleanup_assets()

        assert result.get("cleaned") == 0
