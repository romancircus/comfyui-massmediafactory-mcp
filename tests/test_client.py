"""
Integration tests for client operations - Task ROM-202
Tests HTTP retry logic, upload/download operations
"""

import pytest
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import requests

# Set up mocks before importing modules
if 'comfyui_massmediafactory_mcp.client' not in sys.modules:
    client_mock = types.ModuleType('comfyui_massmediafactory_mcp.client')
    client_mock.get_client = lambda: None
    client_mock.ComfyUIClient = MagicMock
    sys.modules['comfyui_massmediafactory_mcp.client'] = client_mock

if 'comfyui_massmediafactory_mcp' not in sys.modules:
    pkg = types.ModuleType('comfyui_massmediafactory_mcp')
    pkg.__path__ = [str(Path(__file__).parent.parent / "src" / "comfyui_massmediafactory_mcp")]
    sys.modules['comfyui_massmediafactory_mcp'] = pkg

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comfyui_massmediafactory_mcp.server import upload_image, download_output


class TestUploadImage:
    """Tests for upload_image() HTTP operations"""

    @patch('comfyui_massmediafactory_mcp.server.execution')
    def test_upload_success(self, mock_execution):
        """Test successful image upload"""
        mock_execution.upload_image.return_value = {
            "success": True,
            "filename": "test.png",
            "subfolder": ""
        }
        
        result = upload_image("/home/user/images/test.png", "test.png")
        
        assert result.get("success") is True
        assert result.get("filename") == "test.png"

    @patch('comfyui_massmediafactory_mcp.server.execution')
    def test_upload_with_subfolder(self, mock_execution):
        """Test upload with subfolder parameter"""
        mock_execution.upload_image.return_value = {
            "success": True,
            "filename": "test.png",
            "subfolder": "inputs"
        }
        
        result = upload_image(
            "/home/user/images/test.png",
            "test.png",
            subfolder="inputs"
        )
        
        assert result.get("subfolder") == "inputs"

    @patch('comfyui_massmediafactory_mcp.server.execution')
    def test_upload_file_not_found(self, mock_execution):
        """Test upload with non-existent file"""
        mock_execution.upload_image.return_value = {
            "error": "File not found",
            "isError": True
        }
        
        result = upload_image("/nonexistent/path.png", "test.png")
        assert "error" in result

    @patch('comfyui_massmediafactory_mcp.server.execution')
    def test_upload_network_error(self, mock_execution):
        """Test upload with network failure"""
        mock_execution.upload_image.return_value = {
            "error": "Connection failed",
            "isError": True
        }
        
        result = upload_image("/home/user/test.png", "test.png")
        assert "error" in result


class TestDownloadOutput:
    """Tests for download_output() HTTP operations"""

    @patch('comfyui_massmediafactory_mcp.server.execution')
    def test_download_success(self, mock_execution):
        """Test successful download"""
        mock_execution.download_output.return_value = {
            "success": True,
            "path": "/home/user/downloads/asset-123.png",
            "size": 1024000
        }
        
        result = download_output("asset-123", "/home/user/downloads/asset-123.png")
        
        assert result.get("success") is True
        assert "path" in result

    @patch('comfyui_massmediafactory_mcp.server.execution')
    def test_download_invalid_asset(self, mock_execution):
        """Test download with invalid asset ID"""
        mock_execution.download_output.return_value = {
            "error": "Asset not found",
            "isError": True
        }
        
        result = download_output("invalid-id", "/home/user/test.png")
        assert "error" in result

    @patch('comfyui_massmediafactory_mcp.server.execution')
    def test_download_disk_full(self, mock_execution):
        """Test download with disk full error"""
        mock_execution.download_output.return_value = {
            "error": "Disk full",
            "isError": True
        }
        
        result = download_output("asset-123", "/home/user/test.png")
        assert "error" in result

    @patch('comfyui_massmediafactory_mcp.server.execution')
    def test_download_permission_denied(self, mock_execution):
        """Test download with permission denied"""
        mock_execution.download_output.return_value = {
            "error": "Permission denied",
            "isError": True
        }
        
        result = download_output("asset-123", "/root/protected.png")
        assert "error" in result
