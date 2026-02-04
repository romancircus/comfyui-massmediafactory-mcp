"""
Pytest fixtures and utilities for integration tests
"""

import pytest
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def mock_client():
    """Mock ComfyUI client for testing"""
    client = MagicMock()
    client.base_url = "http://localhost:8188"
    client.session = MagicMock()
    return client


@pytest.fixture
def mock_response():
    """Factory for creating mock HTTP responses"""
    def _make_response(status_code=200, json_data=None, content=None):
        response = MagicMock()
        response.status_code = status_code
        if json_data is not None:
            response.json.return_value = json_data
        if content is not None:
            response.content = content
        return response
    return _make_response


@pytest.fixture
def sample_workflow():
    """Sample workflow for testing"""
    return {
        "1": {
            "class_type": "LTXVLoader",
            "inputs": {"unet_name": "ltx2.safetensors"}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "a cat walking"}
        }
    }


@pytest.fixture
def sample_asset():
    """Sample asset metadata for testing"""
    return {
        "asset_id": "test-asset-123",
        "type": "image",
        "path": "/tmp/test.png",
        "created_at": "2026-02-04T17:00:00",
        "workflow": {"nodes": []},
        "parameters": {"prompt": "test"}
    }
