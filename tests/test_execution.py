"""
Integration tests for execution tools - Task ROM-202
Tests execute_workflow(), wait_for_completion(), regenerate()
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

from comfyui_massmediafactory_mcp.server import execute_workflow, wait_for_completion, regenerate


class TestExecuteWorkflow:
    """Tests for execute_workflow() tool"""

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_execute_workflow_success(self, mock_execution):
        """Test successful workflow execution"""
        mock_execution.execute_workflow.return_value = {"prompt_id": "test-123"}

        workflow = {"1": {"class_type": "TestNode", "inputs": {}}}
        result = execute_workflow(workflow, client_id="test-client")

        assert result.get("prompt_id") == "test-123"
        mock_execution.execute_workflow.assert_called_once()

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_execute_workflow_error(self, mock_execution):
        """Test workflow execution with error"""
        mock_execution.execute_workflow.return_value = {"error": "Invalid workflow", "isError": True}

        workflow = {"invalid": "workflow"}
        result = execute_workflow(workflow)

        assert "error" in result
        assert result.get("isError") is True

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_execute_workflow_empty_workflow(self, mock_execution):
        """Test execution with empty workflow"""
        mock_execution.execute_workflow.return_value = {"error": "Empty workflow"}

        result = execute_workflow({})
        assert "error" in result


class TestWaitForCompletion:
    """Tests for wait_for_completion() tool"""

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_wait_success(self, mock_execution):
        """Test successful completion waiting"""
        mock_execution.wait_for_completion.return_value = {
            "status": "completed",
            "outputs": [{"asset_id": "asset-123", "filename": "test.png"}],
        }

        result = wait_for_completion("prompt-123", timeout_seconds=60)

        assert result.get("status") == "completed"
        assert "outputs" in result
        mock_execution.wait_for_completion.assert_called_once()

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_wait_timeout(self, mock_execution):
        """Test timeout handling"""
        mock_execution.wait_for_completion.return_value = {"error": "Timeout waiting for completion", "isError": True}

        result = wait_for_completion("prompt-123", timeout_seconds=1)

        assert "error" in result
        assert result.get("isError") is True

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_wait_invalid_prompt_id(self, mock_execution):
        """Test with invalid prompt ID"""
        mock_execution.wait_for_completion.return_value = {"error": "Invalid prompt_id"}

        result = wait_for_completion("", timeout_seconds=60)
        assert "error" in result


class TestRegenerate:
    """Tests for regenerate() tool"""

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_regenerate_with_new_seed(self, mock_execution):
        """Test regeneration with new seed"""
        mock_execution.regenerate.return_value = {"prompt_id": "new-123"}

        result = regenerate(asset_id="asset-123", seed=42, prompt="New prompt", cfg=3.5)

        assert result.get("prompt_id") == "new-123"
        mock_execution.regenerate.assert_called_once()

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_regenerate_with_random_seed(self, mock_execution):
        """Test regeneration with random seed (None)"""
        mock_execution.regenerate.return_value = {"prompt_id": "new-456"}

        result = regenerate(asset_id="asset-123", seed=None)

        assert result.get("prompt_id") == "new-456"

    @patch("comfyui_massmediafactory_mcp.server.execution")
    def test_regenerate_invalid_asset(self, mock_execution):
        """Test regeneration with invalid asset_id"""
        mock_execution.regenerate.return_value = {"error": "Asset not found"}

        result = regenerate(asset_id="invalid-id")
        assert "error" in result
