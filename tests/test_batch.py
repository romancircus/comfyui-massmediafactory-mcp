"""
Integration tests for batch execution - Task ROM-202
Tests batch execution, sweep mode, seeds mode
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comfyui_massmediafactory_mcp.server import batch_execute


class TestBatchExecute:
    """Tests for batch_execute() modes"""

    @patch('comfyui_massmediafactory_mcp.server.batch')
    def test_batch_mode_success(self, mock_batch):
        """Test batch mode execution"""
        mock_batch.execute_batch.return_value = {
            "results": [
                {"prompt_id": "batch-1"},
                {"prompt_id": "batch-2"}
            ],
            "success": True
        }
        
        workflow = {"1": {"class_type": "TestNode"}}
        parameter_sets = [
            {"seed": 1, "cfg": 3.5},
            {"seed": 2, "cfg": 4.0}
        ]
        
        result = batch_execute(
            workflow=workflow,
            mode="batch",
            parameter_sets=parameter_sets,
            parallel=1
        )
        
        assert result.get("success") is True
        assert len(result.get("results", [])) == 2

    @patch('comfyui_massmediafactory_mcp.server.batch')
    def test_sweep_mode_success(self, mock_batch):
        """Test sweep mode execution"""
        mock_batch.execute_sweep.return_value = {
            "results": [
                {"prompt_id": "sweep-1", "params": {"cfg": 2.5}},
                {"prompt_id": "sweep-2", "params": {"cfg": 3.5}},
                {"prompt_id": "sweep-3", "params": {"cfg": 4.5}}
            ],
            "success": True
        }
        
        workflow = {"1": {"class_type": "TestNode"}}
        sweep_params = {"cfg": [2.5, 3.5, 4.5]}
        
        result = batch_execute(
            workflow=workflow,
            mode="sweep",
            sweep_params=sweep_params,
            fixed_params={"steps": 20}
        )
        
        assert result.get("success") is True
        assert len(result.get("results", [])) == 3

    @patch('comfyui_massmediafactory_mcp.server.batch')
    def test_seeds_mode_success(self, mock_batch):
        """Test seeds mode execution"""
        mock_batch.execute_seed_variations.return_value = {
            "results": [
                {"prompt_id": "seed-1", "seed": 42},
                {"prompt_id": "seed-2", "seed": 43},
                {"prompt_id": "seed-3", "seed": 44}
            ],
            "success": True
        }
        
        workflow = {"1": {"class_type": "TestNode"}}
        
        result = batch_execute(
            workflow=workflow,
            mode="seeds",
            num_variations=3,
            start_seed=42
        )
        
        assert result.get("success") is True
        assert len(result.get("results", [])) == 3

    def test_invalid_mode(self):
        """Test with invalid mode parameter"""
        workflow = {"1": {"class_type": "TestNode"}}
        
        result = batch_execute(
            workflow=workflow,
            mode="invalid_mode"
        )
        
        assert "error" in result
        assert "Invalid mode" in result["error"]

    @patch('comfyui_massmediafactory_mcp.server.batch')
    def test_batch_missing_parameter_sets(self, mock_batch):
        """Test batch mode without parameter_sets"""
        workflow = {"1": {"class_type": "TestNode"}}
        
        result = batch_execute(
            workflow=workflow,
            mode="batch"
        )
        
        assert "error" in result
        assert "parameter_sets required" in result["error"]

    @patch('comfyui_massmediafactory_mcp.server.batch')
    def test_sweep_missing_sweep_params(self, mock_batch):
        """Test sweep mode without sweep_params"""
        workflow = {"1": {"class_type": "TestNode"}}
        
        result = batch_execute(
            workflow=workflow,
            mode="sweep"
        )
        
        assert "error" in result
        assert "sweep_params required" in result["error"]

    @patch('comfyui_massmediafactory_mcp.server.batch')
    def test_batch_parallel_execution(self, mock_batch):
        """Test batch execution with parallel processing"""
        mock_batch.execute_batch.return_value = {
            "results": [
                {"prompt_id": f"parallel-{i}"} for i in range(4)
            ],
            "success": True
        }
        
        workflow = {"1": {"class_type": "TestNode"}}
        parameter_sets = [{"seed": i} for i in range(4)]
        
        result = batch_execute(
            workflow=workflow,
            mode="batch",
            parameter_sets=parameter_sets,
            parallel=2
        )
        
        assert result.get("success") is True
        assert len(result.get("results", [])) == 4
