"""
Tests for patterns module
"""

import pytest
import sys
import types
from pathlib import Path

# Set up mocks before importing modules
if "comfyui_massmediafactory_mcp.client" not in sys.modules:
    client_mock = types.ModuleType("comfyui_massmediafactory_mcp.client")
    client_mock.get_client = lambda: None
    sys.modules["comfyui_massmediafactory_mcp.client"] = client_mock

if "comfyui_massmediafactory_mcp" not in sys.modules:
    pkg = types.ModuleType("comfyui_massmediafactory_mcp")
    pkg.__path__ = [str(Path(__file__).parent.parent / "src" / "comfyui_massmediafactory_mcp")]
    sys.modules["comfyui_massmediafactory_mcp"] = pkg

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comfyui_massmediafactory_mcp import patterns


class TestWorkflowSkeletons:
    """Tests for workflow skeleton retrieval"""

    def test_get_ltx2_txt2vid(self):
        """Test getting LTX-2 txt2vid skeleton"""
        skeleton = patterns.get_workflow_skeleton("ltx2", "txt2vid")
        assert "error" not in skeleton
        assert "_meta" in skeleton

    def test_get_flux2_txt2img(self):
        """Test getting FLUX.2 txt2img skeleton"""
        skeleton = patterns.get_workflow_skeleton("flux2", "txt2img")
        assert "error" not in skeleton
        assert "_meta" in skeleton

    def test_get_hunyuan15_txt2vid(self):
        """Test getting HunyuanVideo txt2vid skeleton"""
        skeleton = patterns.get_workflow_skeleton("hunyuan15", "txt2vid")
        assert "error" not in skeleton
        assert "_meta" in skeleton

    def test_get_hunyuan15_img2vid(self):
        """Test getting HunyuanVideo img2vid skeleton"""
        skeleton = patterns.get_workflow_skeleton("hunyuan15", "img2vid")
        assert "error" not in skeleton
        assert "_meta" in skeleton

    def test_get_sdxl_txt2img(self):
        """Test getting SDXL txt2img skeleton"""
        skeleton = patterns.get_workflow_skeleton("sdxl", "txt2img")
        assert "error" not in skeleton
        assert "_meta" in skeleton

    def test_get_wan21_img2vid(self):
        """Test getting Wan 2.1 img2vid skeleton"""
        skeleton = patterns.get_workflow_skeleton("wan21", "img2vid")
        assert "error" not in skeleton
        assert "_meta" in skeleton

    def test_get_invalid_skeleton(self):
        """Test getting invalid skeleton returns error"""
        skeleton = patterns.get_workflow_skeleton("invalid", "invalid")
        assert "error" in skeleton

    def test_skeleton_has_nodes(self):
        """Test that skeleton has numbered nodes"""
        skeleton = patterns.get_workflow_skeleton("ltx2", "txt2vid")
        # Should have numbered node keys
        node_keys = [k for k in skeleton.keys() if k.isdigit()]
        assert len(node_keys) > 0


class TestModelConstraints:
    """Tests for model constraints retrieval"""

    def test_get_ltx2_constraints(self):
        """Test getting LTX-2 constraints"""
        constraints = patterns.get_model_constraints("ltx2")
        assert "error" not in constraints
        assert "cfg" in constraints
        assert "resolution" in constraints

    def test_get_flux2_constraints(self):
        """Test getting FLUX.2 constraints"""
        constraints = patterns.get_model_constraints("flux2")
        assert "error" not in constraints
        assert constraints["cfg"]["via"] == "FluxGuidance"

    def test_get_hunyuan15_constraints(self):
        """Test getting HunyuanVideo constraints"""
        constraints = patterns.get_model_constraints("hunyuan15")
        assert "error" not in constraints
        assert constraints["resolution"]["divisible_by"] == 16

    def test_get_invalid_constraints(self):
        """Test getting invalid model constraints returns error"""
        constraints = patterns.get_model_constraints("invalid")
        assert "error" in constraints


class TestNodeChains:
    """Tests for node chain retrieval"""

    def test_get_ltx2_txt2vid_chain(self):
        """Test getting LTX-2 txt2vid node chain"""
        chain = patterns.get_node_chain("ltx2", "txt2vid")
        assert isinstance(chain, list)
        assert len(chain) > 0

    def test_get_flux2_txt2img_chain(self):
        """Test getting FLUX.2 txt2img node chain"""
        chain = patterns.get_node_chain("flux2", "txt2img")
        assert isinstance(chain, list)
        assert len(chain) > 0

    def test_get_hunyuan15_txt2vid_chain(self):
        """Test getting HunyuanVideo txt2vid node chain"""
        chain = patterns.get_node_chain("hunyuan15", "txt2vid")
        assert isinstance(chain, list)
        assert len(chain) > 0

    def test_get_invalid_chain(self):
        """Test getting invalid node chain returns error"""
        chain = patterns.get_node_chain("invalid", "invalid")
        assert "error" in chain


class TestPatternValidation:
    """Tests for pattern validation"""

    def test_validate_valid_workflow(self):
        """Test validating a valid workflow with all required nodes"""
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
            "2": {"class_type": "LTXVScheduler", "inputs": {}},
            "3": {"class_type": "LTXVConditioning", "inputs": {}},
            "4": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
            "5": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 768, "height": 512, "length": 97}},
            "6": {"class_type": "SaveVideo", "inputs": {}},
            "7": {"class_type": "LTXAVTextEncoderLoader", "inputs": {}},
            "8": {"class_type": "LTXVAudioVAELoader", "inputs": {}},
            "9": {"class_type": "CFGGuider", "inputs": {}},
            "10": {"class_type": "LTXVConcatAVLatent", "inputs": {}},
            "11": {"class_type": "LTXVSeparateAVLatent", "inputs": {}},
        }
        result = patterns.validate_against_pattern(workflow, "ltx2")
        assert result["valid"]

    def test_validate_forbidden_node(self):
        """Test validating workflow with forbidden node"""
        workflow = {
            "1": {"class_type": "LTXVLoader", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {}},  # Forbidden for LTX
        }
        result = patterns.validate_against_pattern(workflow, "ltx2")
        assert not result["valid"]
        assert any("KSampler" in e for e in result["errors"])

    def test_validate_cfg_range(self):
        """Test validating CFG range"""
        workflow = {
            "1": {"class_type": "LTXVLoader", "inputs": {}},
            "2": {"class_type": "SamplerCustom", "inputs": {"cfg": 15.0}},  # Too high for LTX
        }
        result = patterns.validate_against_pattern(workflow, "ltx2")
        assert len(result["errors"]) > 0

    def test_validate_resolution_divisibility(self):
        """Test validating resolution divisibility"""
        workflow = {
            "1": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1000, "height": 1000}}  # Not divisible by 16
        }
        result = patterns.validate_against_pattern(workflow, "flux2")
        assert len(result["errors"]) > 0


class TestListAvailablePatterns:
    """Tests for listing available patterns"""

    def test_list_patterns(self):
        """Test listing available patterns"""
        result = patterns.list_available_patterns()
        assert "skeletons" in result
        assert "models" in result
        assert "total" in result
        assert result["total"] > 0

    def test_all_models_in_list(self):
        """Test that all expected models are listed"""
        result = patterns.list_available_patterns()
        expected_models = ["ltx2", "flux2", "wan21", "qwen", "sdxl", "hunyuan15"]
        for model in expected_models:
            assert model in result["models"], f"Model {model} not in available patterns"

    def test_skeleton_has_metadata(self):
        """Test that skeletons in list have metadata"""
        result = patterns.list_available_patterns()
        for skeleton in result["skeletons"]:
            assert "model" in skeleton
            assert "task" in skeleton
            assert "description" in skeleton


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
