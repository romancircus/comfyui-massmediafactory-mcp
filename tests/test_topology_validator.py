"""
Tests for topology_validator module
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comfyui_massmediafactory_mcp import topology_validator


class TestLTXFrameValidation:
    """Tests for LTX frame count validation (8n+1 rule)"""

    def test_valid_frame_counts(self):
        """Test that valid 8n+1 frame counts pass validation"""
        valid_counts = [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]
        for count in valid_counts:
            is_valid, errors = topology_validator.validate_ltx_frames(count)
            assert is_valid, f"Frame count {count} should be valid but got errors: {errors}"
            assert len(errors) == 0

    def test_invalid_frame_counts(self):
        """Test that invalid frame counts fail validation"""
        invalid_counts = [10, 16, 24, 32, 40, 50, 80, 96, 100, 120]
        for count in invalid_counts:
            is_valid, errors = topology_validator.validate_ltx_frames(count)
            assert not is_valid, f"Frame count {count} should be invalid"
            assert len(errors) > 0

    def test_frame_count_too_low(self):
        """Test that frame counts below minimum fail"""
        is_valid, errors = topology_validator.validate_ltx_frames(8)
        assert not is_valid
        assert "too low" in errors[0].lower() or "minimum" in errors[0].lower()


class TestResolutionValidation:
    """Tests for resolution validation"""

    def test_flux_resolution_divisibility(self):
        """Test FLUX resolution must be divisible by 16"""
        # Valid
        is_valid, errors = topology_validator.validate_resolution(1024, 1024, "flux")
        assert is_valid
        assert len(errors) == 0

        # Invalid - not divisible by 16
        is_valid, errors = topology_validator.validate_resolution(1000, 1000, "flux")
        assert not is_valid
        assert len(errors) > 0

    def test_ltx_resolution_divisibility(self):
        """Test LTX resolution must be divisible by 8"""
        # Valid
        is_valid, errors = topology_validator.validate_resolution(768, 512, "ltx")
        assert is_valid

        # Invalid - not divisible by 8
        is_valid, errors = topology_validator.validate_resolution(770, 510, "ltx")
        assert not is_valid

    def test_hunyuan_resolution_divisibility(self):
        """Test Hunyuan resolution must be divisible by 16"""
        # Valid
        is_valid, errors = topology_validator.validate_resolution(1280, 720, "hunyuan")
        assert is_valid

        # Invalid - not divisible by 16
        is_valid, errors = topology_validator.validate_resolution(1280, 718, "hunyuan")
        assert not is_valid

    def test_resolution_range(self):
        """Test resolution range validation"""
        # Too small
        is_valid, errors = topology_validator.validate_resolution(128, 128, "flux")
        assert not is_valid

        # Too large
        is_valid, errors = topology_validator.validate_resolution(4096, 4096, "flux")
        assert not is_valid


class TestCFGValidation:
    """Tests for CFG value validation"""

    def test_ltx_cfg_range(self):
        """Test LTX CFG range (2.0-5.0)"""
        # Valid
        is_valid, warnings = topology_validator.validate_cfg(3.0, "ltx")
        assert is_valid
        assert len(warnings) == 0

        # Out of range
        is_valid, warnings = topology_validator.validate_cfg(10.0, "ltx")
        assert is_valid  # Still valid, just warnings
        assert len(warnings) > 0

    def test_flux_uses_guidance(self):
        """Test FLUX uses FluxGuidance instead of CFG"""
        is_valid, warnings = topology_validator.validate_cfg(3.5, "flux")
        assert is_valid
        assert any("FluxGuidance" in w for w in warnings)


class TestSamplerValidation:
    """Tests for sampler validation"""

    def test_video_unsafe_samplers(self):
        """Test that video-unsafe samplers fail for video models"""
        unsafe_samplers = ["euler_ancestral", "dpmpp_2m_sde", "dpmpp_sde"]
        for sampler in unsafe_samplers:
            is_valid, errors = topology_validator.validate_sampler(sampler, "ltx", is_video=True)
            assert not is_valid, f"Sampler {sampler} should be invalid for video"

    def test_euler_safe_for_video(self):
        """Test that euler sampler is safe for video"""
        is_valid, errors = topology_validator.validate_sampler("euler", "ltx", is_video=True)
        assert is_valid


class TestModelDetection:
    """Tests for model type detection"""

    def test_detect_ltx(self):
        """Test detecting LTX model from workflow"""
        workflow = {
            "1": {"class_type": "LTXVLoader", "inputs": {}},
            "2": {"class_type": "LTXVScheduler", "inputs": {}}
        }
        model = topology_validator.detect_model_type(workflow)
        assert model == "ltx"

    def test_detect_flux(self):
        """Test detecting FLUX model from workflow"""
        workflow = {
            "1": {"class_type": "UNETLoader", "inputs": {}},
            "2": {"class_type": "DualCLIPLoader", "inputs": {}},
            "3": {"class_type": "FluxGuidance", "inputs": {}}
        }
        model = topology_validator.detect_model_type(workflow)
        assert model == "flux"

    def test_detect_hunyuan(self):
        """Test detecting HunyuanVideo model from workflow"""
        workflow = {
            "1": {"class_type": "HunyuanVideoModelLoader", "inputs": {}},
            "2": {"class_type": "HunyuanVideoSampler", "inputs": {}}
        }
        model = topology_validator.detect_model_type(workflow)
        assert model == "hunyuan"

    def test_detect_wan(self):
        """Test detecting Wan model from workflow"""
        workflow = {
            "1": {"class_type": "WanVideoModelLoader", "inputs": {}},
            "2": {"class_type": "WanVAEDecode", "inputs": {}}
        }
        model = topology_validator.detect_model_type(workflow)
        assert model == "wan"


class TestTopologyValidation:
    """Tests for full topology validation"""

    def test_valid_workflow(self):
        """Test that a valid workflow passes validation with all required nodes"""
        workflow = {
            "1": {"class_type": "LTXVLoader", "inputs": {}},
            "2": {"class_type": "LTXVScheduler", "inputs": {"steps": 30}},
            "3": {"class_type": "LTXVConditioning", "inputs": {}},
            "4": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 768, "height": 512, "length": 97}},
            "5": {"class_type": "SamplerCustom", "inputs": {"cfg": 3.0}},
            "6": {"class_type": "VHS_VideoCombine", "inputs": {}}
        }
        result = topology_validator.validate_topology(workflow, "ltx")
        assert result["valid"]

    def test_forbidden_node_detection(self):
        """Test that forbidden nodes are detected"""
        workflow = {
            "1": {"class_type": "LTXVLoader", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {}},  # Forbidden for LTX
        }
        result = topology_validator.validate_topology(workflow, "ltx")
        assert not result["valid"]
        assert any("KSampler" in e for e in result["errors"])


class TestConnectionTypeValidation:
    """Tests for connection type validation"""

    def test_valid_connection(self):
        """Test that valid connections pass"""
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}}  # CLIP from slot 1
        }
        errors = topology_validator.validate_connection_types(workflow)
        assert len(errors) == 0

    def test_missing_source_node(self):
        """Test that missing source node is detected"""
        workflow = {
            "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}}  # Node 1 doesn't exist
        }
        errors = topology_validator.validate_connection_types(workflow)
        assert len(errors) > 0
        assert any("non-existent" in e for e in errors)


class TestAutoCorrection:
    """Tests for auto-correction of parameters"""

    def test_auto_correct_resolution(self):
        """Test auto-correction of resolution"""
        workflow = {
            "1": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 770, "height": 510}}
        }
        result = topology_validator.auto_correct_parameters(workflow, "ltx")
        corrected = result["workflow"]
        assert corrected["1"]["inputs"]["width"] % 8 == 0
        assert corrected["1"]["inputs"]["height"] % 8 == 0
        assert len(result["corrections"]) > 0

    def test_auto_correct_ltx_frames(self):
        """Test auto-correction of LTX frame count"""
        workflow = {
            "1": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"length": 100}}
        }
        result = topology_validator.auto_correct_parameters(workflow, "ltx")
        corrected = result["workflow"]
        frames = corrected["1"]["inputs"]["length"]
        assert (frames - 1) % 8 == 0  # Must be 8n+1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
