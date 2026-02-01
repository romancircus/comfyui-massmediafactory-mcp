"""
Tests for workflow_generator module
"""

import pytest
import sys
import types
from pathlib import Path

# Set up mocks before importing modules
if 'comfyui_massmediafactory_mcp.client' not in sys.modules:
    client_mock = types.ModuleType('comfyui_massmediafactory_mcp.client')
    client_mock.get_client = lambda: None
    sys.modules['comfyui_massmediafactory_mcp.client'] = client_mock

if 'comfyui_massmediafactory_mcp' not in sys.modules:
    pkg = types.ModuleType('comfyui_massmediafactory_mcp')
    pkg.__path__ = [str(Path(__file__).parent.parent / "src" / "comfyui_massmediafactory_mcp")]
    sys.modules['comfyui_massmediafactory_mcp'] = pkg

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comfyui_massmediafactory_mcp import workflow_generator


class TestEdgeCaseHandling:
    """Tests for edge case handling in generate_workflow"""

    def test_empty_prompt_error(self):
        """Test that empty prompt returns error"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "")
        assert "error" in result
        assert "prompt" in result["error"].lower()

    def test_whitespace_prompt_error(self):
        """Test that whitespace-only prompt returns error"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "   ")
        assert "error" in result
        assert "prompt" in result["error"].lower()

    def test_zero_width_error(self):
        """Test that width=0 returns error"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "A cat", width=0)
        assert "error" in result
        assert "width" in result["error"].lower()

    def test_negative_width_error(self):
        """Test that negative width returns error"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "A cat", width=-100)
        assert "error" in result
        assert "width" in result["error"].lower()

    def test_zero_height_error(self):
        """Test that height=0 returns error"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "A cat", height=0)
        assert "error" in result
        assert "height" in result["error"].lower()

    def test_zero_frames_error(self):
        """Test that frames=0 returns error"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "A cat", frames=0)
        assert "error" in result
        assert "frame" in result["error"].lower()

    def test_zero_steps_error(self):
        """Test that steps=0 returns error"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "A cat", steps=0)
        assert "error" in result
        assert "steps" in result["error"].lower()

    def test_zero_cfg_error(self):
        """Test that cfg=0 returns error"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "A cat", cfg=0)
        assert "error" in result
        assert "cfg" in result["error"].lower()


class TestWorkflowGeneration:
    """Tests for successful workflow generation"""

    def test_ltx2_txt2vid_generation(self):
        """Test LTX-2 text-to-video workflow generation"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "A cat walking")
        assert "workflow" in result
        assert "parameters_used" in result
        assert result["workflow"] is not None

    def test_flux2_txt2img_generation(self):
        """Test FLUX.2 text-to-image workflow generation"""
        result = workflow_generator.generate_workflow("flux2", "txt2img", "A portrait")
        assert "workflow" in result
        assert result["workflow"] is not None

    def test_sdxl_txt2img_generation(self):
        """Test SDXL text-to-image workflow generation"""
        result = workflow_generator.generate_workflow("sdxl", "txt2img", "A landscape")
        assert "workflow" in result
        assert result["workflow"] is not None

    def test_hunyuan_txt2vid_generation(self):
        """Test HunyuanVideo text-to-video workflow generation"""
        result = workflow_generator.generate_workflow("hunyuan15", "txt2vid", "A dragon flying")
        assert "workflow" in result
        assert result["workflow"] is not None

    def test_wan_txt2vid_generation(self):
        """Test Wan text-to-video workflow generation"""
        result = workflow_generator.generate_workflow("wan26", "txt2vid", "A sunset timelapse")
        assert "workflow" in result
        assert result["workflow"] is not None


class TestModelAliases:
    """Tests for model name aliases"""

    def test_ltx_alias(self):
        """Test that 'ltx' maps to 'ltx2'"""
        result = workflow_generator.generate_workflow("ltx", "t2v", "A cat")
        assert "workflow" in result

    def test_flux_alias(self):
        """Test that 'flux' maps to 'flux2'"""
        result = workflow_generator.generate_workflow("flux", "t2i", "A portrait")
        assert "workflow" in result

    def test_wan_alias(self):
        """Test that 'wan' maps to 'wan26'"""
        result = workflow_generator.generate_workflow("wan", "t2v", "A landscape")
        assert "workflow" in result

    def test_hunyuan_alias(self):
        """Test that 'hunyuan' maps to 'hunyuan15'"""
        result = workflow_generator.generate_workflow("hunyuan", "t2v", "A dragon")
        assert "workflow" in result


class TestWorkflowTypeAliases:
    """Tests for workflow type aliases"""

    def test_t2v_alias(self):
        """Test that 't2v' works as alias for 'txt2vid'"""
        result = workflow_generator.generate_workflow("ltx2", "t2v", "A cat")
        assert "workflow" in result

    def test_i2v_alias(self):
        """Test that 'i2v' works as alias for 'img2vid'"""
        result = workflow_generator.generate_workflow("ltx2", "i2v", "A cat")
        assert "workflow" in result

    def test_t2i_alias(self):
        """Test that 't2i' works as alias for 'txt2img'"""
        result = workflow_generator.generate_workflow("flux2", "t2i", "A portrait")
        assert "workflow" in result


class TestParameterInjection:
    """Tests for parameter injection into workflows"""

    def test_prompt_injection(self):
        """Test that prompt is injected into workflow"""
        prompt = "A majestic dragon breathing fire"
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", prompt)
        workflow_str = str(result["workflow"])
        assert prompt in workflow_str

    def test_seed_injection(self):
        """Test that seed is injected into workflow"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "A cat", seed=12345)
        assert result["parameters_used"]["SEED"] == 12345

    def test_random_seed_generation(self):
        """Test that random seed is generated when not provided"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "A cat")
        assert "SEED" in result["parameters_used"]
        assert isinstance(result["parameters_used"]["SEED"], int)

    def test_resolution_injection(self):
        """Test that resolution is injected into workflow"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "A cat", width=1024, height=768)
        assert result["parameters_used"]["WIDTH"] == 1024
        assert result["parameters_used"]["HEIGHT"] == 768


class TestAutoCorrection:
    """Tests for auto-correction of parameters"""

    def test_auto_correct_resolution(self):
        """Test that resolution is auto-corrected"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "A cat", width=770, auto_correct=True)
        # Should be corrected to nearest divisible by 8
        assert result["parameters_used"]["WIDTH"] % 8 == 0
        assert len(result["auto_corrections"]) > 0

    def test_auto_correct_frames(self):
        """Test that LTX frame count is auto-corrected to 8n+1"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "A cat", frames=100, auto_correct=True)
        frames = result["parameters_used"]["FRAMES"]
        assert (frames - 1) % 8 == 0


class TestDefaultValues:
    """Tests for default parameter values"""

    def test_ltx_defaults(self):
        """Test LTX default values"""
        result = workflow_generator.generate_workflow("ltx2", "txt2vid", "A cat")
        params = result["parameters_used"]
        assert params["WIDTH"] == 768
        assert params["HEIGHT"] == 512
        assert params["FRAMES"] == 97

    def test_flux_defaults(self):
        """Test FLUX default values"""
        result = workflow_generator.generate_workflow("flux2", "txt2img", "A portrait")
        params = result["parameters_used"]
        assert params["WIDTH"] == 1024
        assert params["HEIGHT"] == 1024

    def test_hunyuan_defaults(self):
        """Test HunyuanVideo default values"""
        result = workflow_generator.generate_workflow("hunyuan15", "txt2vid", "A dragon")
        params = result["parameters_used"]
        assert params["WIDTH"] == 1280
        assert params["HEIGHT"] == 720
        assert params["FRAMES"] == 81


class TestSkeletonLoading:
    """Tests for skeleton loading"""

    def test_load_skeleton_success(self):
        """Test successful skeleton loading"""
        skeleton, name = workflow_generator.load_skeleton("ltx2", "txt2vid")
        assert skeleton is not None
        assert name is not None

    def test_load_skeleton_invalid_model(self):
        """Test skeleton loading with invalid model"""
        skeleton, error = workflow_generator.load_skeleton("invalid_model", "txt2vid")
        assert skeleton is None
        assert "No skeleton" in error

    def test_load_skeleton_invalid_type(self):
        """Test skeleton loading with invalid workflow type"""
        skeleton, error = workflow_generator.load_skeleton("ltx2", "invalid_type")
        assert skeleton is None
        assert "No skeleton" in error


class TestListSupportedWorkflows:
    """Tests for listing supported workflows"""

    def test_list_workflows(self):
        """Test listing supported workflows"""
        result = workflow_generator.list_supported_workflows()
        assert "workflows" in result
        assert "count" in result
        assert result["count"] > 0

    def test_all_models_present(self):
        """Test that all expected models are in the list"""
        result = workflow_generator.list_supported_workflows()
        models = {w["model"] for w in result["workflows"]}
        expected = {"ltx", "ltx2", "flux", "flux2", "wan", "wan26", "qwen", "sdxl", "hunyuan", "hunyuan15"}
        for model in expected:
            assert model in models, f"Model {model} not in supported workflows"


class TestCacheClearing:
    """Tests for skeleton cache management"""

    def test_clear_cache(self):
        """Test that cache can be cleared without error"""
        # Load a skeleton to populate cache
        workflow_generator.load_skeleton("ltx2", "txt2vid")

        # Clear cache
        workflow_generator.clear_skeleton_cache()

        # Should still work after clearing
        skeleton, name = workflow_generator.load_skeleton("ltx2", "txt2vid")
        assert skeleton is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
