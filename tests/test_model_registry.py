"""
Tests for model_registry module - Single Source of Truth for Model Definitions

These tests ensure the model registry:
1. Contains all expected models
2. Returns correct constraints and defaults
3. Properly resolves aliases
4. Maintains backwards compatibility with existing code
"""

import pytest
import sys
import types
from pathlib import Path

# Set up mocks before importing modules
# Mock the client module to avoid importing mcp
if "comfyui_massmediafactory_mcp.client" not in sys.modules:
    client_mock = types.ModuleType("comfyui_massmediafactory_mcp.client")
    client_mock.get_client = lambda: None
    sys.modules["comfyui_massmediafactory_mcp.client"] = client_mock

# Create package namespace if needed
if "comfyui_massmediafactory_mcp" not in sys.modules:
    pkg = types.ModuleType("comfyui_massmediafactory_mcp")
    pkg.__path__ = [str(Path(__file__).parent.parent / "src" / "comfyui_massmediafactory_mcp")]
    sys.modules["comfyui_massmediafactory_mcp"] = pkg

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import model_registry
from comfyui_massmediafactory_mcp import model_registry


class TestModelConstraints:
    """Tests for get_model_constraints()"""

    def test_ltx2_constraints_exist(self):
        """Test LTX-2 constraints are defined"""
        constraints = model_registry.get_model_constraints("ltx2")
        assert "error" not in constraints
        assert "cfg" in constraints
        assert "resolution" in constraints
        assert "frames" in constraints

    def test_flux2_constraints_exist(self):
        """Test FLUX.2 constraints are defined"""
        constraints = model_registry.get_model_constraints("flux2")
        assert "error" not in constraints
        assert constraints["cfg"]["via"] == "FluxGuidance"

    def test_wan21_constraints_exist(self):
        """Test Wan 2.1 constraints are defined"""
        constraints = model_registry.get_model_constraints("wan21")
        assert "error" not in constraints
        assert "sampler_params" in constraints

    def test_qwen_constraints_exist(self):
        """Test Qwen constraints are defined"""
        constraints = model_registry.get_model_constraints("qwen")
        assert "error" not in constraints
        assert "shift" in constraints

    def test_sdxl_constraints_exist(self):
        """Test SDXL constraints are defined"""
        constraints = model_registry.get_model_constraints("sdxl")
        assert "error" not in constraints

    def test_hunyuan15_constraints_exist(self):
        """Test HunyuanVideo 1.5 constraints are defined"""
        constraints = model_registry.get_model_constraints("hunyuan15")
        assert "error" not in constraints
        assert constraints["resolution"]["divisible_by"] == 16

    def test_qwen_edit_constraints_exist(self):
        """Test Qwen Edit constraints are defined"""
        constraints = model_registry.get_model_constraints("qwen_edit")
        assert "error" not in constraints
        assert "denoise" in constraints
        assert "workflow_notes" in constraints

    def test_invalid_model_returns_error(self):
        """Test invalid model returns error with available models"""
        constraints = model_registry.get_model_constraints("invalid_model")
        assert "error" in constraints
        assert "available" in constraints

    def test_constraints_have_required_fields(self):
        """Test all constraints have required fields"""
        required_fields = ["display_name", "type", "cfg", "resolution", "steps"]
        for model in model_registry.list_supported_models():
            constraints = model_registry.get_model_constraints(model)
            for field in required_fields:
                assert field in constraints, f"Model {model} missing field {field}"


class TestModelDefaults:
    """Tests for get_model_defaults()"""

    def test_ltx2_defaults(self):
        """Test LTX-2 default values"""
        defaults = model_registry.get_model_defaults("ltx2")
        assert defaults["width"] == 768
        assert defaults["height"] == 512
        assert defaults["frames"] == 97
        assert defaults["cfg"] == 3.0

    def test_flux2_defaults(self):
        """Test FLUX.2 default values"""
        defaults = model_registry.get_model_defaults("flux2")
        assert defaults["width"] == 1024
        assert defaults["height"] == 1024
        assert defaults["guidance"] == 3.5

    def test_wan21_defaults(self):
        """Test Wan 2.1 default values"""
        defaults = model_registry.get_model_defaults("wan21")
        assert defaults["width"] == 832
        assert defaults["height"] == 480
        assert defaults["frames"] == 81

    def test_qwen_defaults(self):
        """Test Qwen default values (1328 is official 1:1 native from HuggingFace)"""
        defaults = model_registry.get_model_defaults("qwen")
        assert defaults["width"] == 1328
        assert defaults["height"] == 1328

    def test_hunyuan15_defaults(self):
        """Test HunyuanVideo 1.5 default values"""
        defaults = model_registry.get_model_defaults("hunyuan15")
        assert defaults["width"] == 1280
        assert defaults["height"] == 720

    def test_invalid_model_returns_empty_dict(self):
        """Test invalid model returns empty dict"""
        defaults = model_registry.get_model_defaults("invalid_model")
        assert defaults == {}


class TestModelAliases:
    """Tests for model alias resolution"""

    def test_ltx_resolves_to_ltx2(self):
        """Test 'ltx' alias resolves to 'ltx2'"""
        assert model_registry.resolve_model_name("ltx") == "ltx2"

    def test_flux_resolves_to_flux2(self):
        """Test 'flux' alias resolves to 'flux2'"""
        assert model_registry.resolve_model_name("flux") == "flux2"

    def test_wan_resolves_to_wan21(self):
        """Test 'wan' alias resolves to 'wan21'"""
        assert model_registry.resolve_model_name("wan") == "wan21"

    def test_hunyuan_resolves_to_hunyuan15(self):
        """Test 'hunyuan' alias resolves to 'hunyuan15'"""
        assert model_registry.resolve_model_name("hunyuan") == "hunyuan15"

    def test_canonical_names_unchanged(self):
        """Test canonical names remain unchanged"""
        assert model_registry.resolve_model_name("ltx2") == "ltx2"
        assert model_registry.resolve_model_name("flux2") == "flux2"
        assert model_registry.resolve_model_name("wan26") == "wan21"
        assert model_registry.resolve_model_name("qwen") == "qwen"
        assert model_registry.resolve_model_name("sdxl") == "sdxl"

    def test_unknown_model_returned_as_is(self):
        """Test unknown model names returned as-is (lowercase)"""
        assert model_registry.resolve_model_name("newmodel") == "newmodel"
        assert model_registry.resolve_model_name("UNKNOWN") == "unknown"

    def test_case_insensitive(self):
        """Test alias resolution is case-insensitive"""
        assert model_registry.resolve_model_name("LTX") == "ltx2"
        assert model_registry.resolve_model_name("Flux") == "flux2"
        assert model_registry.resolve_model_name("WAN") == "wan21"


class TestWorkflowTypeAliases:
    """Tests for workflow type alias resolution"""

    def test_t2v_resolves_to_txt2vid(self):
        """Test 't2v' alias resolves to 'txt2vid'"""
        assert model_registry.resolve_workflow_type("t2v") == "txt2vid"

    def test_i2v_resolves_to_img2vid(self):
        """Test 'i2v' alias resolves to 'img2vid'"""
        assert model_registry.resolve_workflow_type("i2v") == "img2vid"

    def test_t2i_resolves_to_txt2img(self):
        """Test 't2i' alias resolves to 'txt2img'"""
        assert model_registry.resolve_workflow_type("t2i") == "txt2img"

    def test_canonical_types_unchanged(self):
        """Test canonical types remain unchanged"""
        assert model_registry.resolve_workflow_type("txt2vid") == "txt2vid"
        assert model_registry.resolve_workflow_type("img2vid") == "img2vid"
        assert model_registry.resolve_workflow_type("txt2img") == "txt2img"


class TestCanonicalModelKey:
    """Tests for get_canonical_model_key()"""

    def test_ltx_t2v_resolves(self):
        """Test ltx/t2v resolves correctly"""
        key = model_registry.get_canonical_model_key("ltx", "t2v")
        assert key == ("ltx2", "txt2vid")

    def test_flux_t2i_resolves(self):
        """Test flux/t2i resolves correctly"""
        key = model_registry.get_canonical_model_key("flux", "t2i")
        assert key == ("flux2", "txt2img")

    def test_wan_i2v_resolves(self):
        """Test wan/i2v resolves correctly"""
        key = model_registry.get_canonical_model_key("wan", "i2v")
        assert key == ("wan21", "img2vid")

    def test_hunyuan_t2v_resolves(self):
        """Test hunyuan/t2v resolves correctly"""
        key = model_registry.get_canonical_model_key("hunyuan", "t2v")
        assert key == ("hunyuan15", "txt2vid")

    def test_invalid_combination_returns_none(self):
        """Test invalid model/type combination returns None"""
        key = model_registry.get_canonical_model_key("ltx", "txt2img")
        assert key is None

    def test_case_insensitive(self):
        """Test key lookup is case-insensitive"""
        key = model_registry.get_canonical_model_key("LTX", "T2V")
        assert key == ("ltx2", "txt2vid")


class TestListFunctions:
    """Tests for list functions"""

    def test_list_supported_models(self):
        """Test listing supported models"""
        models = model_registry.list_supported_models()
        expected = ["ltx2", "flux2", "wan21", "qwen", "sdxl", "hunyuan15", "qwen_edit"]
        for model in expected:
            assert model in models, f"Model {model} not in supported models"

    def test_list_model_aliases(self):
        """Test listing model aliases"""
        aliases = model_registry.list_model_aliases()
        assert "ltx" in aliases
        assert aliases["ltx"] == "ltx2"
        assert "flux" in aliases
        assert aliases["flux"] == "flux2"


class TestUtilityFunctions:
    """Tests for utility functions"""

    def test_is_video_model_ltx2(self):
        """Test LTX-2 is identified as video model"""
        assert model_registry.is_video_model("ltx2") is True
        assert model_registry.is_video_model("ltx") is True

    def test_is_video_model_flux2(self):
        """Test FLUX.2 is not identified as video model"""
        assert model_registry.is_video_model("flux2") is False
        assert model_registry.is_video_model("flux") is False

    def test_is_video_model_wan21(self):
        """Test Wan 2.1 is identified as video model"""
        assert model_registry.is_video_model("wan21") is True

    def test_is_video_model_qwen(self):
        """Test Qwen is not identified as video model"""
        assert model_registry.is_video_model("qwen") is False

    def test_validate_model_exists_valid(self):
        """Test validate_model_exists with valid model"""
        exists, result = model_registry.validate_model_exists("ltx2")
        assert exists is True
        assert result == "ltx2"

    def test_validate_model_exists_alias(self):
        """Test validate_model_exists with alias"""
        exists, result = model_registry.validate_model_exists("ltx")
        assert exists is True
        assert result == "ltx2"

    def test_validate_model_exists_invalid(self):
        """Test validate_model_exists with invalid model"""
        exists, result = model_registry.validate_model_exists("invalid")
        assert exists is False
        assert "Unknown model" in result

    def test_get_resolution_spec(self):
        """Test get_resolution_spec returns correct data"""
        spec = model_registry.get_resolution_spec("flux2")
        assert spec is not None
        assert "divisible_by" in spec
        assert spec["divisible_by"] == 16


class TestBackwardsCompatibility:
    """Tests ensuring backwards compatibility with existing code"""

    def test_model_constraints_dict_exported(self):
        """Test MODEL_CONSTRAINTS dict is exported"""
        assert hasattr(model_registry, "MODEL_CONSTRAINTS")
        assert isinstance(model_registry.MODEL_CONSTRAINTS, dict)
        assert "ltx2" in model_registry.MODEL_CONSTRAINTS
        assert "flux2" in model_registry.MODEL_CONSTRAINTS

    def test_model_defaults_dict_exported(self):
        """Test MODEL_DEFAULTS dict is exported"""
        assert hasattr(model_registry, "MODEL_DEFAULTS")
        assert isinstance(model_registry.MODEL_DEFAULTS, dict)
        assert "ltx2" in model_registry.MODEL_DEFAULTS
        assert "flux2" in model_registry.MODEL_DEFAULTS

    def test_model_skeleton_map_exported(self):
        """Test MODEL_SKELETON_MAP dict is exported"""
        assert hasattr(model_registry, "MODEL_SKELETON_MAP")
        assert isinstance(model_registry.MODEL_SKELETON_MAP, dict)
        assert ("ltx2", "txt2vid") in model_registry.MODEL_SKELETON_MAP

    def test_model_aliases_dict_exported(self):
        """Test MODEL_ALIASES dict is exported"""
        assert hasattr(model_registry, "MODEL_ALIASES")
        assert isinstance(model_registry.MODEL_ALIASES, dict)

    def test_model_resolution_specs_exported(self):
        """Test MODEL_RESOLUTION_SPECS dict is exported"""
        assert hasattr(model_registry, "MODEL_RESOLUTION_SPECS")
        assert isinstance(model_registry.MODEL_RESOLUTION_SPECS, dict)


class TestConstraintStructure:
    """Tests for constraint structure consistency"""

    def test_cfg_structure(self):
        """Test CFG constraints have consistent structure"""
        for model in model_registry.list_supported_models():
            constraints = model_registry.get_model_constraints(model)
            cfg = constraints["cfg"]
            assert "min" in cfg
            assert "max" in cfg
            assert "default" in cfg
            assert cfg["min"] <= cfg["default"] <= cfg["max"], f"CFG default out of range for {model}"

    def test_resolution_structure(self):
        """Test resolution constraints have consistent structure"""
        for model in model_registry.list_supported_models():
            constraints = model_registry.get_model_constraints(model)
            res = constraints["resolution"]
            assert "divisible_by" in res
            assert "native" in res
            assert isinstance(res["native"], list)
            assert len(res["native"]) == 2

    def test_steps_structure(self):
        """Test steps constraints have consistent structure"""
        for model in model_registry.list_supported_models():
            constraints = model_registry.get_model_constraints(model)
            steps = constraints["steps"]
            assert "min" in steps
            assert "max" in steps
            assert "default" in steps
            assert steps["min"] <= steps["default"] <= steps["max"], f"Steps default out of range for {model}"

    def test_video_models_have_frames(self):
        """Test video models have frame constraints"""
        video_models = ["ltx2", "wan21", "hunyuan15"]
        for model in video_models:
            constraints = model_registry.get_model_constraints(model)
            assert "frames" in constraints, f"Video model {model} missing frames constraint"
            assert "default" in constraints["frames"]


class TestSingleSourceOfTruth:
    """Tests ensuring this is the single source of truth"""

    def test_adding_model_requires_one_location(self):
        """
        Verify that model data structure allows adding models in one place.
        This test documents the expected workflow for adding new models.
        """
        # Count distinct model definitions
        models = model_registry.list_supported_models()

        # Each model should have:
        # 1. Entry in _MODEL_REGISTRY (the single source)
        # 2. Optionally aliases in MODEL_ALIASES
        # 3. Entry in MODEL_DEFAULTS (derived from _MODEL_REGISTRY)
        # 4. Entry in MODEL_CONSTRAINTS (derived from _MODEL_REGISTRY)

        # All models should be accessible through all public interfaces
        for model in models:
            assert model_registry.get_model_constraints(model) is not None
            # Note: get_model_defaults returns {} for models without explicit defaults
            # which is acceptable behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
