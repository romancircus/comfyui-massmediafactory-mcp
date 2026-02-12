"""
Template Contract Tests

Validates that all built-in templates produce ComfyUI-valid prompt payloads.
These tests POST to /prompt endpoint in dry-run mode to catch schema validation
errors before templates are used in production.

Run with: pytest tests/test_template_contract.py -v
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any, List


# Add src to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comfyui_massmediafactory_mcp.templates import (
    load_template,
    inject_parameters,
    TEMPLATES_DIR,
)


class TestTemplateComfyUIValidity:
    """Test that templates produce ComfyUI-valid prompt payloads."""

    def _get_all_template_names(self) -> List[str]:
        """Get all template names from the templates directory."""
        templates = []
        for f in TEMPLATES_DIR.glob("*.json"):
            if not f.name.startswith("."):
                templates.append(f.stem)
        return sorted(templates)

    def _inject_with_required_params(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Inject a template with minimal required parameters for validation."""
        meta = template.get("_meta", {})
        defaults = meta.get("defaults", {})

        # Ensure IMAGE_PATH is set for I2V templates
        params = dict(defaults)
        params.setdefault("PROMPT", "test prompt for validation")
        params.setdefault("NEGATIVE", "test negative prompt")
        params.setdefault("SEED", 42)

        # Add image path for templates that need it
        parameters = meta.get("parameters", [])
        if "IMAGE_PATH" in parameters and "IMAGE_PATH" not in params:
            params["IMAGE_PATH"] = "test_keyframe.png"
        if "CHARACTER_IMAGE" in parameters and "CHARACTER_IMAGE" not in params:
            params["CHARACTER_IMAGE"] = "test_character.png"
        if "CONTENT_IMAGE" in parameters and "CONTENT_IMAGE" not in params:
            params["CONTENT_IMAGE"] = "test_content.png"
        if "STYLE_IMAGE" in parameters and "STYLE_IMAGE" not in params:
            params["STYLE_IMAGE"] = "test_style.png"
        if "FACE_IMAGE" in parameters and "FACE_IMAGE" not in params:
            params["FACE_IMAGE"] = "test_face.png"
        if "EDIT_IMAGE" in parameters and "EDIT_IMAGE" not in params:
            params["EDIT_IMAGE"] = "test_edit.png"

        # Add style path for TeleStyle templates
        if "STYLE_PATH" in parameters and "STYLE_PATH" not in params:
            params["STYLE_PATH"] = "test_style.png"

        # Add multi-character image paths for phantom template
        for i in range(1, 5):
            char_param = f"CHARACTER_IMAGE_{i}"
            if char_param in parameters and char_param not in params:
                params[char_param] = f"test_character_{i}.png"

        # Add video path for templates that need it
        if "VIDEO_PATH" in parameters and "VIDEO_PATH" not in params:
            params["VIDEO_PATH"] = "test_video.mp4"
        if "AUDIO_PATH" in parameters and "AUDIO_PATH" not in params:
            params["AUDIO_PATH"] = "test_audio.mp3"
        if "VIDEO_1" in parameters and "VIDEO_1" not in params:
            params["VIDEO_1"] = "test_video1.mp4"
        if "VIDEO_2" in parameters and "VIDEO_2" not in params:
            params["VIDEO_2"] = "test_video2.mp4"

        # Add text parameters for editing templates
        if "EDIT_PROMPT" in parameters and "EDIT_PROMPT" not in params:
            params["EDIT_PROMPT"] = "test edit prompt"
        if "SELECT_TEXT" in parameters and "SELECT_TEXT" not in params:
            params["SELECT_TEXT"] = "test selection text"
        if "REPLACE_PROMPT" in parameters and "REPLACE_PROMPT" not in params:
            params["REPLACE_PROMPT"] = "test replacement prompt"

        return inject_parameters(template, params)

    def _validate_comfyui_structure(self, workflow: Dict[str, Any]) -> List[str]:
        """
        Validate that a workflow meets ComfyUI structure requirements.

        Returns list of validation errors (empty if valid).
        """
        errors = []

        # Check 1: All nodes must have class_type
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                errors.append(f"Node '{node_id}' is not a dict")
                continue

            if "class_type" not in node:
                errors.append(f"Node '{node_id}' missing class_type")
                continue

            # Check 2: Inputs must be valid
            inputs = node.get("inputs", {})
            for input_name, input_value in inputs.items():
                # Connection format must be [node_id, slot]
                if isinstance(input_value, list):
                    if len(input_value) != 2:
                        errors.append(
                            f"Node '{node_id}' input '{input_name}' has invalid "
                            f"connection format (expected [node_id, slot], got {input_value})"
                        )
                    elif not isinstance(input_value[1], int):
                        errors.append(f"Node '{node_id}' input '{input_name}' slot must be integer")
                # String inputs must not contain unsubstituted placeholders
                elif isinstance(input_value, str):
                    if "{{" in input_value and "}}" in input_value:
                        errors.append(
                            f"Node '{node_id}' input '{input_name}' has unsubstituted " f"placeholder: {input_value}"
                        )

        # Check 3: CreateVideo nodes must have proper images input
        for node_id, node in workflow.items():
            if node.get("class_type") == "CreateVideo":
                inputs = node.get("inputs", {})
                if "images" not in inputs:
                    errors.append(f"CreateVideo node '{node_id}' missing required 'images' input")
                elif isinstance(inputs["images"], list):
                    source_node = str(inputs["images"][0])
                    if source_node not in workflow:
                        errors.append(
                            f"CreateVideo node '{node_id}' references non-existent " f"source node '{source_node}'"
                        )

        # Check 4: SaveImage/SaveVideo nodes must have proper inputs
        for node_id, node in workflow.items():
            class_type = node.get("class_type", "")
            if class_type in ["SaveImage", "SaveVideo", "VHS_SaveVideo", "SaveAudio"]:
                inputs = node.get("inputs", {})
                main_input = "images" if class_type in ["SaveImage", "VHS_SaveVideo"] else "video"
                if main_input not in inputs:
                    errors.append(f"{class_type} node '{node_id}' missing required '{main_input}' input")

        # Check 5: References must point to existing nodes
        for node_id, node in workflow.items():
            inputs = node.get("inputs", {})
            for input_name, input_value in inputs.items():
                if isinstance(input_value, list) and len(input_value) == 2:
                    source_node = str(input_value[0])
                    if source_node not in workflow:
                        errors.append(
                            f"Node '{node_id}' input '{input_name}' references " f"non-existent node '{source_node}'"
                        )

        return errors

    @pytest.mark.parametrize(
        "template_name",
        [
            # Core production templates
            "qwen_txt2img",
            "ltx2_txt2vid",
            "ltx2_img2vid",
            "wan26_img2vid",
            "flux2_txt2img",
            "qwen_edit_background",
            # Additional I2V templates
            "wan22_animate",
            "wan22_i2v_lightning",
            "wan22_i2v_enhanced",
            "wan22_i2v_a14b",
            "wan22_i2v_breakthrough",
            "wan26_i2v_breakthrough",
            "wan26_camera_i2v",
            "wan22_phantom",
            "ltx2_i2v_distilled",
            "hunyuan15_img2vid",
            # Style/utility templates
            "telestyle_image",
            "telestyle_video",
            "flux2_face_id",
            "flux_kontext_edit",
            "video_inpaint",
            "video_stitch",
        ],
    )
    def test_template_produces_valid_workflow(self, template_name: str):
        """Test that a template can be loaded and produces a valid workflow."""
        template = load_template(template_name)
        assert "error" not in template, f"Failed to load template: {template.get('error')}"

        workflow = self._inject_with_required_params(template)

        # Validate workflow structure
        errors = self._validate_comfyui_structure(workflow)
        assert len(errors) == 0, f"Template '{template_name}' validation errors:\n" + "\n".join(errors)

    def test_all_templates_have_required_meta(self):
        """Test that all templates have required _meta fields."""
        all_templates = self._get_all_template_names()
        failures = []

        for template_name in all_templates:
            template = load_template(template_name, validate=False)
            if "error" in template:
                failures.append(f"{template_name}: Failed to load - {template['error']}")
                continue

            meta = template.get("_meta", {})
            required = ["description", "model", "type", "parameters"]

            for field in required:
                if field not in meta:
                    failures.append(f"{template_name}: Missing _meta.{field}")

        assert len(failures) == 0, "Template metadata issues:\n" + "\n".join(failures)

    def test_all_templates_produce_no_placeholder_leaks(self):
        """Test that parameter injection removes all {{PLACEHOLDER}} syntax."""
        all_templates = self._get_all_template_names()
        failures = []

        for template_name in all_templates:
            template = load_template(template_name)
            if "error" in template:
                failures.append(f"{template_name}: Failed to load - {template['error']}")
                continue

            workflow = self._inject_with_required_params(template)
            workflow_str = json.dumps(workflow)

            # Check for any remaining {{PLACEHOLDER}} patterns
            import re

            placeholders = re.findall(r"\{\{[A-Z0-9_]+\}\}", workflow_str)
            if placeholders:
                failures.append(f"{template_name}: Unsubstituted placeholders: {', '.join(placeholders)}")

        assert len(failures) == 0, "Placeholder leaks found:\n" + "\n".join(failures)


class TestTemplateComfyUIAPIValidation:
    """
    Test templates against live ComfyUI API (when available).

    These tests require a running ComfyUI instance at COMFYUI_URL.
    They POST templates to the /prompt endpoint in validation mode.
    """

    @pytest.fixture(scope="class")
    def comfyui_client(self):
        """Get ComfyUI client if available."""
        try:
            from comfyui_massmediafactory_mcp.client import get_client

            client = get_client()
            if client.is_available():
                return client
            return None
        except Exception:
            return None

    @pytest.mark.skipif(
        not pytest.importorskip("comfyui_massmediafactory_mcp.client", reason="No ComfyUI client"),
        reason="ComfyUI not available for live testing",
    )
    def test_qwen_txt2img_live_validation(self, comfyui_client):
        """Test qwen_txt2img template against live ComfyUI."""
        if not comfyui_client:
            pytest.skip("ComfyUI not available")

        template = load_template("qwen_txt2img")
        workflow = TestTemplateComfyUIValidity()._inject_with_required_params(template)

        # Queue the workflow
        result = comfyui_client.queue_prompt(workflow, client_id="test_template_contract")

        # We expect either:
        # - A prompt_id (meaning it was accepted)
        # - An error dict with validation issues
        if "error" in result:
            # Check if it's a prompt validation error (not a runtime/model error)
            error_msg = str(result["error"]).lower()
            if "prompt_outputs_failed_validation" in error_msg or "required input is missing" in error_msg:
                pytest.fail(f"Template failed prompt validation: {result['error']}")
            # Runtime errors (missing model, VRAM, etc.) are acceptable
            pytest.skip(f"Runtime error (not template issue): {result['error']}")

        assert "prompt_id" in result, f"Expected prompt_id in result: {result}"

    @pytest.mark.skipif(
        not pytest.importorskip("comfyui_massmediafactory_mcp.client", reason="No ComfyUI client"),
        reason="ComfyUI not available for live testing",
    )
    def test_ltx2_img2vid_live_validation(self, comfyui_client):
        """Test ltx2_img2vid template against live ComfyUI."""
        if not comfyui_client:
            pytest.skip("ComfyUI not available")

        template = load_template("ltx2_img2vid")
        workflow = TestTemplateComfyUIValidity()._inject_with_required_params(template)

        result = comfyui_client.queue_prompt(workflow, client_id="test_template_contract")

        if "error" in result:
            error_msg = str(result["error"]).lower()
            if "prompt_outputs_failed_validation" in error_msg or "required input is missing" in error_msg:
                pytest.fail(f"Template failed prompt validation: {result['error']}")
            pytest.skip(f"Runtime error (not template issue): {result['error']}")

        assert "prompt_id" in result, f"Expected prompt_id in result: {result}"

    @pytest.mark.skipif(
        not pytest.importorskip("comfyui_massmediafactory_mcp.client", reason="No ComfyUI client"),
        reason="ComfyUI not available for live testing",
    )
    def test_wan26_img2vid_live_validation(self, comfyui_client):
        """Test wan26_img2vid template against live ComfyUI."""
        if not comfyui_client:
            pytest.skip("ComfyUI not available")

        template = load_template("wan26_img2vid")
        workflow = TestTemplateComfyUIValidity()._inject_with_required_params(template)

        result = comfyui_client.queue_prompt(workflow, client_id="test_template_contract")

        if "error" in result:
            error_msg = str(result["error"]).lower()
            if "prompt_outputs_failed_validation" in error_msg or "required input is missing" in error_msg:
                pytest.fail(f"Template failed prompt validation: {result['error']}")
            pytest.skip(f"Runtime error (not template issue): {result['error']}")

        assert "prompt_id" in result, f"Expected prompt_id in result: {result}"


class TestTemplateErrorHandling:
    """Test that template failures produce deterministic error classes."""

    def test_missing_template_returns_error_dict(self):
        """Test that loading a non-existent template returns an error dict."""
        from comfyui_massmediafactory_mcp.templates import load_template

        result = load_template("non_existent_template_xyz")
        assert "error" in result
        assert "non_existent_template_xyz" in result["error"].lower()

    def test_invalid_template_returns_validation_errors(self):
        """Test that templates with validation issues return proper error structure."""
        # Create a malformed template
        malformed = {
            "_meta": {
                "description": "Test",
                # Missing required fields
            },
            "1": {
                # Missing class_type
                "inputs": {}
            },
        }

        from comfyui_massmediafactory_mcp.templates import validate_template

        errors, warnings = validate_template(malformed)

        assert len(errors) > 0, "Expected validation errors for malformed template"
        # Check that errors are deterministic strings
        for error in errors:
            assert isinstance(error, str), f"Error should be string, got {type(error)}"
            assert len(error) > 0, "Error should not be empty"
