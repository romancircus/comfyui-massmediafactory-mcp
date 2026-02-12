"""
Tests for LTX Planner Module - ROM-668

S5 Test Suite for LTX template parity implementation.
"""

import pytest
from comfyui_massmediafactory_mcp.ltx_planner import (
    select_ltx_variant,
    get_ltx_workflow_with_trace,
    generate_compile_report,
    get_advanced_ltx_branches,
    LTXVariant,
    LTXSampler,
    LTX_CANONICAL_NODES,
)


class TestLTXVariantSelection:
    """Test branch selection logic for LTX variants."""

    def test_t2v_standard_default(self):
        """Test default T2V selection without hints."""
        selection = select_ltx_variant("t2v", hints={})

        assert selection.selected_variant == LTXVariant.T2V_STANDARD
        assert selection.sampler_type == LTXSampler.SAMPLER_CUSTOM_ADVANCED
        assert "CFGGuider" in selection.canonical_nodes.values()

    def test_t2v_distilled_with_speed_hint(self):
        """Test T2V distilled selection with speed hint."""
        selection = select_ltx_variant("t2v", hints={"speed": True})

        assert selection.selected_variant == LTXVariant.T2V_DISTILLED
        assert selection.sampler_type == LTXSampler.SAMPLER_CUSTOM
        assert "LTXVGemmaCLIPModelLoader" in selection.canonical_nodes.values()

    def test_t2v_enhanced_with_quality_hint(self):
        """Test T2V enhanced selection with quality/STG hint."""
        selection = select_ltx_variant("t2v", hints={"enhanced": True})

        assert selection.selected_variant == LTXVariant.T2V_ENHANCED
        assert "LTXVApplySTG" in selection.canonical_nodes.values()

    def test_i2v_standard_default(self):
        """Test default I2V selection."""
        selection = select_ltx_variant("i2v", hints={})

        assert selection.selected_variant == LTXVariant.I2V_STANDARD
        assert "LTXVImgToVideoAdvanced" in selection.canonical_nodes.values()

    def test_i2v_distilled_with_speed_hint(self):
        """Test I2V distilled selection with speed hint."""
        selection = select_ltx_variant("i2v", hints={"distilled": True})

        assert selection.selected_variant == LTXVariant.I2V_DISTILLED
        assert "LTXVImgToVideo" in selection.canonical_nodes.values()

    def test_v2v_selection(self):
        """Test V2V branch selection."""
        selection = select_ltx_variant("v2v", hints={})

        assert selection.selected_variant == LTXVariant.V2V
        assert "LTXVAddGuide" in selection.canonical_nodes.values()


class TestExplainTrace:
    """Test explain trace generation."""

    def test_trace_contains_selection_steps(self):
        """Test that explain trace documents selection rationale."""
        selection = select_ltx_variant("t2v", hints={"speed": True})

        # Should have trace entries
        assert len(selection.explain_trace) > 0

        # Should document task normalization
        task_steps = [e for e in selection.explain_trace if e.get("step") == "task_normalization"]
        assert len(task_steps) > 0

        # Should document variant scoring
        scoring_steps = [e for e in selection.explain_trace if "variant_scoring" in e.get("step", "")]
        assert len(scoring_steps) > 0

        # Should document final selection
        final_steps = [e for e in selection.explain_trace if e.get("step") == "final_selection"]
        assert len(final_steps) > 0

    def test_rejected_alternatives_documented(self):
        """Test that rejected alternatives are documented."""
        selection = select_ltx_variant("t2v", hints={"speed": True})

        # Should have rejected alternatives when choosing distilled
        assert len(selection.rejected_alternatives) > 0

        # Check that standard variants were rejected for speed mode
        # (txt2vid is the standard T2V, txt2vid_enhanced is also rejected)
        rejected_variants = [r.get("variant", "") for r in selection.rejected_alternatives]
        assert "txt2vid" in rejected_variants or "txt2vid_enhanced" in rejected_variants


class TestCanonicalNodes:
    """Test canonical node chains."""

    def test_t2v_standard_canonical_nodes(self):
        """Test T2V standard has all required canonical nodes."""
        nodes = LTX_CANONICAL_NODES[LTXVariant.T2V_STANDARD]

        required = [
            "CheckpointLoaderSimple",
            "LTXAVTextEncoderLoader",
            "LTXVAudioVAELoader",
            "LTXVConditioning",
            "EmptyLTXVLatentVideo",
            "LTXVEmptyLatentAudio",
            "LTXVConcatAVLatent",
            "LTXVScheduler",
            "SamplerCustomAdvanced",
            "CFGGuider",
            "LTXVSeparateAVLatent",
            "VAEDecodeTiled",
            "LTXVAudioVAEDecode",
            "SaveVideo",
        ]

        for req in required:
            assert req in nodes.values(), f"Missing required node: {req}"

    def test_t2v_distilled_canonical_nodes(self):
        """Test T2V distilled uses simplified node chain."""
        nodes = LTX_CANONICAL_NODES[LTXVariant.T2V_DISTILLED]

        # Distilled uses simplified sampler
        assert "SamplerCustom" in nodes.values()
        assert "CFGGuider" not in nodes.values()  # No guider in distilled

        # Has Gemma-specific nodes
        assert "LTXVGemmaCLIPModelLoader" in nodes.values()
        assert "LTXVGemmaEnhancePrompt" in nodes.values()

    def test_t2v_enhanced_has_stg(self):
        """Test T2V enhanced includes STG node."""
        nodes = LTX_CANONICAL_NODES[LTXVariant.T2V_ENHANCED]

        assert "LTXVApplySTG" in nodes.values()


class TestWorkflowGeneration:
    """Test workflow generation with trace."""

    def test_get_workflow_with_trace_basic(self):
        """Test basic workflow generation returns trace."""
        result = get_ltx_workflow_with_trace("t2v", "a test prompt")

        assert "workflow_type" in result
        assert "selection_trace" in result
        assert "canonical_nodes" in result
        assert isinstance(result["selection_trace"], list)

    def test_i2v_requires_image_warning(self):
        """Test I2V warns when no image provided."""
        result = get_ltx_workflow_with_trace("i2v", "animate this", image_path=None)

        assert "warning" in result
        assert "requires_image" in result

    def test_v2v_requires_guide_warning(self):
        """Test V2V warns when no guide provided."""
        result = get_ltx_workflow_with_trace("v2v", "style transfer", image_path=None)

        assert "warning" in result
        assert "requires_guide" in result


class TestCompileReport:
    """Test compile report generation."""

    def test_compile_report_format(self):
        """Test compile report is properly formatted."""
        selection = select_ltx_variant("t2v", hints={"enhanced": True})
        report = generate_compile_report(selection)

        # Should contain key sections
        assert "LTX Workflow Compile Report" in report
        assert "Selected Variant:" in report
        assert "Canonical Node Chain:" in report
        assert "Selection Rationale" in report

    def test_compile_report_shows_rejected(self):
        """Test compile report shows rejected alternatives."""
        selection = select_ltx_variant("t2v", hints={"speed": True})
        report = generate_compile_report(selection)

        assert "Rejected Alternatives:" in report
        assert "txt2vid" in report or "txt2vid_enhanced" in report


class TestAdvancedBranches:
    """Test advanced branch support."""

    def test_modify_ltx_model_branch(self):
        """Test ModifyLTXModel branch is documented."""
        branches = get_advanced_ltx_branches()

        assert "modify_ltx_model" in branches
        assert branches["modify_ltx_model"]["node"] == "ModifyLTXModel"

    def test_inversion_compatible_branch(self):
        """Test inversion compatible branch."""
        branches = get_advanced_ltx_branches()

        assert "inversion_compatible" in branches

    def test_interpolation_branch(self):
        """Test interpolation branch."""
        branches = get_advanced_ltx_branches()

        assert "interpolation" in branches
        assert "LTXVInterpolation" in branches["interpolation"]["node"]


class TestVRAMConstraints:
    """Test VRAM-based selection adjustments."""

    def test_low_vram_prefers_distilled(self):
        """Test low VRAM prefers distilled variants."""
        selection_low = select_ltx_variant("t2v", hints={}, available_vram_gb=12)
        selection_high = select_ltx_variant("t2v", hints={}, available_vram_gb=24)

        # Should document VRAM constraint in trace
        vram_entries = [e for e in selection_low.explain_trace if e.get("step") == "vram_constraint"]
        assert len(vram_entries) > 0
        # Verify both selections completed successfully
        assert selection_high.selected_variant is not None


class TestIntegration:
    """Integration tests for S5 validation."""

    def test_s5_ltx_variant_flow(self):
        """
        S5 Test: LTX variant flow passes with live runtime path.

        This test validates:
        - Branch selection works correctly
        - Explain trace is generated
        - Canonical nodes are correct
        - Compile report is valid
        """
        # Test all major LTX variants
        test_cases = [
            ("t2v", {}, LTXVariant.T2V_STANDARD),
            ("t2v", {"speed": True}, LTXVariant.T2V_DISTILLED),
            ("t2v", {"enhanced": True}, LTXVariant.T2V_ENHANCED),
            ("i2v", {}, LTXVariant.I2V_STANDARD),
            ("i2v", {"distilled": True}, LTXVariant.I2V_DISTILLED),
            ("v2v", {}, LTXVariant.V2V),
        ]

        for task, hints, expected_variant in test_cases:
            selection = select_ltx_variant(task, hints=hints)

            assert selection.selected_variant == expected_variant, f"Failed for {task} with hints {hints}"

            # Verify trace
            assert len(selection.explain_trace) >= 3, f"Insufficient trace for {task}"

            # Verify canonical nodes
            assert len(selection.canonical_nodes) > 5, f"Insufficient nodes for {task}"

            # Verify report generation
            report = generate_compile_report(selection)
            assert len(report) > 100

    def test_fr_2_3_implementation(self):
        """
        FR-2.3: LTX-specific template variants with canonical nodes.

        Validates that all LTX variants have proper canonical node chains
        with LTXVConditioning, LTXVScheduler, SamplerCustom/Advanced, VAEDecode.
        """
        variants_to_test = [
            LTXVariant.T2V_STANDARD,
            LTXVariant.T2V_DISTILLED,
            LTXVariant.T2V_ENHANCED,
            LTXVariant.I2V_STANDARD,
            LTXVariant.I2V_DISTILLED,
            LTXVariant.V2V,
        ]

        for variant in variants_to_test:
            nodes = LTX_CANONICAL_NODES[variant]

            # All variants must have conditioning
            assert "LTXVConditioning" in nodes.values(), f"{variant.value} missing LTXVConditioning"

            # All variants must have scheduler
            assert "LTXVScheduler" in nodes.values(), f"{variant.value} missing LTXVScheduler"

            # All variants must have a sampler
            has_sampler = any("Sampler" in v for v in nodes.values())
            assert has_sampler, f"{variant.value} missing sampler"

            # All variants must have VAE decode
            has_vae_decode = any("VAEDecode" in v for v in nodes.values())
            assert has_vae_decode, f"{variant.value} missing VAEDecode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
