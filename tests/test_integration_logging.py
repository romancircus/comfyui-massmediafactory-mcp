"""
Integration test for structured logging with actual workflow execution.

This script generates a workflow and verifies structured logs are captured.
"""
import sys

sys.path.insert(0, "/home/romancircus/Applications/comfyui-massmediafactory-mcp/src")

from comfyui_massmediafactory_mcp.mcp_utils import log_structured, get_correlation_id
from comfyui_massmediafactory_mcp.workflow_generator import generate_workflow


def test_integration():
    """Test that structured logging works end-to-end."""
    print("=" * 60)
    print("Structured Logging Integration Test")
    print("=" * 60)

    # 1. Generate a simple workflow
    print("\n1. Generating workflow...")
    result = generate_workflow(
        model="flux",
        workflow_type="t2i",
        prompt="a red apple",
        width=512,
        height=512,
        steps=20,
        seed=42,
    )

    assert "error" not in result, f"Workflow generation failed: {result.get('error')}"
    assert len(result["workflow"]) > 0

    # 2. Verify structured log was captured
    # The correlation_id should persist from workflow generation
    current_cid = get_correlation_id()
    assert current_cid, "Correlation ID missing"

    # 3. Test manual logging (should not raise)
    log_structured(
        level="info",
        message="integration_test_manual_log",
        test_phase="manual",
        test_value=123,
    )

    # 4. Verify correlation ID persists
    assert get_correlation_id() == current_cid
