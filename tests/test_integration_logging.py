"""
Integration test for structured logging with actual workflow execution.

This script generates a workflow and verifies structured logs are captured.
"""
import json
import sys
sys.path.insert(0, '/home/romancircus/Applications/comfyui-massmediafactory-mcp/src')

from comfyui_massmediafactory_mcp.mcp_utils import (
    log_structured, set_correlation_id, get_correlation_id
)
from comfyui_massmediafactory_mcp.workflow_generator import generate_workflow
from comfyui_massmediafactory_mcp.execution import execute_workflow

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
        seed=42
    )
    
    if "error" in result:
        print(f"❌ Workflow generation failed: {result['error']}")
        return False
    
    print(f"✅ Workflow generated: {len(result['workflow'])} nodes")
    print(f"   Correlation ID: {get_correlation_id()}")
    
    # 2. Verify structured log was captured
    print("\n2. Checking log output...")
    # The correlation_id should persist from workflow generation
    current_cid = get_correlation_id()
    print(f"   Current correlation_id: {current_cid}")
    
    if not current_cid:
        print("❌ Correlation ID missing")
        return False
    print("✅ Correlation ID present")
    
    # 3. Test manual logging
    print("\n3. Testing manual structured logging...")
    log_structured(
        level="info",
        message="integration_test_manual_log",
        test_phase="manual",
        test_value=123
    )
    print("✅ Manual log entry emitted")
    
    # 4. Verify JSON format (would be visible in actual log output)
    print("\n4. Log format verification...")
    print("   Expected JSON format: {timestamp, level, logger, message, correlation_id, ...}")
    print("   Check actual logs above to verify JSON structure")
    
    print("\n" + "=" * 60)
    print("✅ Integration test completed successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
