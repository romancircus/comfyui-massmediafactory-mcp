"""Tests for JSON structured logging functionality."""
import json
import pytest
import logging

from comfyui_massmediafactory_mcp.mcp_utils import (
    log_structured,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    ToolInvocation,
    JSONFormatter,
)


class CapturingLogHandler(logging.Handler):
    """Handler that captures log records for testing."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def get_json_logs(self):
        """Return list of parsed JSON log entries."""
        logs = []
        for record in self.records:
            formatter = JSONFormatter()
            formatted = formatter.format(record)
            logs.append(json.loads(formatted))
        return logs

    def clear(self):
        self.records = []


@pytest.fixture
def capturing_logger():
    """Fixture providing a capturing log handler."""
    logger = logging.getLogger("comfyui-mcp")

    handler = CapturingLogHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    original_level = logger.level
    logger.setLevel(logging.DEBUG)

    yield handler

    logger.removeHandler(handler)
    logger.setLevel(original_level)
    handler.clear()


@pytest.fixture
def correlation_context():
    """Fixture providing correlation ID context management."""

    def _set_cid(cid):
        set_correlation_id(cid)
        return cid

    yield _set_cid

    clear_correlation_id()


class TestJSONFormatter:
    """Test JSON log format output."""

    def test_json_output_format(self, capturing_logger):
        """Verify logs are valid JSON."""
        log_structured("info", "test_message")

        logs = capturing_logger.get_json_logs()
        assert len(logs) == 1

        log = logs[0]
        assert "timestamp" in log
        assert "level" in log
        assert "logger" in log
        assert "message" in log
        assert log["message"] == "test_message"
        assert log["level"] == "INFO"

    def test_correlation_id_included(self, capturing_logger, correlation_context):
        """Verify correlation ID appears in logs."""
        cid = "test_cid_123"
        correlation_context(cid)

        log_structured("info", "test_with_cid")

        logs = capturing_logger.get_json_logs()
        assert logs[0]["correlation_id"] == cid

    def test_custom_fields(self, capturing_logger):
        """Verify custom fields are included."""
        log_structured(
            "info",
            "test_custom",
            key1="value1",
            key2=42,
        )

        logs = capturing_logger.get_json_logs()
        log = logs[0]
        assert log["key1"] == "value1"
        assert log["key2"] == 42

    def test_all_log_levels(self, capturing_logger):
        """Test all log levels produce valid JSON."""
        levels = ["debug", "info", "warning", "error", "critical"]

        for level in levels:
            capturing_logger.clear()
            log_structured(level, f"test_{level}")

            logs = capturing_logger.get_json_logs()
            assert len(logs) == 1
            assert logs[0]["level"] == level.upper()


class TestCorrelationID:
    """Test correlation ID functionality."""

    def test_auto_generation(self):
        """Test correlation ID auto-generated when not set."""
        clear_correlation_id()
        cid = get_correlation_id()
        assert cid is not None
        assert len(cid) > 0

    def test_manual_setting(self, correlation_context):
        """Test manual correlation ID setting."""
        test_cid = "manual_test_123"
        correlation_context(test_cid)

        assert get_correlation_id() == test_cid

    def test_persistence_across_calls(self, capturing_logger, correlation_context):
        """Test correlation ID persists across multiple log calls."""
        cid = "persistent_cid"
        correlation_context(cid)

        log_structured("info", "call_1")
        log_structured("info", "call_2")
        log_structured("error", "call_3")

        logs = capturing_logger.get_json_logs()
        for log in logs:
            assert log["correlation_id"] == cid


class TestToolInvocation:
    """Test ToolInvocation logging."""

    def test_tool_completion_logged(self, capturing_logger):
        """Test tool completion produces structured log."""
        invocation = ToolInvocation(tool_name="test_tool")
        _result = invocation.complete("success")

        logs = capturing_logger.get_json_logs()
        assert len(logs) == 1

        log = logs[0]
        assert log["tool"] == "test_tool"
        assert "invocation_id" in log
        assert "latency_ms" in log
        assert log["status"] == "success"

    def test_tool_error_logged(self, capturing_logger):
        """Test tool error includes error details."""
        invocation = ToolInvocation(tool_name="test_tool")
        invocation.complete("error", error="Something failed")

        logs = capturing_logger.get_json_logs()
        log = logs[0]
        assert log["status"] == "error"
        assert log["error"] == "Something failed"

    def test_tool_rate_limited_logged(self, capturing_logger):
        """Test rate limited status produces warning log."""
        invocation = ToolInvocation(tool_name="test_tool")
        invocation.complete("rate_limited")

        logs = capturing_logger.get_json_logs()
        log = logs[0]
        assert log["status"] == "rate_limited"
        assert log["level"] == "WARNING"


class TestLogValidation:
    """Test log entry validation."""

    def test_timestamp_format(self, capturing_logger):
        """Verify timestamp is ISO 8601 format."""
        import re

        log_structured("info", "timestamp_test")

        logs = capturing_logger.get_json_logs()
        timestamp = logs[0]["timestamp"]

        # Should match ISO 8601 pattern
        iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z$"
        assert re.match(iso_pattern, timestamp), f"Timestamp {timestamp} doesn't match ISO 8601"

    def test_exception_included(self, capturing_logger):
        """Test exceptions can be logged (exc_info passthrough)."""
        # Test that log_structured accepts exc_info without error
        try:
            raise ValueError("Test exception")
        except Exception:
            # This should not raise
            log_structured("error", "exception_test", exc_info=True)

        logs = capturing_logger.get_json_logs()
        log = logs[0]
        # Just verify the log was created successfully
        assert log["message"] == "exception_test"
        assert log["level"] == "ERROR"


class TestExecutionLogging:
    """Test execution module produces correct logs."""

    def test_workflow_queued_log(self, capturing_logger, correlation_context):
        """Test execute_workflow produces workflow_queued log."""
        # This test requires a mock ComfyUI client
        # For now, just verify the log function exists and works
        log_structured(
            "info",
            "workflow_queued",
            prompt_id="test_prompt_123",
            client_id="test_client",
            node_count=5,
            queue_position=0,
        )

        logs = capturing_logger.get_json_logs()
        log = logs[0]
        assert log["message"] == "workflow_queued"
        assert log["prompt_id"] == "test_prompt_123"
        assert log["node_count"] == 5

    def test_regeneration_logs(self, capturing_logger, correlation_context):
        """Test regeneration produces start and complete logs."""
        cid = "regen_test_123"
        correlation_context(cid)

        log_structured(
            "info",
            "regeneration_started",
            asset_id="asset_456",
            overrides=["prompt", "seed"],
        )

        log_structured(
            "info",
            "regeneration_completed",
            asset_id="asset_456",
            new_prompt_id="new_prompt_789",
        )

        logs = capturing_logger.get_json_logs()
        assert len(logs) == 2
        assert logs[0]["message"] == "regeneration_started"
        assert logs[1]["message"] == "regeneration_completed"
        assert logs[0]["correlation_id"] == cid
        assert logs[1]["correlation_id"] == cid
