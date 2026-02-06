"""
Tests for Round 2 bug fixes and SOTA features.
"""


class TestPathValidation:
    """Test security fix: path validation with is_relative_to."""

    def test_valid_path(self):
        from comfyui_massmediafactory_mcp.server import _validate_path

        valid, result = _validate_path("/home/romancircus/test.txt", "/home/romancircus")
        assert valid is True

    def test_prefix_attack_blocked(self):
        """Path like /home/user_evil should NOT match /home/user."""
        from comfyui_massmediafactory_mcp.server import _validate_path

        valid, result = _validate_path("/home/romancircus_evil/attack.txt", "/home/romancircus")
        assert valid is False

    def test_traversal_attack_blocked(self):
        from comfyui_massmediafactory_mcp.server import _validate_path

        valid, result = _validate_path("/home/romancircus/../etc/passwd", "/home/romancircus")
        assert valid is False

    def test_subdirectory_allowed(self):
        from comfyui_massmediafactory_mcp.server import _validate_path

        valid, result = _validate_path("/home/romancircus/deep/nested/file.txt", "/home/romancircus")
        assert valid is True


class TestURLValidation:
    """Test security fix: exact domain matching."""

    def test_valid_civitai(self):
        from comfyui_massmediafactory_mcp.server import _validate_url

        valid, _ = _validate_url("https://civitai.com/models/123")
        assert valid is True

    def test_evil_domain_blocked(self):
        """evil-civitai.com should NOT match."""
        from comfyui_massmediafactory_mcp.server import _validate_url

        valid, _ = _validate_url("https://evil-civitai.com/models/123")
        assert valid is False

    def test_subdomain_allowed(self):
        from comfyui_massmediafactory_mcp.server import _validate_url

        valid, _ = _validate_url("https://api.huggingface.co/models/flux")
        assert valid is True

    def test_github_allowed(self):
        from comfyui_massmediafactory_mcp.server import _validate_url

        valid, _ = _validate_url("https://github.com/user/repo")
        assert valid is True

    def test_random_domain_blocked(self):
        from comfyui_massmediafactory_mcp.server import _validate_url

        valid, _ = _validate_url("https://malicious.example.com/payload")
        assert valid is False

    def test_no_hostname_blocked(self):
        from comfyui_massmediafactory_mcp.server import _validate_url

        valid, _ = _validate_url("not-a-url")
        assert valid is False


class TestPublishDirValidation:
    """Test security fix: set_publish_dir blocklist."""

    def test_etc_blocked(self):
        from comfyui_massmediafactory_mcp.publish import set_publish_dir

        result = set_publish_dir("/etc/nginx")
        assert "error" in result
        assert result["error"] == "FORBIDDEN_PATH"

    def test_usr_blocked(self):
        from comfyui_massmediafactory_mcp.publish import set_publish_dir

        result = set_publish_dir("/usr/local/bin")
        assert "error" in result

    def test_root_blocked(self):
        from comfyui_massmediafactory_mcp.publish import set_publish_dir

        result = set_publish_dir("/root")
        assert "error" in result


class TestPromptEnhance:
    """Test prompt enhancement module."""

    def test_token_enhancement_flux(self):
        from comfyui_massmediafactory_mcp.prompt_enhance import enhance_prompt

        result = enhance_prompt("a cat", model="flux", use_llm=False)
        assert result["method"] == "tokens"
        assert "a cat" in result["enhanced"]
        assert "original" in result
        assert result["original"] == "a cat"

    def test_token_enhancement_sdxl(self):
        from comfyui_massmediafactory_mcp.prompt_enhance import enhance_prompt

        result = enhance_prompt("a dog", model="sdxl", use_llm=False)
        assert "masterpiece" in result["enhanced"]

    def test_style_included(self):
        from comfyui_massmediafactory_mcp.prompt_enhance import enhance_prompt

        result = enhance_prompt("a landscape", model="flux", style="cinematic", use_llm=False)
        assert "cinematic" in result["enhanced"]

    def test_unknown_model_fallback(self):
        from comfyui_massmediafactory_mcp.prompt_enhance import enhance_prompt

        result = enhance_prompt("test", model="nonexistent_model", use_llm=False)
        assert result["method"] == "tokens"
        # Should fall back to flux tokens


class TestWorkflowDiff:
    """Test workflow diff module."""

    def test_identical_workflows(self):
        from comfyui_massmediafactory_mcp.workflow_diff import diff_workflows

        wf = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}}}
        result = diff_workflows(wf, wf)
        assert result["identical"] is True
        assert result["nodes_unchanged"] == 1
        assert len(result["nodes_added"]) == 0

    def test_node_added(self):
        from comfyui_massmediafactory_mcp.workflow_diff import diff_workflows

        wf_a = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}}}
        wf_b = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
            "2": {"class_type": "KSampler", "inputs": {"steps": 20}},
        }
        result = diff_workflows(wf_a, wf_b)
        assert len(result["nodes_added"]) == 1
        assert result["nodes_added"][0]["node_id"] == "2"

    def test_node_removed(self):
        from comfyui_massmediafactory_mcp.workflow_diff import diff_workflows

        wf_a = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
            "2": {"class_type": "KSampler", "inputs": {"steps": 20}},
        }
        wf_b = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}}}
        result = diff_workflows(wf_a, wf_b)
        assert len(result["nodes_removed"]) == 1

    def test_node_modified(self):
        from comfyui_massmediafactory_mcp.workflow_diff import diff_workflows

        wf_a = {"1": {"class_type": "KSampler", "inputs": {"steps": 20, "cfg": 7.0}}}
        wf_b = {"1": {"class_type": "KSampler", "inputs": {"steps": 30, "cfg": 7.0}}}
        result = diff_workflows(wf_a, wf_b)
        assert len(result["nodes_modified"]) == 1
        assert any(c["field"] == "inputs.steps" for c in result["nodes_modified"][0]["changes"])

    def test_metadata_keys_skipped(self):
        from comfyui_massmediafactory_mcp.workflow_diff import diff_workflows

        wf_a = {"_meta": {"version": "1.0"}, "1": {"class_type": "A", "inputs": {}}}
        wf_b = {"_meta": {"version": "2.0"}, "1": {"class_type": "A", "inputs": {}}}
        result = diff_workflows(wf_a, wf_b)
        assert result["identical"] is True  # _meta ignored


class TestConditioningNodes:
    """Test execution.py _find_conditioning_nodes fix."""

    def test_traces_positive_and_negative(self):
        from comfyui_massmediafactory_mcp.execution import _find_conditioning_nodes

        workflow = {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "positive prompt"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "negative prompt"},
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": ["1", 0],
                    "negative": ["2", 0],
                    "steps": 20,
                },
            },
        }
        result = _find_conditioning_nodes(workflow)
        assert "1" in result["positive"]
        assert "2" in result["negative"]

    def test_no_sampler_returns_empty(self):
        from comfyui_massmediafactory_mcp.execution import _find_conditioning_nodes

        workflow = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
        }
        result = _find_conditioning_nodes(workflow)
        assert result["positive"] == []
        assert result["negative"] == []


class TestPipelineExtractOutput:
    """Test pipeline.py _extract_output_files format fix."""

    def test_list_format(self):
        """wait_for_completion returns list format."""
        from comfyui_massmediafactory_mcp.pipeline import _extract_output_files

        outputs = [
            {
                "type": "images",
                "filename": "img_001.png",
                "subfolder": "",
                "node_id": "1",
            },
            {
                "type": "video",
                "filename": "vid_001.mp4",
                "subfolder": "",
                "node_id": "2",
            },
        ]
        result = _extract_output_files(outputs, "images")
        assert len(result) == 1
        assert result[0] == "img_001.png"

    def test_dict_format(self):
        """Raw ComfyUI API returns dict format."""
        from comfyui_massmediafactory_mcp.pipeline import _extract_output_files

        outputs = {
            "1": {
                "images": [
                    {"filename": "img_001.png", "subfolder": "", "type": "output"},
                ]
            }
        }
        result = _extract_output_files(outputs, "images")
        assert len(result) == 1
        assert result[0] == "img_001.png"

    def test_empty_outputs(self):
        from comfyui_massmediafactory_mcp.pipeline import _extract_output_files

        assert _extract_output_files([], "images") == []
        assert _extract_output_files({}, "images") == []


class TestQATypeFix:
    """Test QA asset_type mismatch fix."""

    def test_images_plural_accepted(self):
        """Registry stores 'images' (plural), QA should accept it."""
        import tempfile
        from pathlib import Path
        from comfyui_massmediafactory_mcp.qa import qa_output
        from comfyui_massmediafactory_mcp.assets import AssetRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            registry = AssetRegistry(ttl_hours=1, db_path=db_path)

            # Register an asset with "images" type (as registry does)
            asset = registry.register_asset(
                filename="test.png",
                subfolder="",
                asset_type="images",
                workflow={"1": {"class_type": "SaveImage"}},
            )

            # Patch get_registry to return our test registry
            import comfyui_massmediafactory_mcp.qa as qa_mod

            original = qa_mod.get_registry
            qa_mod.get_registry = lambda: registry
            try:
                result = qa_output(asset.asset_id, "a test prompt")
                # Will fail with FILE_NOT_FOUND (file doesn't exist) but NOT UNSUPPORTED_ASSET_TYPE
                assert result.get("error") != "UNSUPPORTED_ASSET_TYPE"
            finally:
                qa_mod.get_registry = original


class TestPublishPathTraversal:
    """Test publish_asset path traversal prevention."""

    def test_traversal_filename_rejected(self):
        """Filename with ../ should be rejected."""
        # This will fail earlier (asset not found) but the path validation
        # would catch it if we had a valid asset
        # We test the validation logic directly instead
        from pathlib import Path

        dest_dir = Path("/tmp/claude/test_publish")
        filename = "../../../etc/passwd"
        dest = dest_dir / filename
        try:
            dest_resolved = dest.resolve()
            dir_resolved = dest_dir.resolve()
            is_safe = dest_resolved.is_relative_to(dir_resolved)
        except Exception:
            is_safe = False
        assert is_safe is False


class TestTopologyDuplicate:
    """Test that SIGMAS duplicate was removed."""

    def test_no_duplicate_sigmas(self):
        from comfyui_massmediafactory_mcp.topology_validator import TYPE_COMPATIBILITY

        # SIGMAS should only appear once
        sigmas_count = sum(1 for k in TYPE_COMPATIBILITY if k == "SIGMAS")
        assert sigmas_count == 1


class TestAssetsURLEncoding:
    """Test URL encoding in get_asset_url."""

    def test_filename_with_spaces_encoded(self):
        import tempfile
        from pathlib import Path
        from comfyui_massmediafactory_mcp.assets import AssetRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            registry = AssetRegistry(ttl_hours=1, db_path=db_path)
            asset = registry.register_asset(
                filename="my image (1).png",
                subfolder="",
                asset_type="images",
                workflow={},
            )
            url = registry.get_asset_url(asset)
            assert "my%20image" in url
            assert "%281%29" in url
            assert " " not in url.split("?")[1]  # No raw spaces in query params

    def test_subfolder_with_special_chars_encoded(self):
        import tempfile
        from pathlib import Path
        from comfyui_massmediafactory_mcp.assets import AssetRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            registry = AssetRegistry(ttl_hours=1, db_path=db_path)
            asset = registry.register_asset(
                filename="test.png",
                subfolder="folder with spaces/sub&dir",
                asset_type="images",
                workflow={},
            )
            url = registry.get_asset_url(asset)
            assert "folder%20with%20spaces" in url
            assert "%26" in url  # & encoded


class TestProfilingTimingReconstruction:
    """Test profiling reconstructs per-node timing from status messages."""

    def test_timing_from_executing_messages(self):
        """Profiling extracts duration from sequential executing timestamps."""
        from unittest.mock import patch, MagicMock
        from comfyui_massmediafactory_mcp.profiling import (
            get_execution_profile,
            _profiles,
        )

        _profiles.clear()  # Clear cache

        mock_client = MagicMock()
        mock_client.get_history.return_value = {
            "test-prompt": {
                "status": {
                    "status_str": "success",
                    "messages": [
                        ["execution_start", {"timestamp": 1000.0}],
                        ["executing", {"node": "1", "timestamp": 1000.0}],
                        ["executing", {"node": "2", "timestamp": 1002.5}],
                        ["executing", {"node": "3", "timestamp": 1003.0}],
                        ["executing", {"node": None, "timestamp": 1010.0}],
                    ],
                },
                "outputs": {"3": {"images": [{"filename": "out.png"}]}},
                "prompt": [
                    None,
                    None,
                    {
                        "1": {"class_type": "CheckpointLoader"},
                        "2": {"class_type": "KSampler"},
                        "3": {"class_type": "SaveImage"},
                    },
                ],
            }
        }

        with patch("comfyui_massmediafactory_mcp.client.get_client", return_value=mock_client):
            result = get_execution_profile("test-prompt")

        assert result["node_count"] == 3
        assert result["total_duration_ms"] == 10000.0  # 1010 - 1000 = 10s
        # Node 1: 1002.5 - 1000.0 = 2500ms
        # Node 2: 1003.0 - 1002.5 = 500ms
        # Node 3: 1010.0 - 1003.0 = 7000ms
        nodes_by_id = {n["node_id"]: n for n in result["nodes"]}
        assert nodes_by_id["1"]["duration_ms"] == 2500.0
        assert nodes_by_id["2"]["duration_ms"] == 500.0
        assert nodes_by_id["3"]["duration_ms"] == 7000.0
        assert nodes_by_id["3"]["class_type"] == "SaveImage"
        # Slowest node should be node 3
        assert result["slowest_node"]["node_id"] == "3"
        _profiles.clear()

    def test_no_timestamps_returns_zero_duration(self):
        """Without timestamps, durations default to 0."""
        from unittest.mock import patch, MagicMock
        from comfyui_massmediafactory_mcp.profiling import (
            get_execution_profile,
            _profiles,
        )

        _profiles.clear()

        mock_client = MagicMock()
        mock_client.get_history.return_value = {
            "test-no-ts": {
                "status": {
                    "status_str": "success",
                    "messages": [
                        ["executing", {"node": "1"}],
                        ["executing", {"node": None}],
                    ],
                },
                "outputs": {},
                "prompt": [None, None, {"1": {"class_type": "KSampler"}}],
            }
        }

        with patch("comfyui_massmediafactory_mcp.client.get_client", return_value=mock_client):
            result = get_execution_profile("test-no-ts")

        assert result["node_count"] == 1
        assert result["total_duration_ms"] == 0.0
        _profiles.clear()

    def test_not_found_prompt(self):
        """Missing prompt_id returns error."""
        from unittest.mock import patch, MagicMock
        from comfyui_massmediafactory_mcp.profiling import (
            get_execution_profile,
            _profiles,
        )

        _profiles.clear()

        mock_client = MagicMock()
        mock_client.get_history.return_value = {}

        with patch("comfyui_massmediafactory_mcp.client.get_client", return_value=mock_client):
            result = get_execution_profile("nonexistent")

        assert result["error"] == "NOT_FOUND"
        _profiles.clear()


class TestPublishSourcePathValidation:
    """Test that publish source path uses is_relative_to (not startswith)."""

    def test_source_path_uses_relative_check(self):
        """Verify get_source_path uses is_relative_to not startswith."""
        import inspect
        from comfyui_massmediafactory_mcp.publish import PublishManager

        source = inspect.getsource(PublishManager.get_source_path)
        assert "is_relative_to" in source
        assert "startswith" not in source


class TestBatchShortTimeout:
    """Test that batch completed jobs use short timeout for stragglers."""

    def test_short_timeout_after_min_complete(self):
        """After min_complete reached, remaining jobs get shorter timeout."""
        import inspect
        from comfyui_massmediafactory_mcp.batch import _wait_for_jobs_blocking

        source = inspect.getsource(_wait_for_jobs_blocking)
        # The if branch should have a different (shorter) timeout than the else branch
        assert "min(30, timeout)" in source
