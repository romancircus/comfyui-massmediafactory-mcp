"""
Tests for Priority 2 features: node_registry, civitai, and new client endpoints.

Covers:
- NodeRegistry: dynamic discovery, fallback, caching, invalidation, stats, type compat
- CivitAI: search, error handling, template conversion, model/task detection
- Client new endpoints: embeddings, model folders, metadata, features, jobs, clear_history
"""

import importlib
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from comfyui_massmediafactory_mcp import node_registry
from comfyui_massmediafactory_mcp.civitai import (
    search_workflows,
    convert_to_template,
    _detect_model_from_workflow,
    _detect_task_from_workflow,
)

# Import the REAL ComfyUIClient class, bypassing conftest's sys.modules mock.
# conftest.py replaces comfyui_massmediafactory_mcp.client with a mock module,
# so we need to load the actual source file directly.
_client_src = Path(__file__).parent.parent / "src" / "comfyui_massmediafactory_mcp" / "client.py"
_spec = importlib.util.spec_from_file_location("_real_client", _client_src)
_real_client_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real_client_mod)
RealComfyUIClient = _real_client_mod.ComfyUIClient

# Patch target for node_registry: it does `from .client import get_client` inside
# function bodies, so we patch at the source module level.
_CLIENT_PATCH = "comfyui_massmediafactory_mcp.client.get_client"


# =============================================================================
# TestNodeRegistry
# =============================================================================


class TestNodeRegistry:
    """Tests for src/comfyui_massmediafactory_mcp/node_registry.py"""

    def setup_method(self):
        """Clear cache before every test to avoid cross-test pollution."""
        node_registry.invalidate_cache()

    def test_dynamic_results_when_comfyui_available(self):
        """get_node_output_types() returns dynamic results when ComfyUI responds."""
        mock_client = MagicMock()
        mock_client.get_object_info.return_value = {
            "CheckpointLoaderSimple": {
                "output": ["MODEL", "CLIP", "VAE"],
                "input": {"required": {}},
            },
            "KSampler": {
                "output": ["LATENT"],
                "input": {"required": {}},
            },
            "MyCustomNode": {
                "output": ["IMAGE", "MASK"],
                "input": {"required": {}},
            },
        }

        with patch(_CLIENT_PATCH, return_value=mock_client):
            result = node_registry.get_node_output_types()

        assert "CheckpointLoaderSimple" in result
        assert result["CheckpointLoaderSimple"] == ["MODEL", "CLIP", "VAE"]
        assert "KSampler" in result
        assert result["KSampler"] == ["LATENT"]
        assert "MyCustomNode" in result
        assert result["MyCustomNode"] == ["IMAGE", "MASK"]
        # Dynamic result should have exactly the 3 nodes from the mock
        assert len(result) == 3

    def test_fallback_when_comfyui_unreachable(self):
        """Falls back to hardcoded when ComfyUI raises an exception."""
        mock_client = MagicMock()
        mock_client.get_object_info.side_effect = ConnectionError("Connection refused")

        with patch(_CLIENT_PATCH, return_value=mock_client):
            result = node_registry.get_node_output_types()

        # Should return fallback entries
        assert "CheckpointLoaderSimple" in result
        assert "KSampler" in result
        assert "VAEDecode" in result
        # Fallback should match the hardcoded count
        assert len(result) == len(node_registry._FALLBACK_OUTPUT_TYPES)

    def test_fallback_when_object_info_returns_error(self):
        """Falls back to hardcoded when /object_info returns an error dict."""
        mock_client = MagicMock()
        mock_client.get_object_info.return_value = {"error": "Server error"}

        with patch(_CLIENT_PATCH, return_value=mock_client):
            result = node_registry.get_node_output_types()

        assert "CheckpointLoaderSimple" in result
        assert len(result) == len(node_registry._FALLBACK_OUTPUT_TYPES)

    def test_cache_behavior_second_call_no_new_api_call(self):
        """Second call returns cached result without making a new API call."""
        mock_client = MagicMock()
        mock_client.get_object_info.return_value = {
            "TestNode": {
                "output": ["IMAGE"],
                "input": {"required": {}},
            },
        }

        with patch(_CLIENT_PATCH, return_value=mock_client):
            result1 = node_registry.get_node_output_types()
            result2 = node_registry.get_node_output_types()

        # get_object_info should only be called once (cached on second call)
        assert mock_client.get_object_info.call_count == 1
        assert result1 is result2
        assert result1["TestNode"] == ["IMAGE"]

    def test_invalidate_cache_forces_refetch(self):
        """invalidate_cache() clears cache so next call re-queries ComfyUI."""
        mock_client = MagicMock()
        mock_client.get_object_info.return_value = {
            "NodeA": {
                "output": ["MODEL"],
                "input": {"required": {}},
            },
        }

        with patch(_CLIENT_PATCH, return_value=mock_client):
            result1 = node_registry.get_node_output_types()
            assert mock_client.get_object_info.call_count == 1

            # Invalidate and fetch again
            node_registry.invalidate_cache()
            result2 = node_registry.get_node_output_types()
            assert mock_client.get_object_info.call_count == 2

        # Both should have the same content (same mock)
        assert "NodeA" in result1
        assert "NodeA" in result2

    def test_get_registry_stats_not_loaded(self):
        """get_registry_stats() returns correct metadata when nothing is loaded."""
        stats = node_registry.get_registry_stats()
        assert stats["loaded"] is False
        assert stats["source"] == "none"
        assert stats["node_count"] == 0

    def test_get_registry_stats_after_dynamic_load(self):
        """get_registry_stats() returns correct metadata after dynamic discovery."""
        mock_client = MagicMock()
        # Return more nodes than fallback to be detected as dynamic
        fallback_count = len(node_registry._FALLBACK_OUTPUT_TYPES)
        nodes = {}
        for i in range(fallback_count + 10):
            nodes[f"Node{i}"] = {"output": ["IMAGE"], "input": {"required": {}}}
        mock_client.get_object_info.return_value = nodes

        with patch(_CLIENT_PATCH, return_value=mock_client):
            node_registry.get_node_output_types()

        stats = node_registry.get_registry_stats()
        assert stats["loaded"] is True
        assert stats["source"] == "dynamic"
        assert stats["node_count"] == fallback_count + 10
        assert stats["fallback_count"] == fallback_count
        assert stats["cache_age_seconds"] >= 0
        assert stats["cache_ttl_seconds"] == 300

    def test_get_registry_stats_after_fallback(self):
        """get_registry_stats() reports 'fallback' source when using hardcoded data."""
        mock_client = MagicMock()
        mock_client.get_object_info.side_effect = ConnectionError("unreachable")

        with patch(_CLIENT_PATCH, return_value=mock_client):
            node_registry.get_node_output_types()

        stats = node_registry.get_registry_stats()
        assert stats["loaded"] is True
        assert stats["source"] == "fallback"

    def test_get_type_compatibility_returns_dict(self):
        """get_type_compatibility() returns a dict with type mappings."""
        mock_client = MagicMock()
        mock_client.get_object_info.return_value = {
            "KSampler": {
                "output": ["LATENT"],
                "input": {
                    "required": {
                        "model": ["MODEL", {}],
                        "positive": ["CONDITIONING", {}],
                        "negative": ["CONDITIONING", {}],
                        "latent_image": ["LATENT", {}],
                    },
                    "optional": {},
                },
            },
        }

        with patch(_CLIENT_PATCH, return_value=mock_client):
            compat = node_registry.get_type_compatibility()

        assert isinstance(compat, dict)
        # MODEL type should map to "model" input name
        assert "MODEL" in compat
        assert "model" in compat["MODEL"]
        # CONDITIONING should map to "positive" and "negative"
        assert "CONDITIONING" in compat
        assert "positive" in compat["CONDITIONING"]
        assert "negative" in compat["CONDITIONING"]

    def test_get_type_compatibility_fallback(self):
        """get_type_compatibility() falls back to hardcoded when ComfyUI unreachable."""
        mock_client = MagicMock()
        mock_client.get_object_info.side_effect = ConnectionError("unreachable")

        with patch(_CLIENT_PATCH, return_value=mock_client):
            compat = node_registry.get_type_compatibility()

        assert isinstance(compat, dict)
        # Fallback should have MODEL, CLIP, VAE etc.
        assert "MODEL" in compat
        assert "CLIP" in compat
        assert "VAE" in compat
        assert "CONDITIONING" in compat

    def test_non_dict_nodes_skipped(self):
        """Non-dict entries in object_info are skipped gracefully."""
        mock_client = MagicMock()
        mock_client.get_object_info.return_value = {
            "ValidNode": {"output": ["IMAGE"], "input": {}},
            "InvalidEntry": "not a dict",
            "AnotherInvalid": 42,
        }

        with patch(_CLIENT_PATCH, return_value=mock_client):
            result = node_registry.get_node_output_types()

        assert "ValidNode" in result
        assert "InvalidEntry" not in result
        assert "AnotherInvalid" not in result

    def test_force_refresh_bypasses_cache(self):
        """force_refresh=True bypasses cache even when cache is valid."""
        mock_client = MagicMock()
        mock_client.get_object_info.return_value = {
            "NodeV1": {"output": ["IMAGE"], "input": {}},
        }

        with patch(_CLIENT_PATCH, return_value=mock_client):
            node_registry.get_node_output_types()
            assert mock_client.get_object_info.call_count == 1

            # Update mock to return different data
            mock_client.get_object_info.return_value = {
                "NodeV2": {"output": ["LATENT"], "input": {}},
            }

            result = node_registry.get_node_output_types(force_refresh=True)
            assert mock_client.get_object_info.call_count == 2
            assert "NodeV2" in result


# =============================================================================
# TestCivitAI
# =============================================================================


class TestCivitAI:
    """Tests for src/comfyui_massmediafactory_mcp/civitai.py"""

    def _mock_urlopen_response(self, data):
        """Create a mock context manager for urllib.request.urlopen."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_search_workflows_parses_response(self):
        """search_workflows() parses CivitAI API response correctly."""
        api_response = {
            "items": [
                {
                    "id": 12345,
                    "url": "https://civitai.com/images/12345",
                    "width": 1024,
                    "height": 1024,
                    "meta": {
                        "prompt": "a beautiful landscape with mountains",
                        "Model": "FLUX.1-dev",
                        "sampler": "euler",
                        "steps": 20,
                        "cfgScale": 7.0,
                    },
                    "stats": {"heartCount": 42, "commentCount": 5},
                },
                {
                    "id": 67890,
                    "url": "https://civitai.com/images/67890",
                    "width": 512,
                    "height": 768,
                    "meta": {
                        "prompt": "cyberpunk city at night",
                        "Model": "SDXL",
                        "sampler": "dpmpp_2m",
                        "steps": 30,
                        "cfgScale": 5.0,
                    },
                    "stats": {"heartCount": 100},
                },
            ],
            "metadata": {"nextCursor": "abc123"},
        }

        mock_resp = self._mock_urlopen_response(api_response)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = search_workflows("flux portrait", limit=10)

        assert result["query"] == "flux portrait"
        assert result["count"] == 2
        assert len(result["results"]) == 2

        first = result["results"][0]
        assert first["id"] == 12345
        assert first["model"] == "FLUX.1-dev"
        assert first["sampler"] == "euler"
        assert first["steps"] == 20
        assert first["cfg"] == 7.0
        assert "beautiful landscape" in first["prompt"]
        assert first["stats"]["heartCount"] == 42

        assert result["metadata"]["cursor"] == "abc123"

    def test_search_workflows_handles_api_error(self):
        """search_workflows() returns error dict on API failure."""
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            result = search_workflows("test query")

        assert "error" in result
        assert "CivitAI API error" in result["error"]
        assert result["query"] == "test query"

    def test_search_workflows_handles_none_meta(self):
        """search_workflows() handles items where meta is None."""
        api_response = {
            "items": [
                {
                    "id": 11111,
                    "url": "https://civitai.com/images/11111",
                    "width": 512,
                    "height": 512,
                    "meta": None,
                    "stats": {},
                },
            ],
            "metadata": {},
        }

        mock_resp = self._mock_urlopen_response(api_response)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = search_workflows("test")

        assert result["count"] == 1
        first = result["results"][0]
        assert first["prompt"] == ""
        assert first["model"] == "unknown"

    def test_convert_to_template_injects_placeholders(self):
        """convert_to_template() injects PROMPT, SEED, CFG placeholders."""
        workflow = {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "a beautiful cat sitting on a table",
                    "clip": ["2", 0],
                },
            },
            "2": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "model.safetensors"},
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 20,
                    "cfg": 7.0,
                    "denoise": 1.0,
                    "positive": ["1", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                },
            },
        }

        result = convert_to_template(workflow, name="test_template")
        template = result["template"]
        placeholders = result["placeholders"]

        # Check that placeholders were injected
        placeholder_names = [p["placeholder"] for p in placeholders]
        assert "PROMPT" in placeholder_names
        assert "SEED" in placeholder_names
        assert "CFG" in placeholder_names
        assert "STEPS" in placeholder_names
        assert "DENOISE" in placeholder_names

        # Check template has _meta
        assert "_meta" in template
        assert template["_meta"]["name"] == "test_template"
        assert template["_meta"]["source"] == "civitai"
        assert template["_meta"]["version"] == "1.0.0"

        # Check actual values in template are placeholder strings
        assert template["3"]["inputs"]["seed"] == "{{SEED}}"
        assert template["3"]["inputs"]["cfg"] == "{{CFG}}"
        assert template["1"]["inputs"]["text"] == "{{PROMPT}}"

    def test_convert_to_template_detects_model_type(self):
        """convert_to_template() auto-detects model type from workflow nodes."""
        flux_workflow = {
            "1": {"class_type": "DualCLIPLoader", "inputs": {}},
            "2": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["3", 0]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "test"}},
        }

        result = convert_to_template(flux_workflow, name="flux_test", model="unknown")
        assert result["model_detected"] == "flux"

    def test_convert_to_template_detects_task_type(self):
        """convert_to_template() auto-detects task type from workflow nodes."""
        # txt2img workflow (no LoadImage, no video output)
        txt2img_workflow = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "test"}},
            "2": {"class_type": "KSampler", "inputs": {"seed": 42}},
            "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "output"}},
        }

        result = convert_to_template(txt2img_workflow, name="t2i_test", task="unknown")
        assert result["task_detected"] == "txt2img"

    def test_convert_to_template_preserves_connections(self):
        """convert_to_template() does not replace connection references (lists)."""
        workflow = {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "hello",
                    "clip": ["2", 0],  # This is a connection, should NOT be replaced
                },
            },
            "2": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "model.safetensors"},
            },
        }

        result = convert_to_template(workflow, name="test")
        template = result["template"]

        # Connection should be preserved as-is
        assert template["1"]["inputs"]["clip"] == ["2", 0]

    def test_convert_to_template_skips_meta_keys(self):
        """convert_to_template() skips keys starting with underscore."""
        workflow = {
            "_meta": {"version": "1.0"},
            "1": {
                "class_type": "KSampler",
                "inputs": {"seed": 42, "cfg": 7.0},
            },
        }

        result = convert_to_template(workflow, name="test")
        # Should not crash on _meta, and _meta should be overwritten
        assert result["template"]["_meta"]["name"] == "test"

    def test_detect_model_from_workflow_flux(self):
        """_detect_model_from_workflow() identifies FLUX from FluxGuidance."""
        workflow = {
            "1": {"class_type": "FluxGuidance", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {}},
        }
        assert _detect_model_from_workflow(workflow) == "flux"

    def test_detect_model_from_workflow_flux_via_dualclip(self):
        """_detect_model_from_workflow() identifies FLUX from DualCLIPLoader."""
        workflow = {
            "1": {"class_type": "DualCLIPLoader", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {}},
        }
        assert _detect_model_from_workflow(workflow) == "flux"

    def test_detect_model_from_workflow_wan(self):
        """_detect_model_from_workflow() identifies WAN from WanVideoModelLoader."""
        workflow = {
            "1": {"class_type": "WanVideoModelLoader", "inputs": {}},
            "2": {"class_type": "WanVideoSampler", "inputs": {}},
        }
        assert _detect_model_from_workflow(workflow) == "wan"

    def test_detect_model_from_workflow_ltx(self):
        """_detect_model_from_workflow() identifies LTX from LTXVScheduler."""
        workflow = {
            "1": {"class_type": "LTXVScheduler", "inputs": {}},
            "2": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
        }
        assert _detect_model_from_workflow(workflow) == "ltx"

    def test_detect_model_from_workflow_ltx_via_conditioning(self):
        """_detect_model_from_workflow() identifies LTX from LTXVConditioning."""
        workflow = {
            "1": {"class_type": "LTXVConditioning", "inputs": {}},
        }
        assert _detect_model_from_workflow(workflow) == "ltx"

    def test_detect_model_from_workflow_qwen(self):
        """_detect_model_from_workflow() identifies Qwen from ModelSamplingAuraFlow."""
        workflow = {
            "1": {"class_type": "ModelSamplingAuraFlow", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {}},
        }
        assert _detect_model_from_workflow(workflow) == "qwen"

    def test_detect_model_from_workflow_hunyuan(self):
        """_detect_model_from_workflow() identifies Hunyuan from HunyuanVideoModelLoader."""
        workflow = {
            "1": {"class_type": "HunyuanVideoModelLoader", "inputs": {}},
        }
        assert _detect_model_from_workflow(workflow) == "hunyuan"

    def test_detect_model_from_workflow_sdxl_fallback(self):
        """_detect_model_from_workflow() identifies SDXL from CheckpointLoaderSimple only."""
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {}},
        }
        assert _detect_model_from_workflow(workflow) == "sdxl"

    def test_detect_model_from_workflow_unknown(self):
        """_detect_model_from_workflow() returns 'unknown' when no model markers found."""
        workflow = {
            "1": {"class_type": "SomeCustomNode", "inputs": {}},
        }
        assert _detect_model_from_workflow(workflow) == "unknown"

    def test_detect_task_from_workflow_txt2img(self):
        """_detect_task_from_workflow() identifies txt2img (no image input, no video output)."""
        workflow = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {}},
            "3": {"class_type": "SaveImage", "inputs": {}},
        }
        assert _detect_task_from_workflow(workflow) == "txt2img"

    def test_detect_task_from_workflow_img2vid(self):
        """_detect_task_from_workflow() identifies img2vid (LoadImage + video output)."""
        workflow = {
            "1": {"class_type": "LoadImage", "inputs": {}},
            "2": {"class_type": "WanVideoSampler", "inputs": {}},
            "3": {"class_type": "SaveVideo", "inputs": {}},
        }
        assert _detect_task_from_workflow(workflow) == "img2vid"

    def test_detect_task_from_workflow_txt2vid(self):
        """_detect_task_from_workflow() identifies txt2vid (no image, video output)."""
        workflow = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {}},
            "2": {"class_type": "WanVideoSampler", "inputs": {}},
            "3": {"class_type": "CreateVideo", "inputs": {}},
        }
        assert _detect_task_from_workflow(workflow) == "txt2vid"

    def test_detect_task_from_workflow_img2img(self):
        """_detect_task_from_workflow() identifies img2img (LoadImage + image output)."""
        workflow = {
            "1": {"class_type": "LoadImage", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {}},
            "3": {"class_type": "SaveImage", "inputs": {}},
        }
        assert _detect_task_from_workflow(workflow) == "img2img"

    def test_detect_task_vhs_video_combine(self):
        """_detect_task_from_workflow() recognizes VHS_VideoCombine as video output."""
        workflow = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {}},
            "2": {"class_type": "VHS_VideoCombine", "inputs": {}},
        }
        assert _detect_task_from_workflow(workflow) == "txt2vid"


# =============================================================================
# TestClientNewEndpoints
# =============================================================================


class TestClientNewEndpoints:
    """Tests for new methods in src/comfyui_massmediafactory_mcp/client.py

    Uses the real ComfyUIClient class (loaded directly from source to bypass
    conftest's sys.modules mock) with request method patched.
    """

    def setup_method(self):
        """Create a fresh real client with a test base URL."""
        self.client = RealComfyUIClient(base_url="http://test:8188")

    def test_get_embeddings_calls_correct_endpoint(self):
        """get_embeddings() calls GET /embeddings."""
        with patch.object(self.client, "get", return_value=["embedding1", "embedding2"]) as mock_get:
            result = self.client.get_embeddings()
            mock_get.assert_called_once_with("/embeddings")
            assert result == ["embedding1", "embedding2"]

    def test_list_model_folders_calls_correct_endpoint(self):
        """list_model_folders() calls GET /models."""
        with patch.object(
            self.client, "get", return_value=["checkpoints", "loras", "vae"]
        ) as mock_get:
            result = self.client.list_model_folders()
            mock_get.assert_called_once_with("/models")
            assert result == ["checkpoints", "loras", "vae"]

    def test_list_models_in_folder_calls_correct_endpoint(self):
        """list_models_in_folder() calls GET /models/<folder> with URL encoding."""
        with patch.object(
            self.client, "get", return_value=["model1.safetensors", "model2.safetensors"]
        ) as mock_get:
            result = self.client.list_models_in_folder("checkpoints")
            mock_get.assert_called_once_with("/models/checkpoints")
            assert result == ["model1.safetensors", "model2.safetensors"]

    def test_list_models_in_folder_url_encodes_folder_name(self):
        """list_models_in_folder() URL-encodes folder names with special characters."""
        with patch.object(self.client, "get", return_value=[]) as mock_get:
            self.client.list_models_in_folder("my folder/sub")
            # urllib.parse.quote encodes spaces as %20 but preserves / (safe by default)
            mock_get.assert_called_once_with("/models/my%20folder/sub")

    def test_get_model_metadata_calls_correct_endpoint(self):
        """get_model_metadata() calls GET /view_metadata/<folder>?filename=<file>."""
        metadata = {"ss_training_steps": 1000, "ss_resolution": "1024x1024"}
        with patch.object(self.client, "get", return_value=metadata) as mock_get:
            result = self.client.get_model_metadata("checkpoints", "model.safetensors")
            mock_get.assert_called_once_with("/view_metadata/checkpoints?filename=model.safetensors")
            assert result == metadata

    def test_get_model_metadata_url_encodes_params(self):
        """get_model_metadata() URL-encodes folder and filename."""
        with patch.object(self.client, "get", return_value={}) as mock_get:
            self.client.get_model_metadata("my models", "file name (1).safetensors")
            expected = "/view_metadata/my%20models?filename=file%20name%20%281%29.safetensors"
            mock_get.assert_called_once_with(expected)

    def test_get_features_calls_correct_endpoint(self):
        """get_features() calls GET /features."""
        features = {"jobs_api": True, "workflow_templates": False}
        with patch.object(self.client, "get", return_value=features) as mock_get:
            result = self.client.get_features()
            mock_get.assert_called_once_with("/features")
            assert result == features

    def test_get_jobs_calls_correct_endpoint_no_filter(self):
        """get_jobs() calls GET /api/jobs with limit only."""
        jobs = {"jobs": [{"id": "job1"}, {"id": "job2"}]}
        with patch.object(self.client, "get", return_value=jobs) as mock_get:
            result = self.client.get_jobs()
            mock_get.assert_called_once_with("/api/jobs?limit=50")
            assert result == jobs

    def test_get_jobs_with_status_filter(self):
        """get_jobs() includes status filter in query params."""
        with patch.object(self.client, "get", return_value={"jobs": []}) as mock_get:
            self.client.get_jobs(status="running", limit=10)
            mock_get.assert_called_once_with("/api/jobs?limit=10&status=running")

    def test_get_jobs_url_encodes_status(self):
        """get_jobs() URL-encodes status parameter."""
        with patch.object(self.client, "get", return_value={"jobs": []}) as mock_get:
            self.client.get_jobs(status="in progress")
            mock_get.assert_called_once_with("/api/jobs?limit=50&status=in%20progress")

    def test_get_job_calls_correct_endpoint(self):
        """get_job() calls GET /api/jobs/<job_id>."""
        job = {"id": "abc-123", "status": "completed"}
        with patch.object(self.client, "get", return_value=job) as mock_get:
            result = self.client.get_job("abc-123")
            mock_get.assert_called_once_with("/api/jobs/abc-123")
            assert result == job

    def test_clear_history_with_prompt_ids(self):
        """clear_history() sends delete list when prompt_ids provided."""
        with patch.object(self.client, "post", return_value={}) as mock_post:
            result = self.client.clear_history(prompt_ids=["id1", "id2"])
            mock_post.assert_called_once_with("/history", {"delete": ["id1", "id2"]})
            assert result == {}

    def test_clear_history_without_prompt_ids(self):
        """clear_history() sends clear=True when no prompt_ids provided."""
        with patch.object(self.client, "post", return_value={}) as mock_post:
            result = self.client.clear_history()
            mock_post.assert_called_once_with("/history", {"clear": True})
            assert result == {}

    def test_clear_history_explicit_none(self):
        """clear_history(prompt_ids=None) sends clear=True."""
        with patch.object(self.client, "post", return_value={}) as mock_post:
            self.client.clear_history(prompt_ids=None)
            mock_post.assert_called_once_with("/history", {"clear": True})

    def test_clear_history_empty_list(self):
        """clear_history(prompt_ids=[]) sends clear=True (empty list is falsy)."""
        with patch.object(self.client, "post", return_value={}) as mock_post:
            self.client.clear_history(prompt_ids=[])
            mock_post.assert_called_once_with("/history", {"clear": True})

    def test_get_jobs_custom_limit(self):
        """get_jobs() respects custom limit parameter."""
        with patch.object(self.client, "get", return_value={"jobs": []}) as mock_get:
            self.client.get_jobs(limit=5)
            mock_get.assert_called_once_with("/api/jobs?limit=5")
