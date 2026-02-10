# Workflow Drift Analysis

**Problem:** When Claude constructs ComfyUI workflows, it drifts significantly from working patterns.

---

## The 10 Systematic Mistakes

### 1. Wrong Model Loaders

| Model | Claude Uses | Should Use |
|-------|-------------|------------|
| LTX-2 | CheckpointLoaderSimple | **LTXVLoader** |
| FLUX.2 | CheckpointLoaderSimple | **UNETLoader + DualCLIPLoader + VAELoader** (3 nodes!) |
| Wan 2.1 | CheckpointLoaderSimple | **WanVideoModelLoader** |

**Why it breaks:** Output slots differ. LTXVLoader: [MODEL=0, CLIP=1, VAE=2]. DualCLIPLoader: [CLIP=0] only.

### 2. Missing Conditioning Wrappers

Claude builds:
```
CLIPTextEncode → KSampler
```

Correct patterns:
```
LTX-2:  CLIPTextEncode → LTXVConditioning → SamplerCustom
FLUX:   CLIPTextEncode → FluxGuidance → BasicGuider → SamplerCustomAdvanced
```

**Why it breaks:** LTXVConditioning injects frame_rate. FluxGuidance handles guidance scaling. Without these, output is garbage.

### 3. Using KSampler Instead of Advanced Samplers

Claude builds:
```json
{"class_type": "KSampler", "inputs": {"cfg": 7.0, "steps": 20, ...}}
```

Correct pattern (3-4 nodes):
```
LTXVScheduler → sigmas
KSamplerSelect → sampler
SamplerCustom (takes model, sampler, sigmas, latent)
```

**Why it breaks:** Video/advanced models need separate scheduler to compute sigmas. KSampler's built-in scheduler doesn't work.

### 4. Wrong CFG Values

| Model | Claude Uses | Should Use |
|-------|-------------|------------|
| LTX-2 | 7.0 | **3.0** (ONLY) |
| FLUX.2 | 7.5 | **3.5** (via FluxGuidance node) |
| Wan 2.1 | 7.0 | **5.0** |

**Why it breaks:** LTX-2 at CFG 7.0 = oversaturated, broken output.

### 5. Wrong Latent Node Names

| Model | Claude Uses | Should Use |
|-------|-------------|------------|
| FLUX.2 | EmptyLatentImage | **EmptySD3LatentImage** |
| LTX-2 | EmptyLatentVideo | **EmptyLTXVLatentVideo** |
| Wan 2.1 | (missing) | **EmptyWanLatentVideo** |

**Why it breaks:** Incompatible latent format = black frames or crash.

### 6. Missing Model-Specific Encoding

| Model | Claude Uses | Should Use |
|-------|-------------|------------|
| FLUX.2 | CLIPTextEncode from CheckpointLoader | **DualCLIPLoader** (clip_l + t5xxl) |
| Wan 2.1 | Generic CLIPTextEncode | **CLIPLoader** with type="wan" |

**Why it breaks:** Wrong text encoder architecture = prompts not understood.

### 7. Wrong Output Nodes

| Output Type | Claude Uses | Should Use |
|-------------|-------------|------------|
| Video | SaveImage | **VHS_VideoCombine** |
| Image | SaveImage | SaveImage (correct) |

**Why it breaks:** SaveImage saves only first frame as PNG.

### 8. Wrong Slot Indices

```
Claude: ["1", 1]  (assumes CLIP at slot 1)
Actual:
  - LTXVLoader: CLIP at slot 1 ✓
  - DualCLIPLoader: CLIP at slot 0 ✗
```

**Why it breaks:** Type mismatch error, node rejects input.

### 9. Missing KSamplerSelect

Claude skips this node entirely. SamplerCustom REQUIRES a SAMPLER object from KSamplerSelect.

### 10. Wrong Scheduler Parameters

| Model | Claude Uses | Should Use |
|-------|-------------|------------|
| LTX-2 | (none) | **LTXVScheduler** with max_shift=2.05, base_shift=0.95 |
| FLUX.2 | (none) | **BasicScheduler** with scheduler="simple" |

**Why it breaks:** Wrong sigma curve = poor quality or artifacts.

---

## Node-by-Node Comparison

### LTX-2 Text-to-Video

**Claude's Wrong Version (6 nodes, broken):**
```json
{
  "1": {"class_type": "CheckpointLoaderSimple"},
  "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
  "3": {"class_type": "EmptyLatentVideo"},
  "4": {"class_type": "KSampler", "inputs": {"cfg": 7.0}},
  "5": {"class_type": "VAEDecode"},
  "6": {"class_type": "SaveImage"}
}
```

**Correct Version (10 nodes, works):**
```json
{
  "1": {"class_type": "LTXVLoader", "inputs": {"dtype": "bfloat16"}},
  "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
  "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
  "4": {"class_type": "LTXVConditioning", "inputs": {"positive": ["2", 0], "negative": ["3", 0], "frame_rate": 24}},
  "5": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"length": 97}},
  "6": {"class_type": "LTXVScheduler", "inputs": {"steps": 30, "max_shift": 2.05, "base_shift": 0.95}},
  "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
  "8": {"class_type": "SamplerCustom", "inputs": {"cfg": 3.0, "positive": ["4", 0], "negative": ["4", 1]}},
  "9": {"class_type": "VAEDecode"},
  "10": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": 24, "format": "video/h264-mp4"}}
}
```

**Differences:** 4 extra nodes (LTXVConditioning, LTXVScheduler, KSamplerSelect, swap to SamplerCustom), different loader, different latent node, different output node, CFG 3.0 not 7.0.

### FLUX.2 Text-to-Image

**Claude's Wrong Version (6 nodes, broken):**
```json
{
  "1": {"class_type": "CheckpointLoaderSimple"},
  "2": {"class_type": "CLIPTextEncode"},
  "3": {"class_type": "EmptyLatentImage"},
  "4": {"class_type": "KSampler", "inputs": {"cfg": 7.5}},
  "5": {"class_type": "VAEDecode"},
  "6": {"class_type": "SaveImage"}
}
```

**Correct Version (13 nodes, works):**
```json
{
  "1": {"class_type": "UNETLoader"},
  "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp16.safetensors"}},
  "3": {"class_type": "VAELoader"},
  "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0]}},
  "5": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5}},
  "6": {"class_type": "EmptySD3LatentImage"},
  "7": {"class_type": "KSamplerSelect"},
  "8": {"class_type": "BasicScheduler", "inputs": {"scheduler": "simple"}},
  "9": {"class_type": "RandomNoise"},
  "10": {"class_type": "BasicGuider", "inputs": {"conditioning": ["5", 0]}},
  "11": {"class_type": "SamplerCustomAdvanced"},
  "12": {"class_type": "VAEDecode"},
  "13": {"class_type": "SaveImage"}
}
```

**Differences:** 7 extra nodes (3 loaders, FluxGuidance, BasicGuider, BasicScheduler, RandomNoise), completely different sampling pipeline.

---

## Solution: MCP Tools for Exact Patterns

### Tool 1: get_workflow_skeleton(model, task)

Returns the exact node structure for a model+task combination:

```python
@mcp.tool()
def get_workflow_skeleton(model: str, task: str) -> dict:
    """Get exact working node structure for model+task.

    Args:
        model: "ltx2", "flux2", "wan21", "qwen", "sdxl"
        task: "txt2vid", "img2vid", "txt2img"

    Returns:
        Complete workflow JSON with placeholders.
    """
```

### Tool 2: get_node_chain(model, task)

Returns ordered list of required nodes with connection info:

```python
@mcp.tool()
def get_node_chain(model: str, task: str) -> list:
    """Get required nodes in order with connections.

    Returns:
        [
            {"id": "1", "class_type": "LTXVLoader", "outputs": {"MODEL": 0, "CLIP": 1, "VAE": 2}},
            {"id": "2", "class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
            ...
        ]
    """
```

### Tool 3: get_model_constraints(model)

Returns hard constraints that must not be violated:

```python
@mcp.tool()
def get_model_constraints(model: str) -> dict:
    """Get model-specific constraints.

    Returns:
        {
            "cfg": {"min": 2.5, "max": 4.0, "default": 3.0},
            "resolution": {"divisible_by": 8, "native": [768, 512]},
            "frames": {"formula": "8n+1", "examples": [97, 121]},
            "sampler": {"required": "SamplerCustom", "not": "KSampler"},
            "conditioning_wrapper": "LTXVConditioning",
            "scheduler": "LTXVScheduler"
        }
    """
```

### Tool 4: validate_against_pattern(workflow, model)

Validates a workflow against known working pattern:

```python
@mcp.tool()
def validate_against_pattern(workflow: dict, model: str) -> dict:
    """Check workflow against working pattern.

    Returns:
        {
            "valid": False,
            "errors": [
                "Using KSampler - should use SamplerCustom",
                "Missing LTXVConditioning node",
                "CFG is 7.0 - should be 3.0"
            ],
            "corrected_workflow": {...}
        }
    """
```

---

## Implementation Priority

1. **P0:** Add get_workflow_skeleton() - returns exact template for model+task
2. **P0:** Add get_model_constraints() - returns CFG, resolution, frame rules
3. **P1:** Add validate_against_pattern() - catches drift before execution
4. **P1:** Add get_node_chain() - shows exact connection flow

These tools ensure Claude queries for exact patterns instead of guessing.
