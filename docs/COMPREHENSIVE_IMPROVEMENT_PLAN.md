# Comprehensive Improvement Plan

**Date:** January 2026
**Scope:** MassMediaFactory MCP + ai-model-docs

---

## Source Verification

### Confirmed Official Sources

| Source | Maintainer | URL | Status |
|--------|------------|-----|--------|
| ComfyUI Examples | comfyanonymous (ComfyUI creator) | comfyanonymous.github.io/ComfyUI_examples | 41 categories verified |
| Lightricks LTX-Video | Lightricks (Model creator) | github.com/Lightricks/ComfyUI-LTXVideo | 6 workflows verified |

### ComfyUI UI Templates

The "templates library" in the ComfyUI UI (http://127.0.0.1:8188) typically comes from:
1. **ComfyUI Manager** extension (most common)
2. **Workflow gallery** (loads from examples site)
3. **Custom extensions** with bundled templates

Our documentation is based on the **official examples site**, which is the authoritative source maintained by ComfyUI's creator.

---

## Part 1: Template Gap Analysis

### Current Templates (22)

| Type | Templates |
|------|-----------|
| Image Gen | qwen_txt2img, flux2_txt2img, qwen_poster_design, flux2_face_id, flux2_edit_by_text, flux2_lora_stack, flux2_union_controlnet, flux2_grounding_dino_inpaint, flux2_ultimate_upscale |
| Video Gen | ltx2_txt2vid, ltx2_txt2vid_distilled, ltx2_img2vid, ltx2_audio_reactive, wan21_img2vid, video_inpaint, video_stitch |
| Audio/TTS | audio_tts_f5, audio_tts_voice_clone, chatterbox_tts, qwen3_tts_custom_voice, qwen3_tts_voice_clone, qwen3_tts_voice_design |

### Missing Templates (30+ from official categories)

**Critical Priority - SOTA models:**

| Category | Templates Needed | Official Source |
|----------|------------------|-----------------|
| HunyuanVideo 1.5 | hunyuan_video_txt2vid, hunyuan_video_img2vid | ComfyUI examples/hunyuan_video |
| Nvidia Cosmos | cosmos_txt2vid, cosmos_img2vid | ComfyUI examples/cosmos |
| Mochi | mochi_txt2vid | ComfyUI examples/mochi |
| SVD | svd_img2vid | ComfyUI examples/svd |
| SDXL | sdxl_txt2img, sdxl_img2img, sdxl_inpaint | ComfyUI examples/sdxl |
| SD3/SD3.5 | sd3_txt2img, sd35_txt2img | ComfyUI examples/sd3 |
| Wan 2.2 | wan22_txt2vid, wan22_img2vid | ComfyUI examples/wan |

**High Priority - Common workflows:**

| Category | Templates Needed | Notes |
|----------|------------------|-------|
| ControlNet (individual) | flux2_canny_controlnet, flux2_depth_controlnet, qwen_depth_controlnet | Only union exists |
| IP-Adapter | flux2_ipadapter, sdxl_ipadapter_face | Character consistency |
| Upscaling | esrgan_upscale, realesrgan_upscale | Multiple approaches |
| LoRA | qwen_lora_txt2img | Qwen LoRA support |

**Medium Priority:**

| Category | Templates Needed |
|----------|------------------|
| 3D | hunyuan3d_txt2mesh, hunyuan3d_img2mesh |
| AuraFlow | auraflow_txt2img |
| Lumina 2.0 | lumina_txt2img |
| Z-Image | z_image_txt2img |
| Omnigen2 | omnigen2_txt2img |

---

## Part 2: Documentation Gaps

### COMFYUI_OFFICIAL_WORKFLOWS.md - Missing Sections

| Section | Status | Priority |
|---------|--------|----------|
| HunyuanVideo 1.5 patterns | Missing | Critical |
| Nvidia Cosmos patterns | Missing | High |
| SDXL workflow patterns | Missing | High |
| SD3/SD3.5 patterns | Missing | High |
| ControlNet detailed patterns | Minimal | High |
| IP-Adapter patterns | Missing | High |
| Audio workflow patterns | Missing | High |
| 3D generation patterns | Missing | Medium |

### MASSMEDIAFACTORY_MCP.md - Issues

| Issue | Current | Needed |
|-------|---------|--------|
| Template list | 4 templates | 22 templates |
| QA tools | Not documented | qa_output(), detect_objects() |
| Style learning | Not documented | Full module docs |
| Model management | Not documented | search_civitai(), download_model() |

---

## Part 3: Code Quality Improvements

### templates/__init__.py

**Current Issues:**
1. No template validation on load
2. Silent JSON parsing failures
3. No type hints for template structure
4. Parameter injection edge cases

**Recommended Changes:**

```python
# Add template validation
def validate_template(template: dict) -> list[str]:
    """Validate template structure, return list of errors."""
    errors = []
    if "_meta" not in template:
        errors.append("Missing _meta section")
    else:
        meta = template["_meta"]
        required_meta = ["description", "model", "type", "parameters", "defaults"]
        for field in required_meta:
            if field not in meta:
                errors.append(f"Missing _meta.{field}")

    # Check placeholders exist in workflow
    workflow_str = json.dumps(template)
    placeholders = re.findall(r'\{\{([A-Z_]+)\}\}', workflow_str)
    declared = set(template.get("_meta", {}).get("parameters", []))
    for p in placeholders:
        if p not in declared:
            errors.append(f"Undeclared placeholder: {{{{{p}}}}}")

    return errors

# Add template categorization
def get_templates_by_type(type_filter: str) -> dict:
    """Filter templates by type: txt2img, img2vid, txt2vid, tts, etc."""
    all_templates = list_templates()
    filtered = [t for t in all_templates["templates"] if t.get("type") == type_filter]
    return {"templates": filtered, "count": len(filtered)}

# Add requirement checking
def check_template_requirements(template_name: str) -> dict:
    """Check which models are required and if they're installed."""
    template = load_template(template_name)
    if "error" in template:
        return template

    # Extract model references from workflow
    workflow_str = json.dumps(template)
    # ... analyze for model file references
    # ... check against installed models
    return {"required": [...], "installed": [...], "missing": [...]}
```

### validation.py

**Add missing model resolution specs:**

```python
MODEL_RESOLUTION_SPECS = {
    # Existing...
    "hunyuan_video": {"native": 1280, "divisible_by": 16, "min": 256, "max": 1920},
    "cosmos": {"native": 1024, "divisible_by": 16, "min": 256, "max": 2048},
    "mochi": {"native": 848, "divisible_by": 16, "min": 256, "max": 1280},
    "sd3": {"native": 1024, "divisible_by": 8, "min": 512, "max": 2048},
    "z_image": {"native": 1024, "divisible_by": 16, "min": 512, "max": 2048},
}
```

**Add video-specific validation:**

```python
def validate_video_params(workflow: dict, model_type: str) -> list[str]:
    """Validate video-specific parameters."""
    warnings = []

    # Check frame count for LTX (must be 8n+1)
    if "ltx" in model_type.lower():
        for node in workflow.values():
            if node.get("class_type") == "EmptyLTXVLatentVideo":
                frames = node.get("inputs", {}).get("length", 0)
                if isinstance(frames, int) and (frames - 1) % 8 != 0:
                    warnings.append(f"LTX frames should be 8n+1, got {frames}")

    return warnings
```

---

## Part 4: Agent Usability Improvements

### Problem: Agents Can't Build Workflows From Scratch

**Solution: Add workflow builder helpers**

```python
# New file: workflow_builder.py

def create_workflow_skeleton(model_type: str) -> dict:
    """Create minimal workflow skeleton for model type."""
    skeletons = {
        "flux": {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "{{MODEL}}"}},
            "2": {"class_type": "DualCLIPLoader", "inputs": {...}},
            # ... minimal structure
        },
        "ltx": {...},
        "sdxl": {...},
    }
    return skeletons.get(model_type, {})

def get_node_sequence(model: str, task: str) -> list[dict]:
    """Return recommended node sequence for model+task."""
    sequences = {
        ("flux", "txt2img"): [
            {"class_type": "UNETLoader", "connects_to": "SamplerCustomAdvanced.model"},
            {"class_type": "DualCLIPLoader", "connects_to": "CLIPTextEncode.clip"},
            # ...
        ],
    }
    return sequences.get((model, task), [])

def explain_workflow(workflow: dict) -> str:
    """Generate natural language description of workflow."""
    # Analyze workflow structure
    # Describe what each node does
    # Explain the data flow
    pass
```

### Problem: No Way to Discover Template Capabilities

**Solution: Add template discovery tools**

```python
# Add to server.py

@mcp.tool()
def get_template_capabilities() -> dict:
    """Get all templates organized by capability."""
    return {
        "txt2img": [...],
        "img2img": [...],
        "txt2vid": [...],
        "img2vid": [...],
        "tts": [...],
        "upscale": [...],
        "inpaint": [...],
        "controlnet": [...],
    }

@mcp.tool()
def find_template_for_task(task_description: str) -> dict:
    """Find best template for a given task description."""
    # Use semantic matching
    # Return ranked templates
    pass
```

---

## Part 5: Cross-Repo Integration

### Current State

| Repo | Purpose | Integration |
|------|---------|-------------|
| ai-model-docs | Prompting guides, conceptual docs | References MCP |
| comfyui-massmediafactory-mcp | Execution tools, templates | References docs |

### Gaps

1. **Template documentation split** - Templates exist in MCP but not documented in ai-model-docs
2. **SOTA knowledge duplication** - MCP has sota.py, ai-model-docs has model guides
3. **Workflow patterns not synced** - COMFYUI_ORCHESTRATION.md recipes don't match templates

### Integration Actions

**A. Create ai-model-docs/comfyui/TEMPLATE_REFERENCE.md**

Complete template matrix with:
- All 22+ templates
- Required models
- Use cases
- Example parameters

**B. Update MASSMEDIAFACTORY_MCP.md**

Add all missing tool documentation:
- Complete template list
- QA tools
- Style learning module
- Model management tools

**C. Sync workflow recipes ↔ templates**

Every recipe in COMFYUI_ORCHESTRATION.md should have a corresponding template.

---

## Implementation Roadmap

### Phase 1: Critical (Week 1)

| Task | Files | Priority |
|------|-------|----------|
| Add template validation | templates/__init__.py | P0 |
| Update MASSMEDIAFACTORY_MCP.md | ai-model-docs | P0 |
| Add sdxl_txt2img template | templates/ | P0 |
| Document HunyuanVideo patterns | COMFYUI_OFFICIAL_WORKFLOWS.md | P0 |

### Phase 2: High (Week 2)

| Task | Files | Priority |
|------|-------|----------|
| Add hunyuan_video_txt2vid template | templates/ | P1 |
| Add controlnet individual templates | templates/ | P1 |
| Add IP-Adapter templates | templates/ | P1 |
| Create TEMPLATE_REFERENCE.md | ai-model-docs | P1 |

### Phase 3: Medium (Week 3-4)

| Task | Files | Priority |
|------|-------|----------|
| Add workflow builder tools | workflow_builder.py | P2 |
| Add cosmos, mochi templates | templates/ | P2 |
| Document audio workflows | COMFYUI_OFFICIAL_WORKFLOWS.md | P2 |
| Sync SOTA with sota-tracker | sota.py | P2 |

---

## Success Metrics

- [ ] All 41 official ComfyUI workflow categories have corresponding templates
- [ ] All templates pass validation on load
- [ ] MASSMEDIAFACTORY_MCP.md documents all tools and templates
- [ ] LLM agents can build workflows from scratch using builder tools
- [ ] Cross-repo references are complete and bidirectional
