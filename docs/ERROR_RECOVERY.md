# Error Recovery Guide

This guide provides actionable recovery steps for common errors when using the ComfyUI MassMediaFactory MCP.

## Error Response Format

All errors follow the MCP specification and include:

```json
{
  "isError": true,
  "code": "ERROR_CODE",
  "error": "Human-readable error message",
  "suggestion": "What to do to fix it",
  "details": { ... },
  "troubleshooting": "Detailed steps if needed"
}
```

---

## MODEL_NOT_FOUND

**When it happens:** The specified model file is not in your ComfyUI models directory.

**Example Error:**
```json
{
  "code": "MODEL_NOT_FOUND",
  "error": "Model 'flux1-dev.safetensors' not found in models/checkpoints/",
  "suggestion": "Run list_models('checkpoints') to see available models.",
  "details": {
    "model_name": "flux1-dev.safetensors",
    "model_type": "checkpoints",
    "available": ["flux2-dev.safetensors", "sdxl-base.safetensors"]
  }
}
```

**Recovery Steps:**

1. **List available models:**
   ```python
   list_models("checkpoints")  # or "unet", "lora", "vae"
   ```

2. **Check for typos:**
   - Verify exact spelling from `list_models()` output
   - Common issue: `flux1` vs `flux2`, `sdxl` vs `sd-xl`

3. **Download missing models:**
   ```python
   # Search Civitai
   search_civitai("flux dev", model_type="checkpoint")
   
   # Download to correct location
   download_model(url, model_type="checkpoint")
   ```

4. **Verify ComfyUI models directory:**
   ```bash
   ls ~/ComfyUI/models/checkpoints/
   ```

---

## CUSTOM_NODE_MISSING

**When it happens:** A required ComfyUI custom node is not installed or not loaded.

**Example Error:**
```json
{
  "code": "CUSTOM_NODE_MISSING",
  "error": "Custom node 'Wan2VideoNode' not installed or not loaded.",
  "suggestion": "Install the required custom node package or restart ComfyUI.",
  "details": {
    "node_type": "Wan2VideoNode",
    "package": "comfyui-wan2-video",
    "install_cmd": "pip install comfyui-wan2-video"
  }
}
```

**Recovery Steps:**

1. **Install the package (if provided):**
   ```bash
   pip install comfyui-wan2-video
   ```

2. **Restart ComfyUI:**
   ```bash
   sudo systemctl restart comfyui
   ```

3. **Check ComfyUI UI:**
   Open ComfyUI → Look for "Missing Nodes" panel → Install required nodes

4. **Verify node is loaded:**
   ```python
   search_nodes("Wan")
   ```

---

## TEMPLATE_METADATA

**When it happens:** A workflow template has invalid or missing metadata structure.

**Example Error:**
```json
{
  "code": "TEMPLATE_METADATA",
  "error": "Template 'flux_basic' missing required fields: description, model, type",
  "suggestion": "Ensure template has _meta section with: description, model, type, parameters, defaults.",
  "details": {
    "template_name": "flux_basic",
    "missing_fields": ["description", "model", "type"],
    "errors": []
  }
}
```

**Recovery Steps:**

1. **List valid templates:**
   ```python
   list_workflow_templates()
   ```

2. **Inspect a valid template structure:**
   ```python
   get_template("flux_t2i")  # or any working template
   ```

3. **Fix template metadata:**
   ```json
   {
     "_meta": {
       "description": "FLUX text-to-image generation",
       "model": "flux",
       "type": "t2i",
       "parameters": ["PROMPT", "SEED", "WIDTH", "HEIGHT"],
       "defaults": {
         "PROMPT": "a dragon in the clouds",
         "SEED": 42,
         "WIDTH": 1024,
         "HEIGHT": 1024
       }
     },
     "...": "workflow nodes here"
   }
   ```

4. **Validate all templates:**
   ```python
   # In a testing context
   from comfyui_massmediafactory_mcp.templates import validate_all_templates
   validate_all_templates()
   ```

---

## TEMPLATE_PARAMETER

**When it happens:** A parameter has an invalid value (wrong type, out of range, etc.).

**Example Error:**
```json
{
  "code": "TEMPLATE_PARAMETER",
  "error": "Parameter 'FRAMES' invalid for template 'ltx_t2v'.",
  "suggestion": "Expected: integer divisible by 8 (8n+1). Run get_template('ltx_t2v') to see valid parameters.",
  "details": {
    "template_name": "ltx_t2v",
    "parameter": "FRAMES",
    "expected": "integer divisible by 8",
    "provided": 97
  }
}
```

**Recovery Steps:**

1. **Check template metadata:**
   ```python
   get_template("ltx_t2v")
   ```
   Look for `_meta.defaults` and `_meta.parameters`

2. **Fix parameter values:**
   ```python
   # Wrong:
   FRAMES=97  # Not divisible by 8
   
   # Right:
   FRAMES=81  # 8n+1 pattern
   FRAMES=89
   FRAMES=97
   ```

3. **Common parameter constraints:**

   | Parameter | Constraint | Valid Values |
   |-----------|------------|--------------|
   | `FRAMES` (LTX) | Divisible by 8 | 81, 89, 97, 105, 113... |
   | `WIDTH` | Divisible by 16 (FLUX) or 8 (LTX) | 512, 768, 1024, 1536... |
   | `HEIGHT` | Divisible by 16 or 8 | 512, 768, 1024... |
   | `CFG` | Min 1.5, depends on model | 2.5-4.0 (LTX), 3.5 (FLUX) |
   | `STEPS` | Min 10, max 100 | 20-50 typical |

4. **Use generate_workflow() for safe defaults:**
   ```python
   generate_workflow(
       model="flux",
       workflow_type="t2i",
       prompt="a dragon in the clouds"
   )
   # Uses safe defaults automatically
   ```

---

## Common Error Codes Reference

| Code | When it happens | Immediate action |
|------|-----------------|------------------|
| `MODEL_NOT_FOUND` | Model missing | Check `list_models()` → download or fix typo |
| `CUSTOM_NODE_MISSING` | Node not loaded | Install package → restart ComfyUI |
| `TEMPLATE_METADATA` | Template invalid | Check `_meta` section structure |
| `TEMPLATE_PARAMETER` | Invalid param value | Check `get_template()` → fix value |
| `CONNECTION_FAILED` | Can't reach ComfyUI | `sudo systemctl status comfyui` |
| `OUT_OF_VRAM` | GPU memory full | Lower res `free_memory(unload_models=True)` |
| `TIMEOUT` | Workflow took too long | Increase `timeout_seconds` |
| `VALIDATION_ERROR` | Input validation failed | Check required params, types |
| `NOT_FOUND` | Resource missing | Verify ID, check `list_*()` functions |
| `RATE_LIMITED` | Too many requests | Wait `retry_after_seconds` |

---

## Debugging Tips

### Enable Verbose Logging

```bash
# View ComfyUI logs
sudo journalctl -u comfyui -f

# Check MCP server logs
tail -f ~/.local/state/comfyui-massmediafactory-mcp/logs/
```

### Validate Workflows Before Execution

```python
# Validate with auto-fix
result = validate_workflow(workflow, auto_fix=True, check_pattern=True)

if not result["valid"]:
    print("Errors:", result["errors"])
    print("Corrections:", result["corrections"])
    # Use corrected workflow
    workflow = result["workflow"]
```

### Check VRAM Before Large Workflows

```python
# Estimate VRAM usage
estimate_vram(workflow)

# Check if model fits
check_model_fits("flux2-dev", precision="fp16")

# Free memory if needed
free_memory(unload_models=True)
```

### Test Templates Independently

```python
# List all templates and their status
templates = list_workflow_templates()
for t in templates["templates"]:
    print(f"✓ {t['name']}: {t['description']}")

# Get template details
template = get_template("ltx_t2v")
print(template["_meta"])  # See parameters and defaults
```

---

## Getting Help

1. **Check this guide first** - Most errors are documented here
2. **Use `list_*()` functions** - See what's available (models, templates, assets)
3. **Run `validate_workflow()`** - Catch errors before execution
4. **Check logs** - `sudo journalctl -u comfyui -f`

---

## Error Response Examples

### Example 1: Model Not Found
```json
{
  "isError": true,
  "code": "MODEL_NOT_FOUND",
  "error": "Model 'flux1-dev.safetensors' not found in models/checkpoints/",
  "suggestion": "Run list_models('checkpoints') to see available models. Common issues: typo in model name, model not downloaded, or wrong model_type.",
  "details": {
    "model_name": "flux1-dev.safetensors",
    "model_type": "checkpoints",
    "available": ["flux2-dev.safetensors", "flux-dev-pro.safetensors"]
  },
  "troubleshooting": "Check ~/ComfyUI/models/checkpoints/ for the file."
}
```

### Example 2: Template Parameter Error
```json
{
  "isError": true,
  "code": "TEMPLATE_PARAMETER",
  "error": "Parameter 'FRAMES' invalid for template 'ltx_t2v'.",
  "suggestion": "Expected: integer divisible by 8 (8n+1). Use 81, 89, 97, 105, etc.",
  "details": {
    "template_name": "ltx_t2v",
    "parameter": "FRAMES",
    "expected": "integer divisible by 8",
    "provided": 100,
    "suggested_values": [81, 89, 97, 105, 113]
  },
  "troubleshooting": "1. Check template parameters: get_template('ltx_t2v')\n2. Use list_workflow_templates() to see all templates"
}
```

### Example 3: Custom Node Missing
```json
{
  "isError": true,
  "code": "CUSTOM_NODE_MISSING",
  "error": "Custom node 'Wan2VideoNode' not installed or not loaded.",
  "suggestion": "Run: pip install comfyui-wan2-video",
  "details": {
    "node_type": "Wan2VideoNode",
    "package": "comfyui-wan2-video",
    "install_cmd": "pip install comfyui-wan2-video"
  },
  "troubleshooting": "1. Install package: pip install <package>\n2. Restart ComfyUI: sudo systemctl restart comfyui\n3. Check nodes are loaded in ComfyUI UI (missing nodes panel)"
}
```