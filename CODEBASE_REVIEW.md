# MassMediaFactory MCP - Codebase Review

**Review Date:** January 2026
**Reviewed By:** Claude Opus 4.5
**Standards:** ai-model-docs/comfyui, Context7 /comfy-org/docs

---

## Summary

The MassMediaFactory MCP is well-architected with strong validation, asset tracking, and execution patterns. This review identifies gaps against ComfyUI standards and recommends improvements.

**Overall Assessment:** Good (7/10)

---

## Strengths

### 1. Clean Architecture
- Modular separation: client, execution, validation, assets, publish, qa
- Global singleton pattern for client/registry (thread-safe)
- Clear separation between low-level API and high-level tools

### 2. Robust Validation (validation.py)
- Checks node types exist in ComfyUI
- Validates model file references
- Verifies node connections (source exists, slot valid)
- Detects orphaned nodes and missing outputs

### 3. Asset Registry (assets.py)
- TTL-based expiration (configurable)
- Identity-based deduplication
- Full workflow/parameter provenance for regeneration

### 4. Error Handling Patterns
- Consistent `{"error": "..."}` return format
- Graceful handling of ComfyUI unavailability

---

## Gaps & Recommendations

### Priority 1: Critical (Should Fix)

#### 1.1 Missing Image Upload API

**Gap:** No tool for uploading reference images to ComfyUI.

**Impact:** Cannot use ControlNet, IP-Adapter, or Image-to-Video workflows.

**Fix:**
```python
# client.py
def upload_image(self, image_path: str, overwrite: bool = True) -> dict:
    """Upload image to ComfyUI input folder."""
    # Implement multipart/form-data POST to /upload/image
```

**Location:** `client.py`, `execution.py`

#### 1.2 Missing Output Download API

**Gap:** No tool to download generated files directly.

**Impact:** Users must manually copy files or use URLs.

**Fix:**
```python
# execution.py
def download_output(asset_id: str, output_path: str) -> dict:
    """Download asset file to local path."""
```

**Location:** `execution.py`

#### 1.3 Connection Type Wildcards

**Gap:** `check_node_compatibility` doesn't handle wildcard types (`*`, `COMBO`).

**Fix:** Update type matching logic to handle:
- `*` matches any type
- `COMBO` type with enum values
- Union types (`IMAGE,MASK`)

**Location:** `validation.py:245-306`

---

### Priority 2: Important (Should Address)

#### 2.1 Workflow Format Conversion

**Gap:** Only supports "API format" (node_id → {class_type, inputs}). ComfyUI native format uses arrays with links table.

**Impact:** Cannot import workflows exported from ComfyUI UI.

**Recommendation:** Add conversion utilities:
```python
# persistence.py
def convert_ui_to_api_format(ui_workflow: dict) -> dict:
    """Convert ComfyUI UI format to API format."""

def convert_api_to_ui_format(api_workflow: dict) -> dict:
    """Convert API format to ComfyUI UI format for visualization."""
```

**Reference:** Context7 `/comfy-org/docs` workflow_json.mdx

#### 2.2 Cycle Detection

**Gap:** Validation doesn't check for circular dependencies in node graph.

**Impact:** Infinite loops at runtime.

**Fix:** Add topological sort validation:
```python
# validation.py
def _check_cycle(workflow: dict) -> List[str]:
    """Detect cycles using DFS."""
```

#### 2.3 Resolution Compatibility Warnings

**Gap:** Doesn't warn about resolution mismatches between connected nodes.

**Example:** Connecting 1024x1024 latent to a node expecting 512x512.

**Fix:** Add heuristic checks for `width`, `height`, `megapixels` parameters.

#### 2.4 WebSocket Support

**Gap:** Only polling-based status checks. No WebSocket real-time updates.

**Impact:** Higher latency, more API calls.

**Recommendation:** Add optional WebSocket monitoring:
```python
# execution.py
async def wait_for_completion_ws(prompt_id: str, on_progress=None) -> dict:
    """Wait with WebSocket progress updates."""
```

---

### Priority 3: Nice-to-Have (Consider)

#### 3.1 Structured Error Codes

**Current:** `{"error": "ASSET_NOT_FOUND_OR_EXPIRED"}`

**Recommended:** Add error code registry:
```python
class MCPError(Enum):
    ASSET_NOT_FOUND = ("E001", "Asset not found or expired")
    MODEL_NOT_FOUND = ("E002", "Model file not found in ComfyUI")
    CONNECTION_FAILED = ("E003", "Cannot connect to ComfyUI")
    VALIDATION_FAILED = ("E004", "Workflow validation failed")
```

#### 3.2 Client Retry Logic

**Gap:** No automatic retry on transient failures.

**Fix:** Add retry decorator with exponential backoff:
```python
@retry(max_attempts=3, backoff=2.0)
def request(...):
```

#### 3.3 Connection Pooling

**Gap:** Creates new connection per request.

**Impact:** Slight performance overhead (minimal for most use cases).

#### 3.4 Type Annotations Improvements

**Files needing attention:**
- `batch.py`: Some `Any` return types
- `pipeline.py`: Missing parameter type hints
- `templates/__init__.py`: Generic dict returns

---

## Code Quality

### PEP 8 Compliance: Good

Minor issues:
- `server.py:1016` exceeds recommended file length (consider splitting template definitions)
- Some functions in `validation.py` exceed 50 lines

### Docstrings: Good

All public functions documented. Private functions (`_helper`) could use brief docstrings.

### Type Hints: Good

Most modules have complete type annotations. Recommend:
- Use `TypedDict` for complex return types
- Add `@overload` decorators for functions with multiple signatures

---

## Refactoring Plan

### Phase 1: Critical APIs (1-2 days)
1. Add `upload_image()` to client and expose as tool
2. Add `download_output()` for asset retrieval
3. Fix connection type wildcard matching

### Phase 2: Validation Enhancements (1 day)
1. Add cycle detection to `validate_workflow()`
2. Add resolution mismatch warnings
3. Improve error messages with suggestions

### Phase 3: Format Support (1 day)
1. Add UI ↔ API workflow format conversion
2. Update `import_workflow()` to auto-detect format

### Phase 4: Polish (Optional)
1. Add structured error codes
2. Add retry logic to client
3. Consider WebSocket support for long generations

---

## Files to Modify

| File | Changes |
|------|---------|
| `client.py` | Add `upload_image()`, `download_file()`, retry logic |
| `execution.py` | Add `upload_image()`, `download_output()` tools |
| `validation.py` | Add cycle detection, wildcard types, resolution warnings |
| `persistence.py` | Add format conversion utilities |
| `server.py` | Register new tools |

---

## Testing Recommendations

1. Add integration tests against running ComfyUI
2. Add unit tests for validation edge cases
3. Test with real workflows from ComfyUI share sites

---

*Review completed. Ready for implementation upon approval.*
