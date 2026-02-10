# DEPRECATED - Skeleton Files

**Status**: These skeleton files are deprecated as of January 2026.

## What Changed

Workflow skeletons have been moved to `src/comfyui_massmediafactory_mcp/patterns.py` for the following reasons:

1. **Single Source of Truth**: All workflow patterns, model constraints, and node chains are now in one place
2. **Better LLM Support**: patterns.py includes validation and drift prevention
3. **Direct API Format**: Skeletons now use ComfyUI API format directly (no conversion needed)
4. **Comprehensive Coverage**: patterns.py includes all models (LTX-2, FLUX.2, Wan 2.1, Qwen, SDXL, HunyuanVideo 1.5)

## Migration

The `workflow_generator.py` module now loads skeletons from `patterns.py` instead of these JSON files.

### Old Flow
```
workflow_generator.py -> docs/library/skeletons/*.json
```

### New Flow
```
workflow_generator.py -> patterns.py (WORKFLOW_SKELETONS)
```

## Files in This Directory

These files are kept for reference only:

| File | Superseded By |
|------|---------------|
| `ltx_video_t2v.json` | `patterns.WORKFLOW_SKELETONS[("ltx2", "txt2vid")]` |
| `ltx_video_i2v.json` | `patterns.WORKFLOW_SKELETONS[("ltx2", "img2vid")]` |
| `flux_dev_t2i.json` | `patterns.WORKFLOW_SKELETONS[("flux2", "txt2img")]` |
| `wan_t2v.json` | `patterns.WORKFLOW_SKELETONS[("wan21", "txt2vid")]` |
| `qwen_t2i.json` | `patterns.WORKFLOW_SKELETONS[("qwen", "txt2img")]` |

## Do Not Modify

Do not modify these files. All changes should be made in `patterns.py`.

## Removal Timeline

These files may be removed in a future version once we verify all workflows are working correctly with patterns.py.
