# QA Checklist - Pre and Post Migration Verification

> **Version:** 1.0
> **Date:** 2026-02-07
> **Scope:** Migration from HTTP clients to mmf CLI across all repos

---

## Summary

This document provides verification steps for:
1. **Baseline state** before migration work begins
2. **Post-migration verification** after parallel execution completes
3. **Regression tests** to ensure functionality is preserved

**Migration Status (Current):**
- ✅ **MCP Main Repo:** Already on mmf CLI (ROM-548 completed)
- ✅ **KDH-Automation:** Already on mmf CLI (mmf.js complete)
- ⏳ **Goat:** Uses direct HTTP client (ComfyUIClient.js) + PonyMCPService
- ⏳ **RobloxChristian:** Uses direct HTTP client (ComfyUIHYMotionClient)

---

## Pre-Migration Verification

### 1. MCP Main Repo (comfyui-massmediafactory-mcp)

**Quick State Check:**

```bash
# Verify mmf CLI is installed
mmf --version

# Test system stats (should return GPU info)
mmf stats --pretty

# Test template listing (should return 38+ templates)
mmf templates list | jq '.count'

# Test quick generation (generates actual image)
mmf run --model qwen --type t2i --prompt "QA test" -o /tmp/qa_test.png

# Verify file was created
ls -la /tmp/qa_test.png
```

**Expected Results:**
- CLI version: 0.1.0+
- Templates count: 38+
- System stats: RTX 5090, 31GB VRAM
- Test image: ~1024x1024 PNG created

**MCP Tool Registration:**
- Active tools: 59
- Migrated to CLI: 25 (commented out in server.py)
- Total historically: 84

---

### 2. KDH-Automation (Reference Implementation)

**Verification:**

```bash
cd ~/Applications/KDH-Automation

# Verify mmf.js exists
ls -la src/core/mmf.js

# Verify mmf.js exports
node -e "const mmf = require('./src/core/mmf.js'); console.log('Exports:', Object.keys(mmf));"

# Check mmf.js line count (should be ~200-400 lines)
wc -l src/core/mmf.js
```

**Key Functions to Verify:**
- `qwenTxt2Img()` - Image generation
- `fluxTxt2Img()` - FLUX image generation
- `faceIdTxt2Img()` - Face ID generation
- `wanI2V()` - Video generation
- `wanS2V()` - Sound-to-video
- `ltxT2V()` - LTX text-to-video
- `ltxI2V()` - LTX image-to-video
- `qwenEditBackground()` - Background replacement
- `telestyleImage()` - Style transfer
- `telestyleVideo()` - Video style transfer
- `chatterboxTTS()` - Text-to-speech
- `stats()` - System stats
- `templates()` - List templates

**Success Criteria:**
- mmf.js exists and is ~200-400 lines (previously 2,000+ lines replaced)
- All export functions available
- No remaining `ComfyUIExecutor.js`, `ComfyUIClient.js` references in imports

---

### 3. Goat (Pre-Migration)

**Current State Documentation:**

```bash
cd ~/Applications/Goat

# Check current HTTP client
ls -la src/clients/ComfyUIClient.js

# Check Pony MCP Service (already uses MCP templates)
ls -la src/services/PonyMCPService.js

# Check MCP Service
ls -la src/services/MCPService.js

# Verify template path hardcoded in PonyMCPService
grep -n "comfyui-massmediafactory-mcp" src/services/PonyMCPService.js
```

**Files to Document:**
- `src/clients/ComfyUIClient.js` - Direct HTTP client (to be replaced)
- `src/services/MCPService.js` - MCP service (may be replaced by mmf.js)
- `src/services/PonyMCPService.js` - Pony template loader (references MCP templates directly)
- `src/generators/CharacterSheetGenerator.js` - Uses ComfyUIClient
- `src/generators/SpriteSheetGenerator.js` - Uses ComfyUIClient
- `src/generators/ReactionGenerator.js` - Uses ComfyUIClient

**Pony Templates Referenced:**
- `pony/turnaround_sheet` - Character turnaround
- `pony/expression_sheet` - Expression variations

**Success Criteria (Pre-Migration):**
- All current files documented
- Template paths recorded
- Generator dependencies mapped
- No mmf.js present yet

---

### 4. RobloxChristian (Pre-Migration)

**Current State Documentation:**

```bash
cd ~/Applications/RobloxChristian

# Check current HTTP client implementation
ls -la src/animation_gen.py

# Check for visual_gen_mcp.py
grep -r "visual_gen_mcp" src/

# Check mcp_templates directory
ls -la mcp_templates/

# Verify ComfyUI client usage
grep -n "comfyui_url\|ComfyUIHYMotionClient" src/animation_gen.py | head -20
```

**Files to Document:**
- `src/animation_gen.py` - Contains `ComfyUIHYMotionClient` class
- `mcp_templates/*.json` - 7 templates present:
  - `flux2_txt2img.json`
  - `ltx2_img2vid.json`
  - `ltx2_txt2vid.json`
  - `qwen_edit_background.json`
  - `qwen_txt2img.json`
  - `wan21_img2vid.json`
  - `wan21_txt2vid.json`

**Success Criteria (Pre-Migration):**
- `visual_gen_mcp.py` not found (not yet created)
- All local templates documented
- ComfyUIHYMotionClient usage mapped
- No mmf CLI integration yet

---

## Post-Migration Verification

### 1. MCP Main Repo (No Changes Expected)

```bash
# Re-verify mmf CLI still works after parallel execution
mmf stats --pretty
mmf templates list | jq '.count'

# Verify no regression in MCP tools
grep -c "@mcp.tool()" src/comfyui_massmediafactory_mcp/server.py
# Expected: 59 (should not decrease)
```

**Regression Tests:**
- [ ] `mmf run --model flux --type t2i` produces valid image
- [ ] `mmf run --template qwen_txt2img` produces valid image
- [ ] `mmf batch seeds` works with workflow.json
- [ ] `mmf pipeline i2v` completes successfully
- [ ] MCP tools still accessible via server
- [ ] Template count unchanged (38+)

---

### 2. KDH-Automation (No Changes Expected - Already Migrated)

```bash
# Verify mmf.js still works
cd ~/Applications/KDH-Automation
node -e "const {qwenTxt2Img} = require('./src/core/mmf.js'); console.log('qwenTxt2Img:', typeof qwenTxt2Img);"

# Verify old files removed
ls src/core/ComfyUIExecutor.js 2>&1 | grep "No such file"
ls src/core/ComfyUIClient.js 2>&1 | grep "No such file"
```

**Regression Tests:**
- [ ] All mmf.js functions still export correctly
- [ ] No residual HTTP client files
- [ ] Integration tests pass (if any)

---

### 3. Goat (Post-Migration)

**Expected New Structure:**

```bash
cd ~/Applications/Goat

# Verify new mmf.js exists
ls -la src/core/mmf.js

# Verify old files removed
ls src/clients/ComfyUIClient.js 2>&1 | grep "No such file"
ls src/services/MCPService.js 2>&1 | grep "No such file"

# Verify PonyMCPService.js updated (if kept) or removed
ls src/services/PonyMCPService.js 2>&1 | grep "No such file"

# Verify generators updated
head -20 src/generators/CharacterSheetGenerator.js | grep -E "import.*mmf|from.*mmf"
```

**Migration Verification Tests:**

```bash
# Test mmf.js directly
node -e "
const mmf = require('./src/core/mmf.js');
const stats = mmf.stats();
console.log('Stats result:', stats.system ? 'OK' : 'FAIL');
"

# Test template listing
node -e "
const mmf = require('./src/core/mmf.js');
const templates = mmf.templates();
console.log('Templates count:', templates.count);
"

# Test generation (dry-run or real)
# node -e "const mmf = require('./src/core/mmf.js'); console.log(mmf.qwenTxt2Img({prompt:'test', seed:42}));"
```

**Success Criteria:**
- [ ] `src/core/mmf.js` exists with all required functions
- [ ] `ComfyUIClient.js` removed or deprecated
- [ ] `MCPService.js` removed or deprecated
- [ ] `PonyMCPService.js` removed or refactored
- [ ] All generators updated to use mmf.js
- [ ] Pony templates accessible via `mmf.run --template` syntax
- [ ] Integration tests pass

**Function Mapping:**

| Old (ComfyUIClient.js) | New (mmf.js) | Status |
|------------------------|--------------|--------|
| `isRunning()` | `mmf.stats()` | ✅ Should exist |
| `getSystemStats()` | `mmf.stats()` | ✅ Should exist |
| `getCheckpoints()` | `mmf.models list` | ✅ Should exist |
| `getLoRAs()` | `mmf.models list --type lora` | ✅ Should exist |
| `queuePrompt()` | `mmf.execute()` | ✅ Should exist |
| `waitForCompletion()` | `mmf.wait()` | ✅ Should exist |
| `getImage()` | `mmf.download()` | ✅ Should exist |
| Pony template loading | `mmf.run --template pony/...` | ✅ Should exist |

---

### 4. RobloxChristian (Post-Migration)

**Expected New Structure:**

```bash
cd ~/Applications/RobloxChristian

# Verify new visual_gen_mcp.py exists
ls -la src/visual_gen_mcp.py

# Verify mmf CLI integration
grep -n "mmf\|massmediafactory" src/visual_gen_mcp.py | head -10

# Verify old HTTP client removed or deprecated
grep -n "ComfyUIHYMotionClient\|requests.post" src/animation_gen.py | head -5 || echo "HTTP client removed"

# Check if mcp_templates still needed
ls mcp_templates/ 2>&1
```

**Migration Verification Tests:**

```bash
# Test Python import
python3 -c "from src.visual_gen_mcp import VisualGeneratorMCP; print('Import OK')"

# Test basic functionality
python3 -c "
from src.visual_gen_mcp import VisualGeneratorMCP
vg = VisualGeneratorMCP()
print('Instance:', vg)
print('Available:', vg.list_available_templates())
"

# Test template generation (dry-run if possible)
# python3 -c "
# from src.visual_gen_mcp import VisualGeneratorMCP
# vg = VisualGeneratorMCP()
# result = vg.generate_from_template('qwen_txt2img', {'PROMPT': 'QA test', 'SEED': 42})
# print('Result:', result)
# "
```

**Success Criteria:**
- [ ] `src/visual_gen_mcp.py` exists and is importable
- [ ] `VisualGeneratorMCP` class instantiates correctly
- [ ] `ComfyUIHYMotionClient` removed from `animation_gen.py` or deprecated
- [ ] All template functions work via mmf CLI
- [ ] Local `mcp_templates/` directory may be removed (using MCP server templates)
- [ ] Python tests pass (if any)

**Function Mapping:**

| Old (ComfyUIHYMotionClient) | New (VisualGeneratorMCP) | Status |
|-----------------------------|--------------------------|--------|
| `check_connection()` | `mmf.stats()` | ✅ |
| `get_available_nodes()` | `mmf.templates list` | ✅ |
| `queue_prompt()` | `mmf.execute()` | ✅ |
| `wait_for_completion()` | `mmf.wait()` | ✅ |
| `download_output()` | `mmf.download()` | ✅ |

---

## Cross-Repo Integration Verification

After all repos are migrated:

```bash
# Verify consistent mmf CLI version across repos
mmf --version  # Should be 0.1.0+
cd ~/Applications/KDH-Automation && node -e "console.log(require('./src/core/mmf.js').stats())"
cd ~/Applications/Goat && node -e "console.log(require('./src/core/mmf.js').stats())"
cd ~/Applications/RobloxChristian && python3 -c "from src.visual_gen_mcp import VisualGeneratorMCP; print(VisualGeneratorMCP().get_stats())"
```

All should return consistent system stats (RTX 5090, 31GB VRAM).

---

## Performance Regression Tests

### Generation Speed

| Task | Pre-Migration Baseline | Post-Migration Target | Variance |
|------|------------------------|----------------------|----------|
| Qwen T2I (50 steps) | ~45s | <50s | +10% acceptable |
| Wan I2V (81 frames) | ~8min | <10min | +25% acceptable |
| LTX T2V (97 frames) | ~6min | <8min | +33% acceptable |
| FLUX T2I (20 steps) | ~15s | <20s | +33% acceptable |

### Template Loading

```bash
# Test template loading speed (should be <1s)
time mmf templates list > /dev/null
```

### Batch Operations

```bash
# Test batch execution (4 seed variations)
time mmf batch seeds workflow.json --count 4 --start-seed 42
```

---

## Error Handling Verification

Verify error codes are consistent:

```bash
# Test invalid template
mmf run --template nonexistent --params '{}' 2>&1 | grep -E "error|code"
# Expected: Exit code 6 (NOT_FOUND)

# Test invalid model
mmf run --model nonexistent --type t2i --prompt "test" 2>&1 | grep -E "error|code"
# Expected: Exit code 3 (VALIDATION)

# Test connection failure (with wrong URL)
mmf --url http://invalid:8188 stats 2>&1 | grep -E "error|code"
# Expected: Exit code 5 (CONNECTION)
```

---

## Documentation Verification

Verify all repos have updated documentation:

- [ ] MCP Main Repo: CLAUDE.md updated if new features added
- [ ] KDH-Automation: CLAUDE.md references mmf.js correctly
- [ ] Goat: CLAUDE.md updated to reference mmf.js instead of ComfyUIClient
- [ ] RobloxChristian: CLAUDE.md updated to reference visual_gen_mcp.py

---

## Rollback Plan

If migration fails:

1. **Goat:** Restore from git backup
   ```bash
   cd ~/Applications/Goat
   git checkout -- src/clients/ComfyUIClient.js
   git checkout -- src/services/MCPService.js
   # Remove mmf.js
   rm src/core/mmf.js
   ```

2. **RobloxChristian:** Restore HTTP client
   ```bash
   cd ~/Applications/RobloxChristian
   git checkout -- src/animation_gen.py
   # Remove visual_gen_mcp.py
   rm src/visual_gen_mcp.py
   ```

3. **MCP Main Repo:** Already stable, no rollback needed

---

## Sign-Off Checklist

**Before Migration Starts:**
- [x] Baseline state documented
- [x] All current files listed
- [x] Test commands validated
- [x] This QA checklist created

**After Migration Completes:**
- [ ] All post-migration tests pass
- [ ] No regression in generation quality
- [ ] Performance within acceptable variance
- [ ] All repos updated
- [ ] Documentation updated
- [ ] Sign-off by: _____________ Date: _______

---

## Appendix: Quick Reference Commands

```bash
# Test everything in one go
mmf stats --pretty && \
mmf templates list | jq '.count' && \
mmf models list --type checkpoint | head -5 && \
echo "✅ All checks passed"

# Test generation pipeline
mmf run --model qwen --type t2i --prompt "verification test" --seed 42 -o /tmp/verify.png && \
mmf qa /tmp/verify.png --prompt "Is this a clear, high-quality image?" && \
echo "✅ Generation and QA passed"

# Test video pipeline
mmf pipeline i2v --image /tmp/verify.png --prompt "gentle motion" -o /tmp/verify.mp4 && \
echo "✅ Video pipeline passed"
```
