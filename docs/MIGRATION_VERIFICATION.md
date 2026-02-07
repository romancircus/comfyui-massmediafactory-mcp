# Migration Verification Guide

> **Version:** 1.0
> **Date:** 2026-02-07
> **Scope:** How to verify each migration is complete and correct

---

## Overview

This document provides step-by-step verification procedures for each repository migration. Use this to confirm migrations are successful before marking issues as complete.

---

## MCP Main Repo (comfyui-massmediafactory-mcp)

**Status:** ✅ Already Migrated (ROM-548 completed)

### Verification Steps

#### 1. CLI Installation
```bash
which mmf
mmf --version
```
**Expected:** Path shows installed location, version 0.1.0+

#### 2. Core Commands
```bash
# System info
mmf stats --pretty

# List templates
mmf templates list | jq '.templates | length'
# Expected: 38+ templates

# List models
mmf models list --type checkpoint | head -5
```

#### 3. End-to-End Generation
```bash
# Quick image generation test
mmf run --model qwen --type t2i \
  --prompt "QA verification test image" \
  --seed 42 \
  -o /tmp/migration_verify.png

# Verify output exists
ls -lh /tmp/migration_verify.png
file /tmp/migration_verify.png
# Expected: PNG image, ~1-2MB, 1024x1024 or similar

# Clean up
rm /tmp/migration_verify.png
```

#### 4. MCP Server Status
```bash
# Count active MCP tools
grep -c "@mcp.tool()" src/comfyui_massmediafactory_mcp/server.py
# Expected: 59

# Verify no critical errors in server startup
python3 -c "from comfyui_massmediafactory_mcp.server import mcp; print('MCP server loads:', type(mcp))"
```

#### 5. Template Validation
```bash
# Test specific templates
mmf run --template qwen_txt2img \
  --params '{"PROMPT":"test","SEED":42}' \
  -o /tmp/template_test.png

mmf run --template flux2_txt2img \
  --params '{"PROMPT":"test","SEED":42}' \
  -o /tmp/flux_test.png

# Clean up
rm /tmp/*_test.png
```

**Migration Complete When:**
- [ ] CLI version 0.1.0+ confirmed
- [ ] 38+ templates listed
- [ ] Image generation completes in <60s
- [ ] MCP server starts without errors
- [ ] 59 MCP tools registered

---

## KDH-Automation

**Status:** ✅ Already Migrated (Reference Implementation)

### Verification Steps

#### 1. File Structure
```bash
cd ~/Applications/KDH-Automation

# Verify mmf.js exists
ls -la src/core/mmf.js
# Expected: File exists, ~200-400 lines

# Verify old files removed
ls src/core/ComfyUIExecutor.js 2>&1 | grep "No such file"
ls src/core/ComfyUIClient.js 2>&1 | grep "No such file"
ls src/core/ComfyUIHttpAdapter.js 2>&1 | grep "No such file"
ls src/core/ComfyUIWorkflow.js 2>&1 | grep "No such file"
ls src/core/SOTAWorkflowTemplates.js 2>&1 | grep "No such file"
# Expected: All "No such file" (files removed)
```

#### 2. Module Exports
```bash
node -e "
const mmf = require('./src/core/mmf.js');
const exports = Object.keys(mmf);
console.log('Exports:', exports.join(', '));
console.log('Total:', exports.length);
"
```
**Expected Exports:**
- `qwenTxt2Img`
- `fluxTxt2Img`
- `faceIdTxt2Img`
- `wanI2V`
- `wanS2V`
- `ltxT2V`
- `ltxI2V`
- `qwenEditBackground`
- `telestyleImage`
- `telestyleVideo`
- `chatterboxTTS`
- `f5TTS`
- `f5TTSClone`
- `qwen3TTSCustomVoice`
- `qwen3TTSVoiceClone`
- `qwen3TTSVoiceDesign`
- `hunyuanTxt2Vid`
- `hunyuanImg2Vid`
- `wanAnimate`
- `stats`
- `templates`

#### 3. Function Execution
```bash
# Test stats function
node -e "
const {stats} = require('./src/core/mmf.js');
const result = stats();
console.log('Stats success:', result.system ? 'YES' : 'NO');
console.log('GPU:', result.devices?.[0]?.name);
"
# Expected: YES, "NVIDIA GeForce RTX 5090"

# Test templates function
node -e "
const {templates} = require('./src/core/mmf.js');
const result = templates();
console.log('Templates count:', result.count);
"
# Expected: 38+
```

#### 4. Integration Test
```bash
# Test generation (if safe to run)
# node -e "
# const {qwenTxt2Img} = require('./src/core/mmf.js');
# const result = qwenTxt2Img({
#   prompt: 'KDH migration verification',
#   seed: 42,
#   output: '/tmp/kdh_verify.png'
# });
# console.log('Success:', result.asset_id ? 'YES' : 'NO');
# console.log('Error:', result.error || 'None');
# "
```

**Migration Complete When:**
- [ ] mmf.js exists and loads
- [ ] All old files removed (5 files)
- [ ] All expected functions export
- [ ] stats() returns GPU info
- [ ] templates() returns 38+ templates
- [ ] Codebase reduced from ~4,000 lines to ~400 lines

---

## Goat (Target for Migration)

**Status:** ⏳ Pending Migration

### Pre-Migration Baseline

```bash
cd ~/Applications/Goat

# Document current state
git status
ls -la src/clients/ComfyUIClient.js
ls -la src/services/MCPService.js
ls -la src/services/PonyMCPService.js

# Count lines in files to be replaced
wc -l src/clients/ComfyUIClient.js
wc -l src/services/MCPService.js
wc -l src/services/PonyMCPService.js
# Total baseline: ~400-500 lines
```

### Post-Migration Verification

#### 1. New Structure
```bash
cd ~/Applications/Goat

# Verify new mmf.js exists
ls -la src/core/mmf.js
wc -l src/core/mmf.js
# Expected: ~200-300 lines

# Verify old files removed
ls src/clients/ComfyUIClient.js 2>&1 | grep "No such file"
ls src/services/MCPService.js 2>&1 | grep "No such file"
ls src/services/PonyMCPService.js 2>&1 | grep "No such file"
```

#### 2. Generator Updates
```bash
# Check generators use mmf.js
grep -l "from.*mmf\|require.*mmf" src/generators/*.js
# Expected: CharacterSheetGenerator.js, SpriteSheetGenerator.js, ReactionGenerator.js

# Verify no old imports remain
grep -r "ComfyUIClient" src/generators/ || echo "No ComfyUIClient references"
grep -r "MCPService" src/generators/ || echo "No MCPService references"
```

#### 3. Pony Template Access
```bash
# Verify Pony templates accessible via mmf CLI
node -e "
const mmf = require('./src/core/mmf.js');
const result = mmf.templates();
const ponyTemplates = result.templates.filter(t => t.tags?.includes('pony') || t.name?.includes('pony'));
console.log('Pony templates found:', ponyTemplates.length);
console.log('Names:', ponyTemplates.map(t => t.name).join(', '));
"
```

#### 4. Function Parity

Verify these functions exist in new mmf.js:

| Function | Purpose | Test Command |
|----------|---------|--------------|
| `isRunning()` | Check ComfyUI status | `mmf.stats()` |
| `getSystemStats()` | Get GPU info | `mmf.stats()` |
| `getCheckpoints()` | List models | `mmf.models list --type checkpoint` |
| `getLoRAs()` | List LoRAs | `mmf.models list --type lora` |
| `queuePrompt()` | Execute workflow | `mmf.execute()` |
| `waitForCompletion()` | Wait for job | `mmf.wait()` |
| `getImage()` | Download output | `mmf.download()` |
| `loadTemplate()` | Load Pony template | `mmf.templates list` + filter |

#### 5. Integration Test
```bash
# Test via Node.js
node -e "
const mmf = require('./src/core/mmf.js');

// Test connection
const stats = mmf.stats();
console.log('Connection:', stats.system ? 'OK' : 'FAIL');

// Test template loading
const templates = mmf.templates();
console.log('Templates:', templates.count > 0 ? 'OK' : 'FAIL');

// Test generation (optional)
// const result = mmf.qwenTxt2Img({ prompt: 'Goat test', seed: 42 });
// console.log('Generation:', result.asset_id ? 'OK' : 'FAIL');
"
```

**Migration Complete When:**
- [ ] src/core/mmf.js created (~200-300 lines)
- [ ] ComfyUIClient.js removed
- [ ] MCPService.js removed
- [ ] PonyMCPService.js removed or refactored
- [ ] All generators updated to use mmf.js
- [ ] Pony templates accessible via mmf CLI
- [ ] All integration tests pass
- [ ] Line count reduced from ~500 to ~300 lines

---

## RobloxChristian (Target for Migration)

**Status:** ⏳ Pending Migration

### Pre-Migration Baseline

```bash
cd ~/Applications/RobloxChristian

# Document current state
ls -la src/animation_gen.py
ls -la mcp_templates/
ls mcp_templates/*.json
# Expected: 7 templates

# Check current HTTP client
grep -n "class ComfyUIHYMotionClient" src/animation_gen.py
wc -l src/animation_gen.py
# Baseline: ~500+ lines with embedded client
```

### Post-Migration Verification

#### 1. New Structure
```bash
cd ~/Applications/RobloxChristian

# Verify visual_gen_mcp.py exists
ls -la src/visual_gen_mcp.py
wc -l src/visual_gen_mcp.py
# Expected: ~200-400 lines

# Check if mcp_templates can be removed
ls mcp_templates/ 2>&1
# If using MCP server templates, directory may be empty or removed
```

#### 2. Python Import Test
```bash
# Test Python import
python3 -c "
from src.visual_gen_mcp import VisualGeneratorMCP
print('✓ Import successful')
"

# Test class instantiation
python3 -c "
from src.visual_gen_mcp import VisualGeneratorMCP
vg = VisualGeneratorMCP()
print('✓ Instantiation successful')
print('Instance:', type(vg))
"

# Test basic methods
python3 -c "
from src.visual_gen_mcp import VisualGeneratorMCP
vg = VisualGeneratorMCP()
templates = vg.list_available_templates()
print('✓ Templates:', len(templates))
"
```

#### 3. Method Parity

Verify these methods exist in VisualGeneratorMCP:

| Old Method (ComfyUIHYMotionClient) | New Method (VisualGeneratorMCP) | Purpose |
|-----------------------------------|----------------------------------|---------|
| `check_connection()` | `get_stats()` | Check ComfyUI status |
| `get_available_nodes()` | `list_available_templates()` | List available templates |
| `queue_prompt()` | `generate_from_template()` | Execute workflow |
| `wait_for_completion()` | Internal in generate_from_template() | Wait for job |
| `download_output()` | `download_asset()` | Download result |

#### 4. Template Generation Test
```bash
# Test template-based generation
python3 << 'EOF'
from src.visual_gen_mcp import VisualGeneratorMCP
import tempfile
import os

vg = VisualGeneratorMCP()

# Test with a simple template
result = vg.generate_from_template(
    template_name='qwen_txt2img',
    params={
        'PROMPT': 'Roblox character test',
        'SEED': 42,
        'WIDTH': 1024,
        'HEIGHT': 1024
    }
)

if result.get('asset_id'):
    print('✓ Generation successful')
    print('  Asset ID:', result['asset_id'])

    # Test download
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        download_result = vg.download_asset(result['asset_id'], f.name)
        if os.path.exists(f.name):
            print('✓ Download successful')
            os.unlink(f.name)
        else:
            print('✗ Download failed')
else:
    print('✗ Generation failed:', result.get('error', 'Unknown error'))
EOF
```

#### 5. animation_gen.py Cleanup
```bash
# Verify ComfyUIHYMotionClient removed or deprecated
grep -n "class ComfyUIHYMotionClient" src/animation_gen.py || echo "✓ Class removed"
grep -n "ComfyUIHYMotionClient" src/animation_gen.py || echo "✓ No references"

# Check if animation_gen.py uses visual_gen_mcp
grep -n "from.*visual_gen_mcp\|import.*VisualGeneratorMCP" src/animation_gen.py
```

**Migration Complete When:**
- [ ] src/visual_gen_mcp.py created (~200-400 lines)
- [ ] VisualGeneratorMCP class importable
- [ ] All required methods implemented
- [ ] ComfyUIHYMotionClient removed from animation_gen.py
- [ ] Template generation works end-to-end
- [ ] mcp_templates/ directory may be removed (using MCP server)
- [ ] Python tests pass (if any)

---

## Cross-Repo Validation

After all migrations complete, verify consistency:

```bash
# 1. Same mmf CLI version everywhere
mmf --version

# 2. Same system stats from all repos
cd ~/Applications/KDH-Automation
node -e "const {stats} = require('./src/core/mmf.js'); console.log('KDH:', stats().devices[0].vram_total_gb);"

cd ~/Applications/Goat
node -e "const mmf = require('./src/core/mmf.js'); console.log('Goat:', mmf.stats().devices[0].vram_total_gb);"

cd ~/Applications/RobloxChristian
python3 -c "from src.visual_gen_mcp import VisualGeneratorMCP; print('Roblox:', VisualGeneratorMCP().get_stats()['devices'][0]['vram_total_gb'])"

# All should show: 31.34 (RTX 5090 32GB)
```

---

## Performance Verification

### Timing Tests

```bash
# Before migration (document baseline)
time (cd ~/Applications/KDH-Automation && node -e "require('./src/core/mmf.js').stats()")

# After migration (should be similar)
time mmf stats
```

**Acceptable Variance:**
- ±20% for simple operations (stats, templates)
- ±30% for generation (network/queue dependent)

---

## Error Handling Verification

### Exit Codes

Test that error codes are consistent across implementations:

```bash
# Test invalid model
mmf run --model invalid --type t2i --prompt "test" 2>&1
echo "Exit code: $?"
# Expected: 3 (VALIDATION)

# Test invalid template
mmf run --template nonexistent --params '{}' 2>&1
echo "Exit code: $?"
# Expected: 6 (NOT_FOUND)

# Test connection failure
mmf --url http://invalid:8188 stats 2>&1
echo "Exit code: $?"
# Expected: 5 (CONNECTION)
```

### Error Propagation

Verify errors propagate correctly through wrappers:

```bash
# KDH
cd ~/Applications/KDH-Automation
node -e "
const {qwenTxt2Img} = require('./src/core/mmf.js');
const result = qwenTxt2Img({ prompt: 'test', seed: 42 });
// With invalid params, should return {error: '...'}
console.log('Has error field:', 'error' in result);
"

# Goat (post-migration)
cd ~/Applications/Goat
node -e "
const mmf = require('./src/core/mmf.js');
const result = mmf.stats({invalid: true});
console.log('Error handling:', result.error ? 'OK' : 'Check');
"

# RobloxChristian (post-migration)
cd ~/Applications/RobloxChristian
python3 -c "
from src.visual_gen_mcp import VisualGeneratorMCP
vg = VisualGeneratorMCP()
result = vg.get_stats()
print('Error handling:', 'error' in result or 'OK')
"
```

---

## Final Sign-Off

**Per-Repo Checklist:**

### MCP Main
- [ ] 59 MCP tools active
- [ ] 38+ templates available
- [ ] All CLI commands work
- [ ] Generation completes successfully

### KDH-Automation
- [ ] mmf.js exists
- [ ] Old files removed
- [ ] All functions export
- [ ] Integration tests pass

### Goat
- [ ] mmf.js created
- [ ] Old clients removed
- [ ] Generators updated
- [ ] Pony templates work
- [ ] Integration tests pass

### RobloxChristian
- [ ] visual_gen_mcp.py created
- [ ] VisualGeneratorMCP works
- [ ] Old client removed
- [ ] Template generation works
- [ ] Python tests pass

**Overall Migration:**
- [ ] All repos using consistent mmf CLI
- [ ] No HTTP client duplication
- [ ] Line count reduced in all repos
- [ ] Performance acceptable
- [ ] Error handling consistent
- [ ] Documentation updated

**Sign-off:**
- Verified by: _____________
- Date: _______
- Notes: _______________________________

---

## Troubleshooting

### Common Issues

#### Issue: mmf CLI not found
```bash
# Check installation
pip show comfyui-massmediafactory-mcp
which mmf

# Reinstall if needed
pip install -e ~/Applications/comfyui-massmediafactory-mcp
```

#### Issue: Import errors (Python)
```bash
# Check Python path
cd ~/Applications/RobloxChristian
python3 -c "import sys; print('\n'.join(sys.path))"

# Add to path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Issue: Node.js require errors
```bash
# Check Node version
node --version
# Required: 18+

# Check module type
cat package.json | grep "type"
# Should be "module" for ES6 imports
```

#### Issue: Template not found
```bash
# List all templates
mmf templates list | jq '.templates[].name'

# Check template name spelling
mmf run --template <name> --params '{}'
```

#### Issue: Generation timeout
```bash
# Increase timeout
mmf run --model wan --type i2v ... --timeout 900

# Or set environment variable
export MMF_TIMEOUT=900
mmf run ...
```

---

## Quick Reference: Migration Commands

```bash
# Full verification in one command
./scripts/verify_migration.sh

# Or manual step-by-step
echo "=== MCP Main ==="
mmf stats --pretty && mmf templates list | jq '.count'

echo "=== KDH ==="
cd ~/Applications/KDH-Automation && node -e "console.log(require('./src/core/mmf.js').stats().system.os)"

echo "=== Goat ==="
cd ~/Applications/Goat && ls src/core/mmf.js && node -e "console.log(require('./src/core/mmf.js').stats().system.os)"

echo "=== RobloxChristian ==="
cd ~/Applications/RobloxChristian && python3 -c "from src.visual_gen_mcp import VisualGeneratorMCP; print(VisualGeneratorMCP().get_stats()['system']['os'])"
```
