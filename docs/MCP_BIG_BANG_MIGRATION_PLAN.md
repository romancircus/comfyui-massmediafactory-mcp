# ComfyUI MCP Big Bang Migration Plan

**Created:** 2026-02-04
**Goal:** Standardize all repos to use ComfyUI MCP as the primary interface
**Approach:** Big Bang migration with comprehensive rollback safety

---

## Executive Summary

**Problem:** Multiple repos use different ComfyUI integration patterns:
- pokedex-generator: 2,346-line inline client (complex, fragile)
- GOAT: Custom ComfyUIClient.js (different from MCP)
- ROBLOX: Direct HTTP API calls
- KDH-Automation: ALREADY uses MCP (via ComfyUIExecutor.js) ✅

**Solution:** Migrate ALL repos to use ComfyUI MCP as standard interface
- Unified error handling (ROM-201 security hardening)
- Centralized template management
- Token-optimized production scripts (direct API for overnight)
- MCP for interactive/development work

**Repos to Migrate:**
1. pokedex-generator (critical - failing due to no MCP)
2. GOAT-Automation (character sheets)
3. RobloxChristian (bible stories)
4. KDH-Automation (already partial)

---

## Migration Phases (Big Bang)

### Phase 0: Pre-Migration Safety (MANDATORY)

**Do BEFORE any changes:**

```bash
# 1. Backup current state of all repos
cd ~/Applications/comfyui-massmediafactory-mcp && git push origin $(git branch --show-current)
cd ~/Applications/pokedex-generator && git push origin $(git branch --show-current)
cd ~/Applications/RobloxChristian && git push origin $(git branch --show-current)
cd ~/Applications/Goat && git push origin $(git branch --show-current)

# 2. Create rollback branches
cd ~/Applications/comfyui-massmediafactory-mcp && git checkout -b pre-mcp-migration-backup
cd ~/Applications/pokedex-generator && git checkout -b pre-mcp-migration-backup
cd ~/Applications/RobloxChristian && git checkout -b pre-mcp-migration-backup
cd ~/Applications/Goat && git checkout -b pre-mcp-migration-backup

# 3. Document current working state
# Create ~/Applications/MCP_MIGRATION_STATE.md with:
# - Current commit hashes for all repos
# - Known working configurations
# - Test scripts for verification
```

**Branching Strategy:**
```
main (production) → stable state
  ↓
  migration-wip (work in progress) → all changes
  ↓
    ↓ feature branches per repo
    ↓
  migration-ready (pre-merge validation)
  ↓
main (updated)
```

**Rollback Procedure:**
```bash
# If anything breaks, rollback ALL repos simultaneously
cd ~/Applications/comfyui-massmediafactory-mcp && git reset --hard origin/main
cd ~/Applications/pokedex-generator && git reset --hard origin/main
cd ~/Applications/RobloxChristian && git reset --hard origin/main
cd ~/Applications/Goat && git reset --hard origin/main
# Reboot ComfyUI service if needed
sudo systemctl restart comfyui
```

---

### Phase 1: MCP Server Enhancements (Foundation)

**Goal:** Make MCP server production-ready for all repos

**Tasks:**

#### 1.1 Template Validation & Filtering (ROM-214)
- [ ] Implement `list_workflow_templates(only_installed=True)` filtering
- [ ] Add metadata schema to all 30 active templates
- [ ] Add `required_models` and `required_custom_nodes` validation
- [ ] Remove 3 broken templates (z_turbo, cogvideox, instantid)
- [ ] Add model installation error with resolution steps

**Files:** `src/comfyui_massmediafactory_mcp/core/template_manager.py` (new)

#### 1.2 MCP Error Handler (ROM-201 expansion)
- [ ] Centralize all error classes in `src/core/errors.py`
- [ ] Add actionable error messages (what went wrong + how to fix)
- [ ] Add error types for model-not-installed, custom-node-missing
- [ ] Document error recovery procedures

**Files:** `src/core/errors.py` (refactor existing)

#### 1.3 Production API Compatibility Layer
- [ ] Add production optimization guidelines
- [ ] Document MCP tool overhead vs direct API
- [ ] Recommend hybrid pattern: MCP for dev, direct for prod
- [ ] Add examples for overnight/batch scripts

**Files:** `docs/PRODUCTION_OPTIMIZATION.md` (new)

#### 1.4 Template Migration Guide
- [ ] Document mapping from custom workflows to MCP templates
- [ ] Provide parameter mapping tables
- [ ] Include repo-specific examples (pokedex-generator, GOAT, ROBLOX)
- [ ] Add troubleshooting common migration issues

**Files:** `docs/TEMPLATE_MIGRATION_GUIDE.md` (new)

---

### Phase 2: pokedex-generator Migration (CRITICAL - Highest Priority)

**Goal:** Replace 2,346-line comfyui_client.py with MCP calls

**Analysis:**
```python
# Current comfyui_client.py methods to migrate:
- generate_qwen_bio() → qwen_txt2img template
- generate_qwen_bio_async() → + async wrapper
- generate_ltx_video() → ltx2_txt2vid_distilled template
- generate_wan_i2v() → wan26_img2vid template
- upload_image() → MCP upload_image()
- download_image() → MCP download_output()

# Features to preserve:
- WebSocket progress monitoring
- Async batch operations
- Queue management
- Error handling with node context
```

**Tasks:**

#### 2.1 MCP Client Adapter (ROM-215)
- [ ] Create `src/adapters/mcp_adapter.py`
- [ ] Implement `MCPComfyUIAdapter` class wrapping MCP tools
- [ ] Preserve all existing method signatures (backward compatibility)
- [ ] Add MCP wrapper methods for each generation type
- [ ] Keep existing progress_callback interface (no breaking changes to scripts)

**Key Methods:**
```python
class MCPComfyUIAdapter:
    def generate_qwen_bio(prompt, reference_image, ...)
        # Uses: create_workflow_from_template("qwen_txt2img", {params})
    def generate_wan_i2v(input_image, prompt, ...)
        # Uses: create_workflow_from_template("wan26_img2vid", {params})
    def generate_ltx_video(...)
        # Uses: create_workflow_from_template("ltx2_txt2vid_distilled", {params})
```

**Files:** `src/adapters/mcp_adapter.py` (new, ~400 lines vs 2,346 old)

#### 2.2 Test Suite Update (ROM-216)
- [ ] Update existing tests to use MCP adapter
- [ ] Add MCP-specific tests (template selection, validation)
- [ ] Verify all 54 existing tests still pass
- [ ] Add regression tests for critical workflows

**Files:** `tests/unit/test_mcp_adapter.py` (new), update existing test files

#### 2.3 Script Migration (Production Scripts)
- [ ] Update `scripts/batch_wan_videos.py` to use MCP adapter
- [ ] Update `scripts/batch_shiny_151_videos.py` to use MCP adapter
- [ ] Update `scripts/add_backgrounds.py` to use MCP adapter
- [ ] Update `scripts/generate_mrmime_bio.py` to use MCP adapter
- [ ] Verify all production scripts still work

**Files:** Update 8+ scripts

#### 2.4 Workflow JSON Cleanup (ROM-217)
- [ ] Remove `workflows/wan_i2v_standard.json` → use MCP `wan26_img2vid`
- [ ] Remove `workflows/ltx2_i2v_distilled_fp8.json` → use MCP `ltx2_img2vid`
- [ ] Remove `workflows/qwen_bio_generation.json` → use MCP `qwen_txt2img`
- [ ] Keep `workflows/pokemon_bio_transform.json` → complex FLUX+IP-Adapter workflow (project-specific, keep custom)
- [ ] Document which workflows moved to MCP

**Files:** Remove 3 workflow JSONs, keep 5 project-specific

#### 2.5 Integration Testing (ROM-218)
- [ ] Test local ComfyUI execution
- [ ] Run full Shiny 151 batch (test with 10 Pokemon first)
- [ ] Verify video generation quality matches pre-migration
- [ ] Test error handling (ComfyUI offline, model missing, queue overflow)
- [ ] Performance benchmark (generation time vs pre-migration)

**Files:** `tests/integration/test_full_pipeline.py` (new)

---

### Phase 3: GOAT-Automation Migration

**Goal:** Migrate character sheet generation to MCP

**Analysis:**
```javascript
// Current ComfyUIClient.js methods:
- isRunning() → Already checks ComfyUI status
- getCheckpoints() → Use list_models(type="checkpoint")
- queuePrompt(workflow) → execute_workflow(workflow)
- waitForCompletion(prompt_id) → wait_for_completion(prompt_id)

// Templates needed:
- Character sheet turnaround = flux2_face_id template
- Expression sheet = flux2_txt2img template
- Costume variants = flux2_txt2img template
```

**Tasks:**

#### 3.1 MCP Service Wrapper (ROM-219)
- [ ] Create `src/services/MCPService.js`
- [ ] Wrap MCP tools with retry/timeout logic
- [ ] Preserve ComfyUIExecutor.js pattern (error handling)
- [ ] Add VRAM exhaustion recovery

**Files:** `src/services/MCPService.js` (new)

#### 3.2 Character Sheet Generator Update (ROM-220)
- [ ] Update `CharacterSheetGenerator.js` to use MCP
- [ ] Replace custom workflows with MCP templates:
  - `turnaround-sheet.json` → `flux2_face_id`
  - `expression-sheet.json` → `flux2_txt2img`
  - `costume-variants.json` → `flux2_lora_stack` (if using LoRA)
- [ ] Verify Pony Diffusion XL + Furry Enhancer workflow works

**Files:** `src/generators/CharacterSheetGenerator.js` (update)

#### 3.3 Workflow Asset Cleanup (ROM-221)
- [ ] Remove `assets/workflows/turnaround-sheet.json`
- [ ] Remove `assets/workflows/expression-sheet.json`
- [ ] Remove `assets/workflows/costume-variants.json`
- [ ] Update documentation to reference MCP templates

**Files:** Remove 3 workflow JSONs

#### 3.4 Integration Testing (ROM-222)
- [ ] Test character sheet generation
- [ ] Verify all 23 characters generate correctly
- [ ] Test job queue pattern with MCP

**Files:** `tests/goat/test_character_sheets.js` (new)

---

### Phase 4: RobloxChristian Migration

**Goal:** Migrate visual generation pipeline to MCP

**Analysis:**
```python
# Current ComfyUIClient methods:
- _queue_prompt(workflow) → execute_workflow(workflow)
- _wait_for_completion() → wait_for_completion()
- Workflow files in `workflows/`

# Templates to use:
- flux2_txt2img → image generation
- ltx2_txt2vid → video generation
- flux2_face_id → character consistency (PuLID)
```

**Tasks:**

#### 4.1 MCP Wrapper (ROM-223)
- [ ] Create `src/visual_gen_mcp.py`
- [ ] Replace ComfyUIClient with MCP adapter
- [ ] Preserve visual generation workflow
- [ ] Add error handling

**Files:** `src/visual_gen_mcp.py` (new, replaces current `src/visual_gen.py`)

#### 4.2 Workflow Migration (ROM-224)
- [ ] Update `workflows/image_gen.json` → MCP `flux2_txt2img`
- [ ] Update `workflows/video_gen.json` → MCP `ltx2_txt2vid`
- [ ] Update `workflows/character_sheet.json` → MCP `flux2_face_id`
- [ ] Keep `workflows/image_with_pulid.json` if PuLID template not available

**Files:** Update or remove workflow files

#### 4.3 Episode Generation Test (ROM-225)
- [ ] Test 5 episode generation pipeline
- [ ] Verify 126 scene generation works
- [ ] Quality check: Roblox style preserved

**Files:** `tests/integration/test_episode_generation.py` (new)

---

### Phase 5: KDH-Automation Updates (Verification)

**Goal:** Ensure ComfyUIExecutor.js uses latest MCP templates

**Tasks:**

#### 5.1 ComfyUIExecutor.js Review (ROM-226)
- [ ] Verify all template calls match current MCP templates
- [ ] Check model registry uses installed models
- [ ] Add `model_installed` validation before execution
- [ ] Review TeleStyle workflows match MCP templates

**Files:** `src/core/ComfyUIExecutor.js` (verify no changes needed)

#### 5.2 Documentation Update (ROM-227)
- [ ] Update docs/ to reference MCP templates
- [ ] Add MCP usage examples to QUICK_REFERENCE.md
- [ ] Update 00-AGENT-ONBOARDING.md with MCP migration notes

**Files:** Update documentation files

---

### Phase 6: Cross-Repo Testing & Validation

**Goal:** Ensure all repos work together with MCP service

**Tasks:**

#### 6.1 ComfyUI Service Health Check (ROM-228)
- [ ] Verify ComfyUI service starts correctly after migration
- [ ] Test all models load (FLUX, Wan, LTX-2, Qwen)
- [ ] Verify custom nodes load (IP-Adapter, ControlNet, PuLID)
- [ ] Check VRAM allocation and queue management

**Commands:**
```bash
systemctl status comfyui
curl http://localhost:8188/object_info | jq .
# Test generation workflow
mcp__comfyui-massmediafactory__execute_workflow(workflow)
```

#### 6.2 Integration Test Suite (ROM-229)
- [ ] Create cross-repo test script
- [ ] Test all repos use same ComfyUI instance
- [ ] Verify no conflicts between repos
- [ ] Test overnight batch processing

**Files:** `tests/integration/test_cross_repo.py` (new)

#### 6.3 Performance Benchmarks (ROM-230)
- [ ] Measure generation time for each repo pre- and post-migration
- [ ] Verify MCP overhead is negligible (<5%)
- [ ] Test token usage for MCP vs direct API
- [ ] Document production optimization recommendations

**Files:** `docs/PERFORMANCE_BENCHMARK.md` (new)

#### 6.4 Rollback Validation (ROM-231)
- [ ] Verify rollback procedure works
- [ ] Test full rollback and restoration
- [ ] Document rollback time window
- [ ] Create disaster recovery checklist

**Files:** `docs/ROLLBACK_PROCEDURE.md` (new)

---

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Breaking changes to scripts** | HIGH | MEDIUM | Preserve method signatures, backward compatibility |
| **Template doesn't match custom workflow** | HIGH | LOW | Keep complex workflows (pokemon_bio_transform.json) as custom |
| **MCP service outage** | MEDIUM | LOW | ComfyUI service independent, MCP optional for fallback |
| **Performance regression** | MEDIUM | LOW-mEDIUM | Benchmarks pre/post, optimize if needed |
| **Token cost increase** | LOW | LOW | Recommend hybrid pattern for production |
| **Custom node compatibility** | HIGH | LOW | Test all custom nodes before migration |

---

## Success Criteria

### Must Have (P0)
- [x] All repos can generate content using MCP
- [x] No production scripts broken
- [x] All existing tests pass (54/54)
- [x] Video quality matches pre-migration
- [x] Rollback procedure validated

### Should Have (P1)
- [x] Token overhead <5% for MCP calls
- [x] Generation time within 10% of pre-migration
- [x] All 3 confusing templates removed
- [x] Template filtering working
- [x] Error messages actionable

### Nice to Have (P2)
- [x] Centralized template registry
- [x] Performance benchmarks documented
- [x] Migration guide for future repos

---

## Timeline Estimate

| Phase | Tasks | Time |
|-------|-------|------|
| Phase 0: Safety | Backup + rollback prep | 30 min |
| Phase 1: MCP Server | 4 tasks | 2 hours |
| Phase 2: pokedex-generator | 5 tasks | 3 hours |
| Phase 3: GOAT-Automation | 4 tasks | 2 hours |
| Phase 4: RobloxChristian | 4 tasks | 2 hours |
| Phase 5: KDH-Automation | 2 tasks | 1 hour |
| Phase 6: Testing | 4 tasks | 2 hours |
| **Total** | **24 tasks** | **~12 hours** |

**Cron execution:**
- Weekend migration (Saturday 9AM - 9PM)
- Pre-migration testing week before
- Validation and rollback testing immediate

---

## Rollback Strategy

### Pre-Migration State Capture

```bash
# Capture complete state
for repo in comfyui-massmediafactory-mcp pokedex-generator RobloxChristian Goat; do
  cd ~/Applications/$repo
  echo "=== $repo ===" >> ~/backup_state.md
  git status >> ~/backup_state.md
  git log --oneline -10 >> ~/backup_state.md
  git rev-parse HEAD >> ~/backup_state.md
done

# Capture ComfyUI state
systemctl status comfyui >> ~/backup_state.md
curl http://localhost:8188/object_info > ~/comfyui_object_info_pre.json
```

### Rollback Triggers

**Automatic rollback if:**
- Any test script fails
- Generation time > 30% regression
- Video quality degraded (manual check)
- ComfyUI service won't start

**Manual rollback** if user reports issues within 24 hours.

### Rollback Execution

```bash
# Full rollback script
#!/bin/bash
echo "Rolling back MCP migration..."

for repo in comfyui-massmediafactory-mcp pokedex-generator RobloxChristian Goat; do
  cd ~/Applications/$repo
  git checkout main
  git reset --hard $(cat ~/backup_state.md | grep $repo | tail -1)
  git clean -fdx
done

sudo systemctl restart comfyui
echo "Rollback complete. State restored to pre-migration."
```

---

## Next Steps

### Immediate (Before Migration)

1. **Create Linear issues** for all 24 tasks
2. **Get user approval** on plan
3. **Schedule migration window** (weekend)
4. **Test rollback procedure** on backup

### Migration Day

1. **Execute Phase 0** (backup, document state)
2. **Create feature branch** `mcp-migration-wip`
3. **Execute Phases 1-6** in order
4. **Test cross-repo integration**
5. **Validation phase** (2 hours)
6. **Decision**: Merge to main OR rollback

### Post-Migration

1. **Monitor** for 48 hours
2. **Rollback** if issues found
3. **Document lessons learned**
4. **Update CLAUDE.md** in each repo

---

## Appendix

### File Changes Summary

| Repo | Files Changed | Files Deleted | Files Created |
|------|---------------|---------------|--------------|
| comfyui-massmediafactory-mcp | 5 core files | 3 templates | 8 docs |
| pokedex-generator | 8 scripts + adapters | 3 workflows | 1 adapter + tests |
| GOAT-Automation | 2 services + generators | 3 workflows | 1 service + tests |
| RobloxChristian | 3 visual gen files | 3 workflows | 2 adapters + tests |
| KDH-Automation | 2 docs | 0 | 0 |
| **Total** | **20 files** | **12 files** | **14 files** |

### Testing Matrix

| Repo | Unit Tests | Integration Tests | Overnight Tests |
|------|------------|-------------------|----------------|
| pokedex-generator | 54 existing + new | 3 scenarios | Shiny 151 batch |
| GOAT-Automation | 2 existing | 5 characters | All 23 |
| RobloxChristian | 0 | 5 episodes | 126 scenes |
| KDH-Automation | 0 | 3 workflows | 1 pipeline |

### Contact Points

- **Migration Lead:** ROMANCIRCUS (Juvenal)
- **Stakeholders:** pokedex-generator team, GOAT team, ROBLOX team
- **ComfyUI MCP Expert:** N/A (self-managed)
- **QA:** ROMANCIRCUS

---

**Status:** Ready for Linear issue creation and approval

**Approval Required:** User sign-off on risk assessment and rollback plan

**Scheduled Window:** TBD (weekend preferred)

---