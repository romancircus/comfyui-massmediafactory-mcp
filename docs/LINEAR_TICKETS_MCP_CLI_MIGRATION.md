# Linear Issues for MCP→CLI Migration

## ROM-20: Document CLI Patterns as Reference Standard
**Labels:** `repo:comfyui`, `docs`, `priority:medium`

### Description:
**TASK:** Create comprehensive CLI patterns documentation based on KDH-Automation's proven mmf.js implementation

Document the production-hardened patterns from KDH-Automation that should be the reference standard for all repos:

**Requirements:**
* Extract patterns from `KDH-Automation/src/core/mmf.js` (688 lines)
* Document timeout hierarchy (11min images, 15min videos, 30min pipelines)
* Document error handling (return error objects, not throwing)
* Document retry patterns (`--retry 3 --retry-on vram,timeout,connection`)
* Document critical parameters (Qwen shift=7.0, TeleStyle cfg=2.0-2.5, LTX 8n+1 frames)
* Document template parameter building patterns
* Document shell escaping strategies
* Document atomic file write patterns
* Create migration guide for repos moving from HTTP to CLI

**Success Criteria:**
* New `docs/CLI_PATTERNS.md` file with comprehensive patterns
* Updated CLAUDE.md with CLI-first guidance
* Code examples for each pattern
* Template reference guide

**Time:** 2 hours

---

## ROM-21: Goat HTTP→CLI Migration (Keep Job Queue)
**Labels:** `repo:goat`, `migration`, `priority:critical`

### Description:
**TASK:** Migrate Goat's HTTP client to CLI while preserving job queue and business logic

Goat currently uses direct HTTP to ComfyUI but documentation says to use CLI. Migrate execution layer while keeping:
- Pony Diffusion template system
- VRAM-aware job queue
- Character-aware prompt building
- Persistent job storage

**Requirements:**
* Replace `ComfyUIClient.js` HTTP calls with CLI subprocess calls
* Keep `PonyMCPService.js` template injection (adapt to CLI `--params`)
* Preserve `JobQueue.js` + `JobScheduler.js` disk-based queue
* Update `CharacterSheetGenerator.js` to pass final prompts to CLI
* Add retry wrapper around CLI calls (match existing retry logic)
* Test character turnaround sheet generation
* Test expression sheet generation
* Test costume variant generation

**CLI Equivalents:**
| HTTP Call | CLI Command |
|-----------|-------------|
| `GET /system_stats` | `mmf stats` |
| `GET /object_info/*` | `mmf models list` |
| `POST /prompt` | `mmf run --template pony/turnaround_sheet --params '{...}'` |
| `GET /view` | `mmf download <asset_id>` |
| `POST /upload/image` | `mmf upload <path>` |

**Success Criteria:**
* `npm run sheets -- --all` works with CLI backend
* Job queue persistence maintained
* VRAM monitoring still functional
* Output directory structure preserved
* No regression in character sheet quality

**Time:** 3 hours

**Note:** Register pony templates in CLI manifest or use absolute paths

---

## ROM-22: RobloxChristian HTTP→CLI Migration
**Labels:** `repo:roblox`, `migration`, `priority:high`

### Description:
**TASK:** Migrate RobloxChristian from HTTP client to CLI, delete broken MCP client

RobloxChristian has three competing implementations:
1. `visual_gen.py` - HTTP client (currently used by orchestrator)
2. `visual_gen_mcp.py` - MCP client (broken, unused)
3. Need: CLI implementation

Migrate to CLI and delete the unused MCP client.

**Requirements:**
* Create `visual_gen_cli.py` wrapper (pattern from KDH's mmf.js)
* Replace orchestrator import: `visual_gen` → `visual_gen_cli`
* Delete `visual_gen_mcp.py` (never used, broken)
* Update `batch_produce.py` to use CLI
* Add `--retry 3` to all generation calls
* Use SOTA templates: `qwen_txt2img` (images), `wan21_img2vid` (video)
* Test single image generation
* Test batch video generation
* Update CLAUDE.md documentation

**Migration Steps:**
1. Create `src/visual_gen_cli.py` with:
   - `generate_image(prompt, output, model="qwen")`
   - `generate_video(image_path, prompt, output)`
   - `generate_character_sheet(character_data, output)`
2. Update `src/orchestrator.py` imports
3. Update `src/batch_produce.py` generation calls
4. Delete `src/visual_gen_mcp.py`
5. Test: `python scripts/batch_produce.py --start 0 --count 1`

**Success Criteria:**
* Bible story generation works end-to-end with CLI
* All 7 mcp_templates/ synced to mmf templates
* No references to HTTP or MCP clients in orchestrator
* Batch production produces videos
* Documentation updated

**Time:** 2 hours

---

## ROM-23: Create Shared mmf.js Wrapper Module
**Labels:** `repo:comfyui`, `infra`, `priority:low`

### Description:
**TASK:** Extract KDH's mmf.js into reusable npm package for all repos

KDH-Automation has a production-hardened 688-line wrapper. Extract it for use across Goat, RobloxChristian, and future repos.

**Requirements:**
* Extract `src/core/mmf.js` from KDH-Automation
* Create new package: `@romancircus/mmf-client`
* Export functions: `qwenTxt2Img()`, `wanI2V()`, `fluxTxt2Img()`, etc.
* Include TypeScript definitions
* Add proper error handling and retry logic
* Document all 22 functions
* Add unit tests
* Publish to private registry or git-based install

**Package Structure:**
```
packages/mmf-client/
├── src/
│   ├── index.js          # Main exports
│   ├── templates.js      # Template constants
│   ├── timeouts.js       # Timeout constants
│   └── errors.js         # Error handling
├── tests/
│   └── mmf.test.js
├── package.json
└── README.md
```

**Success Criteria:**
* All repos can `npm install @romancircus/mmf-client`
* KDH-Automation uses shared package (not local file)
* Goat can import and use wrapper
* RobloxChristian can import and use wrapper
* Tests pass

**Time:** 4 hours

---

## Issue Routing Summary

| Issue | Repo | Labels | Parallel Safe |
|-------|------|--------|---------------|
| ROM-20 | comfyui-massmediafactory-mcp | `repo:comfyui`, `docs` | ✅ Yes |
| ROM-21 | Goat | `repo:goat`, `migration` | ✅ Yes |
| ROM-22 | RobloxChristian | `repo:roblox`, `migration` | ✅ Yes |
| ROM-23 | comfyui-massmediafactory-mcp | `repo:comfyui`, `infra` | ❌ Wait for ROM-20 |

**Total Time:** 11 hours (ROM-20, ROM-21, ROM-22 in parallel = 3-4 hours; ROM-23 = 4 hours after)

---

## Delegation Commands

```python
# After creating in Linear, delegate to Cyrus for execution:
mcp__linear__update_issue("ROM-20", delegate="Cyrus")
mcp__linear__update_issue("ROM-21", delegate="Cyrus")  # Will wait for ROM-20
mcp__linear__update_issue("ROM-22", delegate="Cyrus")  # Will wait for ROM-20
mcp__linear__update_issue("ROM-23", delegate="Cyrus")  # Can run in parallel
```

## Execution Order

**Phase 1 (Parallel):**
1. ROM-20 - Document patterns (needed for reference)
2. ROM-23 - Create shared package (can run parallel with docs)

**Phase 2 (After ROM-20 complete):**
3. ROM-21 - Goat migration (uses patterns from ROM-20)
4. ROM-22 - RobloxChristian migration (uses patterns from ROM-20)

**Why this order:**
- ROM-20 creates the reference documentation needed for migrations
- ROM-21 and ROM-22 can use shared package if ROM-23 finishes first
- Goat and RobloxChristian are independent (different repos)
