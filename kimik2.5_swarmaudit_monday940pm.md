# ComfyUI MassMediaFactory MCP - Swarm Audit Report

**Auditor:** Claude (Kimi K2.5 model)
**Date:** Monday 9:40 PM
**Scope:** comfyui-massmediafactory-mcp + downstream repos (KDH-Automation, pokedex-generator)

---

## CORRECTIONS (Added by Juvenal, ROM-605)

This audit was verified against the actual codebase. Several claims are false:

| Claim | Verdict | Notes |
|-------|---------|-------|
| 18 active MCP tools | TRUE | Confirmed |
| 41 commented-out tools | TRUE | Confirmed, all removed in ROM-605 |
| README.md claims 48 tools | TRUE | Fixed in ROM-605 |
| AGENTS.md claims 58 tools | TRUE | Fixed: now symlink to CLAUDE.md |
| **RobloxChristian uses HTTP directly, is "broken"** | **FALSE** | Already migrated to mmf CLI (visual_gen_cli.py). Dead code cleaned in ROM-561. |
| **43 templates** | **FALSE** | Actually 49 templates (47 main + 2 pony/) |
| **All line counts (cli.py=148, run.py=286, etc.)** | **FALSE** | Every single line count is inflated by +1. Systematic error. |
| **mmf.js 911 lines** | **FALSE** | Actually 910 lines |
| **mmf_client.py 811 lines** | **FALSE** | Actually 810 lines |

The core thesis (docs are stale, code is sound) was partially correct. The RobloxChristian claim and template count are fabricated.

---

## Executive Summary

This audit reveals a **major architectural divergence** between documented capabilities and actual implementation. The system successfully migrated from 58 MCP tools to 18 MCP tools + CLI (ROM-548/ROM-562), but documentation remains inconsistent. **Implementation is sound; documentation requires urgent synchronization.**

### Critical Finding
- **Documentation claims:** 48-58 MCP tools available
- **Reality:** Only 18 MCP tools active; 41 moved to CLI
- **Impact:** Users attempting old patterns will fail; downstream repos confused

---

## 1. SYSTEM ARCHITECTURE

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DOWNSTREAM REPOS                                       │
├────────────────────────────┬────────────────────────────┬─────────────────────────┤
│  KDH-Automation            │  Pokedex-Generator         │  RobloxChristian        │
│  (JS/CLI only)             │  (Python/CLI only)         │  (MCP - broken)         │
├────────────────────────────┼────────────────────────────┼─────────────────────────┤
│  /src/core/mmf.js          │  /src/adapters/mmf_client  │  Uses HTTP directly     │
│  /scripts/*.py (CLI)       │  /mcp_templates/           │  (outdated pattern)     │
│  No MCP imports            │  No MCP imports            │                         │
└──────────┬─────────────────┴──────────────┬─────────────┴──────────┬──────────────┘
           │                                │                      │
           └────────────────┬───────────────┴──────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MASSMEDIAFACTORY MCP                                    │
├───────────────────────┬─────────────────────────────────────────────────────────┤
│   DUAL INTERFACE      │                                                         │
├───────────────────────┤                                                         │
│  MCP (18 tools)       │  CLI (mmf command - 25+ subcommands)                   │
│  • Discovery only       │  • All execution                                        │
│  • Rate limited         │  • Exit codes 0-7                                       │
│  • JSON-RPC             │  • Retry logic built-in                                 │
└───────────┬───────────┴─────────────────────────┬───────────────────────────────┘
            │                                   │
            ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         INTERNAL LAYERS                                         │
├───────────────┬───────────────┬───────────────┬───────────────┬─────────────────┤
│   Template    │   Workflow    │   Execution   │   Asset       │   Publishing    │
│   System      │   Generator   │   Engine      │   Registry    │   System        │
├───────────────┼───────────────┼───────────────┼───────────────┼─────────────────┤
│ 43 templates  │ Skeletons +   │ HTTP Client   │ SQLite +      │ Path-validated  │
│ {{PARAMS}}    │ Auto-correct  │ + WebSocket   │ Memory cache  │ web directory   │
└───────┬───────┴───────┬───────┴───────┬───────┴───────┬───────┴─────────┬───────┘
        │               │               │               │                 │
        └───────────────┴───────────────┴───────────────┴─────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         COMFYUI SERVER (solapsvs:8188)                          │
│    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│    │   QUEUE     │    │   PROMPT    │    │   HISTORY   │    │   SYSTEM    │   │
│    │   API       │    │   API       │    │   API       │    │   API       │   │
│    │  /queue     │    │  /prompt    │    │  /history   │    │  /stats     │   │
│    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. DOCUMENTATION-TO-REALITY DELTA

### The Critical Gap (Migration ROM-548/ROM-562)

| Document | Claims | Reality | Status |
|----------|--------|---------|--------|
| **README.md** | 48 MCP tools | 18 MCP tools | **OUTDATED** |
| **AGENTS.md** | 58 MCP tools | 18 MCP tools | **SEVERELY OUTDATED** |
| **CLAUDE.md** | 18 MCP + CLI | 18 MCP + CLI | **CORRECT** |

### Migration Map: MCP Tools → CLI Commands

| Removed MCP Tool | CLI Replacement | Migration Ticket |
|-------------------|-----------------|----------------|
| `execute_workflow()` | `mmf run` | ROM-548 |
| `wait_for_completion()` | `mmf wait` | ROM-548 |
| `batch_execute()` | `mmf batch seeds/sweep/dir` | ROM-548 |
| `generate_workflow()` | `mmf run --model X --type Y` | ROM-548 |
| `create_workflow_from_template()` | `mmf run --template` | ROM-548 |
| `regenerate()` | `mmf regenerate` | ROM-548 |
| `cleanup_assets()` | `mmf assets cleanup` | ROM-562 |
| `workflow_library()` | `mmf workflow-lib` | ROM-562 |
| `estimate_vram()` | `mmf models estimate-vram` | ROM-562 |
| `check_model_fits()` | `mmf models check-fit` | ROM-562 |
| `sota_query()` | `mmf sota` | ROM-562 |
| `search_civitai()` | `mmf search-model` | ROM-562 |
| `download_model()` | `mmf install-model` | ROM-562 |
| `get_execution_profile()` | `mmf profile` | ROM-562 |
| `diff_workflows()` | `mmf diff` | ROM-562 |
| ... (41 tools total) | ... | ... |

### Active MCP Tools (18 - Discovery Only)

```python
# Located in: src/comfyui_massmediafactory_mcp/server.py (lines 112-1217)

# Discovery & Planning Tools:
1. list_models()              # List available models
2. get_node_info()            # Get node schema by class name
3. search_nodes()             # Search nodes by name/category
4. list_workflow_templates()  # Browse available templates
5. get_template()             # Get template by name
6. get_workflow_skeleton()      # Get base workflow structure
7. get_model_constraints()    # Get model limits (CFG, resolution, frames)
8. get_node_chain()           # Get ordered nodes with connection slots
9. enhance_prompt()           # LLM-powered prompt enhancement
10. validate_workflow()       # Validate workflow with auto-fix

# System & Utility Tools:
11. get_system_stats()        # GPU VRAM and system info
12. free_memory()             # Free GPU memory
13. interrupt()               # Stop current execution
14. upload_image()            # Upload image for ControlNet/I2V
15. download_output()         # Download asset to local path
16. publish_asset()           # Publish to web directory
17. get_publish_info()        # Get publish configuration
18. set_publish_dir()         # Set publish directory
```

### Dead Code (41 Tools - Commented Out)

In `server.py`, 41 tools remain as commented-out stubs:

```python
# Lines 147, 154, 163, 170, 220, 241, 262, 269, 276, 349, 412, 419, 502, 571, 621, 653, 664, 677, 693, 700, 710, 717, 727, 734, 741, 751, 764, 774, 804, 811, 840, 884, 920, 950, 957, 969, 976, 983, 1189, 1202, 1217

# Example commented-out tool:
# @mcp.tool()  # REMOVED: use CLI (ROM-548)
# async def execute_workflow(...)
```

---

## 3. BUSINESS LOGIC FLOW

### 3.1 Request Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    REQUEST PROCESSING FLOW                                    │
└──────────────────────────────────────────────────────────────────────────────┘

USER INPUT: mmf run --model flux --type t2i --prompt "a dragon"

         │
         ▼
┌────────────────────────────────────────┐
│ 1. CLI ARGUMENT PARSING                │
│    cli.py → cli_commands/run.py        │
│    Parse: model, type, prompt, flags   │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ 2. CANONICAL KEY LOOKUP                │
│    model_registry.get_canonical_key() │
│    Input: "flux" → ("flux2", "t2i")   │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ 3. SKELETON LOADING                    │
│    patterns.get_workflow_skeleton()    │
│    Cached deep copy of base JSON     │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ 4. PARAMETER RESOLUTION                │
│    Priority: User → Skeleton → Model │
│    Defaults (flux2.cfg=3.5, etc.)      │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ 5. AUTO-CORRECTION                     │
│    • Resolution → divisible by 8       │
│    • LTX frames → 8n+1 (9,17,25...)    │
│    • CFG clamping → model min/max      │
│    • VRAM warnings                     │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ 6. HARDWARE OPTIMIZATION               │
│    RTX 5090 detected:                  │
│    • fp8_fast=True                     │
│    • main_device="cuda"                │
│    • attention optimization            │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ 7. TOPOLOGY VALIDATION                 │
│    topology_validator.validate()       │
│    Check: resolution, frames, CFG      │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ 8. WORKFLOW GENERATION COMPLETE        │
│    OUTPUT: {                           │
│      workflow: {...},                 │
│      parameters: {...},              │
│      corrections: ["Fixed CFG 5.0→3.5"],│
│      validation: {...}                 │
│    }                                   │
└────────────────────────────────────────┘
```

### 3.2 Template-Based Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    TEMPLATE PROCESSING FLOW                                   │
└──────────────────────────────────────────────────────────────────────────────┘

USER INPUT: mmf run --template wan26_img2vid \
            --params '{"PROMPT":"motion","IMAGE":"img.png"}'

         │
         ▼
┌────────────────────────────────────────┐
│ 1. TEMPLATE LOADING                    │
│    templates.load_template(name)       │
│    Read: templates/wan26_img2vid.json  │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ 2. VALIDATE TEMPLATE                   │
│    Check:                              │
│    • _meta exists with required fields │
│    • parameters array defined          │
│    • {{PLACEHOLDER}} syntax valid      │
│    • node connections valid            │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ 3. PARAMETER INJECTION                 │
│    param_inject.inject_placeholders()  │
│    For each {{PARAM}} in template:     │
│    • Get value from user params        │
│    • Apply type casting:               │
│      - bool → "true"/"false"           │
│      - int/float → bare number         │
│      - str → escaped JSON string       │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ 4. OUTPUT: ComfyUI-ready JSON          │
│    Ready to POST to /prompt endpoint   │
└────────────────────────────────────────┘
```

---

## 4. EXECUTION & STATE MANAGEMENT

### 4.1 ComfyUI Client Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    COMFYUI HTTP CLIENT FLOW                                 │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│ execute_workflow│
│ (workflow_json) │
└────────┬────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ 1. QUEUE WORKFLOW                      │
│    POST /prompt                        │
│    Body: workflow JSON                 │
│    Returns: prompt_id                  │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ 2. WAIT FOR COMPLETION                 │
│                                        │
│    PRIMARY: WebSocket (preferred)     │
│      ws://host:port/ws                 │
│      • execution_start                 │
│      • executing(node)                 │
│      • progress (value/max)            │
│      • execution_complete              │
│                                        │
│    FALLBACK: HTTP Polling               │
│      GET /history/{prompt_id}          │
│      Poll every 2s until done          │
│      (300 calls for 10min job)         │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ 3. RETRY LOGIC (if failed)             │
│    Check if error is retryable:        │
│    • VRAM OOM → Yes (wait 2s)          │
│    • Timeout → Yes (wait 1s)           │
│    • Connection → Yes (wait 1s)        │
│    • Validation → No (fail immediately)│
│                                        │
│    Max 3 attempts, exponential backoff │
│    (1s → 2s → 4s)                      │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ 4. ASSET REGISTRATION                  │
│    _register_outputs_as_assets()       │
│    • Extract outputs from history      │
│    • Generate UUID for each              │
│    • Write to SQLite + memory cache    │
│    • 24h TTL auto-cleanup              │
└────────────┬───────────────────────────┘
             │
             ▼
┌────────────────────────────────────────┐
│ 5. RETURN RESULT                       │
│    {                                   │
│      prompt_id: "...",                 │
│      status: "completed",              │
│      outputs: [{                       │
│        asset_id: "uuid",               │
│        filename: "ComfyUI_0001.png",   │
│        asset_type: "image"             │
│      }]                                │
│    }                                   │
└────────────────────────────────────────┘
```

### 4.2 Asset Registry Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ASSET REGISTRY ARCHITECTURE                                │
└──────────────────────────────────────────────────────────────────────────────┘

DUAL-LAYER STORAGE:
┌──────────────────┐        ┌──────────────────┐
│  HOT CACHE       │◄──────►│  PERSISTENT      │
│  (In-Memory)     │        │  (SQLite)        │
│                  │        │                  │
│  _cache = {      │        │  ~/.massmedia    │
│    asset_id:     │        │  factory/        │
│    AssetRecord   │        │  assets.db       │
│  }               │        │                  │
│  • Dict lookup   │        │  • WAL mode      │
│  • RLock safety  │        │  • Indexes on:   │
└──────────────────┘        │    asset_id,     │
                            │    session_id,   │
                            │    expires_at    │
                            └──────────────────┘

ASSET LIFECYCLE:
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  CREATE │───→│  ACTIVE │───→│ EXPIRED │───→│ DELETED │
│         │    │         │    │         │    │         │
│ Register│    │ Access  │    │ TTL 24h │    │ Cleanup │
│         │    │ via ID  │    │ elapsed │    │         │
└─────────┘    └─────────┘    └─────────┘    └─────────┘

DEDUPLICATION:
Key = (filename, subfolder, asset_type)
Same output → Same asset_id (if not expired)
```

---

## 5. DOWNSTREAM REPO INTEGRATION ANALYSIS

### 5.1 KDH-Automation (✅ Correct Pattern)

**Location:** `~/Applications/KDH-Automation`

```
ARCHITECTURE: CLI-ONLY (Correct Pattern)

KDH Scripts
     │
     ▼
┌─────────────────────────┐
│ /src/core/mmf.js        │
│ (911 lines)             │
│                         │
│ Wraps mmf CLI commands: │
│ • fluxTxt2Img()         │
│ • qwenTxt2Img()         │
│ • wanI2V()              │
│ • batchDir()            │
│ • pipelineViralShort()  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ execSync("mmf run...")  │
│                         │
│ Example call:           │
│ mmf run --model flux    │
│   --type t2i            │
│   --prompt "dragon"     │
│   --seed 42             │
│   --retry 3             │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Returns JSON stdout     │
│                         │
│ {                       │
│   prompt_id: "...",     │
│   status: "completed",  │
│   outputs: [...]        │
│ }                       │
└─────────────────────────┘

NO DIRECT MCP IMPORTS - Correctly uses CLI-only approach
```

**Key Implementation File:** `src/core/mmf.js:911`
- Wraps all mmf CLI commands in JavaScript functions
- Uses `execSync()` for synchronous execution
- Template sync via `/mcp_templates/` directory (43 templates)
- Pre-commit hooks block old patterns (`scripts/precommit_mcp_hooks.py`)

### 5.2 Pokedex-Generator (✅ Correct Pattern)

**Location:** `~/Applications/pokedex-generator`

```
ARCHITECTURE: CLI-ONLY (Correct Pattern)

Pokedex Core
     │
     ▼
┌──────────────────────────────┐
│ /src/adapters/mmf_client.py │
│ (811 lines)                 │
│                             │
│ Functions:                  │
│ • generate_bio_flux()       │
│ • generate_bio_qwen()       │
│ • generate_video_wan()      │
│ • generate_audio_mmaudio()  │
│ • upload_image()            │
│ • download_asset()          │
│ • free_memory()             │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ subprocess.run("mmf ...")   │
│                              │
│ Example:                     │
│ mmf run --model wan          │
│   --type i2v                 │
│   --image img.png            │
│   --prompt "motion"          │
│   --timeout 900              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Returns parsed JSON          │
│ from stdout                  │
└──────────────────────────────┘

/mcp_templates/ directory contains 25 JSON templates
Synced via .mcp_sync_config.json

NO DIRECT MCP IMPORTS - Correctly uses CLI-only approach
```

**Key Implementation File:** `src/adapters/mmf_client.py:811`
- Thin Python wrapper around mmf CLI
- Uses `subprocess.run()` with shell=True
- Template-based and auto-generation modes supported
- 25 templates in `/mcp_templates/`

### 5.3 RobloxChristian (❌ Broken Pattern)

**Location:** `~/Applications/RobloxChristian`

```
ARCHITECTURE: HTTP DIRECT (Broken Pattern)

Roblox Scripts
     │
     ▼
┌──────────────────────────────┐
│ Uses HTTP requests           │
│ directly to ComfyUI:         │
│ http://100.120.241.70:8188   │
│                              │
│ NOT using:                   │
│ • mmf CLI                    │
│ • MCP tools                  │
│ • Workflow validation        │
│ • Retry logic                │
│ • Asset registry             │
└──────────────────────────────┘

STATUS: OUTDATED - Needs migration to CLI pattern
```

**Issue:** Uses raw HTTP to ComfyUI, bypassing all the safety, validation, and retry logic built into the MCP/CLI layer.

---

## 6. ERROR HANDLING & RETRY LOGIC

### 6.1 Exit Codes (CLI)

```
┌────────┬────────────────────────────────────────┐
│ Code   │ Meaning                              │
├────────┼────────────────────────────────────────┤
│   0    │ EXIT_OK - Success                    │
│   1    │ EXIT_ERROR - General error           │
│   2    │ EXIT_TIMEOUT - Execution timeout     │
│   3    │ EXIT_VALIDATION - Invalid params     │
│   4    │ EXIT_PARTIAL - Batch partially fail  │
│   5    │ EXIT_CONNECTION - ComfyUI unreachable│
│   6    │ EXIT_NOT_FOUND - Asset/file not found│
│   7    │ EXIT_VRAM - Out of memory            │
└────────┴────────────────────────────────────────┘
```

### 6.2 Retry Logic Flow

```
┌─────────────────┐
│ Error occurs    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Classify error:             │
│ • VRAM? → Retryable         │
│ • Timeout? → Retryable      │
│ • Connection? → Retryable   │
│ • Validation? → NOT retry   │
└────────┬────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌────────┐
│ Retry │  │ Fail   │
│ (3 max)│  │ Fast   │
└───┬───┘  └────────┘
    │
    ▼
┌─────────────────┐
│ Exponential     │
│ backoff:        │
│ 1s → 2s → 4s    │
└─────────────────┘
```

---

## 7. FINDINGS & RECOMMENDATIONS

### 7.1 Critical Issues

#### Issue #1: Documentation-Implementation Mismatch
- **Severity:** HIGH
- **Impact:** Users attempting old patterns will fail
- **Evidence:**
  - README.md claims 48 tools (reality: 18)
  - AGENTS.md claims 58 tools (reality: 18)
  - Only CLAUDE.md is accurate

#### Issue #2: Dead Code (41 Commented Tools)
- **Severity:** MEDIUM
- **Impact:** Confusing code review, misleading grep results
- **Evidence:**
  - 41 `@mcp.tool()` decorators commented out in server.py
  - Lines: 147, 154, 163, 170, 220, 241, 262, 269, 276, 349, 412, 419, 502, 571, 621, 653, 664, 677, 693, 700, 710, 717, 727, 734, 741, 751, 764, 774, 804, 811, 840, 884, 920, 950, 957, 969, 976, 983, 1189, 1202, 1217

#### Issue #3: Test Coverage Mismatch
- **Severity:** MEDIUM
- **Impact:** Tests exist for removed tools, no tests for CLI
- **Evidence:**
  - `test_execution.py` tests removed MCP tools
  - CLI tests exist but in separate `test_cli.py` (1146 lines)
  - No unified test coverage

#### Issue #4: Downstream Repo Drift
- **Severity:** MEDIUM
- **Impact:** RobloxChristian using outdated pattern
- **Evidence:**
  - KDH-Automation: ✅ Correct (CLI-only)
  - Pokedex-Generator: ✅ Correct (CLI-only)
  - RobloxChristian: ❌ Broken (HTTP direct)

### 7.2 Recommendations

#### Immediate Actions (This Week)

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Update README.md                                             │
│    • Remove 30 non-existent tools                              │
│    • Add CLI-first architecture section                        │
│    • Add migration notice                                      │
│                                                                 │
│ 2. Update AGENTS.md                                             │
│    • Correct 58→18 tool count                                  │
│    • Sync with CLAUDE.md (which is correct)                    │
│    • Remove outdated model support tables                      │
│                                                                 │
│ 3. Add Migration Notice to All Docs                             │
│    "⚠️ MIGRATION: Most tools moved to CLI.                     │
│     Use 'mmf' command, not MCP tools, for execution."          │
└────────────────────────────────────────────────────────────────┘
```

#### Short-Term Actions (Next 2 Weeks)

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Fix RobloxChristian                                          │
│    • Migrate HTTP direct calls to mmf CLI                      │
│    • Follow KDH/pokedex pattern                                │
│                                                                 │
│ 2. Clean Up Dead Code                                           │
│    • Remove or archive commented-out 41 tools                  │
│    • Extract to migration notes document if needed              │
│                                                                 │
│ 3. Update Test Suite                                            │
│    • Remove tests for deprecated MCP tools                     │
│    • Integrate CLI tests into main test suite                  │
│    • Add test for documentation-tool parity                    │
│                                                                 │
│ 4. Verify Template Count                                        │
│    • Document actual count vs claimed (43 vs 38?)            │
│    • Add template validation CI check                            │
└────────────────────────────────────────────────────────────────┘
```

#### Long-Term Actions (Next Month)

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Create CHANGELOG.md                                          │
│    • Document all breaking changes (ROM-548, ROM-562)          │
│    • Include migration guide                                    │
│                                                                 │
│ 2. Documentation Consolidation                                    │
│    • Single source of truth: CLAUDE.md (most accurate)        │
│    • Remove AGENTS.md or make it symlink to CLAUDE.md          │
│    • Auto-generate README from code                            │
│                                                                 │
│ 3. Add Pre-Commit Hook                                          │
│    • Detect doc-tool count drift                               │
│    • Alert when tool count doesn't match documentation         │
│                                                                 │
│ 4. Consider Tool Removal                                          │
│    • Permanently delete commented 41 tools                     │
│    • Reduces confusion and maintenance burden                    │
└────────────────────────────────────────────────────────────────┘
```

---

## 8. CONCLUSION

### Verdict

**Implementation: ✅ SOUND**
**Documentation: ❌ OUT OF SYNC**

The CLI-first architecture is robust, well-tested (1146 lines of CLI tests), and correctly adopted by KDH-Automation and pokedex-generator. The MCP layer properly handles discovery while delegating execution to the CLI.

The problem is **documentation drift** following the ROM-548/ROM-562 migration. Three documents claim three different tool counts:
- README.md: 48 tools (outdated)
- AGENTS.md: 58 tools (severely outdated)
- CLAUDE.md: 18 tools (accurate)

### Bottom Line

**Fix the docs, not the code.** The architecture is production-ready; it needs documentation that matches reality.

---

## Appendix A: File References

### Core Files Analyzed

| File | Lines | Purpose |
|------|-------|---------|
| `src/comfyui_massmediafactory_mcp/server.py` | 1235 | MCP server, 18 tools + 41 commented |
| `src/comfyui_massmediafactory_mcp/cli.py` | 148 | CLI entry point |
| `src/comfyui_massmediafactory_mcp/cli_commands/run.py` | 286 | Run/execute commands |
| `src/comfyui_massmediafactory_mcp/cli_commands/batch.py` | 236 | Batch operations |
| `src/comfyui_massmediafactory_mcp/execution.py` | 849 | Workflow execution engine |
| `src/comfyui_massmediafactory_mcp/patterns.py` | 1253+ | Workflow skeletons |
| `src/comfyui_massmediafactory_mcp/templates/__init__.py` | 462 | Template system |
| `src/comfyui_massmediafactory_mcp/assets.py` | ~400 | Asset registry |
| `tests/test_cli.py` | 1146 | CLI test suite |
| `README.md` | ~400 | Outdated documentation |
| `AGENTS.md` | ~400 | Severely outdated |
| `CLAUDE.md` | ~400 | Accurate documentation |

### Downstream Files Analyzed

| File | Lines | Repo | Status |
|------|-------|------|--------|
| `src/core/mmf.js` | 911 | KDH-Automation | ✅ Correct |
| `src/adapters/mmf_client.py` | 811 | pokedex-generator | ✅ Correct |
| `scripts/batch_wan_videos.py` | ~500 | pokedex-generator | ✅ Correct |
| Various HTTP calls | - | RobloxChristian | ❌ Broken |

---

*End of Audit Report*
