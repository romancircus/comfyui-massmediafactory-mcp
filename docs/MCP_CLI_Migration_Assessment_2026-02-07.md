# MCP→CLI Migration Strategic Assessment
**Project:** ComfyUI MassMediaFactory MCP Server Migration
**Date:** 2026-02-07
**Status:** Assessment Phase → Linear Task Generation

---

## Executive Summary

The migration from MCP tools to the unified `mmf` CLI is approximately **90-95% complete** across the ecosystem. Four downstream repositories consume ComfyUI functionality, with varying degrees of migration success. This assessment identifies completion gaps, architectural patterns, best practices, and strategic recommendations to finish the migration and establish a unified execution paradigm.

**Key Metrics:**
- 4 repositories assessed
- 1 fully migrated (KDH-Automation)
- 1 near-complete (pokedex-generator via mmf_client.py)
- 1 divergent (Goat - docs claim CLI, code uses HTTP)
- 1 hybrid (RobloxChristian - MCP client exists but orchestrator doesn't use it)

---

## Section 1: Repository Assessment Summary

### 1.1 comfyui-massmediafactory-mcp (Main Repository)
**Status:** ~95% Complete

**Current State:**
- Primary MCP server implementation with 34+ tools
- `mmf` CLI fully implemented and documented
- All core functionality available via CLI (`run`, `batch`, `pipeline`, `qa`, `assets`, `visualize`)
- Templates system migrated (34 templates)
- Workflows validated across 12 model+type combinations

**Remaining Work:**
- Migration of remaining legacy MCP tools to CLI equivalents
- Complete documentation refresh to reflect CLI-first paradigm
- Template system drift detection validation
- Final cleanup of deprecated MCP paths

**Key Files:**
- `mmf/` - CLI package (95% coverage)
- `massmediafactory_mcp_server.py` - Legacy MCP server
- `AGENTS.md` - Architecture documentation

---

### 1.2 KDH-Automation
**Status:** COMPLETE ✅

**Current State:**
- Fully migrated to `mmf` CLI
- Uses `mmf run`, `mmf batch`, `mmf pipeline` commands
- No direct HTTP calls to ComfyUI
- Orchestration scripts use CLI exclusively
- Cleanest implementation in the ecosystem

**Evidence:**
- All generation workflows call `mmf` commands
- Batch processing via `mmf batch`
- Viral shorts and 3D music video pipelines CLI-native

**Key Files:**
- Orchestration scripts in `scripts/` directory
- Template definitions in `templates/`

---

### 1.3 pokedex-generator
**Status:** ~95% Complete

**Current State:**
- Uses `mmf_client.py` wrapper (CLI abstraction layer)
- Effectively CLI-first through wrapper pattern
- Minor dependency cleanup pending (torch/diffusers)
- Shiny 151 pipeline working via mmf_client

**Remaining Work:**
- Remove unused torch/diffusers dependencies
- Verify direct CLI usage vs wrapper (wrapper is acceptable pattern)
- Final cleanup pass

**Key Files:**
- `mmf_client.py` - CLI abstraction wrapper
- `scripts/` - Generation orchestration

**Note:** This is effectively complete; wrapper pattern is valid architectural choice.

---

### 1.4 Goat
**Status:** NOT ADOPTED ⚠️

**Current State:**
- **Documentation/code divergence** - AGENTS.md claims CLI usage
- **Reality:** Uses direct HTTP client (`httpx`/`requests` to ComfyUI server)
- Not using `mmf` CLI despite documented migration
- Channel launch work ongoing but without CLI adoption

**Gap Analysis:**
- No evidence of `mmf` commands in codebase
- Custom HTTP implementation duplicating CLI functionality
- Drift from documented architecture

**Required Action:**
- Audit all HTTP calls to ComfyUI
- Replace with `mmf` CLI calls
- Update AGENTS.md or fix code (whichever is wrong)
- Priority: High - blocking ecosystem standardization

**Key Files:**
- `AGENTS.md` - Claims CLI usage (verify accuracy)
- Search for `httpx`, `requests`, `http://` patterns

---

### 1.5 RobloxChristian
**Status:** Hybrid ⚠️

**Current State:**
- MCP client library exists (`comfyui_mcp_client.py`)
- **Orchestrator does NOT use the MCP client**
- Bible Stories generation uses direct HTTP calls
- Client library orphaned/abandoned in codebase

**Gap Analysis:**
- Duplication: Client library exists but unused
- Orchestrator bypasses client entirely
- Pattern shows incomplete migration attempt

**Required Action:**
- Option A: Remove MCP client, migrate orchestrator to `mmf` CLI
- Option B: Fix orchestrator to use existing MCP client
- Recommendation: Option A (align with ecosystem)

**Key Files:**
- `comfyui_mcp_client.py` - Orphaned client
- `orchestrator/` - Main generation logic (bypasses client)

---

## Section 2: Key Architectural Patterns Identified

### 2.1 Hub-and-Spoke Template System
**Pattern:** Central template repository with downstream consumption

```
comfyui-massmediafactory-mcp (Hub)
├── templates/ (34 templates)
│   ├── flux2_txt2img.json
│   ├── ltx2_txt2vid.json
│   ├── wan21_img2vid.json
│   └── qwen_txt2img.json
│
├── manifest.json (drift detection)
└── cache/ (template versioning)
    ↓
KDH-Automation (Spoke)
pokedex-generator (Spoke)
Goat (Spoke) ← Not consuming properly
RobloxChristian (Spoke) ← Not consuming properly
```

**Drift Detection Mechanism:**
- `manifest.json` tracks template versions, checksums, compatibility
- Repos validate templates against manifest before execution
- Prevents silent failures from template updates

**Implementation:**
- Template validation before execution
- Version pinning support
- Automatic rollback on incompatibility

---

### 2.2 Three-Tier Execution Approach
**Pattern:** Segmented execution strategies based on scale and timing

| Tier | Mode | Tools | Use Case |
|------|------|-------|----------|
| **Interactive** | Real-time | `mmf run`, manual prompt | Human-in-the-loop, iteration |
| **Batch** | Multi-job | `mmf batch seeds`, `mmf batch sweep` | Parameter exploration, variations |
| **Overnight** | Autonomous | `mmf pipeline`, jinyang orchestration | Large-scale, agentic execution |

**Benefits:**
- Interactive tier: Fast feedback loop for creative iteration
- Batch tier: Systematic exploration without human attention
- Overnight tier: Scales beyond human monitoring capacity

---

### 2.3 Template Hub with Drift Detection
**Pattern:** Version-controlled templates with automated validation

**Components:**
1. **Template Registry:** JSON templates with metadata
2. **Manifest System:** Version tracking, compatibility matrix
3. **Validation Layer:** Pre-execution constraint checking
4. **Cache Invalidation:** Automatic refresh on drift

**Workflow:**
```
1. Template loaded from hub
2. Checksum validated against manifest
3. Constraint validation (VRAM, resolution, CFG)
4. Execution or graceful degradation
5. Results logged back to hub for learning
```

---

### 2.4 Retry Logic with VRAM Recovery
**Pattern:** Resilient execution with automatic recovery

**Mechanism:**
```python
def execute_with_recovery(workflow):
    try:
        return execute(workflow)
    except OOMError:
        free_memory(unload_models=True)
        return execute(workflow)
    except ComfyUIUnavailable:
        sleep(RETRY_DELAY)
        return execute(workflow)
```

**VRAM Recovery Strategies:**
- Automatic model unloading on OOM
- Batch size reduction with retry
- Queue position preservation during recovery
- Graceful fallback to lower precision (FP16 → FP8)

---

## Section 3: Critical Gaps and Pain Points

### 3.1 Documentation/Code Divergence (GOAT) 🔴 CRITICAL

**Issue:** AGENTS.md documents CLI usage, code uses HTTP

**Impact:**
- Team members follow docs, write CLI commands
- Actual execution uses different code path
- Debugging confusion (works locally, fails remotely)
- Maintenance burden (two code paths)

**Root Cause:**
- Partial migration started but not completed
- Docs updated before code migration
- No validation mechanism for doc/code sync

**Resolution:**
1. Audit all ComfyUI HTTP calls in Goat
2. Replace with `mmf` CLI equivalents
3. Add CI check for HTTP patterns in codebase
4. Document the discrepancy if intentional

**Estimated Effort:** 2-4 hours
**Priority:** High (blocks standardization)

---

### 3.2 Pokedex Dependency Cleanup

**Issue:** Unused torch/diffusers dependencies present

**Impact:**
- Bloat in virtual environment
- Potential version conflicts
- Slower CI builds

**Resolution:**
1. Audit requirements.txt
2. Remove torch, diffusers, transformers if unused
3. Verify mmf_client.py doesn't need them

**Estimated Effort:** 30 minutes
**Priority:** Low (cleanup task)

---

### 3.3 RobloxChristian Orchestrator Not Using MCP Client

**Issue:** Client library exists but orchestrator bypasses it

**Impact:**
- Code duplication (client library vs orchestrator HTTP)
- Orphaned code (comfyui_mcp_client.py)
- No standardization benefits

**Resolution:**
1. Decide: Use client or remove it
2. If use: Refactor orchestrator to use client
3. If remove: Delete comfyui_mcp_client.py, migrate to CLI

**Recommendation:** Remove client, migrate to CLI (aligns with ecosystem)

**Estimated Effort:** 1-2 hours
**Priority:** Medium (technical debt)

---

### 3.4 No Centralized CLI Wrapper Pattern

**Issue:** Each repo has different abstraction layers

**Current State:**
- KDH: Direct `mmf` CLI calls
- Pokedex: `mmf_client.py` wrapper
- Goat: Direct HTTP (no CLI)
- Roblox: Unused MCP client

**Impact:**
- Inconsistent error handling
- Different retry logic per repo
- Knowledge silos (KDH pattern not known to others)
- Maintenance burden (4 patterns to maintain)

**Desired State:**
```
┌─────────────────────────────────────┐
│  Single Standard: mmf CLI            │
│  (or documented wrapper pattern)     │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│  Downstream Repos                   │
│  - Direct CLI usage (KDH style)     │
│  - Or minimal wrapper (Pokedex)    │
└─────────────────────────────────────┘
```

**Resolution:**
1. Document KDH pattern as canonical
2. Migrate all repos to consistent approach
3. Add wrapper pattern to AGENTS.md if needed
4. Create shared utilities package (optional)

**Estimated Effort:** 3-4 hours across all repos
**Priority:** High (architectural consistency)

---

## Section 4: Best-in-Class Findings from Code Research

### 4.1 CLI Frameworks

**Typer (Recommended for New Projects)**
```python
import typer
app = typer.Typer()

@app.command()
def run(model: str, prompt: str, width: int = 1024):
    """Generate image from prompt"""
    pass
```
**Pros:** Type hints, automatic help, modern Pythonic
**Cons:** Less extensibility than Click

**Click (Current `mmf` Choice)**
```python
import click

@click.command()
@click.option('--model', required=True)
@click.option('--prompt', required=True)
def run(model, prompt):
    pass
```
**Pros:** Battle-tested, extensive plugin ecosystem, backwards compatible
**Cons:** Verbose, no native type hints (uses decorators)

**Verdict:** Click is correct choice for `mmf` given stability requirements

---

### 4.2 ComfyUI Optimizations

**NVFP4 (Blackwell Architecture)**
- 4-bit floating point precision
- 2x speedup vs FP8 on RTX 5090
- Best for: Real-time generation, large batches

**FP8 (RTX 40/50 Series)**
- NVIDIA-native 8-bit precision
- Minimal quality loss vs FP16
- Best for: Balanced speed/quality

**Implementation Pattern:**
```python
# Detect GPU architecture
gpu_info = get_gpu_info()
if gpu_info.is_blackwell:
    precision = "nvfp4"
elif gpu_info.is_rtx_40_series or gpu_info.is_rtx_50_series:
    precision = "fp8"
else:
    precision = "fp16"
```

---

### 4.3 Async Offloading and Pinned Memory

**Pattern for NVIDIA GPUs:**
```python
# Pinned memory for faster CPU→GPU transfers
import torch
x = torch.randn(1000, device='cpu', pin_memory=True)
x = x.cuda(non_blocking=True)  # Async transfer

# Overlap compute and transfer
with torch.cuda.stream(torch.cuda.Stream()):
    model(x)
```

**Benefits:**
- 15-20% throughput improvement
- Reduced CPU bottleneck
- Better GPU utilization

---

### 4.4 SOTA Models (Updated 2026-02-07)

| Model | VRAM | Use Case | Status |
|-------|------|----------|--------|
| **WAN 2.2** | ~24GB | Human motion quality | ✅ Recommended |
| **LTX-2** | ~12GB | Speed-focused video | ✅ Recommended |
| **FLUX.2-dev** | ~24GB | Character consistency | ✅ Recommended |
| **Qwen Image 2512** | ~40GB | Portrait quality | ✅ Recommended |

**Model-Task Matrix:**
- **Text-to-Image:** Qwen Image (portrait), FLUX.2 (general)
- **Text-to-Video:** WAN 2.2 (motion), LTX-2 (speed)
- **Image-to-Video:** WAN 2.1 I2V (motion quality)

**RTX 5090 32GB Optimal Settings:**
```
Precision: FP8 (native support)
Batch: 2-4 for images, 1 for video
CFG: 2.5-4.5 (model dependent)
Steps: 20-30 ( diminishing returns after 30)
```

---

## Section 5: Strategic Recommendations

### 5.1 Completing the Migration (Priority Matrix)

| Task | Effort | Impact | Priority |
|------|--------|--------|----------|
| **Goat HTTP→CLI migration** | 2-4h | High | 🔴 CRITICAL |
| **RobloxChristian client cleanup** | 1-2h | Medium | 🟡 MEDIUM |
| **Pokedex dependency cleanup** | 30m | Low | 🟢 LOW |
| **Main repo final cleanup** | 2h | Medium | 🟡 MEDIUM |
| **Documentation sync** | 1h | High | 🔴 CRITICAL |

**Recommended Order:**
1. Goat migration (blocking)
2. RobloxChristian cleanup
3. Documentation sync
4. Main repo finalization
5. Pokedex cleanup

---

### 5.2 Standardizing CLI Patterns

**Canonical Pattern (KDH-Style):**
```bash
# Direct CLI usage - no wrapper needed for most cases
mmf run --model flux --type t2i --prompt "..."
mmf batch seeds workflow.json --count 8
mmf pipeline viral-short --prompt "..."
```

**When to Use Wrapper (Pokedex-Style):**
```python
# mmf_client.py - Only if:
# 1. Complex parameter transformation needed
# 2. Multiple CLI calls batched internally
# 3. Error handling custom to your domain
```

**Migration Guide for Remaining Repos:**

**Goat:**
```python
# BEFORE (HTTP)
response = httpx.post("http://comfyui:8188/prompt", json=workflow)

# AFTER (CLI)
import subprocess
result = subprocess.run([
    "mmf", "run",
    "--template", "flux2_txt2img",
    "--params", json.dumps(params)
], capture_output=True, text=True)
```

**RobloxChristian:**
1. Remove `comfyui_mcp_client.py`
2. Refactor orchestrator to use `mmf` CLI
3. Update AGENTS.md

---

### 5.3 Enabling Agentic ComfyUI Usage

**Current Capabilities:**
- ✅ jinyang can execute overnight batches
- ✅ Template-based generation with validation
- ✅ Progress tracking via WebSocket
- ✅ Automatic retry and recovery

**Gaps for Full Agentic Usage:**
- ❌ No prompt optimization feedback loop
- ❌ Limited VLM-based QA automation
- ❌ No automatic pipeline selection

**Recommendations:**

1. **Prompt Enhancement Pipeline**
   ```python
   # Before generation, enhance via LLM
   enhanced = subprocess.run([
       "mmf", "enhance",
       "--prompt", user_prompt,
       "--model", target_model
   ], capture_output=True)
   ```

2. **VLM-Based QA Loop**
   ```python
   # After generation, validate output
   result = subprocess.run([
       "mmf", "qa",
       "--asset", asset_id,
       "--criteria", "check_prompt_adherence"
   ])
   if result.returncode != 0:
       regenerate_with_adjustments()
   ```

3. **Automatic Model Selection**
   ```python
   # Based on prompt content, select model
   if "human" in prompt or "person" in prompt:
       model = "wan"  # Better for humans
   elif "landscape" in prompt:
       model = "flux"  # Better for scenery
   ```

4. **Agent Tools:**
   - Add `mmf agent-status` for queue visibility
   - Add `mmf agent-priority` for priority queuing
   - Add `mmf agent-cancel` for job management

---

### 5.4 Optimization Opportunities

**Immediate Wins (< 1 day):**

1. **FP8/NVFP4 Precision Adoption**
   - Update templates to use FP8 for compatible GPUs
   - 1.5-2x speedup on RTX 40/50 series

2. **Template Manifest Optimization**
   - Pre-compute optimal settings per GPU
   - Cache validated templates locally

3. **Batch Pipeline Optimization**
   - Parallel template validation
   - Async asset publishing

**Medium-Term (1-2 weeks):**

1. **VRAM Pool Management**
   - Shared model cache across batch jobs
   - Predictive model loading

2. **Queue Optimization**
   - Priority lanes (interactive > batch > overnight)
   - Job coalescing for similar prompts

3. **Template Learning**
   - Log generation results
   - Auto-adjust parameters based on success rate
   - Style drift detection

**Long-Term (1-2 months):**

1. **Distributed Execution**
   - Multi-GPU support
   - Remote ComfyUI nodes

2. **Model Hot-Swapping**
   - Keep multiple models in VRAM
   - Instant switching between FLUX/WAN/LTX

---

## Section 6: Linear Task Generation

### Proposed Issues

#### ROM-XXX: [CRITICAL] Goat Repository - Complete CLI Migration
**Type:** Infrastructure
**Project:** Infra: ComfyUI MCP
**Priority:** High
**Estimate:** 3h

**Description:**
Goat repository has documentation/code divergence. AGENTS.md claims CLI usage but code uses direct HTTP to ComfyUI.

**Acceptance Criteria:**
- [ ] Audit all HTTP calls to ComfyUI in Goat codebase
- [ ] Replace with `mmf` CLI equivalents
- [ ] Verify no direct HTTP patterns remain
- [ ] Update AGENTS.md if needed
- [ ] CI check added for HTTP patterns

**Labels:** `infra`, `comfyui`, `migration`, `blocked`

---

#### ROM-XXX: [MEDIUM] RobloxChristian - Remove Orphaned MCP Client
**Type:** Infrastructure
**Project:** Infra: ComfyUI MCP
**Priority:** Medium
**Estimate:** 2h

**Description:**
RobloxChristian has `comfyui_mcp_client.py` that is not used by the orchestrator. Orchestrator uses direct HTTP instead.

**Acceptance Criteria:**
- [ ] Decision: Use client or remove it
- [ ] If remove: Delete `comfyui_mcp_client.py`
- [ ] Refactor orchestrator to use `mmf` CLI
- [ ] Update AGENTS.md with correct pattern
- [ ] Verify Bible Stories generation works

**Labels:** `infra`, `comfyui`, `cleanup`

---

#### ROM-XXX: [LOW] Pokedex-Generator - Dependency Cleanup
**Type:** Infrastructure
**Project:** Infra: ComfyUI MCP
**Priority:** Low
**Estimate:** 30m

**Description:**
Remove unused torch/diffusers dependencies from pokedex-generator.

**Acceptance Criteria:**
- [ ] Audit requirements.txt
- [ ] Remove torch, diffusers, transformers if unused
- [ ] Verify mmf_client.py still works
- [ ] Test Shiny 151 pipeline

**Labels:** `infra`, `comfyui`, `cleanup`

---

#### ROM-XXX: [MEDIUM] ComfyUI MCP - Final Migration Cleanup
**Type:** Infrastructure
**Project:** Infra: ComfyUI MCP
**Priority:** Medium
**Estimate:** 2h

**Description:**
Final cleanup of main repository to complete MCP→CLI migration.

**Acceptance Criteria:**
- [ ] Remove deprecated MCP tools (mark as legacy)
- [ ] Update all documentation to CLI-first
- [ ] Template drift detection validation
- [ ] Final test of all 34 templates

**Labels:** `infra`, `comfyui`, `migration`

---

#### ROM-XXX: [HIGH] Standardize CLI Patterns Across Ecosystem
**Type:** Infrastructure
**Project:** Infra: ComfyUI MCP
**Priority:** High
**Estimate:** 4h

**Description:**
Document and enforce canonical CLI pattern across all 4 downstream repos.

**Acceptance Criteria:**
- [ ] Document KDH pattern as canonical in AGENTS.md
- [ ] Create migration guide for remaining repos
- [ ] CI checks for standardization
- [ ] Update all repo documentation

**Labels:** `infra`, `comfyui`, `documentation`, `standards`

---

#### ROM-XXX: [MEDIUM] Agentic ComfyUI - Prompt Enhancement Pipeline
**Type:** Infrastructure
**Project:** Infra: ComfyUI MCP
**Priority:** Medium
**Estimate:** 4h

**Description:**
Enable automatic prompt enhancement before generation for better results.

**Acceptance Criteria:**
- [ ] Integrate `mmf enhance` into generation pipeline
- [ ] Add VLM-based QA validation loop
- [ ] Document agentic usage patterns
- [ ] Test with overnight batches

**Labels:** `infra`, `comfyui`, `agentic`, `enhancement`

---

#### ROM-XXX: [MEDIUM] ComfyUI Optimizations - FP8/NVFP4 Adoption
**Type:** Infrastructure
**Project:** Infra: ComfyUI MCP
**Priority:** Medium
**Estimate:** 3h

**Description:**
Update templates and CLI to automatically use optimal precision for detected GPU.

**Acceptance Criteria:**
- [ ] Auto-detect GPU architecture
- [ ] Select optimal precision (NVFP4/FP8/FP16)
- [ ] Update all 34 templates with precision hints
- [ ] Benchmark and document improvements

**Labels:** `infra`, `comfyui`, `optimization`

---

## Section 7: Success Metrics

**Completion Criteria:**
- [ ] 4/4 repos using consistent CLI pattern
- [ ] 0 direct HTTP calls to ComfyUI (except mmf itself)
- [ ] All 34 templates passing validation
- [ ] Documentation synchronized across repos
- [ ] 1.5x speedup from FP8 adoption on compatible GPUs

**Monitoring:**
- Track CLI usage vs HTTP usage per repo (grep patterns)
- Monitor generation success rates
- Track template drift detection hits
- Measure queue throughput

---

## Appendix A: Quick Reference

### Migration Checklist per Repository

```
□ Search for HTTP calls to ComfyUI
□ Identify all generation entry points
□ Replace with mmf CLI equivalents
□ Update error handling
□ Update documentation
□ Add CI check for HTTP patterns
□ Test end-to-end generation
□ Commit and push
```

### Common CLI Patterns

```bash
# Single generation
mmf run --model flux --type t2i --prompt "..." -o output.png

# Batch variations
mmf batch seeds workflow.json --count 8 --start-seed 42

# Pipeline
mmf pipeline viral-short --prompt "..." --style-image style.png -o video.mp4

# QA
mmf qa --asset ASSET_ID --criteria "check_prompt_adherence"

# System info
mmf stats --pretty
mmf models constraints wan --pretty
```

---

**Document Version:** 1.0
**Next Review:** After Goat migration completion
**Owner:** Infra: ComfyUI MCP Project
