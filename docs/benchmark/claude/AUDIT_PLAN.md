# Codebase Audit Plan

**Date:** February 2026
**Auditor:** Claude Opus 4.5
**Repository:** comfyui-massmediafactory-mcp
**Baseline Review:** January 2026 (9/10 rating)

---

## Audit Objectives

1. **Verify prior review findings** - Confirm all critical issues from January 2026 review are resolved
2. **Identify new gaps** - Find issues introduced since the baseline review
3. **Assess token reduction progress** - Evaluate progress on the 77% token reduction plan
4. **Test coverage analysis** - Map untested code paths
5. **Security review** - Check for injection vulnerabilities, unsafe patterns
6. **Performance bottlenecks** - Identify slow operations

---

## Phase 1: Architecture Review (2 hours)

### 1.1 Core Module Audit

| File | Purpose | Lines | Priority | Time |
|------|---------|-------|----------|------|
| `server.py` | MCP tool registration, main entry | ~500 | HIGH | 30m |
| `client.py` | ComfyUI HTTP client, retry logic | ~200 | HIGH | 20m |
| `execution.py` | Workflow execution, polling | ~250 | HIGH | 20m |
| `mcp_utils.py` | MCP response formatting, pagination | ~200 | MEDIUM | 15m |

**Checklist:**
- [ ] Verify all 48 tools are registered correctly
- [ ] Check error handling consistency (isError flag)
- [ ] Validate rate limiting implementation
- [ ] Review retry logic with exponential backoff
- [ ] Check pagination cursor implementation
- [ ] Verify correlation ID logging

### 1.2 Validation Module Audit

| File | Purpose | Lines | Priority | Time |
|------|---------|-------|----------|------|
| `validation.py` | Parameter/connection validation | ~300 | HIGH | 25m |
| `topology_validator.py` | Topology checks, model detection | ~460 | HIGH | 30m |
| `workflow_generator.py` | Meta-template workflow generation | ~460 | HIGH | 30m |

**Checklist:**
- [ ] Verify cycle detection algorithm correctness
- [ ] Check resolution compatibility logic for all models
- [ ] Review connection type wildcard handling
- [ ] Audit edge case handling (empty prompt, zero values)
- [ ] Validate model alias resolution
- [ ] Check auto-correction behavior

---

## Phase 2: Feature Module Audit (2 hours)

### 2.1 Asset Management

| File | Purpose | Lines | Priority | Time |
|------|---------|-------|----------|------|
| `assets.py` | Asset registry, TTL expiration | ~200 | MEDIUM | 15m |
| `publish.py` | Web directory export | ~150 | LOW | 10m |

**Checklist:**
- [ ] Verify TTL expiration logic
- [ ] Check deduplication by content hash
- [ ] Review file path sanitization (security)
- [ ] Audit publish_dir validation

### 2.2 Batch & Pipeline

| File | Purpose | Lines | Priority | Time |
|------|---------|-------|----------|------|
| `batch.py` | Batch execution, sweeps | ~250 | MEDIUM | 20m |
| `pipeline.py` | Multi-stage pipelines | ~200 | MEDIUM | 15m |

**Checklist:**
- [ ] Verify parallel execution safety
- [ ] Check parameter sweep cartesian product
- [ ] Review stage dependency resolution
- [ ] Audit VRAM resource management

### 2.3 Discovery & Models

| File | Purpose | Lines | Priority | Time |
|------|---------|-------|----------|------|
| `discovery.py` | Model listing, node search | ~200 | MEDIUM | 15m |
| `models.py` | Civitai search, model download | ~200 | MEDIUM | 15m |
| `vram.py` | VRAM estimation | ~150 | LOW | 10m |

**Checklist:**
- [ ] Review model path validation
- [ ] Check download security (URL validation)
- [ ] Verify VRAM estimation accuracy
- [ ] Audit Civitai API error handling

### 2.4 Quality & Analysis

| File | Purpose | Lines | Priority | Time |
|------|---------|-------|----------|------|
| `qa.py` | VLM-based QA via Ollama | ~200 | MEDIUM | 15m |
| `analysis.py` | Image/video metadata | ~150 | LOW | 10m |

**Checklist:**
- [ ] Review VLM prompt injection risks
- [ ] Check Ollama connection handling
- [ ] Verify image dimension extraction

---

## Phase 3: Support Module Audit (1.5 hours)

### 3.1 Documentation & Reference

| File | Purpose | Lines | Priority | Time |
|------|---------|-------|----------|------|
| `patterns.py` | Workflow patterns by model | ~250 | MEDIUM | 15m |
| `node_specs.py` | Node input/output specs | ~430 | MEDIUM | 20m |
| `reference_docs.py` | Documentation access | ~200 | LOW | 10m |

**Checklist:**
- [ ] Verify pattern accuracy against ComfyUI
- [ ] Check node spec completeness
- [ ] Review search functionality

### 3.2 Style Learning

| File | Purpose | Lines | Priority | Time |
|------|---------|-------|----------|------|
| `style_learning.py` | Generation logging, suggestions | ~300 | LOW | 20m |

**Checklist:**
- [ ] Review data persistence mechanism
- [ ] Check rating aggregation logic
- [ ] Verify suggestion algorithm

### 3.3 Templates & Persistence

| File | Purpose | Lines | Priority | Time |
|------|---------|-------|----------|------|
| `templates/__init__.py` | Template loading | ~100 | MEDIUM | 10m |
| `persistence.py` | Workflow save/load/convert | ~250 | MEDIUM | 15m |

**Checklist:**
- [ ] Verify UI ↔ API format conversion accuracy
- [ ] Check workflow format detection
- [ ] Review import security (arbitrary file paths)

---

## Phase 4: Test Coverage Analysis (1 hour)

### 4.1 Current Test Inventory

| Test File | Coverage | Gaps |
|-----------|----------|------|
| `test_workflow_generator.py` | Edge cases, aliases, injection | Integration tests |
| `test_topology_validator.py` | Frame/resolution/CFG validation | Connection types |
| `test_patterns.py` | Pattern loading | Pattern accuracy |

### 4.2 Missing Test Coverage

| Module | Missing Tests | Priority |
|--------|---------------|----------|
| `client.py` | HTTP retry, upload/download | HIGH |
| `execution.py` | execute_workflow, regenerate | HIGH |
| `batch.py` | Parallel execution, sweeps | MEDIUM |
| `pipeline.py` | Multi-stage pipelines | MEDIUM |
| `assets.py` | TTL expiration, dedup | MEDIUM |
| `qa.py` | VLM integration | LOW |
| `style_learning.py` | All functions | LOW |

### 4.3 Integration Test Plan

**Required integration tests:**
1. Full workflow: generate → execute → wait → regenerate
2. Batch sweep with 4+ variations
3. Image-to-video pipeline
4. Asset lifecycle: create → list → view → cleanup
5. Model upload → use in workflow → download output

**Time estimate:** 2 hours to write, requires running ComfyUI

---

## Phase 5: Security Audit (1 hour)

### 5.1 Injection Vectors

| Area | Risk | Check |
|------|------|-------|
| File paths | Path traversal | `upload_image()`, `download_output()`, `publish_asset()` |
| Workflow JSON | Arbitrary code exec | Template injection, node class_type validation |
| Civitai URLs | SSRF | `download_model()` URL validation |
| Ollama prompts | Prompt injection | `qa_output()` prompt sanitization |

### 5.2 Resource Exhaustion

| Area | Risk | Mitigation |
|------|------|------------|
| Batch execution | DoS via large param sets | Check batch limit enforcement |
| Timeout handling | Hung connections | Verify timeout_seconds enforcement |
| Asset storage | Disk exhaustion | Check TTL cleanup frequency |

### 5.3 Authentication

| Area | Status | Check |
|------|--------|-------|
| ComfyUI connection | No auth | Verify COMFYUI_URL validation |
| Civitai API | API key optional | Check credential handling |
| Ollama | No auth | Local-only assumption valid? |

---

## Phase 6: Performance Analysis (30 min)

### 6.1 Hot Paths

| Operation | Expected Time | Check |
|-----------|---------------|-------|
| `list_models()` | <500ms | Cache status |
| `validate_workflow()` | <100ms | Complexity |
| `generate_workflow()` | <50ms | Skeleton cache |
| `load_skeleton()` | <10ms | File I/O caching |

### 6.2 Memory Usage

| Concern | Check |
|---------|-------|
| Asset registry growth | Memory-bound or disk-based? |
| Skeleton cache | Unbounded growth? |
| Style learning data | Persistence mechanism |

---

## Phase 7: Token Reduction Verification (30 min)

### 7.1 Current vs Target

| Metric | January 2026 | Target | Audit Task |
|--------|--------------|--------|------------|
| Tool count | 100 | 32 | Count actual tools |
| Token usage | 23,462 | <10,000 | Measure via Claude Code |
| Docstring length | ~180 tokens/tool | ~30 tokens/tool | Sample 10 tools |

### 7.2 Phase Completion Status

| Phase | Description | Status to Verify |
|-------|-------------|------------------|
| Phase 1 | Consolidate discovery (5→1) | Check `list_models()` exists |
| Phase 2 | Reference tools → Resources | Count MCP resources |
| Phase 3 | Consolidate style learning (10→4) | Check tool names |
| Phase 4 | Minimize docstrings | Sample docstring lengths |
| Phase 5 | Update annotations | Check annotations.py |
| Phase 6 | Workflow library consolidation | Optional |

---

## Deliverables

1. **AUDIT_FINDINGS.md** - Issues found with severity ratings
2. **TEST_COVERAGE_REPORT.md** - Coverage gaps with priority
3. **SECURITY_REPORT.md** - Vulnerability assessment
4. **TOKEN_REDUCTION_STATUS.md** - Progress measurement

---

## Time Summary

| Phase | Description | Time |
|-------|-------------|------|
| Phase 1 | Architecture Review | 2h |
| Phase 2 | Feature Module Audit | 2h |
| Phase 3 | Support Module Audit | 1.5h |
| Phase 4 | Test Coverage Analysis | 1h |
| Phase 5 | Security Audit | 1h |
| Phase 6 | Performance Analysis | 0.5h |
| Phase 7 | Token Reduction Verification | 0.5h |
| **Total** | | **8.5h** |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Incomplete January fixes | Low | High | Verify each resolved item |
| New regression bugs | Medium | Medium | Run test suite first |
| Token reduction incomplete | Medium | Low | Measure actual context |
| Security gaps | Low | High | Focus on injection vectors |

---

## Pre-Audit Checklist

- [ ] Clone fresh repo (ensure no local modifications)
- [ ] Run `pip install -e ".[dev]"` to install dev dependencies
- [ ] Run `pytest tests/ -v` to establish baseline
- [ ] Start ComfyUI for integration testing
- [ ] Record initial token count via Claude Code context warnings
