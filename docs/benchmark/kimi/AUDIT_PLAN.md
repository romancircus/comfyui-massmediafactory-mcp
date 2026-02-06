# AUDIT_PLAN.md - Comprehensive Codebase Audit

**Auditor:** KIMI
**Repository:** comfyui-massmediafactory-mcp
**Date:** February 2026
**Estimated Time:** 10 hours

---

## Executive Summary

This audit plan focuses on identifying architectural debt, testing gaps, and optimization opportunities in the MCP server. Unlike traditional code reviews, this audit emphasizes **systemic patterns** that affect maintainability and extensibility.

---

## Phase 1: Architecture & Design Patterns (2.5 hours)

### 1.1 Module Dependency Analysis

**Files to Review:**
| File | Lines | Focus Area | Time |
|------|-------|------------|------|
| `server.py` | 930 | Tool registration, coupling | 30m |
| `workflow_generator.py` | 580 | Template system | 25m |
| `patterns.py` | 1275+ | Constraint definitions | 30m |
| `topology_validator.py` | 460 | Validation logic | 25m |

**Patterns to Identify:**
- [ ] **Circular dependencies** - Check imports between modules
- [ ] **God modules** - Modules with too many responsibilities
- [ ] **Duplicated logic** - Same constraints in multiple files
- [ ] **Inconsistent abstractions** - Mixed levels of abstraction

**Key Questions:**
1. Why are MODEL_CONSTRAINTS defined in both `patterns.py` and `topology_validator.py`?
2. Is the skeleton caching in `workflow_generator.py` actually effective?
3. Are the error handling patterns consistent across all modules?

### 1.2 API Surface Area Analysis

**Current State:** 48 tools exposed via MCP

**Audit Tasks:**
- [ ] Categorize tools by usage frequency (from CLAUDE.md examples)
- [ ] Identify "dead" tools never referenced in documentation
- [ ] Map tool dependencies (which tools call which)
- [ ] Check for tool overlap (multiple ways to do same thing)

**Expected Findings:**
- 20% of tools may be rarely used
- Several tools could be consolidated
- Resource vs Tool boundaries are unclear

### 1.3 Data Flow Analysis

**Trace these critical paths:**
```
1. generate_workflow() → load_skeleton() → expand_skeleton_to_workflow()
2. execute_workflow() → client.py → ComfyUI API
3. validate_workflow() → topology_validator.py → patterns.py
```

**Check for:**
- Unnecessary data transformations
- Missing validation at boundaries
- Error propagation consistency

---

## Phase 2: Code Quality & Maintainability (2 hours)

### 2.1 Technical Debt Inventory

**Debt Categories:**

| Category | Location | Severity | Effort to Fix |
|----------|----------|----------|---------------|
| **Magic numbers** | `patterns.py` (CFG values, resolutions) | Medium | 2h |
| **String literals** | Error messages, node types | Low | 1h |
| **Deep nesting** | `workflow_generator.py:267-372` | Medium | 3h |
| **Long functions** | `expand_skeleton_to_workflow()` | High | 4h |
| **Type inconsistency** | Returns dict vs TypedDict | Medium | 3h |

### 2.2 Documentation Quality

**Check these anti-patterns:**
- [ ] Docstrings that repeat function names
- [ ] Missing parameter documentation
- [ ] "TODO" or "FIXME" comments without issues
- [ ] Outdated examples in CLAUDE.md

**Files to Review:**
- `CLAUDE.md` - Check all code examples still work
- `README.md` - Verify tool counts match reality
- `docs/reference/*.md` - Check for stale information

### 2.3 Naming Conventions

**Inconsistencies to Find:**
- `workflow_type` vs `task` vs `type` parameter names
- `model` vs `model_name` vs `model_type`
- `prompt_id` vs `job_id` vs `execution_id`
- File naming: `snake_case.py` consistency

---

## Phase 3: Testing & Validation (2 hours)

### 3.1 Test Coverage Analysis

**Current Test Files:**
| Test File | Lines | Coverage | Gaps |
|-----------|-------|----------|------|
| `test_workflow_generator.py` | ~200 | Edge cases | Integration |
| `test_topology_validator.py` | ~150 | Validation | Connection types |
| `test_patterns.py` | ~100 | Loading | Pattern accuracy |

**Coverage Targets:**
- **Critical paths:** 90% (workflow generation, execution)
- **Validation logic:** 85% (topology, constraints)
- **Utility functions:** 60% (helpers, formatting)
- **Error handling:** 80% (edge cases, failures)

### 3.2 Test Quality Assessment

**Review each test for:**
- [ ] **Assertions** - Are they checking the right things?
- [ ] **Mocking** - Are external calls properly isolated?
- [ ] **Fixtures** - Is test data reusable?
- [ ] **Naming** - Do test names describe behavior?

**Red Flags:**
- Tests that pass when they should fail
- Tests with no assertions
- Tests that depend on execution order
- Tests that require external services

### 3.3 Integration Test Gaps

**Missing Integration Scenarios:**
1. Full workflow lifecycle (generate → execute → wait → cleanup)
2. Error recovery (ComfyUI disconnect, timeout)
3. Resource exhaustion (OOM, disk full)
4. Concurrent execution (multiple workflows)
5. Model switching (unload/load different models)

---

## Phase 4: Performance & Scalability (1.5 hours)

### 4.1 Performance Hotspots

**Profile These Operations:**
| Operation | Expected | Actual | Optimization |
|-----------|----------|--------|--------------|
| `generate_workflow()` | <50ms | ? | Skeleton caching |
| `validate_workflow()` | <100ms | ? | Topology checks |
| `list_models()` | <500ms | ? | File system cache |
| `execute_workflow()` | <1s | ? | HTTP overhead |

### 4.2 Memory Usage Patterns

**Potential Issues:**
- [ ] Skeleton cache growth (unbounded?)
- [ ] Asset registry memory footprint
- [ ] Workflow JSON size (large video workflows)
- [ ] String concatenation in loops

### 4.3 Scalability Limits

**Test These Scenarios:**
- 100+ workflows in library
- 1000+ assets in registry
- 50+ concurrent executions
- 100+ node workflow

---

## Phase 5: Security & Robustness (1.5 hours)

### 5.1 Input Validation Gaps

**Check These Inputs:**
| Function | Input | Current Validation | Gap |
|----------|-------|-------------------|-----|
| `upload_image()` | `image_path` | Basic existence | Path traversal |
| `download_model()` | `url` | None | SSRF, malicious URLs |
| `execute_workflow()` | `workflow` | Schema validation | Deep nesting |
| `create_workflow_from_template()` | `parameters` | Type checking | Injection |

### 5.2 Error Handling Robustness

**Test These Failure Modes:**
- [ ] ComfyUI returns 500 error
- [ ] Network timeout during upload
- [ ] Invalid JSON in workflow
- [ ] Missing required model file
- [ ] Disk full during asset save

### 5.3 Resource Leaks

**Check For:**
- [ ] Unclosed HTTP connections
- [ ] File handles left open
- [ ] Temporary files not cleaned
- [ ] Memory not released after execution

---

## Phase 6: Extensibility Analysis (1 hour)

### 6.1 Adding New Models - Current Process

**Steps Required:**
1. Add to `MODEL_CONSTRAINTS` in patterns.py
2. Add to `MODEL_DEFAULTS` in workflow_generator.py
3. Add skeleton to `WORKFLOW_SKELETONS`
4. Add node chain to `NODE_CHAINS`
5. Add template file
6. Update `MODEL_SKELETON_MAP`
7. Add tests

**Problems:**
- 7 different files to modify
- Risk of inconsistency
- No validation that all required pieces exist

### 6.2 Plugin Architecture Assessment

**Current Extensibility:**
- Templates: Good (file-based)
- Models: Poor (scattered definitions)
- Validation: Poor (hardcoded rules)
- Workflows: Good (JSON-based)

**Recommendations:**
- Centralize model definitions
- Create validation schema registry
- Support custom node types

---

## Deliverables

### 1. AUDIT_FINDINGS.md
- Categorized issues (Critical, High, Medium, Low)
- Specific file:line references
- Effort estimates for fixes

### 2. TECHNICAL_DEBT_REGISTER.md
- Debt items with interest rates (cost of not fixing)
- Prioritized repayment schedule
- Prevention strategies

### 3. ARCHITECTURE_IMPROVEMENTS.md
- Refactoring recommendations
- Module restructuring proposals
- Abstraction level adjustments

### 4. TEST_STRATEGY.md
- Coverage improvement plan
- Integration test scenarios
- Test data management strategy

---

## Time Breakdown

| Phase | Focus | Time |
|-------|-------|------|
| 1 | Architecture & Design | 2.5h |
| 2 | Code Quality | 2.0h |
| 3 | Testing | 2.0h |
| 4 | Performance | 1.5h |
| 5 | Security | 1.5h |
| 6 | Extensibility | 1.0h |
| **Total** | | **10.5h** |

---

## Tools & Commands

```bash
# Static analysis
pip install pylint mypy
pylint src/comfyui_massmediafactory_mcp/
mypy src/comfyui_massmediafactory_mcp/

# Test coverage
pip install pytest-cov
pytest --cov=src/comfyui_massmediafactory_mcp --cov-report=html

# Complexity analysis
pip install radon
radon cc src/comfyui_massmediafactory_mcp/ -a

# Dependency graph
pip install pydeps
pydeps src/comfyui_massmediafactory_mcp/server.py
```

---

## Success Criteria

- [ ] All critical paths identified and documented
- [ ] Test coverage gaps mapped with priorities
- [ ] Security vulnerabilities found and rated
- [ ] Performance bottlenecks identified
- [ ] Technical debt quantified
- [ ] Extensibility improvements proposed
