# IMPROVEMENT_PLAN.md - Prioritized Improvements

**Author:** KIMI
**Repository:** comfyui-massmediafactory-mcp
**Date:** February 2026
**Total Effort:** 95 hours

---

## Priority Framework

- **P0 (Critical):** Must fix immediately - blocks usage or poses risk
- **P1 (High Value):** Significant user impact, clear ROI
- **P2 (Important):** Quality improvements, technical debt
- **P3 (Nice to Have):** Polish, future-proofing

---

## P0: Critical Issues (22 hours)

### P0-1: Type Safety Overhaul (8 hours)

**Problem:** 100+ type errors in codebase, risking runtime failures

**Current State:**
- `server.py`: 50+ type errors (None assigned to non-optional params)
- `workflow_generator.py`: 20+ type errors
- `patterns.py`: Return type mismatches

**Solution:**
```python
# Before (error-prone)
def regenerate(asset_id: str, prompt: str = None):  # type: ignore

# After (type-safe)
from typing import Optional
def regenerate(asset_id: str, prompt: Optional[str] = None) -> dict:
```

**Implementation:**
1. Add `Optional[]` to all optional parameters (3h)
2. Fix return type annotations (2h)
3. Add TypedDict for complex returns (2h)
4. Run mypy in CI (1h)

**Impact:** Prevents runtime crashes, improves IDE support

---

### P0-2: Centralize Model Constraints (6 hours)

**Problem:** Same constraints defined in 3+ places, risk of drift

**Current Duplication:**
- `patterns.py`: MODEL_CONSTRAINTS (303 lines)
- `topology_validator.py`: MODEL_CONSTRAINTS (duplicated)
- `workflow_generator.py`: MODEL_DEFAULTS (separate)

**Solution:**
```python
# New file: model_definitions.py
MODEL_REGISTRY = {
    "flux2": {
        "constraints": {...},  # From patterns.py
        "defaults": {...},     # From workflow_generator.py
        "skeleton": {...},     # Reference to patterns.py
    }
}
```

**Implementation:**
1. Create unified model registry (3h)
2. Refactor patterns.py to use registry (1.5h)
3. Refactor workflow_generator.py (1.5h)

**Impact:** Single source of truth, easier to add models

---

### P0-3: Fix Parameter Injection Vulnerability (4 hours)

**Problem:** Template injection could allow code execution

**Vulnerable Code:**
```python
# workflow_generator.py:362-370
workflow_str = json.dumps(workflow)
for param_name, param_value in params.items():
    placeholder = f"{{{{{param_name}}}}}"
    workflow_str = workflow_str.replace(...)  # No sanitization!
```

**Attack Vector:**
```python
params = {"PROMPT": "test"}}"}}; import os; os.system('rm -rf /')  #"
```

**Solution:**
1. Validate parameter names (alphanumeric only)
2. Escape parameter values before injection
3. Use JSON manipulation instead of string replacement

**Implementation:**
- Add parameter validation (2h)
- Implement safe injection (2h)

**Impact:** Prevents security exploits

---

### P0-4: Add Missing Error Handling (4 hours)

**Problem:** Many edge cases unhandled, causing cryptic failures

**Missing Handlers:**
| Function | Missing Case | Impact |
|----------|--------------|--------|
| `execute_workflow()` | ComfyUI 500 error | Crash |
| `upload_image()` | File not found | Unclear error |
| `download_model()` | Network timeout | Hangs forever |
| `wait_for_completion()` | Connection lost | Infinite loop |

**Implementation:**
1. Add try/except blocks with context (2h)
2. Create error translation layer (1.5h)
3. Add timeout handling (0.5h)

---

## P1: High Value Features (45 hours)

### P1-1: Smart Model Recommendation Engine (10 hours)

**Problem:** Users don't know which model to use

**Solution:** Context-aware recommendations

```python
def recommend_model(
    task_description: str,
    priority: str = "quality",  # quality|speed|balanced
    available_vram_gb: float = None,
    has_reference_image: bool = False
) -> dict:
    """
    Returns:
    {
        "recommended_model": "flux2",
        "confidence": 0.92,
        "reasoning": "Best for photorealistic portraits",
        "alternatives": [...],
        "estimated_time": "45s",
        "estimated_vram": "12GB"
    }
    """
```

**Features:**
- [ ] Natural language task parsing (3h)
- [ ] VRAM-aware filtering (2h)
- [ ] Quality/speed tradeoff options (2h)
- [ ] Learning from past generations (3h)

---

### P1-2: Interactive Progress Streaming (8 hours)

**Problem:** Long generations (video) have no progress visibility

**Solution:** WebSocket-based progress updates

```python
@mcp.tool()
async def execute_workflow_with_progress(workflow: dict) -> AsyncIterator[dict]:
    """Yields progress updates during execution."""
    yield {"status": "queued", "position": 3}
    yield {"status": "running", "step": 5, "total_steps": 30, "percent": 16}
    yield {"status": "completed", "outputs": [...]}
```

**Implementation:**
1. WebSocket client module (3h)
2. Async iterator wrapper (2h)
3. Progress parsing from ComfyUI (2h)
4. Client-side integration example (1h)

---

### P1-3: LoRA Management System (8 hours)

**Problem:** No support for LoRA (Low-Rank Adaptation) models

**Solution:** Full LoRA lifecycle management

```python
# Discovery
list_loras()  # Already exists, enhance

# Workflow integration
generate_with_lora(
    model="flux2",
    lora_paths=["loras/style_cyberpunk.safetensors"],
    lora_strengths=[0.8],
    prompt="cyberpunk portrait"
)

# Management
activate_lora(lora_path, trigger_word="cyberpunk")
download_lora_from_civitai(civitai_url)
```

**Implementation:**
1. LoRA node injection in workflows (3h)
2. Civitai LoRA search/download (2h)
3. Strength optimization suggestions (2h)
4. Documentation and examples (1h)

---

### P1-4: Batch Processing Dashboard (6 hours)

**Problem:** Batch jobs are fire-and-forget, no visibility

**Solution:** Job tracking and management

```python
# Submit batch
batch_id = submit_batch(workflow, parameter_sets=[...])

# Check status
get_batch_status(batch_id)
# Returns: {"completed": 45, "failed": 2, "pending": 3, "progress": 90}

# Get results
get_batch_results(batch_id)
# Returns: All outputs with metadata
```

**Features:**
- [ ] Persistent batch state (2h)
- [ ] Failure recovery and retry (2h)
- [ ] Result aggregation (1.5h)
- [ ] Export to CSV/JSON (0.5h)

---

### P1-5: Intelligent Prompt Enhancement (5 hours)

**Problem:** User prompts often produce suboptimal results

**Solution:** AI-powered prompt optimization

```python
enhance_prompt(
    prompt="a cat",
    model="flux2",
    style_target="photorealistic",
    enhance_level="moderate"  # subtle|moderate|aggressive
)
# Returns: "Professional pet photography of a fluffy domestic cat,
#           soft natural lighting, shallow depth of field, 85mm lens"
```

**Implementation:**
1. Prompt analysis and expansion (2h)
2. Model-specific keyword injection (1.5h)
3. Style database (1h)
4. User feedback loop (0.5h)

---

### P1-6: Workflow Diff & Versioning (8 hours)

**Problem:** Can't track workflow changes or compare versions

**Solution:** Git-like workflow versioning

```python
# Save version
workflow_library(action="commit", name="my_workflow", message="Added LoRA")

# View history
workflow_library(action="log", name="my_workflow")

# Compare versions
diff_workflows(version_a="v1", version_b="v2")
# Returns: {"added_nodes": [...], "removed_nodes": [...], "modified_params": {...}}
```

**Implementation:**
1. Version storage schema (2h)
2. Diff algorithm for workflows (3h)
3. UI for viewing changes (2h)
4. Rollback capability (1h)

---

## P2: Important Improvements (20 hours)

### P2-1: Skeleton Caching with Invalidation (3 hours)

**Problem:** File I/O on every workflow generation

**Current:**
```python
_SKELETON_CACHE = {}  # No TTL, no invalidation
```

**Improved:**
```python
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=32)
def load_skeleton(model: str, workflow_type: str) -> dict:
    cache_key = f"{model}:{workflow_type}"
    file_path = SKELETONS_DIR / f"{cache_key}.json"

    # Check file modification time
    mtime = file_path.stat().st_mtime
    # ... cache invalidation logic
```

---

### P2-2: Workflow Visualization (4 hours)

**Problem:** Hard to understand complex workflows

**Solution:** Generate visual diagrams

```python
def workflow_to_mermaid(workflow: dict) -> str:
    """Generate Mermaid flowchart for visualization."""
    # Returns: graph TD; 1[UNETLoader] --> 2[CLIPTextEncode]...

def workflow_to_graphviz(workflow: dict) -> str:
    """Generate DOT format for advanced visualization."""
```

**Features:**
- Mermaid diagrams (2h)
- Interactive HTML viewer (2h)

---

### P2-3: Comprehensive Logging (4 hours)

**Problem:** Debugging is difficult without structured logs

**Solution:** Structured JSON logging with correlation IDs

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "workflow_executed",
    prompt_id=prompt_id,
    model=model,
    duration_ms=45000,
    vram_peak_gb=14.2,
    correlation_id=correlation_id
)
```

**Implementation:**
1. Structured logging setup (1h)
2. Add correlation IDs (1h)
3. Instrument all major operations (1.5h)
4. Log aggregation setup (0.5h)

---

### P2-4: Configuration Management (3 hours)

**Problem:** Hardcoded paths and settings throughout codebase

**Solution:** Centralized config with environment overrides

```python
# config.py
from pydantic import BaseSettings

class Config(BaseSettings):
    comfyui_url: str = "http://localhost:8188"
    model_cache_dir: Path = Path("~/.cache/comfyui-models")
    max_concurrent_jobs: int = 3
    default_timeout_seconds: int = 600

    class Config:
        env_prefix = "COMFYUI_MCP_"
```

---

### P2-5: Test Coverage Expansion (6 hours)

**Current:** ~40% coverage
**Target:** 80% coverage

**Missing Tests:**
| Module | Coverage | Tests Needed |
|--------|----------|--------------|
| `client.py` | 20% | HTTP retry, error handling |
| `execution.py` | 30% | Full workflow lifecycle |
| `batch.py` | 25% | Parallel execution |
| `pipeline.py` | 15% | Multi-stage pipelines |
| `qa.py` | 10% | VLM integration |

---

## P3: Nice to Have (8 hours)

### P3-1: Plugin System (4 hours)

Allow third-party extensions:

```python
# Plugin API
class MCPPlugin:
    def register_tools(self, mcp_server):
        pass

    def register_resources(self, mcp_server):
        pass

# Load plugins
load_plugins_from_directory("~/.comfyui-mcp/plugins/")
```

### P3-2: Multi-GPU Support (2 hours)

Distribute workloads across multiple GPUs:

```python
execute_workflow(workflow, gpu_id=1)  # or "auto" for load balancing
```

### P3-3: Workflow Marketplace Integration (2 hours)

Import from popular workflow sites:

```python
import_workflow_from_url("https://civitai.com/workflows/1234")
```

---

## Implementation Roadmap

### Sprint 1: Foundation (Week 1-2) - 22 hours
- P0-1: Type safety overhaul
- P0-2: Centralize model constraints
- P0-3: Fix injection vulnerability
- P0-4: Error handling

### Sprint 2: Core Features (Week 3-5) - 28 hours
- P1-1: Model recommendation (partial)
- P1-3: LoRA management
- P1-5: Prompt enhancement
- P2-1: Skeleton caching

### Sprint 3: Experience (Week 6-7) - 20 hours
- P1-2: Progress streaming (partial)
- P1-4: Batch dashboard
- P2-2: Workflow visualization
- P2-3: Logging

### Sprint 4: Polish (Week 8) - 15 hours
- P1-6: Versioning
- P2-4: Configuration
- P2-5: Test coverage
- P3 items

---

## Success Metrics

| Metric | Current | 3-Month Target |
|--------|---------|----------------|
| Type errors | 100+ | 0 |
| Test coverage | 40% | 80% |
| User-reported bugs | 5/week | 1/week |
| Model support | 4 | 8 |
| Avg response time | - | <50ms |
| Documentation completeness | 70% | 95% |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Type changes break existing code | High | Medium | Gradual migration |
| Model constraint refactoring causes bugs | Medium | High | Comprehensive tests |
| Progress streaming adds complexity | Medium | Low | Feature flag |
| LoRA support requires major rework | Low | Medium | Research first |

---

## Dependencies

- **P0-1:** None
- **P0-2:** None
- **P0-3:** None
- **P0-4:** None
- **P1-1:** P0-2 (model registry)
- **P1-2:** WebSocket library
- **P1-3:** LoRA node research
- **P1-4:** P0-4 (error handling)
- **P1-5:** NLP library (optional)
- **P1-6:** None
- **P2 items:** Mostly independent
