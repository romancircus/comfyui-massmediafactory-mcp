# Research Plan

**Date:** February 2026
**Researcher:** Claude Opus 4.5
**Repository:** comfyui-massmediafactory-mcp

---

## Research Objectives

1. **SOTA Model Investigation** - Ensure the MCP supports current generation models
2. **MCP Ecosystem Analysis** - Study best practices from other MCP servers
3. **Competing Tools Analysis** - Understand the ComfyUI automation landscape
4. **ComfyUI API Evolution** - Track breaking changes in ComfyUI
5. **User Pain Points** - Identify common failure modes and UX issues

---

## Section 1: SOTA Model Research (4 hours)

### 1.1 Image Generation Models

| Model | Release Date | Status in MCP | Research Needed |
|-------|--------------|---------------|-----------------|
| **Qwen-Image-2512** | Jan 2026 | Supported | Verify constraints correct |
| **Z-Image-Turbo** | Dec 2025 | Not supported | Add support? |
| **FLUX.2-dev** | Nov 2025 | Supported | Verify CFG/guidance settings |
| **Stable Diffusion 3.5** | Oct 2025 | Not supported | Add support? |
| **DALL-E 4** (API) | Jan 2026 | N/A | Not applicable (API only) |

**Research Tasks:**
- [ ] Query `mcp__sota-tracker__query_sota("image_gen")` for current rankings
- [ ] Compare MCP model list vs SOTA tracker
- [ ] Identify models with ComfyUI node support but missing MCP templates
- [ ] Research optimal settings for each model (CFG, sampler, scheduler)

**Time estimate:** 1.5 hours

### 1.2 Video Generation Models

| Model | Release Date | Status in MCP | Research Needed |
|-------|--------------|---------------|-----------------|
| **LTX-2** | Dec 2025 | Supported | Verify 19B vs 8B settings |
| **Wan 2.1** | Jan 2026 | Supported | Check I2V constraints |
| **HunyuanVideo 1.5** | Dec 2025 | Supported | Verify resolution limits |
| **CogVideoX-5B** | Sep 2024 | Not supported | Add support? |
| **AnimateDiff v3** | 2024 | Not supported | Add support? |
| **Kimi Video** | Jan 2026 | Unknown | Investigate availability |

**Research Tasks:**
- [ ] Query `mcp__sota-tracker__query_sota("video")` for current rankings
- [ ] Research frame count constraints for each model
- [ ] Document resolution/aspect ratio requirements
- [ ] Identify models with quality improvements over current support

**Time estimate:** 1.5 hours

### 1.3 Supporting Models (Upscalers, ControlNets)

| Category | Models to Research | Priority |
|----------|-------------------|----------|
| **Upscalers** | RealESRGAN 4x, ESRGAN, SwinIR | HIGH |
| **ControlNets** | ControlNet for FLUX, depth, canny, pose | MEDIUM |
| **Face Restoration** | CodeFormer, GFPGAN | MEDIUM |
| **IP-Adapter** | FLUX IP-Adapter, FaceID | HIGH |

**Research Tasks:**
- [ ] Identify which upscaler models have ComfyUI nodes
- [ ] Document ControlNet → Model compatibility matrix
- [ ] Research IP-Adapter for FLUX workflow patterns

**Time estimate:** 1 hour

---

## Section 2: MCP Ecosystem Research (3 hours)

### 2.1 Reference MCP Servers to Study

| Server | Features | Why Study |
|--------|----------|-----------|
| **filesystem MCP** | File operations | Pagination patterns |
| **puppeteer MCP** | Browser automation | Async task handling |
| **slack MCP** | Chat integration | Event notification patterns |
| **github MCP** | API integration | Error handling, rate limiting |
| **memory MCP** | Knowledge persistence | State management |

**Research Tasks:**
- [ ] Study pagination implementation in filesystem MCP
- [ ] Review error handling patterns in github MCP
- [ ] Analyze async task patterns in puppeteer MCP
- [ ] Study resource vs tool decisions in official MCPs

**Time estimate:** 1.5 hours

### 2.2 MCP Protocol Best Practices

| Area | Research Questions |
|------|-------------------|
| **Tool Design** | When to use consolidated vs separate tools? |
| **Resources** | When should data be a resource vs tool? |
| **Error Handling** | Best practices for isError, error codes? |
| **Pagination** | Cursor-based vs offset-based? Token limits? |
| **Progress** | How to report progress on long operations? |

**Research Tasks:**
- [ ] Read MCP specification for tool/resource guidelines
- [ ] Find token reduction strategies from community
- [ ] Research streaming response patterns
- [ ] Study progress notification patterns

**Time estimate:** 1 hour

### 2.3 MCP Server Performance

| Metric | Question |
|--------|----------|
| **Context size** | What's the ideal tool count? |
| **Response time** | Acceptable latency for tools? |
| **Batching** | Can MCP batch tool calls? |
| **Concurrency** | How do MCPs handle parallel tool calls? |

**Research Tasks:**
- [ ] Benchmark tool discovery latency
- [ ] Test parallel tool invocation support
- [ ] Measure context overhead at different tool counts

**Time estimate:** 0.5 hours

---

## Section 3: Competing Tools Analysis (2 hours)

### 3.1 ComfyUI Automation Tools

| Tool | Type | Strengths | Gaps vs MCP |
|------|------|-----------|-------------|
| **ComfyUI-Manager** | Web UI plugin | Node management, model download | Not programmable |
| **ComfyScript** | Python SDK | Type-safe workflow building | No MCP integration |
| **comfy-cli** | CLI tool | Headless execution | No AI assistant integration |
| **ComfyUI-API** | HTTP API | Direct workflow execution | No abstraction layer |

**Research Tasks:**
- [ ] Study ComfyScript's type-safe workflow builder
- [ ] Analyze comfy-cli's batch execution model
- [ ] Review ComfyUI-Manager's model catalog API

**Time estimate:** 1 hour

### 3.2 AI Image Generation Platforms

| Platform | Integration | What to Learn |
|----------|-------------|---------------|
| **Automatic1111 API** | HTTP | SD-native workflow patterns |
| **InvokeAI** | HTTP | Node-based workflow model |
| **Fooocus** | HTTP | Simplified prompt handling |
| **Replicate** | API | Hosted model abstraction |

**Research Tasks:**
- [ ] Compare workflow abstraction levels
- [ ] Study simplified prompt-to-image patterns
- [ ] Research model version management approaches

**Time estimate:** 1 hour

---

## Section 4: ComfyUI API Evolution (1.5 hours)

### 4.1 API Changelog Review

| Version | Key Changes | Impact on MCP |
|---------|-------------|---------------|
| **0.2.0** (Dec 2025) | New node system | Check node compatibility |
| **0.1.9** | WebSocket improvements | Consider WS support? |
| **0.1.8** | Upload API changes | Verify upload_image() |

**Research Tasks:**
- [ ] Review ComfyUI changelog since January 2026
- [ ] Identify deprecated endpoints
- [ ] Check for new /object_info fields
- [ ] Test current API against running ComfyUI

**Time estimate:** 0.5 hours

### 4.2 New Node Types

| Node Category | New Nodes | Supported? |
|---------------|-----------|------------|
| **FLUX.2** | FluxGuidance v2? | Check |
| **LTX-2** | LTXVLoader 19B | Verify |
| **Wan 2.1** | WanVideo new nodes | Verify |
| **HunyuanVideo** | HunyuanVideo v1.5 | Verify |

**Research Tasks:**
- [ ] Query `get_node_info()` for all supported models
- [ ] Compare node specs vs documentation
- [ ] Identify new nodes not in MCP patterns

**Time estimate:** 1 hour

---

## Section 5: User Pain Points Research (1.5 hours)

### 5.1 Common Failure Modes

| Failure | Root Cause | Research Needed |
|---------|------------|-----------------|
| **"Model not found"** | File path mismatch | Standardize model path detection |
| **"Out of memory"** | VRAM estimation wrong | Improve estimation accuracy |
| **"Connection refused"** | ComfyUI not running | Better connection handling |
| **"Invalid workflow"** | Schema validation | Improve error messages |
| **"Timeout"** | Long generation | Better progress reporting |

**Research Tasks:**
- [ ] Review GitHub issues for common errors
- [ ] Test error messages for clarity
- [ ] Identify confusing parameter names

**Time estimate:** 0.5 hours

### 5.2 UX Improvement Research

| Area | Pain Point | Research |
|------|------------|----------|
| **Workflow generation** | Which model for my task? | Recommendation engine |
| **Parameter tuning** | What CFG should I use? | Parameter guides per model |
| **Quality issues** | Why is my output blurry? | QA feedback integration |
| **Iteration** | How to improve this image? | Regeneration suggestions |

**Research Tasks:**
- [ ] Study how Midjourney guides users
- [ ] Research parameter recommendation systems
- [ ] Explore quality feedback loops

**Time estimate:** 1 hour

---

## Section 6: Documentation Research (1 hour)

### 6.1 ComfyUI Documentation Gaps

| Area | Current State | Research Needed |
|------|---------------|-----------------|
| **Node reference** | Partial in MCP | Compare to ComfyUI docs |
| **Model patterns** | 4 models documented | Add missing models |
| **Troubleshooting** | Minimal | Collect common issues |

**Research Tasks:**
- [ ] Review ComfyUI official docs
- [ ] Check community wikis (GitHub, Discord)
- [ ] Identify undocumented node behaviors

**Time estimate:** 0.5 hours

### 6.2 Agent Integration Guides

| Agent | Documentation | Needed |
|-------|---------------|--------|
| **Claude Code** | CLAUDE.md exists | Verify accuracy |
| **Cursor** | None | Consider adding |
| **Continue** | None | Consider adding |
| **Aider** | None | Consider adding |

**Research Tasks:**
- [ ] Test CLAUDE.md accuracy with Claude Code
- [ ] Research Cursor MCP integration
- [ ] Document agent-specific tips

**Time estimate:** 0.5 hours

---

## Research Deliverables

1. **SOTA_MODEL_REPORT.md** - Current model rankings, MCP support gaps
2. **MCP_BEST_PRACTICES.md** - Patterns from reference implementations
3. **COMPETITOR_ANALYSIS.md** - Feature comparison matrix
4. **API_COMPATIBILITY_REPORT.md** - ComfyUI version compatibility
5. **USER_PAIN_POINTS.md** - Prioritized UX issues

---

## Time Summary

| Section | Description | Time |
|---------|-------------|------|
| Section 1 | SOTA Model Research | 4h |
| Section 2 | MCP Ecosystem Research | 3h |
| Section 3 | Competing Tools Analysis | 2h |
| Section 4 | ComfyUI API Evolution | 1.5h |
| Section 5 | User Pain Points Research | 1.5h |
| Section 6 | Documentation Research | 1h |
| **Total** | | **13h** |

---

## Research Tools & Resources

### Tools to Use

| Tool | Purpose |
|------|---------|
| `mcp__sota-tracker__*` | Current SOTA rankings |
| `mcp__context7__query-docs` | Up-to-date library docs |
| `WebSearch` | Recent model announcements |
| `WebFetch` | Official documentation |
| ComfyUI `/object_info` | Live node specs |

### Key URLs to Review

| Resource | URL |
|----------|-----|
| ComfyUI GitHub | https://github.com/comfyanonymous/ComfyUI |
| ComfyUI Examples | https://github.com/comfyanonymous/ComfyUI_examples |
| MCP Specification | https://modelcontextprotocol.io/docs |
| MCP Servers Registry | https://github.com/modelcontextprotocol/servers |
| Civitai API Docs | https://wiki.civitai.com/wiki/Civitai_REST_API_Reference |

### Community Resources

| Platform | Channel | Purpose |
|----------|---------|---------|
| Discord | ComfyUI server | Real user issues |
| Reddit | r/comfyui | Common questions |
| GitHub Issues | ComfyUI repo | Bug reports, feature requests |
| Civitai | Model pages | Model-specific settings |

---

## Research Priority Matrix

| Research Area | Value | Effort | Priority |
|---------------|-------|--------|----------|
| SOTA video models | High | Medium | P0 |
| MCP token reduction patterns | High | Low | P0 |
| New image models | Medium | Medium | P1 |
| ComfyUI API changes | Medium | Low | P1 |
| Competing tools | Low | High | P2 |
| Documentation gaps | Low | Low | P2 |
