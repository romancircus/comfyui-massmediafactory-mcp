# Deep Planning Blueprint: Agent-Native ComfyUI for Downstream Repos (2026-02-11)

## 1) Your Core Problem (Restated Clearly)

You are not trying to build "more CLI commands."

You are trying to let coding agents in any repo take a directive like:
- "make a 30s viral short",
- "make an 8 minute story music video",
- "keep character consistency in 3D style",

and reliably turn it into ComfyUI workflows that are:
- model-correct (Wan/LTX/Qwen-specific),
- hardware-correct (RTX 5090 32GB constraints),
- quality-correct (style, motion, identity consistency).

## 2) The Difference You Asked About

The difference is:

- **Old framing**: "CLI/MCP architecture problem" (tool plumbing)
- **Real framing**: "Generation intelligence problem" (planning + policy + recipes)

CLI plumbing matters, but it is no longer the main bottleneck.
The bottleneck is that agents still have to guess:
- model selection,
- prompt style by stage,
- exact params per model family,
- fallback strategy when quality fails.

## 3) What I Found in This Repo (Current State)

### 3.1 Command Surface Is Not the Main Problem

- Top-level CLI commands: **30** (`mmf --help`)
- Total parser registrations across command files: **47**

This is not "100+ core CLI commands."  
The "100+" feeling comes from:
- downstream wrappers,
- historical docs,
- repo-specific helper functions layered on top.

### 3.2 Real Contract Problems (Confirmed)

1. **Exit-code mismatch for automation**  
Some commands return JSON errors but exit code `0`.
- Reproduced: `mmf stats` returned error payload with `EXIT:0`.
- Reproduced: `mmf models list` returned errors with `EXIT:0`.

2. **Template discovery misses nested templates**  
Loader/listing currently uses top-level `glob("*.json")`.
- `pony/*.json` exists on disk but does not appear in `templates list`.

3. **Downstream wrapper drift**
- `pokedex-generator` calls `batch seeds --template ... --params ...`
- Current CLI expects workflow file/stdin for `batch seeds`
- Reproduced parser failure (`unrecognized arguments`) for wrapper shape.

4. **Doc drift**
- README/docs template counts differ from actual repository files and behavior.

### 3.3 Downstream Reality (KDH + Pokedex)

Both repos are already modeling different production needs:

- **KDH**: mixed pipelines (`T2I -> I2V`, `T2V`, programmatic stitching), style families (`photorealistic`, `3d_pixar`), concept-level orchestration.
- **Pokedex**: bio image generation + shiny edits + video animation + audio, with multiple model paths and heavy wrapper logic.

This confirms your objective is valid: one base platform must serve very different content programs.

## 4) External Research: What It Implies for Design

## 4.1 ComfyUI Is a Strong Runtime Primitive Layer

ComfyUI server routes provide what we need for execution backplane:
- queue/submit (`/prompt`)
- websocket progress (`/ws`)
- node schema (`/object_info`)
- outputs/history (`/history`, `/view`)

Implication: do not reinvent runtime; build intelligence above it.

## 4.2 Model Families Require Distinct Policy

- **LTX**: resolution divisible by 32, frame count `8n+1`, distilled/dev behavior differs.
- **Wan 2.2**: multiple task families (T2V/I2V/TI2V/S2V/Animate), variant-specific behavior.
- **Qwen-Image**: aspect ratio presets and stable defaults (`num_inference_steps`, `true_cfg_scale`) matter.

Implication: one generic prompt/param path will keep failing.

## 4.3 Good CLI Pattern (gogcli)

Useful pattern from `gogcli`:
- command groups with clear discoverability,
- consistent UX for automation,
- machine-friendly output modes.

Implication: keep CLI stable, predictable, and narrow at top level.

## 4.4 Other Agentic Comfy Attempts

Projects like VibeComfy and other "ComfyUI for agents" efforts converge on:
- workflow registry,
- prompt-driven generation,
- CLI/MCP bridge.

Implication: direction is correct, but robust policy/recipe intelligence is usually the missing piece.

## 5) Architecture Decision (Recommended)

## 5.1 Hybrid Migration (Not Big-Bang Rewrite)

Build a new agent-native core while preserving existing CLI compatibility:

- Keep current CLI working for downstream stability.
- Add a new planner/policy/recipe layer.
- Migrate KDH and Pokedex to new plan-based API incrementally.

This avoids production breakage while fixing architecture debt.

## 5.2 Three-Layer Agentic Stack

1. **Intent + Planning layer (MCP / planner API)**
- Input: goal-level directive (duration/style/consistency/hardware/time budget)
- Output: staged plan JSON

2. **Policy + Recipe Intelligence layer (new core)**
- Model/task/style policy packs
- Prompt policies by stage
- Fallback/retry strategy by failure class
- Hardware tiering (5090 32GB)

3. **Deterministic runtime layer (CLI + ComfyUI routes)**
- Execute stage plan
- Block/wait/progress/download
- Strict exit codes

## 6) Command Strategy (What to Keep vs Add)

Do not keep expanding one-off commands.

### Stable Agent-Core Commands

- `mmf plan create`
- `mmf plan validate`
- `mmf plan explain`
- `mmf run plan`
- `mmf run stage`
- `mmf qa run`
- `mmf artifacts pull`

Keep old commands as compatibility aliases during migration.

## 7) Knowledge Enrichment Strategy (Critical)

To solve "agents don't know model nuance," build explicit policy registries:

- `policies/models/*.yaml`  
  Model constraints + defaults + forbidden patterns

- `policies/tasks/*.yaml`  
  Task recipes (t2i, i2v, t2v, edit, long-form extension, etc.)

- `policies/styles/*.yaml`  
  Style packs (`photoreal`, `3d_story`, etc.) with prompt templates

- `policies/hardware/*.yaml`  
  5090 tiered fit/perf policy (`tier_a`, `tier_b`, `blocked`)

- `policies/recipes/*.yaml`  
  Multi-stage graphs for known production goals (KDH/Pokedex first)

Then add:
- prompt linter by stage/model,
- parameter linter by model/task,
- QA classifier + targeted replan.

## 8) Proposed First 90 Days

## Phase 0 (Week 1-2): Contract Integrity
- Fix exit codes to match error states.
- Fix nested template discovery policy (or enforce flat policy with validation).
- Publish machine-readable CLI contract for wrapper generation.
- Add cross-repo contract tests for KDH + Pokedex adapters.

## Phase 1 (Week 2-4): Policy Backbone
- Implement policy schemas (`model/task/style/hardware/recipe`).
- Seed with Wan/LTX/Qwen + KDH/Pokedex real recipes.
- Implement `plan create` + `plan validate`.

## Phase 2 (Week 4-8): Plan Compiler + Runtime
- Compile plan JSON to workflow stages.
- Execute stages with queue-safe blocking.
- Capture full provenance: model/template/params/prompt/fallback decisions.

## Phase 3 (Week 8-12): QA + Auto-Repair
- Add quality checks: motion/style/identity/prompt adherence.
- Route failures to targeted retries (not full rerun).
- Add long-form recipes (story/music) with scene-level continuity control.

## 9) Immediate Next Implementation Step

Build a thin vertical slice now:

1. `plan.create` for two recipes:
- `viral_short_photoreal_v1`
- `story_music_longform_3d_v1`

2. `run plan` execution path with current runtime.

3. KDH and Pokedex adapter prototype against this plan API.

This will test the architecture on real workloads before broader migration.

## 10) Sources

- ComfyUI server routes:
  - https://docs.comfy.org/development/comfyui-server/comms_routes
- ComfyUI workflow templates:
  - https://docs.comfy.org/custom-nodes/workflow_templates
- ComfyUI template feature:
  - https://docs.comfy.org/interface/features/template
- ComfyUI Wan2.2 tutorial:
  - https://docs.comfy.org/tutorials/video/wan/wan2_2
- ComfyUI LTX-2 tutorial:
  - https://docs.comfy.org/tutorials/video/ltx/ltx-2
- ComfyUI Qwen-Image tutorial:
  - https://docs.comfy.org/tutorials/image/qwen/qwen-image
- Wan2.2 official repo:
  - https://github.com/Wan-Video/Wan2.2
- LTX model card / ecosystem:
  - https://huggingface.co/Lightricks/LTX-Video
  - https://github.com/Lightricks/ComfyUI-LTXVideo
- Qwen-Image official repo:
  - https://github.com/QwenLM/Qwen-Image
- gogcli reference:
  - https://github.com/steipete/gogcli
  - https://gogcli.sh/
- VibeComfy reference:
  - https://github.com/peteromallet/VibeComfy
