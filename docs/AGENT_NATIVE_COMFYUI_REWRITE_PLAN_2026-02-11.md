# Agent-Native ComfyUI Plan (Rewrite vs Refactor) — 2026-02-11

## Plain-English Problem

You are not trying to build “a CLI with lots of commands.”

You are trying to build:
- a system where an agent can take a high-level directive,
- choose the right model(s) and workflow strategy for that directive,
- stay within 32GB RTX 5090 constraints,
- execute reliably through ComfyUI,
- and iterate toward quality with minimal human intervention.

That is an **autonomous generation system problem**, not only a command-surface problem.

## The Real Gap

Current systems (including this repo) are mostly:
- command wrappers,
- workflow/template runners,
- utility tooling.

What is missing is a first-class **decision layer**:
- Which model family to use for this concept?
- Which chain of stages (T2I -> style -> I2V? TI2V direct? T2V + control?)
- What prompts/settings are valid for that model and stage?
- What fallback path to use when generation fails or quality is poor?

Without this, even perfect CLI plumbing still leaves agents guessing.

## Research Highlights (Source-backed)

## ComfyUI fundamentals

- ComfyUI exposes stable queue/history/ws/object_info routes and workflow template endpoints.
- This is enough to build robust planning+execution on top.

Source:
- https://docs.comfy.org/development/comfyui-server/comms_routes
- https://docs.comfy.org/custom-nodes/workflow_templates
- https://docs.comfy.org/interface/features/template

## WAN 2.2

- Distinct model/task paths matter (T2V-A14B, I2V-A14B, TI2V-5B, S2V-14B, Animate-14B).
- Official docs show TI2V-5B can run at 720p on consumer GPUs (24GB-class), while A14B examples are much heavier.
- Prompt extension is explicitly recommended in official examples.

Source:
- https://github.com/Wan-Video/Wan2.2
- https://docs.comfy.org/tutorials/video/wan/wan2_2

## LTX / LTX-2

- Hard constraints: width/height divisible by 32, frames follow 8n+1 rule.
- Distilled workflows are faster and use fewer steps.
- LTX-2 is integrated into ComfyUI core; control/upscaler/IC-LoRA ecosystem is active.

Source:
- https://huggingface.co/Lightricks/LTX-Video
- https://docs.comfy.org/tutorials/video/ltx/ltx-2
- https://github.com/Lightricks/ComfyUI-LTXVideo

## Qwen-Image

- Official examples repeatedly use clear defaults (e.g., `num_inference_steps=50`, `true_cfg_scale=4.0`) and fixed aspect-ratio presets.
- Editing variants have meaningful differences (e.g., better consistency in newer edit variants).
- ComfyUI docs include native workflow guidance and measured VRAM timings on 24GB class GPUs.

Source:
- https://github.com/QwenLM/Qwen-Image
- https://docs.comfy.org/tutorials/image/qwen/qwen-image
- https://blog.comfy.org/p/qwen-image-in-comfyui-new-era-of

## External comparable attempts

- `VibeComfy` positions itself as CLI+MCP for workflow discovery/editing/submission through Claude Code.
- Signal: market demand exists for “agent-assisted workflow manipulation.”
- Risk: many such repos focus on wrappers and prompt UX, not robust model-policy intelligence.

Source:
- https://github.com/peteromallet/VibeComfy

## Decision: Rewrite, Refactor, or Hybrid?

## Option A: Full Rewrite

Pros:
- Clean design, no migration baggage.
- Easier to enforce one architecture and contracts.

Cons:
- Highest delivery risk/time.
- Breaks downstream repos unless adapters are rebuilt fast.

## Option B: Incremental Refactor

Pros:
- Lowest disruption.
- Existing commands/templates keep working.

Cons:
- Legacy complexity remains.
- Harder to enforce a strict architecture boundary.

## Option C: Hybrid (Recommended)

Approach:
- Build a new “agent-native core” inside this repo (or sibling package).
- Keep current CLI as compatibility shell.
- Migrate downstream repos to the new core plan API progressively.

Why this is best:
- You get clean architecture without freezing production.
- KDH and pokedex can migrate by adapter, not big-bang rewrite.

## Target Architecture (Agent-Native)

## Layer 1: Intent API (new)

Input is a high-level goal spec, not raw ComfyUI params:

```json
{
  "goal": "30-second photoreal viral short about father/son baseball",
  "format": "vertical_9_16",
  "duration_sec": 30,
  "style": "photoreal",
  "consistency": { "character": "high", "style": "high" },
  "hardware": { "gpu": "rtx_5090", "vram_gb": 32 },
  "constraints": { "max_render_minutes": 45 }
}
```

## Layer 2: Capability/Constraint Registry (new)

Machine-readable registry with:
- model capabilities by task (t2i/i2v/t2v/ti2v/s2v/edit),
- validated parameter ranges,
- memory/runtime profiles by hardware tier,
- incompatibility flags.

This is where 32GB RTX 5090 policy lives.

## Layer 3: Recipe + Policy Engine (new, most important)

Recipes are reusable stage graphs:
- `viral_short_photoreal_v1`
- `story_music_8min_character_v2`
- `qwen_keyframe_then_wan_i2v_v3`

Policy resolves per-stage settings:
- model variant selection,
- prompt template style,
- sampler/cfg/steps/frame/resolution defaults,
- fallback rules.

## Layer 4: Workflow Compiler (new)

Compiles recipe stages into executable workflows by:
- selecting official/native template where possible,
- applying validated parameter injection,
- composing subgraphs when template is missing.

No direct node-guessing by agents.

## Layer 5: Runtime (existing + tighten)

Keep current strengths:
- queue submission,
- blocking waits,
- artifact download,
- retries and structured exit codes.

## Layer 6: QA + Auto-Repair Loop (new)

After each stage:
- run quality checks (prompt adherence, identity/style/motion consistency),
- classify failure type,
- replan or retune only what is needed.

## CLI/MCP Roles in This Design

## MCP = Planner Brain

- `plan.create`
- `plan.validate`
- `plan.explain` (why this model/recipe?)
- `capabilities.search`
- `constraints.resolve`

## CLI = Deterministic Executor

- `run plan.json`
- `run stage stage.json`
- `wait`, `status`, `progress`
- `artifacts pull`
- `qa run`

Keep CLI surface narrow and stable.

## What “Success” Looks Like

Agent prompt:
- “Create a new 8-minute 3D story music video with consistent characters.”

System behavior:
1. Planner decomposes into shot/stage plan.
2. Policy picks model recipe chain under 32GB constraints.
3. Compiler builds workflows per stage.
4. Runtime executes with queue-safe blocking.
5. QA loop catches drift and triggers targeted regenerate.
6. Outputs final sequence + provenance of model/settings.

## 32GB RTX 5090 Practical Strategy

Use profile tiers:
- **Tier A (preferred):** models/workflows known to fit with margin.
- **Tier B (allowed):** fit with offload/quantization and slower throughput.
- **Tier C (blocked):** skip unless explicit override.

Initial guidance:
- Qwen fp8 image paths: Tier A.
- LTX-2 fp8/distilled and low-vram loaders: Tier A/B depending on recipe.
- Wan2.2 TI2V-5B: Tier A for 720p-style use.
- Wan2.2 A14B: Tier B/C depending on resolution/offload and throughput goals.

This should be benchmarked and codified as policy, not handled ad hoc.

## Build Plan (90-day)

## Phase 0 (Week 1-2): Contract Hardening

- Fix CLI exit-code correctness.
- Freeze stable command contract and publish JSON spec.
- Add cross-repo adapter conformance tests (KDH, pokedex).

## Phase 1 (Week 2-4): Policy Backbone

- Define schema for capability/constraint/recipe/prompt policy.
- Seed with WAN/LTX/Qwen core recipes.
- Implement planner that outputs staged execution plan JSON.

## Phase 2 (Week 4-8): Compiler + Runtime Integration

- Compile plan -> workflow instances.
- Execute via existing runtime path.
- Add provenance logging per run.

## Phase 3 (Week 8-12): QA/Autotune + Long-form

- Implement stage QA + auto-repair.
- Add long-form narrative recipe packs (music/story mode).
- Add style packs (photoreal, 3D storytelling) and consistency policies.

## Immediate Next Move

If approved, start with a hybrid migration track:
1. Implement `plan.create` + `run plan` prototype for two recipes:
   - `viral_short_photoreal`
   - `story_music_longform_3d`
2. Keep existing CLI commands working.
3. Move KDH first, then pokedex, to prove cross-repo portability.

## Sources

- ComfyUI routes:
  - https://docs.comfy.org/development/comfyui-server/comms_routes
- ComfyUI template systems:
  - https://docs.comfy.org/custom-nodes/workflow_templates
  - https://docs.comfy.org/interface/features/template
- Official ComfyUI Wan2.2 tutorial:
  - https://docs.comfy.org/tutorials/video/wan/wan2_2
- Official ComfyUI LTX-2 tutorial:
  - https://docs.comfy.org/tutorials/video/ltx/ltx-2
- Wan2.2 official repo:
  - https://github.com/Wan-Video/Wan2.2
- LTX official/model docs:
  - https://huggingface.co/Lightricks/LTX-Video
  - https://github.com/Lightricks/ComfyUI-LTXVideo
- Qwen-Image official/model docs:
  - https://github.com/QwenLM/Qwen-Image
  - https://docs.comfy.org/tutorials/image/qwen/qwen-image
  - https://blog.comfy.org/p/qwen-image-in-comfyui-new-era-of
- VibeComfy reference:
  - https://github.com/peteromallet/VibeComfy

