# ComfyUI Agentic Interface Strategy (2026-02-11)

## 1) Problem Framing (Interactive)

Your concern is valid and points to a deeper issue than command count.

What I believe you are saying:
- The current migration may have over-indexed on “CLI-ify everything”.
- Even with architecture fixes, coding agents still struggle because they lack:
  - model-specific generation knowledge,
  - task-specific workflow assembly knowledge,
  - prompt strategy by model/task/style.
- You want one reusable interface that works for many downstream use cases:
  - short viral clips,
  - longer narrative videos (music storytelling),
  - style-specific production (photoreal, 3D storytelling),
  - character consistency across shots.

This implies the core missing layer is:
- **not another command**, but a **policy and planning layer** that converts creative intent into model-aware, validated execution plans.

## 2) Research Findings

## 2.1 ComfyUI: Ground Truth for Programmatic Control

ComfyUI itself is already a strong execution backend with stable primitives:
- Queue + validation via `POST /prompt`
- Real-time status via `/ws`
- Node schema discovery via `/object_info`
- Queue/history/asset system via `/queue`, `/history`, `/view`

Source:
- https://docs.comfy.org/development/comfyui-server/comms_routes

Implication:
- Your wrapper should remain thin around these primitives.
- Intelligence should live in a **planning/constraint layer**, not in ad-hoc wrappers.

## 2.2 ComfyUI Template System Supports Discoverability

ComfyUI supports discoverable workflow templates through extension folders and an endpoint for template collection.

Source:
- https://docs.comfy.org/custom-nodes/workflow_templates

Implication:
- Template-centric execution is a good base.
- But templates must be enriched with machine-readable metadata (task/style/model constraints), not just placeholders.

## 2.3 LTX Model-Specific Realities

Official LTX references highlight hard constraints and workflow patterns:
- Resolution divisible by 32
- Frames must be `8n + 1`
- Works best below 720x1280 and below 257 frames
- Prompts should be detailed, in English
- Distilled/full/fp8 variants have different speed/quality/VRAM tradeoffs
- Official recommendation favors ComfyUI workflows for best results

Sources:
- https://huggingface.co/Lightricks/LTX-Video
- https://github.com/Lightricks/LTX-Video
- https://github.com/Lightricks/ComfyUI-LTXVideo

Implication:
- Agents need model rules encoded as policy, not tribal knowledge.

## 2.4 Wan2.2 Model-Specific Realities

Wan2.2 ecosystem includes distinct task families:
- T2V / I2V / TI2V / S2V / Animate
- A14B and 5B variants with different performance profiles
- 480p/720p support patterns, TI2V 5B emphasizing 720p@24fps efficiency
- Official scripts include offload/dtype options and task-specific flags

Sources:
- https://github.com/Wan-Video/Wan2.2
- https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
- https://huggingface.co/Wan-AI/Wan2.2-S2V-14B

Implication:
- “Wan” is not one behavior; it is a family of behavior profiles by task and model variant.

## 2.5 Qwen-Image Model-Specific Realities

Official Qwen examples encode concrete defaults and prompt behavior:
- Supported aspect ratio presets
- Typical setup: `num_inference_steps=50`, `true_cfg_scale=4.0`
- Optional quality suffix tokens (“positive magic”)
- Distinct editing workflow paths for image edit variants

Source:
- https://github.com/QwenLM/Qwen-Image

Implication:
- Qwen prompt and parameter strategy should be policy-driven and distinct from Wan/LTX.

## 2.6 CLI Pattern Reference (gogcli)

Strong CLI traits visible in `gogcli`:
- Clear top-level groups + drill-down help
- Stable UX with predictable command grammar
- JSON output for automation
- Explicit “full help” discoverability mode

Sources:
- https://github.com/steipete/gogcli
- https://gogcli.sh/

Implication:
- Your CLI should be highly navigable and stable, while model complexity is pushed into policies/profiles.

## 2.7 Other Comfy Agent Efforts (VibeComfy)

VibeComfy appears to combine:
- CLI tooling
- workflow artifacts
- registry-style logic
- test-backed structure

Source:
- https://github.com/peteromallet/VibeComfy

Implication:
- The direction (registry + workflows + tooling) is correct, but your implementation needs stricter contracts and stronger model intelligence.

## 3) Core Diagnosis

Your main bottleneck is:
- **Missing “generation intelligence layer”**, not just command design.

Today’s stack is roughly:
- Intent -> command invocation -> template/workflow execution

Needed stack:
- Intent -> task/style decomposition -> model selection -> policy-constrained parameterization -> workflow synthesis/selection -> execution -> QA feedback -> iteration

## 4) What the Right Interface Should Look Like

## 4.1 Two-Plane Architecture

1. Discovery/Planning Plane (MCP)
- capabilities graph
- model/task/style constraints
- recipe discovery
- node/template introspection
- planning + validation only

2. Execution Plane (CLI)
- deterministic, idempotent operations
- queue-safe blocking behavior
- download/publish/artifact lifecycle
- retry/recovery with explicit exit codes

Do not merge these planes.

## 4.2 Add a Third Layer: Policy/Recipe Engine (Missing Today)

This new layer should be the canonical intelligence for agents:
- Encodes model-specific rules and defaults
- Encodes use-case recipes (viral short, 8-min story arc, photoreal portrait, 3D character narrative)
- Encodes prompting policies by task:
  - I2V motion-only prompts
  - T2I scene/content prompts
  - style-transfer constraints
  - character consistency guidance

### Recommended object model

- `Capability`
  - what each model/task can do
- `Constraint`
  - allowed resolutions, frame rules, cfg/steps bounds, VRAM tiers
- `Recipe`
  - multi-stage graph (e.g. keyframes -> style lock -> i2v -> temporal stitch)
- `Policy`
  - parameter defaults and fallbacks per model/task/style/tier
- `PromptPolicy`
  - prompt templates and anti-patterns per stage

## 4.3 CLI Should Be Small at the Surface, Rich in Inputs

Instead of adding many commands, keep a narrow command grammar:

- `mmf recipe list`
- `mmf recipe describe <id>`
- `mmf plan create --goal ... --style ... --duration ... --constraints ...`
- `mmf plan validate <plan.json>`
- `mmf run plan <plan.json>`
- `mmf run stage <stage.json>`
- `mmf qa run <asset|run_id>`
- `mmf tune suggest <failed_run_id>`

This keeps command count manageable while supporting many use cases.

## 5) Use-Case Coverage Strategy

## 5.1 Short-Form Viral (10-30s)

Primary needs:
- fast iteration
- hook-first visual payoff
- style consistency over few shots

Policy examples:
- LTX distilled / Wan I2V fast paths
- shorter frame windows
- stronger motion prompt templates

## 5.2 Mid/Long Narrative (1-8 min)

Primary needs:
- shot continuity
- character/environment consistency
- multi-stage composition and stitching

Policy examples:
- anchor keyframe generation with identity locks
- reference-bank reuse across scenes
- staged generation + inter-shot QA + regenerate-on-fail

## 5.3 Style Families (Photoreal vs 3D Storytelling)

Primary needs:
- style-specific prompting and model selection
- reusable style recipes
- style drift checks

Policy examples:
- distinct prompt templates by style family
- style-specific model shortlist and defaults
- style-preservation QA checks

## 6) Why Agents Fail Today and How to Fix It

Failure mode A:
- Agent chooses a syntactically valid command with semantically wrong settings.
Fix:
- hard constraints + policy auto-correction.

Failure mode B:
- Agent uses wrong prompt mode for stage (e.g. appearance-heavy I2V prompt).
Fix:
- stage-specific prompt policy and linting.

Failure mode C:
- Agent cannot discover best recipe for concept.
Fix:
- capability + recipe search in MCP with ranking.

Failure mode D:
- drift between docs/wrappers/core CLI.
Fix:
- machine-readable command contract + generated wrappers + cross-repo contract tests.

## 7) Concrete Build Plan

## Phase 1: Contract Hardening (Immediate)

- Fix CLI exit-code semantics.
- Fix template discovery consistency (including nested templates or enforce flat-only).
- Publish CLI spec JSON.
- Add wrapper conformance tests in downstream repos.

## Phase 2: Policy Engine (Core Investment)

- Create `policies/` registry:
  - `models/<model>.yaml`
  - `tasks/<task>.yaml`
  - `styles/<style>.yaml`
  - `recipes/<recipe>.yaml`
- Add policy resolver used by both MCP planning and CLI run paths.

## Phase 3: Planner + Recipe Compiler

- Add plan abstraction:
  - `goal`, `duration`, `style`, `character_constraints`, `quality_tier`, `latency_tier`
- Compile plan to staged workflow graph.
- Validate against model constraints and hardware profile.

## Phase 4: QA + Feedback Loop

- Automated QA checks by stage:
  - style consistency
  - identity consistency
  - motion coherence
  - prompt adherence
- Auto-regenerate with tuned params based on failure class.

## 8) Proposed MCP/CLI Split for This Repo

### MCP (planning/discovery)
- search capabilities (model/task/style)
- resolve constraints
- list/rank recipes
- generate/validate plan
- explain recommended model/params

### CLI (execution)
- run plan/stage
- queue/wait/progress
- artifact export
- batch execution
- QA execution

This keeps LLM-facing reasoning in MCP and deterministic actions in CLI.

## 9) Key Decision

You likely do **not** need many new commands.  
You need:
- a strict stable command contract,
- a first-class policy/recipe intelligence layer,
- generated wrappers and conformance tests across downstream repos.

That combination is what will let coding agents handle varied creative workloads reliably.

## 10) Sources

- ComfyUI Routes:
  - https://docs.comfy.org/development/comfyui-server/comms_routes
- ComfyUI workflow templates:
  - https://docs.comfy.org/custom-nodes/workflow_templates
- ComfyUI node docs support:
  - https://docs.comfy.org/custom-nodes/help_page
- Wan2.2 official repo:
  - https://github.com/Wan-Video/Wan2.2
- Wan2.2 model card:
  - https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
- Wan2.2 S2V:
  - https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
- LTX official repo:
  - https://github.com/Lightricks/LTX-Video
- LTX model card:
  - https://huggingface.co/Lightricks/LTX-Video
- ComfyUI-LTXVideo:
  - https://github.com/Lightricks/ComfyUI-LTXVideo
- Qwen-Image official repo:
  - https://github.com/QwenLM/Qwen-Image
- gogcli:
  - https://github.com/steipete/gogcli
  - https://gogcli.sh/
- VibeComfy:
  - https://github.com/peteromallet/VibeComfy

