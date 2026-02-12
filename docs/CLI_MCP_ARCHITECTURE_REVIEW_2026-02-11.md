# CLI/MCP Architecture Review (2026-02-11)

## Scope

This review covers:
- `comfyui-massmediafactory-mcp` CLI and MCP architecture
- Downstream usage patterns in `KDH-Automation` and `pokedex-generator`
- Agent reliability implications for autonomous image/video generation

## Executive Summary

The direction is correct: **MCP for discovery/planning, CLI for execution**.  
The current risk is not the overall strategy, but **contract drift** and **too many wrapper layers** across repos.

Core facts:
- Main CLI currently exposes **43 concrete command paths**.
- MCP server is intentionally reduced to discovery/planning resources and tools.
- Downstream repos add many wrapper functions, creating the perception of “100+ CLI functions”.

## What Is Working

- Single `mmf` entrypoint with grouped subcommands (`run`, `batch`, `models`, `templates`, `pipeline`, etc.).
- Clear discovery-vs-execution split in architecture docs.
- Blocking execution pattern (`run` + `wait_for_completion`) aligns with queue-safe behavior.
- Strong test coverage on CLI parsing and core command behavior in this repo.

## Critical Gaps Found

### 1) Exit Code Contract Inconsistency

Several commands return `EXIT_OK` even when payload contains `"error"`, which can break autonomous agent workflows that rely on shell exit status.

Examples:
- `src/comfyui_massmediafactory_mcp/cli_commands/system.py`
  - `cmd_stats`, `cmd_free`, `cmd_interrupt`, `cmd_enhance`
- `src/comfyui_massmediafactory_mcp/cli_commands/models.py`
  - `cmd_models_list`, `cmd_models_compatibility`

Observed behavior in this environment:
- `mmf stats` produced an error JSON while the process exited with code `0`.

### 2) Template Discovery Misses Subdirectories

Template loading/listing currently uses top-level globbing and does not include nested templates.

Examples:
- `src/comfyui_massmediafactory_mcp/templates/__init__.py`
  - `TEMPLATES_DIR / f"{name}.json"`
  - `TEMPLATES_DIR.glob("*.json")`

Impact:
- Nested templates (e.g., `templates/pony/*.json`) are not discoverable through current list/load flow.

### 3) Downstream Contract Drift (Batch Seeds)

`pokedex-generator` wrapper builds a `batch seeds` command signature that does not match current `mmf` CLI parser.

Example:
- `pokedex-generator/src/adapters/mmf_client.py`
  - builds: `batch seeds --template ... --params ...`
- current CLI expects:
  - `mmf batch seeds [workflow] --count --start-seed ...`

### 4) Docs/Code Drift

Counts and claims differ across docs vs implementation:
- Template count claims mismatch current files.
- Python version claims differ between docs and `pyproject.toml`.
- Pipeline help text includes entries not present in pipeline dispatcher.

## Architecture Diagnosis

### Strategy is Correct

The dual-path architecture remains the right foundation:
- **Discovery plane (MCP):** capabilities, constraints, templates, node schemas
- **Execution plane (CLI):** queueing, wait semantics, retries, download/publish

### Main Problem is Surface Area Fragmentation

There are 3 layers of command/function surfaces:
- Hub CLI command surface
- Downstream wrappers (`mmf.js`, `mmf_client.py`)
- Historical docs/snippets still referencing older patterns

This causes:
- stale examples
- incompatible wrappers
- agent uncertainty when selecting commands

## Recommendations (Priority Order)

1. **Enforce strict exit-code semantics** across all commands.
2. **Normalize template registry behavior** (top-level + nested, or explicit flat-only policy with validation).
3. **Publish a machine-readable CLI contract** (JSON schema/spec) and auto-generate wrappers from it.
4. **Define a minimal stable “agent-core” command set** and treat other commands as advanced/experimental.
5. **Add cross-repo contract tests** that validate downstream wrappers against current CLI parser/help.

## Proposed “Agent-Core” Command Set (Stable)

Execution:
- `run`
- `execute`
- `wait`
- `status`
- `progress`
- `upload`
- `download`

Batch:
- `batch seeds`
- `batch dir`

Planning/Selection:
- `templates list`
- `templates get`
- `models constraints`
- `models optimize`

Higher-level:
- `pipeline` (small, curated set only)

## Why This Matters for LLM/Coding Agents

LLMs fail most often when:
- command contracts are ambiguous,
- docs and implementation diverge,
- model-specific defaults are implicit, not encoded.

A smaller stable surface plus explicit model/task contracts improves:
- tool selection accuracy
- successful first-run generation
- automated recovery behavior

## Follow-up Research Agenda

Next phase should define:
- model-specific workflow policies (WAN/LTX/Qwen/etc.)
- prompt policies by task (I2V motion-only vs T2I full-scene prompts)
- long-form composition patterns (shorts vs multi-minute storytelling)
- an execution graph abstraction that downstream repos can target without ComfyUI node-level expertise

