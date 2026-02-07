#!/usr/bin/env python3
"""
MCP Big Bang Migration - Template Sync Script
Hub-and-Spoke Architecture: Sync templates from MCP Hub to consumer repo Spokes

Usage:
    python scripts/sync_templates.py [--hub PATH] [--spokes REPO1,REPO2,...]

This script copies workflow templates from the central MCP Hub repository
to local copies in each consumer repo (Spokes). Maintains "Local Copies,
Global Standards" architecture.

Architecture:
- Hub: comfyui-massmediafactory-mcp (master templates)
- Spokes: pokedex-generator, KDH-Automation, Goat, RobloxChristian (local copies)

Benefits:
1. Offline operation (no MCP calls needed for template access)
2. Faster execution (local filesystem vs MCP overhead)
3. Works with Cyrus (no MCP dependency in worktrees)
4. Version controlled (templates tracked in repo)
"""

import shutil
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict
import json


# Default paths
DEFAULT_HUB = Path("/home/romancircus/Applications/comfyui-massmediafactory-mcp")
DEFAULT_SPOKES = [
    Path("/home/romancircus/Applications/pokedex-generator"),
    Path("/home/romancircus/Applications/KDH-Automation"),
    Path("/home/romancircus/Applications/Goat"),
    Path("/home/romancircus/Applications/RobloxChristian"),
]

# Tier 1: Always sync (core production templates)
TIER1_TEMPLATES = [
    "qwen_txt2img.json",
    "ltx2_txt2vid.json",
    "ltx2_img2vid.json",
    "wan26_txt2vid.json",
    "wan26_img2vid.json",
    "flux2_txt2img.json",
    "qwen_edit_background.json",
]

# Tier 2: Advanced templates (opt-in per repo)
TIER2_TEMPLATES = [
    "ltx2_txt2vid_distilled.json",
    "ltx2_i2v_distilled.json",
    "ltx2_audio_reactive.json",
    "flux2_lora_stack.json",
    "flux2_union_controlnet.json",
    "flux2_ultimate_upscale.json",
    "flux2_edit_by_text.json",
    "flux2_face_id.json",
    "flux2_grounding_dino_inpaint.json",
    "hunyuan15_txt2vid.json",
    "hunyuan15_img2vid.json",
    "qwen_poster_design.json",
    "sdxl_txt2img.json",
    "telestyle_image.json",
    "telestyle_video.json",
    "video_inpaint.json",
    "video_stitch.json",
    "chatterbox_tts.json",
    "audio_tts_f5.json",
    "audio_tts_voice_clone.json",
    "qwen3_tts_custom_voice.json",
    "qwen3_tts_voice_clone.json",
    "qwen3_tts_voice_design.json",
    "wan22_s2v.json",
    "wan22_i2v_a14b.json",
    "wan26_i2v_breakthrough.json",
    "qwen_controlnet_bio.json",
    "flux_kontext_edit.json",
    "z_turbo_txt2img.json",
    "mmaudio_v2a.json",
]

# Backward compat alias
CORE_TEMPLATES = TIER1_TEMPLATES


def compute_template_hash(template: dict) -> str:
    """Compute SHA256 hash of template content (excluding _meta)."""
    workflow_only = {k: v for k, v in template.items() if not k.startswith("_")}
    content = json.dumps(workflow_only, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def stamp_template(template_path: Path, version: str = "1.0.0") -> dict:
    """Add version and hub_hash to template _meta. Returns the stamped template."""
    template = json.loads(template_path.read_text())
    if "_meta" not in template:
        return template

    template["_meta"]["version"] = version
    template["_meta"]["hub_hash"] = compute_template_hash(template)

    template_path.write_text(json.dumps(template, indent=2) + "\n")
    return template


def detect_drift(hub_template: dict, spoke_template: dict) -> dict:
    """Compare Hub and Spoke templates. Returns drift info."""
    hub_hash = hub_template.get("_meta", {}).get("hub_hash")
    spoke_hash = spoke_template.get("_meta", {}).get("hub_hash")
    hub_version = hub_template.get("_meta", {}).get("version")
    spoke_version = spoke_template.get("_meta", {}).get("version")

    if not hub_hash or not spoke_hash:
        return {
            "status": "unknown",
            "reason": "Missing hub_hash in one or both templates",
        }

    if hub_hash == spoke_hash:
        return {"status": "in_sync", "version": hub_version}

    # Compute fresh hashes to check for local modifications
    fresh_hub_hash = compute_template_hash(hub_template)
    fresh_spoke_hash = compute_template_hash(spoke_template)

    if fresh_hub_hash != fresh_spoke_hash:
        return {
            "status": "diverged",
            "hub_version": hub_version,
            "spoke_version": spoke_version,
            "hub_hash": hub_hash,
            "spoke_hash": spoke_hash,
        }

    return {
        "status": "stale_hash",
        "reason": "Content matches but hashes differ (re-stamp needed)",
    }


def load_repo_sync_config(repo_path: Path) -> Dict:
    """Load per-repo sync config (.mcp_sync_config.json)."""
    config_path = repo_path / ".mcp_sync_config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def get_templates_for_repo(repo_path: Path) -> List[str]:
    """Get templates to sync for a repo based on its config."""
    config = load_repo_sync_config(repo_path)

    # Start with Tier 1 (always included)
    templates = list(TIER1_TEMPLATES)

    # Add Tier 2 if opted in
    tier2_opt_in = config.get("tier2", [])
    if tier2_opt_in == "all":
        templates.extend(TIER2_TEMPLATES)
    elif isinstance(tier2_opt_in, list):
        for t in tier2_opt_in:
            if t in TIER2_TEMPLATES and t not in templates:
                templates.append(t)

    # Add any extra templates specified
    extras = config.get("extra_templates", [])
    for t in extras:
        if t not in templates:
            templates.append(t)

    # Remove any excluded templates
    excludes = config.get("exclude", [])
    templates = [t for t in templates if t not in excludes]

    return templates


def find_template_dir(repo_path: Path) -> Path:
    """Find templates directory in repo (various locations)."""
    candidates = [
        repo_path / "templates",
        repo_path / "mcp_templates",
        repo_path / "workflows" / "templates",
        repo_path / "src" / "templates",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Create default
    default = repo_path / "mcp_templates"
    default.mkdir(parents=True, exist_ok=True)
    return default


def sync_templates(hub_path: Path, spoke_paths: List[Path], dry_run: bool = False) -> Dict:
    """Sync templates from Hub to all Spokes. Stamps versions and detects drift."""
    results = {
        "hub": str(hub_path),
        "spokes": {},
        "templates_synced": [],
        "drift_detected": [],
        "errors": [],
    }

    # Find source templates in Hub
    hub_templates_dir = hub_path / "src" / "comfyui_massmediafactory_mcp" / "templates"
    if not hub_templates_dir.exists():
        hub_templates_dir = hub_path / "templates"

    if not hub_templates_dir.exists():
        results["errors"].append(f"Hub templates not found at {hub_templates_dir}")
        return results

    results["hub_templates_dir"] = str(hub_templates_dir)

    # Stamp Hub templates with version/hash if missing
    all_syncable = list(set(TIER1_TEMPLATES + TIER2_TEMPLATES))
    manifest = {}
    for template_name in all_syncable:
        source = hub_templates_dir / template_name
        if source.exists():
            hub_template = json.loads(source.read_text())
            meta = hub_template.get("_meta", {})
            if "hub_hash" not in meta:
                hub_template = stamp_template(source, version="1.0.0")
            manifest[template_name] = {
                "version": meta.get("version", "1.0.0"),
                "hub_hash": meta.get("hub_hash", compute_template_hash(hub_template)),
            }

    # Write manifest to Hub
    if not dry_run and manifest:
        manifest_path = hub_templates_dir / ".manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Sync to each spoke
    for spoke_path in spoke_paths:
        spoke_name = spoke_path.name
        results["spokes"][spoke_name] = {"synced": [], "drift": [], "errors": []}

        spoke_templates_dir = find_template_dir(spoke_path)
        results["spokes"][spoke_name]["templates_dir"] = str(spoke_templates_dir)

        repo_templates = get_templates_for_repo(spoke_path)
        results["spokes"][spoke_name]["tier1_count"] = len([t for t in repo_templates if t in TIER1_TEMPLATES])
        results["spokes"][spoke_name]["tier2_count"] = len([t for t in repo_templates if t in TIER2_TEMPLATES])

        for template_name in repo_templates:
            source = hub_templates_dir / template_name
            if not source.exists():
                source = hub_templates_dir / f"{template_name}.json"

            if source.exists():
                dest = spoke_templates_dir / template_name

                # Check for drift before overwriting
                if dest.exists():
                    try:
                        hub_data = json.loads(source.read_text())
                        spoke_data = json.loads(dest.read_text())
                        drift = detect_drift(hub_data, spoke_data)
                        if drift["status"] == "diverged":
                            results["spokes"][spoke_name]["drift"].append(
                                {
                                    "template": template_name,
                                    **drift,
                                }
                            )
                            results["drift_detected"].append(f"{spoke_name}/{template_name}")
                    except (json.JSONDecodeError, Exception):
                        pass

                if dry_run:
                    results["spokes"][spoke_name]["synced"].append(
                        {
                            "template": template_name,
                            "action": "would_copy",
                            "source": str(source),
                            "dest": str(dest),
                        }
                    )
                else:
                    try:
                        shutil.copy2(source, dest)
                        results["spokes"][spoke_name]["synced"].append(
                            {
                                "template": template_name,
                                "action": "copied",
                                "source": str(source),
                                "dest": str(dest),
                            }
                        )
                        if template_name not in results["templates_synced"]:
                            results["templates_synced"].append(template_name)
                    except Exception as e:
                        results["spokes"][spoke_name]["errors"].append(
                            {
                                "template": template_name,
                                "error": str(e),
                            }
                        )
            else:
                results["spokes"][spoke_name]["errors"].append(
                    {
                        "template": template_name,
                        "error": f"Source not found: {source}",
                    }
                )

    return results


def verify_templates(spoke_path: Path) -> Dict:
    """Verify templates are valid JSON."""
    templates_dir = find_template_dir(spoke_path)
    results = {"valid": [], "invalid": []}

    for template_file in templates_dir.glob("*.json"):
        try:
            with open(template_file) as f:
                data = json.load(f)
                results["valid"].append(
                    {
                        "template": template_file.name,
                        "nodes": len(data.get("nodes", [])),
                    }
                )
        except json.JSONDecodeError as e:
            results["invalid"].append(
                {
                    "template": template_file.name,
                    "error": str(e),
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Sync MCP templates from Hub to Spoke repos")
    parser.add_argument(
        "--hub",
        type=Path,
        default=DEFAULT_HUB,
        help="Path to MCP Hub repo (default: ~/Applications/comfyui-massmediafactory-mcp)",
    )
    parser.add_argument(
        "--spokes",
        type=lambda s: [Path(p) for p in s.split(",")],
        default=DEFAULT_SPOKES,
        help="Comma-separated list of spoke repo paths",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without copying",
    )
    parser.add_argument("--verify", action="store_true", help="Verify templates after sync")
    parser.add_argument(
        "--templates",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated list of templates to sync (overrides tier system)",
    )
    parser.add_argument(
        "--stamp",
        action="store_true",
        help="Stamp all Hub templates with version and hub_hash",
    )

    args = parser.parse_args()

    print("MCP Template Sync")
    print(f"Hub: {args.hub}")

    # Stamp-only mode
    if args.stamp:
        hub_templates_dir = args.hub / "src" / "comfyui_massmediafactory_mcp" / "templates"
        stamped = 0
        for f in sorted(hub_templates_dir.glob("*.json")):
            template = stamp_template(f)
            if "_meta" in template:
                stamped += 1
        print(f"Stamped {stamped} templates with version and hub_hash")
        return 0

    print(f"Spokes: {', '.join(p.name for p in args.spokes)}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    # Perform sync
    results = sync_templates(args.hub, args.spokes, dry_run=args.dry_run)

    # Print results
    print(f"Templates synced: {len(results['templates_synced'])}")
    for template in results["templates_synced"]:
        print(f"  ✓ {template}")

    if results.get("drift_detected"):
        print(f"\nDrift detected in {len(results['drift_detected'])} templates:")
        for drift_item in results["drift_detected"]:
            print(f"  ⚠ {drift_item}")

    print()
    print("Per-spoke results:")
    for spoke_name, spoke_results in results["spokes"].items():
        print(f"\n  {spoke_name}:")
        print(f"    Templates dir: {spoke_results.get('templates_dir', 'N/A')}")
        print(f"    Synced: {len(spoke_results['synced'])}")

        if spoke_results.get("drift"):
            print(f"    ⚠ Drift: {len(spoke_results['drift'])} templates modified locally")
            for d in spoke_results["drift"]:
                print(f"      - {d['template']}: hub={d.get('hub_version','?')} spoke={d.get('spoke_version','?')}")

        if spoke_results["errors"]:
            print(f"    ⚠ Errors: {len(spoke_results['errors'])}")
            for error in spoke_results["errors"]:
                print(f"      - {error.get('template', 'unknown')}: {error.get('error', 'unknown error')}")

        # Verify if requested
        if args.verify and not args.dry_run:
            spoke_path = next((p for p in args.spokes if p.name == spoke_name), None)
            if spoke_path:
                verify_results = verify_templates(spoke_path)
                print(f"    Valid templates: {len(verify_results['valid'])}")
                if verify_results["invalid"]:
                    print(f"    ⚠ Invalid templates: {len(verify_results['invalid'])}")

    # Summary
    print("\n" + "=" * 50)
    total_synced = sum(len(s["synced"]) for s in results["spokes"].values())
    total_errors = sum(len(s["errors"]) for s in results["spokes"].values())

    print(f"Total: {total_synced} files synced, {total_errors} errors")

    if total_errors == 0:
        print("✓ Sync completed successfully")
        return 0
    else:
        print("⚠ Sync completed with errors")
        return 1


if __name__ == "__main__":
    exit(main())
