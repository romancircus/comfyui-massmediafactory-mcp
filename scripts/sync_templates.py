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

# Templates to sync (subset most commonly used)
CORE_TEMPLATES = [
    "qwen_txt2img.json",
    "ltx2_txt2vid.json", 
    "ltx2_img2vid.json",
    "wan26_txt2vid.json",
    "wan26_img2vid.json",
    "flux2_txt2img.json",
    "qwen_edit_background.json",
]


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
    """Sync templates from Hub to all Spokes."""
    results = {
        "hub": str(hub_path),
        "spokes": {},
        "templates_synced": [],
        "errors": [],
    }
    
    # Find source templates in Hub
    hub_templates_dir = hub_path / "src" / "comfyui_massmediafactory_mcp" / "templates"
    if not hub_templates_dir.exists():
        # Try alternate locations
        hub_templates_dir = hub_path / "templates"
    
    if not hub_templates_dir.exists():
        results["errors"].append(f"Hub templates not found at {hub_templates_dir}")
        return results
    
    results["hub_templates_dir"] = str(hub_templates_dir)
    
    # Sync to each spoke
    for spoke_path in spoke_paths:
        spoke_name = spoke_path.name
        results["spokes"][spoke_name] = {"synced": [], "errors": []}
        
        # Find/create templates dir in spoke
        spoke_templates_dir = find_template_dir(spoke_path)
        results["spokes"][spoke_name]["templates_dir"] = str(spoke_templates_dir)
        
        # Copy each template
        for template_name in CORE_TEMPLATES:
            source = hub_templates_dir / template_name
            if not source.exists():
                # Try with .json extension if not present
                source = hub_templates_dir / f"{template_name}.json"
            
            if source.exists():
                dest = spoke_templates_dir / template_name
                
                if dry_run:
                    results["spokes"][spoke_name]["synced"].append({
                        "template": template_name,
                        "action": "would_copy",
                        "source": str(source),
                        "dest": str(dest),
                    })
                else:
                    try:
                        shutil.copy2(source, dest)
                        results["spokes"][spoke_name]["synced"].append({
                            "template": template_name,
                            "action": "copied",
                            "source": str(source),
                            "dest": str(dest),
                        })
                        if template_name not in results["templates_synced"]:
                            results["templates_synced"].append(template_name)
                    except Exception as e:
                        results["spokes"][spoke_name]["errors"].append({
                            "template": template_name,
                            "error": str(e),
                        })
            else:
                results["spokes"][spoke_name]["errors"].append({
                    "template": template_name,
                    "error": f"Source not found: {source}",
                })
    
    return results


def verify_templates(spoke_path: Path) -> Dict:
    """Verify templates are valid JSON."""
    templates_dir = find_template_dir(spoke_path)
    results = {"valid": [], "invalid": []}
    
    for template_file in templates_dir.glob("*.json"):
        try:
            with open(template_file) as f:
                data = json.load(f)
                results["valid"].append({
                    "template": template_file.name,
                    "nodes": len(data.get("nodes", [])),
                })
        except json.JSONDecodeError as e:
            results["invalid"].append({
                "template": template_file.name,
                "error": str(e),
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sync MCP templates from Hub to Spoke repos"
    )
    parser.add_argument(
        "--hub",
        type=Path,
        default=DEFAULT_HUB,
        help="Path to MCP Hub repo (default: ~/Applications/comfyui-massmediafactory-mcp)"
    )
    parser.add_argument(
        "--spokes",
        type=lambda s: [Path(p) for p in s.split(",")],
        default=DEFAULT_SPOKES,
        help="Comma-separated list of spoke repo paths"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without copying"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify templates after sync"
    )
    parser.add_argument(
        "--templates",
        type=lambda s: s.split(","),
        default=CORE_TEMPLATES,
        help="Comma-separated list of templates to sync"
    )
    
    args = parser.parse_args()
    
    print(f"MCP Template Sync")
    print(f"Hub: {args.hub}")
    print(f"Spokes: {', '.join(p.name for p in args.spokes)}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()
    
    # Perform sync
    results = sync_templates(args.hub, args.spokes, dry_run=args.dry_run)
    
    # Print results
    print(f"Templates synced: {len(results['templates_synced'])}")
    for template in results["templates_synced"]:
        print(f"  ✓ {template}")
    
    print()
    print("Per-spoke results:")
    for spoke_name, spoke_results in results["spokes"].items():
        print(f"\n  {spoke_name}:")
        print(f"    Templates dir: {spoke_results.get('templates_dir', 'N/A')}")
        print(f"    Synced: {len(spoke_results['synced'])}")
        
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
    print("\n" + "="*50)
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
