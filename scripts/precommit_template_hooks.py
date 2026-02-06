#!/usr/bin/env python3
"""
Pre-commit hook for template compliance.

Validates that committed template JSON files have:
- Required _meta fields (description, model, type, parameters)
- version and hub_hash stamped
- Valid JSON structure
- Placeholder/parameter consistency
"""

import json
import re
import sys
from pathlib import Path


REQUIRED_META_FIELDS = ["description", "model", "type", "parameters"]
RECOMMENDED_META_FIELDS = ["version", "hub_hash", "defaults"]

TEMPLATES_DIR = Path("src/comfyui_massmediafactory_mcp/templates")


def validate_template_file(filepath: Path) -> list[str]:
    """Validate a single template file. Returns list of errors."""
    errors = []

    try:
        data = json.loads(filepath.read_text())
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    # Check _meta exists
    if "_meta" not in data:
        return ["Missing _meta section"]

    meta = data["_meta"]

    # Check required fields
    for field in REQUIRED_META_FIELDS:
        if field not in meta:
            errors.append(f"Missing _meta.{field}")

    # Check recommended fields (warnings, not errors)
    warnings = []
    for field in RECOMMENDED_META_FIELDS:
        if field not in meta:
            warnings.append(f"Missing _meta.{field} (run: python scripts/sync_templates.py --stamp)")

    # Check parameters is a list
    if "parameters" in meta and not isinstance(meta["parameters"], list):
        errors.append("_meta.parameters must be a list")

    # Check placeholder/parameter consistency
    if "parameters" in meta:
        workflow_str = json.dumps(data)
        placeholders = set(re.findall(r"\{\{([A-Z_]+)\}\}", workflow_str))
        declared = set(meta["parameters"])

        undeclared = placeholders - declared
        if undeclared:
            errors.append(f"Undeclared placeholders: {', '.join(sorted(undeclared))}")

    # Check at least one node exists
    nodes = [k for k in data.keys() if not k.startswith("_")]
    if not nodes:
        errors.append("Template has no nodes")

    # Check node structure
    for key in nodes:
        node = data[key]
        if not isinstance(node, dict):
            errors.append(f"Node '{key}' must be a dict")
            continue
        if "class_type" not in node:
            errors.append(f"Node '{key}' missing class_type")

    return errors


def main():
    """Check all template files passed as arguments."""
    if len(sys.argv) < 2:
        # No files to check
        return 0

    files = sys.argv[1:]
    has_errors = False

    for filepath_str in files:
        filepath = Path(filepath_str)

        # Only check template JSON files
        if not filepath.suffix == ".json":
            continue
        if not str(filepath).startswith(str(TEMPLATES_DIR)):
            continue

        errors = validate_template_file(filepath)
        if errors:
            has_errors = True
            print(f"\n{filepath}:")
            for error in errors:
                print(f"  ERROR: {error}")

    if has_errors:
        print("\nTemplate validation failed. Fix errors above before committing.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
