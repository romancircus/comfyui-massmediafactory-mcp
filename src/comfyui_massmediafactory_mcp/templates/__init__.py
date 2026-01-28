"""
Workflow Templates

Pre-built workflow templates for SOTA models with {{PLACEHOLDER}} syntax.
Includes validation to ensure templates are well-formed.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


TEMPLATES_DIR = Path(__file__).parent


def validate_template(template: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Validate template structure.

    Args:
        template: Template dict to validate

    Returns:
        Tuple of (errors, warnings)
    """
    errors = []
    warnings = []

    # Check _meta section exists
    if "_meta" not in template:
        errors.append("Missing _meta section")
        return errors, warnings

    meta = template["_meta"]

    # Check required _meta fields
    required_meta = ["description", "model", "type", "parameters", "defaults"]
    for field in required_meta:
        if field not in meta:
            if field == "defaults":
                warnings.append(f"Missing _meta.{field} (optional but recommended)")
            else:
                errors.append(f"Missing _meta.{field}")

    # Check that parameters is a list
    if "parameters" in meta and not isinstance(meta["parameters"], list):
        errors.append("_meta.parameters must be a list")

    # Check that defaults is a dict
    if "defaults" in meta and not isinstance(meta["defaults"], dict):
        errors.append("_meta.defaults must be a dict")

    # Find all placeholders in workflow
    workflow_str = json.dumps(template)
    placeholders = set(re.findall(r'\{\{([A-Z_]+)\}\}', workflow_str))

    # Check placeholders are declared in parameters
    declared_params = set(meta.get("parameters", []))
    for placeholder in placeholders:
        if placeholder not in declared_params:
            warnings.append(f"Undeclared placeholder: {{{{{placeholder}}}}}")

    # Check declared parameters have placeholders
    for param in declared_params:
        if param not in placeholders:
            warnings.append(f"Declared parameter '{param}' not used in workflow")

    # Check node structure
    node_count = 0
    for key, value in template.items():
        if key.startswith("_"):
            continue

        node_count += 1

        if not isinstance(value, dict):
            errors.append(f"Node '{key}' must be a dict")
            continue

        if "class_type" not in value:
            errors.append(f"Node '{key}' missing class_type")

        if "inputs" not in value:
            warnings.append(f"Node '{key}' has no inputs (may be intentional)")

        # Check connections format
        inputs = value.get("inputs", {})
        for input_name, input_value in inputs.items():
            if isinstance(input_value, list):
                if len(input_value) != 2:
                    errors.append(f"Node '{key}' input '{input_name}' has invalid connection format (expected [node_id, slot])")
                elif not isinstance(input_value[1], int):
                    errors.append(f"Node '{key}' input '{input_name}' slot must be integer")

    if node_count == 0:
        errors.append("Template has no nodes")

    return errors, warnings


def load_template(name: str, validate: bool = True) -> Dict[str, Any]:
    """
    Load a template by name with optional validation.

    Args:
        name: Template name (without .json extension)
        validate: Whether to validate the template (default True)

    Returns:
        Template dict, or {"error": ...} if not found or invalid
    """
    template_file = TEMPLATES_DIR / f"{name}.json"
    if not template_file.exists():
        return {"error": f"Template '{name}' not found"}

    try:
        template = json.loads(template_file.read_text())
    except json.JSONDecodeError as e:
        return {"error": f"Template '{name}' has invalid JSON: {e}"}

    if validate:
        errors, warnings = validate_template(template)
        if errors:
            return {
                "error": f"Template '{name}' has validation errors",
                "validation_errors": errors,
                "validation_warnings": warnings
            }
        # Attach warnings to template metadata if any
        if warnings:
            template["_validation_warnings"] = warnings

    return template


def list_templates(validate: bool = False) -> Dict[str, Any]:
    """
    List all available templates.

    Args:
        validate: Whether to validate each template (default False for speed)

    Returns:
        Dict with templates list and count
    """
    templates = []
    validation_issues = []

    for f in TEMPLATES_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())

            template_info = {
                "name": f.stem,
                "description": data.get("_meta", {}).get("description", ""),
                "model": data.get("_meta", {}).get("model", ""),
                "type": data.get("_meta", {}).get("type", ""),
                "parameters": data.get("_meta", {}).get("parameters", []),
            }

            if validate:
                errors, warnings = validate_template(data)
                if errors or warnings:
                    template_info["validation"] = {
                        "valid": len(errors) == 0,
                        "errors": errors,
                        "warnings": warnings
                    }
                    if errors:
                        validation_issues.append(f.stem)

            templates.append(template_info)

        except json.JSONDecodeError:
            validation_issues.append(f.stem)

    result = {"templates": templates, "count": len(templates)}
    if validation_issues:
        result["templates_with_issues"] = validation_issues

    return result


def inject_parameters(template: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inject parameters into a template.

    Replaces {{PARAM_NAME}} placeholders with actual values.

    Args:
        template: Template dict with placeholders
        params: Dict of parameter values

    Returns:
        Workflow dict with parameters injected
    """
    # Remove metadata before injection
    workflow = {k: v for k, v in template.items() if not k.startswith("_")}

    # Convert to string for replacement
    workflow_str = json.dumps(workflow)

    for param_name, param_value in params.items():
        placeholder = f"{{{{{param_name}}}}}"

        if isinstance(param_value, (int, float, bool)):
            # Numeric/boolean: remove quotes around placeholder
            if isinstance(param_value, bool):
                workflow_str = workflow_str.replace(f'"{placeholder}"', str(param_value).lower())
            else:
                workflow_str = workflow_str.replace(f'"{placeholder}"', str(param_value))
            workflow_str = workflow_str.replace(placeholder, str(param_value))
        else:
            # String: escape for JSON
            escaped = json.dumps(str(param_value))[1:-1]
            workflow_str = workflow_str.replace(placeholder, escaped)

    return json.loads(workflow_str)


def get_template_parameters(name: str) -> Dict[str, Any]:
    """
    Get parameters and defaults for a template.

    Args:
        name: Template name

    Returns:
        Dict with parameters list and defaults
    """
    template = load_template(name, validate=False)
    if "error" in template:
        return template

    meta = template.get("_meta", {})
    return {
        "name": name,
        "parameters": meta.get("parameters", []),
        "defaults": meta.get("defaults", {}),
        "description": meta.get("description", ""),
        "type": meta.get("type", "")
    }


def validate_all_templates() -> Dict[str, Any]:
    """
    Validate all templates in the directory.

    Returns:
        Dict with validation results for each template
    """
    results = {}

    for f in TEMPLATES_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            errors, warnings = validate_template(data)
            results[f.stem] = {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings
            }
        except json.JSONDecodeError as e:
            results[f.stem] = {
                "valid": False,
                "errors": [f"Invalid JSON: {e}"],
                "warnings": []
            }

    valid_count = sum(1 for r in results.values() if r["valid"])
    return {
        "results": results,
        "total": len(results),
        "valid": valid_count,
        "invalid": len(results) - valid_count
    }
