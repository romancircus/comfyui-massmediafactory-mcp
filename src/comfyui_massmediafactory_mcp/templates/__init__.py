"""
Workflow Templates

Pre-built workflow templates for SOTA models with {{PLACEHOLDER}} syntax.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


TEMPLATES_DIR = Path(__file__).parent


def load_template(name: str) -> Dict[str, Any]:
    """Load a template by name."""
    template_file = TEMPLATES_DIR / f"{name}.json"
    if not template_file.exists():
        return {"error": f"Template '{name}' not found"}

    return json.loads(template_file.read_text())


def list_templates() -> Dict[str, Any]:
    """List all available templates."""
    templates = []
    for f in TEMPLATES_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            templates.append({
                "name": f.stem,
                "description": data.get("_meta", {}).get("description", ""),
                "model": data.get("_meta", {}).get("model", ""),
                "type": data.get("_meta", {}).get("type", ""),
            })
        except json.JSONDecodeError:
            pass

    return {"templates": templates, "count": len(templates)}


def inject_parameters(template: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inject parameters into a template.

    Replaces {{PARAM_NAME}} placeholders with actual values.
    """
    # Remove metadata before injection
    workflow = {k: v for k, v in template.items() if not k.startswith("_")}

    # Convert to string for replacement
    workflow_str = json.dumps(workflow)

    for param_name, param_value in params.items():
        placeholder = f"{{{{{param_name}}}}}"

        if isinstance(param_value, (int, float, bool)):
            # Numeric/boolean: remove quotes around placeholder
            workflow_str = workflow_str.replace(f'"{placeholder}"', str(param_value).lower() if isinstance(param_value, bool) else str(param_value))
            workflow_str = workflow_str.replace(placeholder, str(param_value))
        else:
            # String: escape for JSON
            escaped = json.dumps(str(param_value))[1:-1]
            workflow_str = workflow_str.replace(placeholder, escaped)

    return json.loads(workflow_str)
