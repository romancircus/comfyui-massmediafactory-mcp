"""
Unified Parameter Injection

Single implementation for {{PLACEHOLDER}} substitution in workflows.
Replaces 4 duplicate implementations across batch.py, pipeline.py,
templates/__init__.py, and workflow_generator.py.
"""

import json
from typing import Dict, Any


def inject_placeholders(
    workflow: Dict[str, Any],
    params: Dict[str, Any],
    strip_meta: bool = False,
) -> Dict[str, Any]:
    """
    Inject parameter values into workflow {{PLACEHOLDER}} fields.

    Handles type-aware substitution:
    - Numeric (int/float): Removes quotes around placeholder for JSON validity
    - Boolean: Converts to lowercase JSON boolean
    - String: JSON-escapes special characters

    Args:
        workflow: Workflow dict with {{PARAM}} placeholders.
        params: Dict of parameter name -> value.
        strip_meta: If True, remove keys starting with '_' (template metadata).

    Returns:
        New workflow dict with placeholders replaced.
    """
    if strip_meta:
        source = {k: v for k, v in workflow.items() if not k.startswith("_")}
    else:
        source = workflow

    workflow_str = json.dumps(source)

    for param_name, param_value in params.items():
        placeholder = f"{{{{{param_name}}}}}"

        if isinstance(param_value, bool):
            # Bool before int/float since bool is subclass of int
            workflow_str = workflow_str.replace(f'"{placeholder}"', str(param_value).lower())
            workflow_str = workflow_str.replace(placeholder, str(param_value).lower())
        elif isinstance(param_value, (int, float)):
            # Numeric: remove surrounding quotes for JSON validity
            workflow_str = workflow_str.replace(f'"{placeholder}"', str(param_value))
            workflow_str = workflow_str.replace(placeholder, str(param_value))
        else:
            # String: JSON-escape for safe inclusion
            escaped = json.dumps(str(param_value))[1:-1]
            workflow_str = workflow_str.replace(placeholder, escaped)

    return json.loads(workflow_str)
