"""
Workflow Diff Tool for MassMediaFactory MCP

Compare two workflows to find node/parameter differences.
Useful for comparing template versions and debugging parameter changes.
Pattern from: nodetool
"""

from typing import Any, List


def diff_workflows(workflow_a: dict, workflow_b: dict) -> dict:
    """
    Compare two workflows and return differences.

    Args:
        workflow_a: First workflow JSON.
        workflow_b: Second workflow JSON.

    Returns:
        {
            "nodes_added": [...],       # In B but not A
            "nodes_removed": [...],     # In A but not B
            "nodes_modified": [...],    # In both but different
            "nodes_unchanged": N,       # Count of identical nodes
            "summary": "..."            # Human-readable summary
        }
    """
    # Skip metadata keys
    keys_a = {k for k in workflow_a if not k.startswith("_")}
    keys_b = {k for k in workflow_b if not k.startswith("_")}

    added = keys_b - keys_a
    removed = keys_a - keys_b
    common = keys_a & keys_b

    nodes_added = []
    for node_id in sorted(added):
        node = workflow_b[node_id]
        if isinstance(node, dict):
            nodes_added.append(
                {
                    "node_id": node_id,
                    "class_type": node.get("class_type", "unknown"),
                }
            )

    nodes_removed = []
    for node_id in sorted(removed):
        node = workflow_a[node_id]
        if isinstance(node, dict):
            nodes_removed.append(
                {
                    "node_id": node_id,
                    "class_type": node.get("class_type", "unknown"),
                }
            )

    nodes_modified = []
    unchanged_count = 0

    for node_id in sorted(common):
        node_a = workflow_a[node_id]
        node_b = workflow_b[node_id]

        if not isinstance(node_a, dict) or not isinstance(node_b, dict):
            if node_a != node_b:
                nodes_modified.append(
                    {
                        "node_id": node_id,
                        "type": "value_changed",
                        "from": str(node_a)[:100],
                        "to": str(node_b)[:100],
                    }
                )
            else:
                unchanged_count += 1
            continue

        changes = _diff_nodes(node_a, node_b)
        if changes:
            nodes_modified.append(
                {
                    "node_id": node_id,
                    "class_type": node_a.get("class_type", "unknown"),
                    "changes": changes,
                }
            )
        else:
            unchanged_count += 1

    # Build summary
    parts = []
    if nodes_added:
        parts.append(f"{len(nodes_added)} node(s) added")
    if nodes_removed:
        parts.append(f"{len(nodes_removed)} node(s) removed")
    if nodes_modified:
        parts.append(f"{len(nodes_modified)} node(s) modified")
    if unchanged_count:
        parts.append(f"{unchanged_count} node(s) unchanged")
    summary = ", ".join(parts) if parts else "Workflows are identical"

    return {
        "nodes_added": nodes_added,
        "nodes_removed": nodes_removed,
        "nodes_modified": nodes_modified,
        "nodes_unchanged": unchanged_count,
        "summary": summary,
        "identical": len(nodes_added) == 0 and len(nodes_removed) == 0 and len(nodes_modified) == 0,
    }


def _diff_nodes(node_a: dict, node_b: dict) -> List[dict]:
    """Compare two node dicts and return list of changes."""
    changes = []

    # Check class_type change
    if node_a.get("class_type") != node_b.get("class_type"):
        changes.append(
            {
                "field": "class_type",
                "from": node_a.get("class_type"),
                "to": node_b.get("class_type"),
            }
        )

    # Compare inputs
    inputs_a = node_a.get("inputs", {})
    inputs_b = node_b.get("inputs", {})

    all_input_keys = set(inputs_a.keys()) | set(inputs_b.keys())
    for key in sorted(all_input_keys):
        val_a = inputs_a.get(key)
        val_b = inputs_b.get(key)

        if val_a != val_b:
            changes.append(
                {
                    "field": f"inputs.{key}",
                    "from": _summarize_value(val_a),
                    "to": _summarize_value(val_b),
                }
            )

    return changes


def _summarize_value(value: Any) -> Any:
    """Summarize a value for display in diff output."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value[:100] + "..." if len(value) > 100 else value
    if isinstance(value, list):
        return value  # Connection references
    if isinstance(value, dict):
        return f"<dict with {len(value)} keys>"
    return str(value)[:100]
