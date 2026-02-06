"""
Workflow Persistence

Save, load, and manage workflows in a local library.
Workflows are stored in ~/.massmediafactory/workflows/
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List


def get_workflows_dir() -> Path:
    """Get the workflows directory, creating if needed."""
    base_dir = Path.home() / ".massmediafactory" / "workflows"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def get_workflow_path(name: str) -> Path:
    """Get the path for a workflow by name."""
    # Sanitize name
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_").lower()
    if not safe_name:
        safe_name = "workflow"
    return get_workflows_dir() / f"{safe_name}.json"


def save_workflow(
    name: str,
    workflow: dict,
    description: str = "",
    tags: Optional[List[str]] = None,
) -> dict:
    """
    Save a workflow to the local library.

    Args:
        name: Unique name for the workflow (e.g., "flux-portrait", "qwen-landscape")
        workflow: The workflow JSON object
        description: Optional description of what this workflow does
        tags: Optional list of tags (e.g., ["image", "flux", "portrait"])

    Returns:
        Success status and file path.
    """
    path = get_workflow_path(name)

    # Create wrapper with metadata
    data = {
        "name": name,
        "description": description,
        "tags": tags or [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "workflow": workflow,
    }

    # Check if updating existing
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            data["created_at"] = existing.get("created_at", data["created_at"])
        except (json.JSONDecodeError, KeyError):
            pass

    try:
        path.write_text(json.dumps(data, indent=2))
        return {
            "success": True,
            "name": name,
            "path": str(path),
            "message": f"Workflow '{name}' saved",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def load_workflow(name: str) -> dict:
    """
    Load a workflow from the local library.

    Args:
        name: The workflow name to load.

    Returns:
        The workflow object with metadata, or error.
    """
    path = get_workflow_path(name)

    if not path.exists():
        return {"error": f"Workflow '{name}' not found"}

    try:
        data = json.loads(path.read_text())
        return {
            "name": data.get("name", name),
            "description": data.get("description", ""),
            "tags": data.get("tags", []),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "workflow": data.get("workflow", {}),
        }
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in workflow file: {e}"}
    except Exception as e:
        return {"error": str(e)}


def list_workflows(tag: Optional[str] = None) -> dict:
    """
    List all saved workflows.

    Args:
        tag: Optional tag to filter by.

    Returns:
        List of workflow summaries.
    """
    workflows_dir = get_workflows_dir()
    workflows = []

    for path in workflows_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            workflow_info = {
                "name": data.get("name", path.stem),
                "description": data.get("description", ""),
                "tags": data.get("tags", []),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            }

            # Filter by tag if specified
            if tag:
                if tag.lower() in [t.lower() for t in workflow_info["tags"]]:
                    workflows.append(workflow_info)
            else:
                workflows.append(workflow_info)

        except (json.JSONDecodeError, KeyError):
            # Include but mark as invalid
            workflows.append(
                {
                    "name": path.stem,
                    "description": "(invalid JSON)",
                    "tags": [],
                    "error": True,
                }
            )

    # Sort by updated_at descending
    workflows.sort(
        key=lambda x: x.get("updated_at", ""),
        reverse=True,
    )

    return {
        "workflows": workflows,
        "count": len(workflows),
        "filter_tag": tag,
    }


def delete_workflow(name: str) -> dict:
    """
    Delete a workflow from the library.

    Args:
        name: The workflow name to delete.

    Returns:
        Success status.
    """
    path = get_workflow_path(name)

    if not path.exists():
        return {"error": f"Workflow '{name}' not found"}

    try:
        path.unlink()
        return {
            "success": True,
            "name": name,
            "message": f"Workflow '{name}' deleted",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def duplicate_workflow(source_name: str, new_name: str) -> dict:
    """
    Duplicate an existing workflow with a new name.

    Args:
        source_name: The workflow to copy.
        new_name: Name for the new workflow.

    Returns:
        Success status.
    """
    source = load_workflow(source_name)

    if "error" in source:
        return source

    return save_workflow(
        name=new_name,
        workflow=source["workflow"],
        description=f"Copy of {source_name}. {source.get('description', '')}",
        tags=source.get("tags", []),
    )


def update_workflow_tags(name: str, tags: List[str]) -> dict:
    """
    Update the tags for a workflow.

    Args:
        name: The workflow name.
        tags: New list of tags.

    Returns:
        Success status.
    """
    data = load_workflow(name)

    if "error" in data:
        return data

    return save_workflow(
        name=name,
        workflow=data["workflow"],
        description=data.get("description", ""),
        tags=tags,
    )


def export_workflow(name: str) -> dict:
    """
    Export a workflow as raw JSON (without metadata wrapper).
    Useful for sharing or using in ComfyUI directly.

    Args:
        name: The workflow name.

    Returns:
        Raw workflow JSON.
    """
    data = load_workflow(name)

    if "error" in data:
        return data

    return {
        "name": name,
        "workflow": data["workflow"],
    }


def import_workflow(
    name: str,
    workflow_json: dict,
    description: str = "",
    tags: Optional[List[str]] = None,
    auto_convert: bool = True,
) -> dict:
    """
    Import a raw workflow JSON (e.g., from ComfyUI export).

    Automatically detects and converts UI format to API format if needed.

    Args:
        name: Name for the workflow.
        workflow_json: The raw workflow JSON.
        description: Optional description.
        tags: Optional tags.
        auto_convert: Auto-convert UI format to API format (default True).

    Returns:
        Success status.
    """
    # Handle both raw workflow and wrapped format
    if "workflow" in workflow_json and "name" in workflow_json:
        # Already wrapped format
        actual_workflow = workflow_json["workflow"]
        description = description or workflow_json.get("description", "")
        tags = tags or workflow_json.get("tags", [])
    else:
        # Raw workflow
        actual_workflow = workflow_json

    # Auto-detect and convert format if needed
    if auto_convert:
        format_type = detect_workflow_format(actual_workflow)
        if format_type == "ui":
            conversion = convert_ui_to_api_format(actual_workflow)
            if "error" in conversion:
                return conversion
            actual_workflow = conversion["workflow"]

    return save_workflow(
        name=name,
        workflow=actual_workflow,
        description=description,
        tags=tags,
    )


# =============================================================================
# Workflow Format Conversion
# =============================================================================


def detect_workflow_format(workflow: dict) -> str:
    """
    Detect the format of a workflow JSON.

    Returns:
        "api" - API format: {node_id: {class_type, inputs}}
        "ui" - UI format: {nodes: [...], links: [...]}
        "unknown" - Unrecognized format
    """
    # UI format has nodes array and links array
    if "nodes" in workflow and isinstance(workflow.get("nodes"), list):
        if "links" in workflow or "extra" in workflow:
            return "ui"

    # API format has string/int keys with class_type objects
    if workflow:
        # Check first few items for API format structure
        for key, value in list(workflow.items())[:5]:
            if isinstance(value, dict) and "class_type" in value:
                return "api"

    # Check if it might be wrapped
    if "workflow" in workflow:
        inner = workflow["workflow"]
        if isinstance(inner, dict):
            return detect_workflow_format(inner)

    return "unknown"


def convert_ui_to_api_format(ui_workflow: dict) -> dict:
    """
    Convert ComfyUI UI format to API format.

    UI format structure:
    {
        "nodes": [{"id": 1, "type": "KSampler", "widgets_values": [...], ...}],
        "links": [[link_id, source_node, source_slot, target_node, target_slot, type], ...],
        "groups": [...],
        "extra": {...}
    }

    API format structure:
    {
        "1": {"class_type": "KSampler", "inputs": {...}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "...", "clip": ["1", 1]}}
    }

    Args:
        ui_workflow: Workflow in UI format.

    Returns:
        {"workflow": {...}, "converted": True} or {"error": "..."}
    """
    if "nodes" not in ui_workflow:
        return {"error": "Invalid UI format: missing 'nodes' array"}

    nodes = ui_workflow.get("nodes", [])
    links = ui_workflow.get("links", [])

    # Build link lookup: link_id -> (source_node, source_slot, target_node, target_slot)
    link_map = {}
    for link in links:
        if len(link) >= 5:
            link_id = link[0]
            link_map[link_id] = {
                "source_node": str(link[1]),
                "source_slot": link[2],
                "target_node": str(link[3]),
                "target_slot": link[4],
            }

    # Build reverse lookup: (target_node, target_slot) -> (source_node, source_slot)
    input_connections = {}
    for link_id, link_info in link_map.items():
        key = (link_info["target_node"], link_info["target_slot"])
        input_connections[key] = (link_info["source_node"], link_info["source_slot"])

    api_workflow = {}

    for node in nodes:
        node_id = str(node.get("id"))
        class_type = node.get("type")

        if not class_type:
            continue

        inputs = {}

        # Get widget values (non-connection inputs)
        widgets_values = node.get("widgets_values", [])

        # Get input definitions from node
        node_inputs = node.get("inputs", [])
        _node_outputs = node.get("outputs", [])

        # Map input slots to connections
        for i, inp in enumerate(node_inputs):
            input_name = inp.get("name")
            link_id = inp.get("link")

            if link_id is not None and link_id in link_map:
                # This input is connected to another node's output
                link_info = link_map[link_id]
                inputs[input_name] = [link_info["source_node"], link_info["source_slot"]]
            # If not connected, value may come from widgets_values

        # Map widget values to inputs (simplified - may need node-specific handling)
        # This is a best-effort mapping since widget order isn't guaranteed
        _widget_idx = 0
        for prop in node.get("properties", {}).items():
            # Properties are sometimes stored separately
            pass

        api_workflow[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
        }

        # Store widgets_values for reference
        if widgets_values:
            api_workflow[node_id]["_widgets_values"] = widgets_values

    return {
        "workflow": api_workflow,
        "converted": True,
        "original_format": "ui",
        "node_count": len(api_workflow),
        "note": "Widget values may need manual mapping to input names",
    }


def convert_api_to_ui_format(api_workflow: dict) -> dict:
    """
    Convert API format to UI format for visualization.

    This is a best-effort conversion for basic visualization.
    Full UI format requires node dimensions, positions, etc.

    Args:
        api_workflow: Workflow in API format.

    Returns:
        {"workflow": {...}, "converted": True} or {"error": "..."}
    """
    nodes = []
    links = []
    link_id = 0

    # Position nodes in a grid layout
    grid_x, grid_y = 100, 100
    grid_spacing = 300

    # Track output connections for link generation
    for node_id, node_data in api_workflow.items():
        class_type = node_data.get("class_type", "Unknown")
        inputs = node_data.get("inputs", {})

        # Create node entry
        node = {
            "id": int(node_id) if node_id.isdigit() else hash(node_id) % 10000,
            "type": class_type,
            "pos": [grid_x, grid_y],
            "size": [200, 150],
            "flags": {},
            "order": int(node_id) if node_id.isdigit() else 0,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {},
            "widgets_values": [],
        }

        # Process inputs - identify connections vs values
        input_slot = 0
        for input_name, input_value in inputs.items():
            if isinstance(input_value, list) and len(input_value) == 2:
                # This is a connection: [source_node_id, source_slot]
                source_node = input_value[0]
                source_slot = input_value[1]

                # Add input slot definition
                node["inputs"].append(
                    {
                        "name": input_name,
                        "type": "*",  # Type unknown without node info
                        "link": link_id,
                    }
                )

                # Add link
                links.append(
                    [
                        link_id,
                        int(source_node) if str(source_node).isdigit() else hash(source_node) % 10000,
                        source_slot,
                        node["id"],
                        input_slot,
                        "*",
                    ]
                )

                link_id += 1
                input_slot += 1
            else:
                # This is a widget value
                node["widgets_values"].append(input_value)

        nodes.append(node)

        # Update grid position for next node
        grid_x += grid_spacing
        if grid_x > 1200:
            grid_x = 100
            grid_y += grid_spacing

    return {
        "workflow": {
            "last_node_id": max((n["id"] for n in nodes), default=0),
            "last_link_id": link_id,
            "nodes": nodes,
            "links": links,
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4,
        },
        "converted": True,
        "original_format": "api",
        "node_count": len(nodes),
        "note": "Positions are auto-generated. Manual adjustment may be needed.",
    }
