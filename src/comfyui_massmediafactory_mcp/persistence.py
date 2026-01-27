"""
Workflow Persistence

Save, load, and manage workflows in a local library.
Workflows are stored in ~/.massmediafactory/workflows/
"""

import json
import os
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
            workflows.append({
                "name": path.stem,
                "description": "(invalid JSON)",
                "tags": [],
                "error": True,
            })

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
) -> dict:
    """
    Import a raw workflow JSON (e.g., from ComfyUI export).

    Args:
        name: Name for the workflow.
        workflow_json: The raw workflow JSON.
        description: Optional description.
        tags: Optional tags.

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

    return save_workflow(
        name=name,
        workflow=actual_workflow,
        description=description,
        tags=tags,
    )
