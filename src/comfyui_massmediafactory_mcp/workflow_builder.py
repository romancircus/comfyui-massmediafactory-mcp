"""
Workflow Builder Module

Helper tools for constructing ComfyUI workflows programmatically.
Helps agents build workflows from scratch without drifting from working patterns.
"""

import json
import copy
from typing import Dict, Any, List, Optional

from .patterns import WORKFLOW_SKELETONS, MODEL_CONSTRAINTS, NODE_CHAINS


def create_workflow_skeleton(model: str, task: str) -> Dict[str, Any]:
    """
    Create minimal workflow skeleton for model+task.

    This is the primary entry point for agents building workflows.
    Returns a complete, tested workflow structure.

    Args:
        model: Model identifier (ltx2, flux2, wan26, qwen, sdxl)
        task: Task type (txt2vid, img2vid, txt2img)

    Returns:
        Complete workflow JSON with {{PLACEHOLDER}} parameters
    """
    key = (model.lower(), task.lower())

    if key not in WORKFLOW_SKELETONS:
        available = [f"{m}/{t}" for m, t in WORKFLOW_SKELETONS.keys()]
        return {
            "error": f"No skeleton for {model}/{task}",
            "available": available,
            "suggestion": "Use list_available_patterns() to see all options"
        }

    return copy.deepcopy(WORKFLOW_SKELETONS[key])


def get_node_sequence(model: str, task: str) -> List[Dict[str, Any]]:
    """
    Get recommended node sequence for model+task.

    Returns nodes in execution order with connection information.

    Args:
        model: Model identifier
        task: Task type

    Returns:
        List of node definitions with inputs/outputs
    """
    key = (model.lower(), task.lower())

    if key not in NODE_CHAINS:
        available = [f"{m}/{t}" for m, t in NODE_CHAINS.keys()]
        return {
            "error": f"No node chain for {model}/{task}",
            "available": available
        }

    return copy.deepcopy(NODE_CHAINS[key])


def explain_workflow(workflow: Dict[str, Any]) -> str:
    """
    Generate natural language description of a workflow.

    Analyzes the workflow structure and describes what each node does
    and how they're connected.

    Args:
        workflow: Workflow JSON to explain

    Returns:
        Human-readable description of the workflow
    """
    lines = []

    # Get metadata
    meta = workflow.get("_meta", {})
    if meta:
        lines.append(f"## {meta.get('description', 'Workflow')}")
        lines.append(f"Model: {meta.get('model', 'Unknown')}")
        lines.append(f"Type: {meta.get('type', 'Unknown')}")
        lines.append("")

    # Analyze nodes
    nodes = {}
    for node_id, node in workflow.items():
        if node_id.startswith("_"):
            continue
        if isinstance(node, dict) and "class_type" in node:
            nodes[node_id] = node

    lines.append("## Node Chain")
    lines.append(f"Total nodes: {len(nodes)}")
    lines.append("")

    # Build execution order (simple topological description)
    for node_id in sorted(nodes.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        node = nodes[node_id]
        class_type = node.get("class_type", "Unknown")
        title = node.get("_meta", {}).get("title", class_type)

        lines.append(f"**Node {node_id}: {title}** ({class_type})")

        inputs = node.get("inputs", {})
        if inputs:
            for input_name, input_value in inputs.items():
                if isinstance(input_value, list) and len(input_value) == 2:
                    lines.append(f"  - {input_name}: ← Node {input_value[0]} (slot {input_value[1]})")
                elif isinstance(input_value, str) and input_value.startswith("{{"):
                    lines.append(f"  - {input_name}: {input_value} (parameter)")
                else:
                    lines.append(f"  - {input_name}: {input_value}")
        lines.append("")

    return "\n".join(lines)


def get_required_nodes_for_model(model: str) -> Dict[str, Any]:
    """
    Get the required nodes for a specific model.

    Args:
        model: Model identifier

    Returns:
        Dict with required and forbidden nodes
    """
    model_lower = model.lower()
    constraints = MODEL_CONSTRAINTS.get(model_lower, {})

    if not constraints:
        return {
            "error": f"No constraints for model '{model}'",
            "available": list(MODEL_CONSTRAINTS.keys())
        }

    return {
        "model": model,
        "display_name": constraints.get("display_name", model),
        "type": constraints.get("type", "unknown"),
        "required_nodes": constraints.get("required_nodes", {}),
        "forbidden_nodes": constraints.get("forbidden_nodes", {}),
        "note": "Use required_nodes, avoid forbidden_nodes"
    }


def get_connection_pattern(source_node: str, target_node: str, model: str = None) -> Dict[str, Any]:
    """
    Get the correct connection pattern between two node types.

    Args:
        source_node: Source node class_type (e.g., "LTXVLoader")
        target_node: Target node class_type (e.g., "CLIPTextEncode")
        model: Optional model for context

    Returns:
        Connection info with correct slot indices
    """
    # Common connection patterns
    COMMON_PATTERNS = {
        ("LTXVLoader", "CLIPTextEncode"): {"output_slot": 1, "input_name": "clip", "note": "CLIP is at slot 1"},
        ("LTXVLoader", "SamplerCustom"): {"output_slot": 0, "input_name": "model", "note": "MODEL is at slot 0"},
        ("LTXVLoader", "VAEDecode"): {"output_slot": 2, "input_name": "vae", "note": "VAE is at slot 2"},

        ("UNETLoader", "BasicScheduler"): {"output_slot": 0, "input_name": "model", "note": "MODEL only output"},
        ("UNETLoader", "BasicGuider"): {"output_slot": 0, "input_name": "model", "note": "MODEL only output"},

        ("DualCLIPLoader", "CLIPTextEncode"): {"output_slot": 0, "input_name": "clip", "note": "CLIP at slot 0"},

        ("CLIPTextEncode", "LTXVConditioning"): {"output_slot": 0, "input_name": "positive/negative", "note": "CONDITIONING output"},
        ("CLIPTextEncode", "FluxGuidance"): {"output_slot": 0, "input_name": "conditioning", "note": "CONDITIONING output"},
        ("CLIPTextEncode", "KSampler"): {"output_slot": 0, "input_name": "positive/negative", "note": "CONDITIONING output"},

        ("LTXVConditioning", "SamplerCustom"): {"output_slots": [0, 1], "input_names": ["positive", "negative"], "note": "Dual conditioning output"},

        ("FluxGuidance", "BasicGuider"): {"output_slot": 0, "input_name": "conditioning", "note": "Modified conditioning"},

        ("LTXVScheduler", "SamplerCustom"): {"output_slot": 0, "input_name": "sigmas", "note": "SIGMAS output"},
        ("BasicScheduler", "SamplerCustomAdvanced"): {"output_slot": 0, "input_name": "sigmas", "note": "SIGMAS output"},

        ("KSamplerSelect", "SamplerCustom"): {"output_slot": 0, "input_name": "sampler", "note": "SAMPLER output"},
        ("KSamplerSelect", "SamplerCustomAdvanced"): {"output_slot": 0, "input_name": "sampler", "note": "SAMPLER output"},

        ("RandomNoise", "SamplerCustomAdvanced"): {"output_slot": 0, "input_name": "noise", "note": "NOISE output"},

        ("BasicGuider", "SamplerCustomAdvanced"): {"output_slot": 0, "input_name": "guider", "note": "GUIDER output"},

        ("SamplerCustom", "VAEDecode"): {"output_slot": 0, "input_name": "samples", "note": "LATENT output"},
        ("SamplerCustomAdvanced", "VAEDecode"): {"output_slot": 0, "input_name": "samples", "note": "LATENT output"},

        ("VAEDecode", "SaveImage"): {"output_slot": 0, "input_name": "images", "note": "IMAGE output"},
        ("VAEDecode", "VHS_VideoCombine"): {"output_slot": 0, "input_name": "images", "note": "IMAGE output"},
    }

    key = (source_node, target_node)
    if key in COMMON_PATTERNS:
        return {
            "source": source_node,
            "target": target_node,
            **COMMON_PATTERNS[key]
        }

    return {
        "source": source_node,
        "target": target_node,
        "error": "Unknown connection pattern",
        "suggestion": "Check the node documentation or use get_node_chain() for complete patterns"
    }


def modify_workflow_parameter(
    workflow: Dict[str, Any],
    node_id: str,
    param_name: str,
    new_value: Any
) -> Dict[str, Any]:
    """
    Safely modify a parameter in a workflow.

    Args:
        workflow: Workflow to modify
        node_id: Node ID to modify
        param_name: Parameter name in inputs
        new_value: New value to set

    Returns:
        Modified workflow (copy)
    """
    result = copy.deepcopy(workflow)

    if node_id not in result:
        return {"error": f"Node '{node_id}' not found in workflow"}

    node = result[node_id]
    if "inputs" not in node:
        node["inputs"] = {}

    node["inputs"][param_name] = new_value

    return result


def add_node_to_workflow(
    workflow: Dict[str, Any],
    class_type: str,
    inputs: Dict[str, Any],
    node_id: str = None,
    title: str = None
) -> Dict[str, Any]:
    """
    Add a new node to a workflow.

    Args:
        workflow: Workflow to modify
        class_type: Node class type
        inputs: Node inputs
        node_id: Optional specific node ID (auto-assigned if None)
        title: Optional title for _meta

    Returns:
        Modified workflow with new node
    """
    result = copy.deepcopy(workflow)

    # Find next available node ID
    if node_id is None:
        existing_ids = [int(k) for k in result.keys() if k.isdigit()]
        node_id = str(max(existing_ids, default=0) + 1)

    new_node = {
        "class_type": class_type,
        "inputs": inputs
    }

    if title:
        new_node["_meta"] = {"title": title}

    result[node_id] = new_node

    return result


def get_workflow_diff(workflow1: Dict[str, Any], workflow2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two workflows and show differences.

    Args:
        workflow1: First workflow
        workflow2: Second workflow

    Returns:
        Dict with added, removed, and modified nodes
    """
    # Get node IDs (excluding metadata)
    nodes1 = {k for k in workflow1.keys() if not k.startswith("_")}
    nodes2 = {k for k in workflow2.keys() if not k.startswith("_")}

    added = nodes2 - nodes1
    removed = nodes1 - nodes2
    common = nodes1 & nodes2

    modified = []
    for node_id in common:
        if workflow1[node_id] != workflow2[node_id]:
            modified.append({
                "node_id": node_id,
                "before": workflow1[node_id],
                "after": workflow2[node_id]
            })

    return {
        "added_nodes": list(added),
        "removed_nodes": list(removed),
        "modified_nodes": modified,
        "summary": f"{len(added)} added, {len(removed)} removed, {len(modified)} modified"
    }
