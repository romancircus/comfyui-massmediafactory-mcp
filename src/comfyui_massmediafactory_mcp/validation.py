"""
Workflow Validation

Validate workflows before execution to catch errors early.

NOTE: Model resolution specs are now centralized in model_registry.py.
This module imports from there for backwards compatibility.
"""

from typing import Dict, List, Optional, Set, Tuple
from .client import get_client
from .model_registry import MODEL_RESOLUTION_SPECS


def validate_workflow(workflow: dict) -> dict:
    """
    Validate a ComfyUI workflow for common errors.

    Checks:
    - All node types exist in ComfyUI
    - All model files referenced exist
    - Node connections are valid (source nodes exist, output slots valid)
    - Required inputs are provided

    Args:
        workflow: The workflow JSON to validate.

    Returns:
        Validation result with errors and warnings.
    """
    errors = []
    warnings = []

    client = get_client()

    # Get all available node types
    object_info = client.get_object_info()
    if "error" in object_info:
        return {
            "valid": False,
            "errors": [f"Could not fetch node info from ComfyUI: {object_info['error']}"],
            "warnings": [],
        }

    available_nodes = set(object_info.keys())

    # Get available models for validation
    models_cache = {}

    def get_available_models(node_type: str, field: str) -> Set[str]:
        """Get available model names for a specific node field."""
        cache_key = f"{node_type}:{field}"
        if cache_key in models_cache:
            return models_cache[cache_key]

        if node_type in object_info:
            node_info = object_info[node_type]
            inputs = node_info.get("input", {})

            for input_type in ["required", "optional"]:
                if input_type in inputs and field in inputs[input_type]:
                    field_info = inputs[input_type][field]
                    if isinstance(field_info, list) and len(field_info) > 0:
                        if isinstance(field_info[0], list):
                            models_cache[cache_key] = set(field_info[0])
                            return models_cache[cache_key]

        models_cache[cache_key] = set()
        return models_cache[cache_key]

    # Validate each node
    node_ids = set(workflow.keys())

    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        # Check node type exists
        if class_type not in available_nodes:
            errors.append({
                "node_id": node_id,
                "type": "unknown_node",
                "message": f"Unknown node type: {class_type}",
            })
            continue

        # Get node schema
        node_schema = object_info[class_type]
        required_inputs = node_schema.get("input", {}).get("required", {})
        optional_inputs = node_schema.get("input", {}).get("optional", {})
        all_inputs = {**required_inputs, **optional_inputs}

        # Check required inputs are provided
        for input_name, input_spec in required_inputs.items():
            if input_name not in inputs:
                # Check if it's a connection type (usually uppercase)
                if isinstance(input_spec, list) and len(input_spec) > 0:
                    if isinstance(input_spec[0], str) and input_spec[0].isupper():
                        errors.append({
                            "node_id": node_id,
                            "type": "missing_connection",
                            "message": f"Missing required input '{input_name}' (type: {input_spec[0]})",
                        })
                    else:
                        errors.append({
                            "node_id": node_id,
                            "type": "missing_input",
                            "message": f"Missing required input '{input_name}'",
                        })

        # Validate each input
        for input_name, input_value in inputs.items():
            # Check if input is a connection (list format: [node_id, slot])
            if isinstance(input_value, list) and len(input_value) == 2:
                source_node_id = str(input_value[0])
                source_slot = input_value[1]

                # Check source node exists
                if source_node_id not in node_ids:
                    errors.append({
                        "node_id": node_id,
                        "type": "invalid_connection",
                        "message": f"Input '{input_name}' references non-existent node '{source_node_id}'",
                    })
                else:
                    # Check output slot is valid
                    source_node = workflow[source_node_id]
                    source_type = source_node.get("class_type", "")

                    if source_type in object_info:
                        source_outputs = object_info[source_type].get("output", [])
                        if source_slot >= len(source_outputs):
                            errors.append({
                                "node_id": node_id,
                                "type": "invalid_slot",
                                "message": f"Input '{input_name}' references invalid output slot {source_slot} on node '{source_node_id}' (max: {len(source_outputs) - 1})",
                            })

            # Check model file exists (for model loader nodes)
            elif input_name in ["ckpt_name", "unet_name", "vae_name", "lora_name", "clip_name",
                                "clip_name1", "clip_name2", "control_net_name"]:
                available = get_available_models(class_type, input_name)
                if available and input_value not in available:
                    # Check for placeholder
                    if "{{" in str(input_value) and "}}" in str(input_value):
                        warnings.append({
                            "node_id": node_id,
                            "type": "placeholder",
                            "message": f"Input '{input_name}' contains placeholder: {input_value}",
                        })
                    else:
                        errors.append({
                            "node_id": node_id,
                            "type": "model_not_found",
                            "message": f"Model '{input_value}' not found for input '{input_name}'",
                        })

    # Check for orphaned nodes (nodes with no connections to output)
    output_nodes = set()
    connected_nodes = set()

    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")

        # Identify output nodes
        if class_type in ["SaveImage", "SaveVideo", "PreviewImage", "VHS_SaveVideo",
                          "SaveAudio", "VHS_SaveAudio"]:
            output_nodes.add(node_id)

        # Track connections
        for input_value in node.get("inputs", {}).values():
            if isinstance(input_value, list) and len(input_value) == 2:
                connected_nodes.add(str(input_value[0]))

    if not output_nodes:
        warnings.append({
            "node_id": None,
            "type": "no_output",
            "message": "No output node found (SaveImage, SaveVideo, etc.)",
        })

    # Check for cycles in the workflow graph
    cycles = _detect_cycles(workflow)
    for cycle in cycles:
        errors.append({
            "node_id": cycle[0],
            "type": "cycle_detected",
            "message": f"Circular dependency detected: {' → '.join(cycle)}",
            "cycle": cycle,
        })

    # Check for resolution compatibility issues
    resolution_warnings = _check_resolution_compatibility(workflow, object_info)
    warnings.extend(resolution_warnings)

    # Check for nodes that aren't connected to anything
    for node_id in node_ids:
        if node_id not in connected_nodes and node_id not in output_nodes:
            node_type = workflow[node_id].get("class_type", "")
            # Skip loader nodes as they're often at the start of the chain
            if not any(x in node_type.lower() for x in ["loader", "load"]):
                warnings.append({
                    "node_id": node_id,
                    "type": "orphaned_node",
                    "message": f"Node '{node_id}' ({node_type}) is not connected to any output",
                })

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "node_count": len(workflow),
        "output_nodes": list(output_nodes),
    }


def validate_and_fix(workflow: dict) -> dict:
    """
    Validate workflow and attempt to fix common issues.

    Currently fixes:
    - String node IDs (ensures all are strings)

    Args:
        workflow: The workflow JSON to validate and fix.

    Returns:
        Fixed workflow and validation results.
    """
    # Ensure all node IDs are strings
    fixed_workflow = {}
    id_mapping = {}

    for node_id, node in workflow.items():
        str_id = str(node_id)
        id_mapping[node_id] = str_id
        fixed_workflow[str_id] = node.copy()

    # Fix connections to use string IDs
    for node_id, node in fixed_workflow.items():
        inputs = node.get("inputs", {})
        fixed_inputs = {}

        for input_name, input_value in inputs.items():
            if isinstance(input_value, list) and len(input_value) == 2:
                source_id = input_value[0]
                if source_id in id_mapping:
                    source_id = id_mapping[source_id]
                fixed_inputs[input_name] = [str(source_id), input_value[1]]
            else:
                fixed_inputs[input_name] = input_value

        fixed_workflow[node_id]["inputs"] = fixed_inputs

    # Validate the fixed workflow
    validation = validate_workflow(fixed_workflow)
    validation["fixed_workflow"] = fixed_workflow
    validation["fixes_applied"] = ["Ensured all node IDs are strings"]

    return validation


def _detect_cycles(workflow: dict) -> List[List[str]]:
    """
    Detect cycles in the workflow graph using DFS.

    Uses color marking: WHITE (unvisited), GRAY (in progress), BLACK (done).

    Args:
        workflow: The workflow to check.

    Returns:
        List of cycles found, each as a list of node IDs.
    """
    WHITE, GRAY, BLACK = 0, 1, 2

    # Build adjacency list (node -> nodes it depends on)
    adj = {str(node_id): [] for node_id in workflow.keys()}

    for node_id, node in workflow.items():
        inputs = node.get("inputs", {})
        for input_value in inputs.values():
            if isinstance(input_value, list) and len(input_value) == 2:
                source_id = str(input_value[0])
                if source_id in adj:
                    adj[str(node_id)].append(source_id)

    color = {node: WHITE for node in adj}
    parent = {node: None for node in adj}
    cycles = []

    def dfs(node: str, path: List[str]) -> None:
        color[node] = GRAY

        for neighbor in adj[node]:
            if color[neighbor] == GRAY:
                # Back edge found - extract cycle
                cycle_start = path.index(neighbor) if neighbor in path else -1
                if cycle_start >= 0:
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
            elif color[neighbor] == WHITE:
                parent[neighbor] = node
                dfs(neighbor, path + [neighbor])

        color[node] = BLACK

    for node in adj:
        if color[node] == WHITE:
            dfs(node, [node])

    return cycles


def _check_resolution_compatibility(workflow: dict, object_info: dict) -> List[dict]:
    """
    Check for resolution compatibility issues based on model type.

    Args:
        workflow: The workflow to check.
        object_info: ComfyUI node information.

    Returns:
        List of resolution warnings.
    """
    warnings = []

    # Detect model type from loader nodes
    detected_model = None
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        # Check UNET/checkpoint loaders for model type hints
        if any(x in class_type.lower() for x in ["unetloader", "checkpointloader"]):
            model_name = inputs.get("unet_name", inputs.get("ckpt_name", "")).lower()
            if "flux" in model_name:
                detected_model = "flux"
            elif "sdxl" in model_name or "xl" in model_name:
                detected_model = "sdxl"
            elif "sd15" in model_name or "1.5" in model_name:
                detected_model = "sd15"
            elif "qwen" in model_name:
                detected_model = "qwen"
            elif "ltx" in model_name:
                detected_model = "ltx"
            elif "wan" in model_name or "hunyuan" in model_name:
                detected_model = "wan"

    if not detected_model:
        return warnings

    spec = MODEL_RESOLUTION_SPECS.get(detected_model, {})
    if not spec:
        return warnings

    # Check EmptyLatentImage and similar nodes for resolution
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if "latent" in class_type.lower() or "empty" in class_type.lower():
            width = inputs.get("width")
            height = inputs.get("height")

            if width is not None and height is not None:
                try:
                    width = int(width)
                    height = int(height)
                except (ValueError, TypeError):
                    continue

                # Check divisibility
                if width % spec["divisible_by"] != 0:
                    warnings.append({
                        "node_id": node_id,
                        "type": "resolution_divisibility",
                        "message": f"Width {width} not divisible by {spec['divisible_by']} (required for {detected_model})",
                    })

                if height % spec["divisible_by"] != 0:
                    warnings.append({
                        "node_id": node_id,
                        "type": "resolution_divisibility",
                        "message": f"Height {height} not divisible by {spec['divisible_by']} (required for {detected_model})",
                    })

                # Check bounds
                if width < spec["min"] or width > spec["max"]:
                    warnings.append({
                        "node_id": node_id,
                        "type": "resolution_bounds",
                        "message": f"Width {width} outside recommended range [{spec['min']}-{spec['max']}] for {detected_model}",
                    })

                if height < spec["min"] or height > spec["max"]:
                    warnings.append({
                        "node_id": node_id,
                        "type": "resolution_bounds",
                        "message": f"Height {height} outside recommended range [{spec['min']}-{spec['max']}] for {detected_model}",
                    })

                # Check if significantly different from native
                native = spec["native"]
                max_dim = max(width, height)
                if max_dim > native * 1.5 or max_dim < native * 0.5:
                    warnings.append({
                        "node_id": node_id,
                        "type": "resolution_native",
                        "message": f"Resolution {width}x{height} differs significantly from native {native}x{native} for {detected_model}",
                    })

    return warnings


def _types_compatible(source_type: str, target_type: str) -> bool:
    """
    Check if source output type is compatible with target input type.

    Handles:
    - Exact match
    - Wildcard (*) matches any type
    - Union types (comma-separated, e.g., "IMAGE,MASK")
    - COMBO types (treated as wildcard for enums)

    Args:
        source_type: The output type from source node.
        target_type: The expected input type on target node.

    Returns:
        True if types are compatible.
    """
    # Wildcard matches everything
    if source_type == "*" or target_type == "*":
        return True

    # Exact match
    if source_type == target_type:
        return True

    # Handle union types (comma-separated)
    if "," in target_type:
        accepted_types = [t.strip() for t in target_type.split(",")]
        if source_type in accepted_types:
            return True

    if "," in source_type:
        provided_types = [t.strip() for t in source_type.split(",")]
        # Source provides multiple types, check if any match target
        if target_type in provided_types:
            return True

    # COMBO is a special type for enum dropdowns, usually compatible
    if target_type == "COMBO":
        return True

    return False


def check_node_compatibility(source_type: str, source_slot: int, target_type: str, target_input: str) -> dict:
    """
    Check if a connection between two nodes is compatible.

    Args:
        source_type: The source node class type
        source_slot: The output slot index on source
        target_type: The target node class type
        target_input: The input name on target

    Returns:
        Compatibility result with types.
    """
    client = get_client()
    object_info = client.get_object_info()

    if "error" in object_info:
        return {"compatible": None, "error": object_info["error"]}

    # Get source output type
    if source_type not in object_info:
        return {"compatible": False, "error": f"Unknown source node type: {source_type}"}

    source_outputs = object_info[source_type].get("output", [])
    if source_slot >= len(source_outputs):
        return {"compatible": False, "error": f"Invalid source slot {source_slot}"}

    source_output_type = source_outputs[source_slot]

    # Get target input type
    if target_type not in object_info:
        return {"compatible": False, "error": f"Unknown target node type: {target_type}"}

    target_inputs = object_info[target_type].get("input", {})
    target_input_spec = None

    for input_category in ["required", "optional"]:
        if input_category in target_inputs and target_input in target_inputs[input_category]:
            target_input_spec = target_inputs[input_category][target_input]
            break

    if target_input_spec is None:
        return {"compatible": False, "error": f"Unknown target input: {target_input}"}

    # Get expected input type
    target_input_type = None
    if isinstance(target_input_spec, list) and len(target_input_spec) > 0:
        if isinstance(target_input_spec[0], str):
            target_input_type = target_input_spec[0]

    if target_input_type is None:
        return {"compatible": None, "note": "Could not determine target input type"}

    # Check compatibility using the enhanced type matching
    compatible = _types_compatible(source_output_type, target_input_type)

    return {
        "compatible": compatible,
        "source_output_type": source_output_type,
        "target_input_type": target_input_type,
        "message": "Types compatible" if compatible else f"Type mismatch: {source_output_type} → {target_input_type}",
    }
