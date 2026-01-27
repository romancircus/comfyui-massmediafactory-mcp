"""
Workflow Validation

Validate workflows before execution to catch errors early.
"""

from typing import Dict, List, Optional, Set, Tuple
from .client import get_client


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

    # Check compatibility
    compatible = source_output_type == target_input_type

    return {
        "compatible": compatible,
        "source_output_type": source_output_type,
        "target_input_type": target_input_type,
        "message": "Types match" if compatible else f"Type mismatch: {source_output_type} → {target_input_type}",
    }
