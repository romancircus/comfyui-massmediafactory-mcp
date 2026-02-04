"""
Workflow Visualization Module

Generate Mermaid flowcharts from ComfyUI workflows for debugging and documentation.

Key functions:
- workflow_to_mermaid(): Convert workflow JSON to Mermaid diagram syntax
- visualize_workflow(): MCP tool that returns Mermaid syntax
"""

import json
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass
class NodeInfo:
    """Information about a workflow node for visualization."""
    node_id: str
    class_type: str
    title: str
    inputs: Dict[str, Any]
    outputs: List[str]


def _sanitize_node_id(node_id: str) -> str:
    """Sanitize node ID for Mermaid syntax."""
    # Mermaid IDs can't start with numbers, so prefix with 'n'
    if node_id[0].isdigit():
        return f"n{node_id}"
    return node_id.replace("-", "_").replace(".", "_")


def _sanitize_label(label: str) -> str:
    """Sanitize label text for Mermaid."""
    # Escape special characters that break Mermaid syntax
    return label.replace('"', '"').replace("[", "[").replace("]", "]")


def _extract_node_info(node_id: str, node_data: Dict[str, Any]) -> NodeInfo:
    """Extract node information from workflow data."""
    meta = node_data.get("_meta", {})
    title = meta.get("title", node_data.get("class_type", "Unknown"))
    
    # Get outputs from the workflow structure
    outputs = []
    inputs = node_data.get("inputs", {})
    
    return NodeInfo(
        node_id=node_id,
        class_type=node_data.get("class_type", "Unknown"),
        title=title,
        inputs=inputs,
        outputs=outputs,
    )


def _is_connection(value: Any) -> bool:
    """Check if a value is a node connection reference."""
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)


def workflow_to_mermaid(
    workflow: Dict[str, Any],
    direction: str = "TD",
    include_placeholders: bool = False,
) -> str:
    """
    Convert ComfyUI workflow JSON to Mermaid diagram syntax.

    Args:
        workflow: ComfyUI workflow JSON dict
        direction: Graph direction (TD=top-down, LR=left-right, BT=bottom-top, RL=right-left)
        include_placeholders: If True, show {{PLACEHOLDER}} values instead of connections

    Returns:
        Mermaid diagram syntax as a string

    Example:
        >>> workflow = {"1": {"class_type": "KSampler", "inputs": {...}}, ...}
        >>> mermaid = workflow_to_mermaid(workflow)
        >>> print(mermaid)
        graph TD
            n1["KSampler"]
            ...
    """
    lines = [f"graph {direction}"]
    nodes: Dict[str, NodeInfo] = {}
    edges: List[Tuple[str, str, str]] = []  # (source, target, label)
    
    # First pass: extract all node info
    for node_id, node_data in workflow.items():
        if node_id.startswith("_"):
            continue  # Skip metadata keys
        
        if not isinstance(node_data, dict):
            continue
        
        nodes[node_id] = _extract_node_info(node_id, node_data)
    
    # Second pass: build edges from connections
    for node_id, node_info in nodes.items():
        mermaid_id = _sanitize_node_id(node_id)
        
        for input_name, input_value in node_info.inputs.items():
            if _is_connection(input_value):
                # This is a connection to another node
                source_id, slot = input_value
                if source_id in nodes:
                    source_mermaid = _sanitize_node_id(source_id)
                    edges.append((source_mermaid, mermaid_id, input_name))
            elif include_placeholders and isinstance(input_value, str):
                # Show placeholder values as dashed connections
                if input_value.startswith("{{") and input_value.endswith("}}"):
                    # Create a virtual node for the parameter
                    param_name = input_value[2:-2]  # Remove {{ and }}
                    param_id = f"param_{param_name}"
                    if param_id not in nodes:
                        lines.append(f'    {param_id}["{param_name}"]')
                        nodes[param_id] = NodeInfo(
                            node_id=param_id,
                            class_type="Parameter",
                            title=param_name,
                            inputs={},
                            outputs=[],
                        )
                    edges.append((param_id, mermaid_id, input_name))
    
    # Generate node definitions
    for node_id, node_info in nodes.items():
        mermaid_id = _sanitize_node_id(node_id)
        label = _sanitize_label(node_info.title)
        
        # Style nodes by type
        if node_info.class_type in ["SaveImage", "VHS_VideoCombine"]:
            # Output nodes
            lines.append(f'    {mermaid_id}["{label}"]')
        elif "Loader" in node_info.class_type or "Load" in node_info.class_type:
            # Input/Loader nodes
            lines.append(f'    {mermaid_id}("{label}")')
        elif node_info.class_type == "Parameter":
            # Parameter nodes
            lines.append(f'    {mermaid_id}(("{label}"))')
        else:
            # Standard nodes
            lines.append(f'    {mermaid_id}["{label}"]')
    
    # Generate edges
    for source, target, label in edges:
        lines.append(f"    {source} -->|{label}| {target}")
    
    # Add styling classes
    lines.append("")
    lines.append("    classDef outputNode fill:#f96,stroke:#333,stroke-width:2px")
    lines.append("    classDef loaderNode fill:#9f9,stroke:#333,stroke-width:1px")
    lines.append("    classDef paramNode fill:#99f,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5")
    
    # Apply classes
    for node_id, node_info in nodes.items():
        mermaid_id = _sanitize_node_id(node_id)
        if node_info.class_type in ["SaveImage", "VHS_VideoCombine"]:
            lines.append(f"    class {mermaid_id} outputNode")
        elif "Loader" in node_info.class_type or "Load" in node_info.class_type:
            lines.append(f"    class {mermaid_id} loaderNode")
        elif node_info.class_type == "Parameter":
            lines.append(f"    class {mermaid_id} paramNode")
    
    return "\n".join(lines)


def workflow_to_mermaid_url(workflow: Dict[str, Any]) -> str:
    """
    Generate a Mermaid Live Editor URL for the workflow diagram.
    
    This allows users to open the diagram in Mermaid's online editor.
    
    Args:
        workflow: ComfyUI workflow JSON
        
    Returns:
        URL to Mermaid Live Editor with the diagram encoded
    """
    mermaid_code = workflow_to_mermaid(workflow)
    
    # Mermaid Live Editor uses base64-encoded state
    import base64
    state = json.dumps({
        "code": mermaid_code,
        "mermaid": {"theme": "default"},
        "autoSync": True,
        "updateDiagram": True,
    })
    
    encoded = base64.b64encode(state.encode()).decode()
    return f"https://mermaid.live/edit#base64:{encoded}"


def visualize_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool: Generate Mermaid diagram from workflow.
    
    Args:
        workflow: ComfyUI workflow JSON
        
    Returns:
        {
            "mermaid": str,  # Mermaid diagram syntax
            "url": str,      # Mermaid Live Editor URL
            "node_count": int,
            "edge_count": int,
        }
    """
    try:
        mermaid_code = workflow_to_mermaid(workflow)
        url = workflow_to_mermaid_url(workflow)
        
        # Count nodes and edges
        node_count = sum(
            1 for k in workflow.keys()
            if not k.startswith("_") and isinstance(workflow[k], dict)
        )
        
        # Count edges by looking for connections in inputs
        edge_count = 0
        for node_data in workflow.values():
            if not isinstance(node_data, dict):
                continue
            inputs = node_data.get("inputs", {})
            for value in inputs.values():
                if _is_connection(value):
                    edge_count += 1
        
        return {
            "mermaid": mermaid_code,
            "url": url,
            "node_count": node_count,
            "edge_count": edge_count,
        }
    except Exception as e:
        return {
            "error": f"Failed to generate visualization: {str(e)}",
            "isError": True,
            "code": "VISUALIZATION_ERROR",
        }


def get_workflow_summary(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a text summary of workflow structure.
    
    Args:
        workflow: ComfyUI workflow JSON
        
    Returns:
        Summary with node types, connections, and parameters
    """
    node_types: Dict[str, int] = {}
    parameters: Set[str] = set()
    
    for node_id, node_data in workflow.items():
        if node_id.startswith("_") or not isinstance(node_data, dict):
            continue
        
        class_type = node_data.get("class_type", "Unknown")
        node_types[class_type] = node_types.get(class_type, 0) + 1
        
        # Find placeholder parameters
        inputs = node_data.get("inputs", {})
        for value in inputs.values():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                parameters.add(value[2:-2])
    
    return {
        "node_types": node_types,
        "total_nodes": sum(node_types.values()),
        "parameters": sorted(parameters),
        "unique_node_types": len(node_types),
    }
