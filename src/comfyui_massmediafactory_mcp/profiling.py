"""
Execution Profiling for MassMediaFactory MCP

Track per-node timing from workflow executions.
Pattern from: comfyui-deploy

ComfyUI history API provides status messages with event_type and node IDs.
We reconstruct per-node timing by tracking "executing" messages which fire
sequentially - node N's duration is the gap between when N started and when
N+1 started (or execution completed).
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class NodeProfile:
    """Profile data for a single node execution."""

    node_id: str
    class_type: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "class_type": self.class_type,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class ExecutionProfile:
    """Complete execution profile for a workflow."""

    prompt_id: str
    total_duration_ms: float = 0.0
    node_profiles: List[NodeProfile] = field(default_factory=list)
    slowest_node: Optional[str] = None
    status: str = "unknown"

    def to_dict(self) -> dict:
        nodes = [n.to_dict() for n in self.node_profiles]
        nodes.sort(key=lambda x: x["duration_ms"], reverse=True)
        return {
            "prompt_id": self.prompt_id,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "total_duration_seconds": round(self.total_duration_ms / 1000, 2),
            "node_count": len(nodes),
            "nodes": nodes,
            "slowest_node": nodes[0] if nodes else None,
            "status": self.status,
        }


# In-memory profile store (recent executions)
_profiles: Dict[str, ExecutionProfile] = {}
MAX_PROFILES = 50


def get_execution_profile(prompt_id: str) -> dict:
    """
    Get execution profile for a completed workflow.

    Builds timing data from ComfyUI execution history.
    Per-node timing is reconstructed from sequential "executing" status
    messages - each node's duration is the gap until the next node starts.

    Args:
        prompt_id: The prompt_id to profile.

    Returns:
        {
            "prompt_id": "...",
            "total_duration_ms": N,
            "total_duration_seconds": N.N,
            "node_count": N,
            "nodes": [{"node_id": "...", "class_type": "...", "duration_ms": N.N}],
            "slowest_node": {...},
            "status": "completed"|"error"|"not_found"
        }
    """
    # Check cache first
    if prompt_id in _profiles:
        return _profiles[prompt_id].to_dict()

    # Build profile from history
    from .client import get_client

    client = get_client()
    history = client.get_history(prompt_id)

    if "error" in history:
        return {
            "error": "HISTORY_UNAVAILABLE",
            "prompt_id": prompt_id,
            "details": history["error"],
        }

    if prompt_id not in history:
        return {"error": "NOT_FOUND", "prompt_id": prompt_id}

    entry = history[prompt_id]
    status_info = entry.get("status", {})
    _outputs = entry.get("outputs", {})
    prompt_data = entry.get("prompt", [])

    # Extract workflow from prompt data (index 2 in the tuple)
    workflow = {}
    if isinstance(prompt_data, (list, tuple)) and len(prompt_data) > 2:
        workflow = prompt_data[2] if isinstance(prompt_data[2], dict) else {}

    # Reconstruct per-node timing from status messages.
    # ComfyUI sends sequential "executing" messages: [["executing", {"node": "5"}], ...]
    # Node N's duration = timestamp of (N+1) start - timestamp of N start.
    # The final node's end is marked by ["executing", {"node": null}] (execution done).
    messages = status_info.get("messages", [])

    # Collect ordered (node_id, timestamp) pairs from executing events
    execution_events = []  # list of (node_id_or_none, timestamp)
    _execution_start_ts = None

    for msg in messages:
        if not isinstance(msg, (list, tuple)) or len(msg) < 2:
            continue
        event_type = msg[0]
        event_data = msg[1] if isinstance(msg[1], dict) else {}

        if event_type == "execution_start":
            _execution_start_ts = event_data.get("timestamp")

        elif event_type == "executing":
            node_id = event_data.get("node")  # None means "all done"
            ts = event_data.get("timestamp")
            execution_events.append((node_id, ts))

    # Build node profiles from sequential executing events
    node_profiles = []
    for i, (node_id, ts) in enumerate(execution_events):
        if node_id is None:
            # This is the "execution complete" sentinel
            break

        class_type = "unknown"
        if node_id in workflow:
            class_type = workflow[node_id].get("class_type", "unknown")

        duration_ms = 0.0
        # If we have timestamps, calculate duration from gap to next event
        if ts is not None and i + 1 < len(execution_events):
            next_ts = execution_events[i + 1][1]
            if next_ts is not None:
                duration_ms = (next_ts - ts) * 1000

        node_profiles.append(
            NodeProfile(
                node_id=node_id,
                class_type=class_type,
                start_time=ts or 0.0,
                end_time=(
                    execution_events[i + 1][1] if i + 1 < len(execution_events) and execution_events[i + 1][1] else 0.0
                ),
                duration_ms=duration_ms,
            )
        )

    # Calculate total duration from first event to last
    total_duration_ms = 0.0
    if execution_events and len(execution_events) >= 2:
        first_ts = execution_events[0][1]
        last_ts = execution_events[-1][1]
        if first_ts is not None and last_ts is not None:
            total_duration_ms = (last_ts - first_ts) * 1000
    # Fallback: sum node durations
    if total_duration_ms == 0.0 and node_profiles:
        total_duration_ms = sum(n.duration_ms for n in node_profiles)

    # Build the profile
    profile = ExecutionProfile(
        prompt_id=prompt_id,
        total_duration_ms=total_duration_ms,
        node_profiles=node_profiles,
        status=status_info.get("status_str", "unknown"),
    )

    # Cache it
    _profiles[prompt_id] = profile
    # Evict old entries
    if len(_profiles) > MAX_PROFILES:
        oldest = sorted(_profiles.keys())[0]
        del _profiles[oldest]

    return profile.to_dict()


def list_profiles(limit: int = 10) -> dict:
    """List recent execution profiles."""
    profiles = list(_profiles.values())[-limit:]
    return {
        "profiles": [
            {
                "prompt_id": p.prompt_id,
                "total_duration_ms": round(p.total_duration_ms, 2),
                "node_count": len(p.node_profiles),
                "status": p.status,
            }
            for p in reversed(profiles)
        ],
        "count": len(profiles),
    }
