#!/usr/bin/env python3
"""
Linear integration hook for ComfyUI MassmediaFactory MCP.
Updates Linear issues when jobs complete.

Usage:
    from scripts.linear_hook import update_linear_on_complete

    # After job completes:
    update_linear_on_complete(
        issue_id="ROM-XX",
        status="done",  # or "qa-failed"
        output_path="/path/to/output.mp4",
        notes="Optional notes"
    )
"""

import os
import json
import requests
from typing import Optional

LINEAR_API_URL = "https://api.linear.app/graphql"
LINEAR_API_KEY = os.environ.get("LINEAR_API_KEY")


def linear_query(query: str, variables: dict = None) -> dict:
    """Execute a GraphQL query against Linear API."""
    if not LINEAR_API_KEY:
        raise ValueError("LINEAR_API_KEY environment variable not set")

    headers = {
        "Content-Type": "application/json",
        "Authorization": LINEAR_API_KEY
    }

    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    response = requests.post(LINEAR_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def get_issue_uuid(identifier: str) -> Optional[str]:
    """Get the UUID of an issue from its identifier (e.g., ROM-64)."""
    query = """
    query GetIssue($id: String!) {
        issue(id: $id) {
            id
        }
    }
    """
    result = linear_query(query, {"id": identifier})
    if result.get("data", {}).get("issue"):
        return result["data"]["issue"]["id"]
    return None


def get_state_id(state_name: str) -> Optional[str]:
    """Get the UUID of a workflow state by name."""
    query = """
    query GetStates($name: String!) {
        workflowStates(filter: { name: { eq: $name } }) {
            nodes {
                id
                name
            }
        }
    }
    """
    result = linear_query(query, {"name": state_name})
    nodes = result.get("data", {}).get("workflowStates", {}).get("nodes", [])
    if nodes:
        return nodes[0]["id"]
    return None


def get_label_id(label_name: str) -> Optional[str]:
    """Get the UUID of a label by name."""
    query = """
    query GetLabels($name: String!) {
        issueLabels(filter: { name: { eq: $name } }) {
            nodes {
                id
            }
        }
    }
    """
    result = linear_query(query, {"name": label_name})
    nodes = result.get("data", {}).get("issueLabels", {}).get("nodes", [])
    if nodes:
        return nodes[0]["id"]
    return None


def update_issue_state(issue_uuid: str, state_id: str) -> bool:
    """Update an issue's state."""
    mutation = """
    mutation UpdateIssue($id: String!, $stateId: String!) {
        issueUpdate(id: $id, input: { stateId: $stateId }) {
            success
        }
    }
    """
    result = linear_query(mutation, {"id": issue_uuid, "stateId": state_id})
    return result.get("data", {}).get("issueUpdate", {}).get("success", False)


def add_issue_comment(issue_uuid: str, body: str) -> bool:
    """Add a comment to an issue."""
    mutation = """
    mutation CreateComment($issueId: String!, $body: String!) {
        commentCreate(input: { issueId: $issueId, body: $body }) {
            success
        }
    }
    """
    result = linear_query(mutation, {"issueId": issue_uuid, "body": body})
    return result.get("data", {}).get("commentCreate", {}).get("success", False)


def add_issue_label(issue_uuid: str, label_id: str) -> bool:
    """Add a label to an issue."""
    # First get current labels
    query = """
    query GetIssueLabels($id: String!) {
        issue(id: $id) {
            labels {
                nodes {
                    id
                }
            }
        }
    }
    """
    result = linear_query(query, {"id": issue_uuid})
    current_labels = [n["id"] for n in result.get("data", {}).get("issue", {}).get("labels", {}).get("nodes", [])]

    # Add new label
    new_labels = list(set(current_labels + [label_id]))

    mutation = """
    mutation UpdateIssue($id: String!, $labelIds: [String!]!) {
        issueUpdate(id: $id, input: { labelIds: $labelIds }) {
            success
        }
    }
    """
    result = linear_query(mutation, {"id": issue_uuid, "labelIds": new_labels})
    return result.get("data", {}).get("issueUpdate", {}).get("success", False)


def update_linear_on_complete(
    issue_id: str,
    status: str = "done",
    output_path: Optional[str] = None,
    notes: Optional[str] = None
) -> bool:
    """
    Update Linear issue when a ComfyUI job completes.

    Args:
        issue_id: Linear issue identifier (e.g., "ROM-64")
        status: "done" or "qa-failed"
        output_path: Path to generated output
        notes: Optional additional notes

    Returns:
        True if update was successful
    """
    if not LINEAR_API_KEY:
        print(f"[Linear Hook] Skipping - no API key configured")
        return False

    try:
        # Get issue UUID
        issue_uuid = get_issue_uuid(issue_id)
        if not issue_uuid:
            print(f"[Linear Hook] Issue not found: {issue_id}")
            return False

        success = True

        # Update state
        if status == "done":
            state_id = get_state_id("Done")
            if state_id:
                success = update_issue_state(issue_uuid, state_id) and success

            # Add qa-passed label
            label_id = get_label_id("qa-passed")
            if label_id:
                add_issue_label(issue_uuid, label_id)

        elif status == "qa-failed":
            # Add qa-failed label but keep in progress
            label_id = get_label_id("qa-failed")
            if label_id:
                add_issue_label(issue_uuid, label_id)

        # Add comment
        comment_parts = []
        if status == "done":
            comment_parts.append("**ComfyUI Job Completed**")
        else:
            comment_parts.append("**ComfyUI Job Failed QA**")

        if output_path:
            comment_parts.append(f"Output: `{output_path}`")
        if notes:
            comment_parts.append(f"Notes: {notes}")

        comment = "\n".join(comment_parts)
        success = add_issue_comment(issue_uuid, comment) and success

        print(f"[Linear Hook] Updated {issue_id} -> {status}")
        return success

    except Exception as e:
        print(f"[Linear Hook] Error updating {issue_id}: {e}")
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python linear_hook.py ISSUE_ID STATUS [OUTPUT_PATH] [NOTES]")
        sys.exit(1)

    issue_id = sys.argv[1]
    status = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    notes = sys.argv[4] if len(sys.argv) > 4 else None

    success = update_linear_on_complete(issue_id, status, output_path, notes)
    sys.exit(0 if success else 1)
