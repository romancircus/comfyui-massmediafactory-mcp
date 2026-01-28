"""
LLM Reference Documentation Module

Exposes documentation files for LLM agents to read patterns,
node specifications, and parameter rules.
"""

import os
import json
from pathlib import Path

# Get the docs directory relative to this module
MODULE_DIR = Path(__file__).parent
DOCS_DIR = MODULE_DIR.parent.parent / "docs"
REFERENCE_DIR = DOCS_DIR / "reference"
SKELETONS_DIR = DOCS_DIR / "library" / "skeletons"
PROMPT_GUIDES_DIR = DOCS_DIR / "prompt_guides"


def get_model_pattern(model: str) -> dict:
    """
    Get pattern documentation for a specific model.

    Args:
        model: Model identifier (ltx, flux, wan, qwen)

    Returns:
        Pattern documentation text or error.
    """
    patterns_file = REFERENCE_DIR / "01_MODEL_PATTERNS.md"
    if not patterns_file.exists():
        return {"error": f"Patterns file not found: {patterns_file}"}

    content = patterns_file.read_text()

    # Extract section for the requested model
    model_lower = model.lower()
    model_map = {
        "ltx": "Pattern: LTX-Video",
        "ltx2": "Pattern: LTX-Video",
        "ltx-video": "Pattern: LTX-Video",
        "ltx-2": "Pattern: LTX-2 Distilled",
        "ltx2-distilled": "Pattern: LTX-2 Distilled",
        "flux": "Pattern: FLUX",
        "flux2": "Pattern: FLUX",
        "wan": "Pattern: Wan 2.1",
        "wan2": "Pattern: Wan 2.1",
        "wan26": "Pattern: Wan 2.1",
        "qwen": "Pattern: Qwen",
    }

    pattern_header = model_map.get(model_lower)
    if not pattern_header:
        return {
            "error": f"Unknown model: {model}",
            "available": list(set(model_map.values()))
        }

    # Find the section
    lines = content.split("\n")
    section_lines = []
    in_section = False
    section_depth = 0

    for line in lines:
        if pattern_header in line and line.startswith("##"):
            in_section = True
            section_depth = line.count("#")
            section_lines.append(line)
        elif in_section:
            if line.startswith("#") and line.count("#") <= section_depth and line.strip() != "":
                # Hit next section at same or higher level
                break
            section_lines.append(line)

    if section_lines:
        return {
            "model": model,
            "pattern": "\n".join(section_lines),
            "source": "01_MODEL_PATTERNS.md"
        }

    return {"error": f"Pattern section not found for: {model}"}


def get_parameter_rules() -> dict:
    """
    Get all parameter validation rules.

    Returns:
        Parameter rules documentation.
    """
    rules_file = REFERENCE_DIR / "03_PARAMETER_RULES.md"
    if not rules_file.exists():
        return {"error": f"Rules file not found: {rules_file}"}

    return {
        "rules": rules_file.read_text(),
        "source": "03_PARAMETER_RULES.md"
    }


def get_skeleton(model: str, task: str = None) -> dict:
    """
    Get a workflow skeleton for a model.

    Args:
        model: Model identifier (ltx, flux, wan, qwen)
        task: Task type (t2v, i2v, t2i) - optional, will try to find matching

    Returns:
        Skeleton JSON or error.
    """
    model_lower = model.lower().replace("-", "_").replace("2", "")

    # Map common model names
    name_map = {
        "ltx": "ltx_video",
        "ltxvideo": "ltx_video",
        "flux": "flux_dev",
        "fluxdev": "flux_dev",
        "wan": "wan",
        "qwen": "qwen",
    }

    base_name = name_map.get(model_lower, model_lower)

    # Map task types
    task_map = {
        "t2v": "t2v",
        "txt2vid": "t2v",
        "text-to-video": "t2v",
        "i2v": "i2v",
        "img2vid": "i2v",
        "image-to-video": "i2v",
        "t2i": "t2i",
        "txt2img": "t2i",
        "text-to-image": "t2i",
    }

    task_suffix = task_map.get(task.lower() if task else "", "")

    # Try to find matching skeleton
    if task_suffix:
        skeleton_name = f"{base_name}_{task_suffix}.json"
    else:
        # Find any skeleton for this model
        for f in SKELETONS_DIR.glob(f"{base_name}*.json"):
            skeleton_name = f.name
            break
        else:
            skeleton_name = None

    if not skeleton_name:
        available = [f.stem for f in SKELETONS_DIR.glob("*.json")]
        return {
            "error": f"No skeleton found for model={model}, task={task}",
            "available": available
        }

    skeleton_file = SKELETONS_DIR / skeleton_name
    if not skeleton_file.exists():
        available = [f.stem for f in SKELETONS_DIR.glob("*.json")]
        return {
            "error": f"Skeleton file not found: {skeleton_name}",
            "available": available
        }

    return {
        "skeleton": json.loads(skeleton_file.read_text()),
        "name": skeleton_name,
        "source": f"library/skeletons/{skeleton_name}"
    }


def get_system_prompt() -> dict:
    """
    Get the system prompt guide for LLM workflow generation.

    Returns:
        System prompt content.
    """
    prompt_file = PROMPT_GUIDES_DIR / "SYSTEM_PROMPT.md"
    if not prompt_file.exists():
        return {"error": f"System prompt not found: {prompt_file}"}

    return {
        "system_prompt": prompt_file.read_text(),
        "source": "prompt_guides/SYSTEM_PROMPT.md"
    }


def list_available_skeletons() -> dict:
    """
    List all available workflow skeletons.

    Returns:
        List of skeleton names with metadata.
    """
    skeletons = []
    for f in SKELETONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            meta = data.get("_meta", {})
            skeletons.append({
                "name": f.stem,
                "model": meta.get("model", "unknown"),
                "workflow": meta.get("workflow", "unknown"),
                "sampler_type": meta.get("sampler_type", "unknown"),
            })
        except Exception:
            skeletons.append({"name": f.stem, "error": "Failed to parse"})

    return {"skeletons": skeletons, "count": len(skeletons)}


def search_patterns(query: str) -> dict:
    """
    Search pattern documentation for a query.

    Args:
        query: Search query (e.g., "image to video with LTX")

    Returns:
        Matching sections from documentation.
    """
    patterns_file = REFERENCE_DIR / "01_MODEL_PATTERNS.md"
    if not patterns_file.exists():
        return {"error": "Patterns file not found"}

    content = patterns_file.read_text()
    query_lower = query.lower()
    keywords = query_lower.split()

    # Find sections containing the keywords
    lines = content.split("\n")
    matches = []
    current_section = []
    current_header = ""

    for line in lines:
        if line.startswith("##"):
            # Save previous section if it matched
            if current_section:
                section_text = "\n".join(current_section)
                score = sum(1 for kw in keywords if kw in section_text.lower())
                if score > 0:
                    matches.append({
                        "header": current_header,
                        "score": score,
                        "preview": section_text[:500] + "..." if len(section_text) > 500 else section_text
                    })
            current_header = line
            current_section = [line]
        else:
            current_section.append(line)

    # Check last section
    if current_section:
        section_text = "\n".join(current_section)
        score = sum(1 for kw in keywords if kw in section_text.lower())
        if score > 0:
            matches.append({
                "header": current_header,
                "score": score,
                "preview": section_text[:500] + "..." if len(section_text) > 500 else section_text
            })

    # Sort by score descending
    matches.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": query,
        "matches": matches[:5],
        "total_matches": len(matches)
    }
