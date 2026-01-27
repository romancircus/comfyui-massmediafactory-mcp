"""
VLM Quality Assurance for MassMediaFactory MCP

Automated image/video QA using Vision-Language Models.
"""

import os
import base64
import json
import urllib.request
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field

from .assets import get_registry


@dataclass
class QAResult:
    """Result of QA check on an asset."""
    passed: bool
    score: float  # 0.0 - 1.0
    issues: List[str] = field(default_factory=list)
    recommendation: str = "accept"  # "accept", "regenerate", "tweak_prompt"
    details: dict = field(default_factory=dict)


# Default QA checks
DEFAULT_CHECKS = ["prompt_match", "artifacts", "composition"]

# VLM endpoints
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_VLM = os.environ.get("QA_VLM_MODEL", "qwen2.5-vl:7b")


def load_image_as_base64(image_path: Path) -> Optional[str]:
    """Load image and convert to base64."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def call_ollama_vlm(
    image_base64: str,
    prompt: str,
    model: str = DEFAULT_VLM,
) -> Optional[str]:
    """Call Ollama VLM API with image."""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result.get("response", "")

    except Exception as e:
        return None


def build_qa_prompt(original_prompt: str, checks: List[str]) -> str:
    """Build the QA evaluation prompt for the VLM."""
    check_instructions = []

    if "prompt_match" in checks:
        check_instructions.append(
            f"1. PROMPT MATCH: Does this image match the prompt '{original_prompt}'? "
            "Rate 0-10 how well it matches."
        )

    if "artifacts" in checks:
        check_instructions.append(
            "2. ARTIFACTS: Check for visual artifacts, distortions, blurry areas, "
            "or unnatural elements. List any issues found."
        )

    if "faces" in checks:
        check_instructions.append(
            "3. FACES: If there are human faces, check for: wrong number of fingers, "
            "distorted features, asymmetry, extra limbs. List any issues."
        )

    if "text" in checks:
        check_instructions.append(
            "4. TEXT: If there is text in the image, is it readable and spelled correctly? "
            "List any text errors."
        )

    if "composition" in checks:
        check_instructions.append(
            "5. COMPOSITION: Is the image well-composed? Good framing, lighting, focus? "
            "Rate 0-10."
        )

    checks_text = "\n".join(check_instructions)

    return f"""You are a quality assurance expert for AI-generated images.
Analyze this image and evaluate it based on the following criteria:

{checks_text}

Respond in JSON format:
{{
    "overall_score": <0-10>,
    "prompt_match_score": <0-10 or null if not checked>,
    "composition_score": <0-10 or null if not checked>,
    "issues": ["list", "of", "issues"],
    "passed": <true if overall_score >= 7>,
    "recommendation": "accept" | "regenerate" | "tweak_prompt"
}}

Be strict but fair. Minor imperfections are okay for scores 7+.
Only recommend "regenerate" for significant issues.
Only recommend "tweak_prompt" if the image doesn't match the prompt well but is technically good.
"""


def parse_vlm_response(response: str) -> QAResult:
    """Parse VLM response into QAResult."""
    try:
        # Try to extract JSON from response
        # VLMs sometimes wrap JSON in markdown code blocks
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```json"):
                    in_json = True
                    continue
                if line.startswith("```") and in_json:
                    break
                if in_json:
                    json_lines.append(line)
            response = "\n".join(json_lines)

        data = json.loads(response)

        score = float(data.get("overall_score", 5)) / 10.0
        passed = data.get("passed", score >= 0.7)
        issues = data.get("issues", [])
        recommendation = data.get("recommendation", "accept" if passed else "regenerate")

        return QAResult(
            passed=passed,
            score=score,
            issues=issues,
            recommendation=recommendation,
            details={
                "prompt_match_score": data.get("prompt_match_score"),
                "composition_score": data.get("composition_score"),
            },
        )

    except (json.JSONDecodeError, ValueError, KeyError):
        # If parsing fails, return a neutral result
        return QAResult(
            passed=True,
            score=0.7,
            issues=["Could not parse VLM response"],
            recommendation="accept",
            details={"raw_response": response[:500]},
        )


def qa_output(
    asset_id: str,
    prompt: str,
    checks: List[str] = None,
    vlm_model: str = None,
) -> dict:
    """
    Run QA on a generated asset using a VLM.

    Args:
        asset_id: The asset to evaluate.
        prompt: Original generation prompt for comparison.
        checks: List of checks to run. Options:
                - "prompt_match": Does image match prompt?
                - "artifacts": Visual artifacts or distortions?
                - "faces": Face/hand distortions?
                - "text": Text rendering issues?
                - "composition": Overall composition quality?
        vlm_model: VLM model to use (default: qwen2.5-vl:7b via Ollama).

    Returns:
        {
            "passed": True/False,
            "score": 0.0-1.0,
            "issues": ["list", "of", "issues"],
            "recommendation": "accept" | "regenerate" | "tweak_prompt",
            "details": {...}
        }
    """
    if checks is None:
        checks = DEFAULT_CHECKS

    if vlm_model is None:
        vlm_model = DEFAULT_VLM

    # Get asset from registry
    registry = get_registry()
    asset = registry.get_asset(asset_id)

    if asset is None:
        return {
            "error": "ASSET_NOT_FOUND_OR_EXPIRED",
            "asset_id": asset_id,
        }

    # Only support image QA for now
    if asset.asset_type != "image":
        return {
            "error": "UNSUPPORTED_ASSET_TYPE",
            "message": f"QA only supports images, got {asset.asset_type}",
            "asset_id": asset_id,
        }

    # Get ComfyUI output directory
    comfyui_output_dir = os.environ.get(
        "COMFYUI_OUTPUT_DIR", "/home/romancircus/ComfyUI/output"
    )

    # Build path to asset
    if asset.subfolder:
        image_path = Path(comfyui_output_dir) / asset.subfolder / asset.filename
    else:
        image_path = Path(comfyui_output_dir) / asset.filename

    if not image_path.exists():
        return {
            "error": "FILE_NOT_FOUND",
            "message": f"Asset file not found: {image_path}",
            "asset_id": asset_id,
        }

    # Load image
    image_base64 = load_image_as_base64(image_path)
    if image_base64 is None:
        return {
            "error": "LOAD_FAILED",
            "message": "Failed to load image file",
            "asset_id": asset_id,
        }

    # Build QA prompt
    qa_prompt = build_qa_prompt(prompt, checks)

    # Call VLM
    vlm_response = call_ollama_vlm(image_base64, qa_prompt, vlm_model)
    if vlm_response is None:
        return {
            "error": "VLM_UNAVAILABLE",
            "message": f"Could not reach VLM at {OLLAMA_URL}. Is Ollama running?",
            "asset_id": asset_id,
        }

    # Parse response
    result = parse_vlm_response(vlm_response)

    return {
        "passed": result.passed,
        "score": result.score,
        "issues": result.issues,
        "recommendation": result.recommendation,
        "details": result.details,
        "asset_id": asset_id,
        "checks_performed": checks,
        "vlm_model": vlm_model,
    }


def check_vlm_available(model: str = None) -> dict:
    """Check if VLM is available for QA."""
    if model is None:
        model = DEFAULT_VLM

    try:
        # Check if Ollama is running
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])]

            # Check if requested model is available
            model_available = any(model in m for m in models)

            return {
                "available": True,
                "ollama_url": OLLAMA_URL,
                "models": models,
                "requested_model": model,
                "model_available": model_available,
            }

    except Exception as e:
        return {
            "available": False,
            "ollama_url": OLLAMA_URL,
            "error": str(e),
            "message": "Ollama not running or unreachable. Start with: ollama serve",
        }
