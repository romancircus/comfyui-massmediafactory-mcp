"""
Prompt Enhancement for MassMediaFactory MCP

Rewrites generation prompts with model-specific quality tokens using local Ollama LLM.
Pattern from: ComfyUI-IF_AI_tools, ComfyUI-Copilot
"""

import os
import json
import urllib.request
from typing import Optional

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_LLM = os.environ.get("ENHANCE_LLM_MODEL", "qwen3:8b")

# Model-specific quality tokens and style guides
MODEL_QUALITY_TOKENS = {
    "flux": {
        "prefix": "",
        "suffix": ", highly detailed, professional photography, sharp focus, 8k uhd",
        "style_guide": "FLUX works best with natural language descriptions. Be descriptive about lighting, composition, and mood.",
    },
    "sdxl": {
        "prefix": "masterpiece, best quality, ",
        "suffix": ", highly detailed, sharp focus",
        "style_guide": "SDXL responds well to quality tags at the start. Use comma-separated descriptors.",
    },
    "qwen": {
        "prefix": "",
        "suffix": ", high quality, detailed, professional",
        "style_guide": "Qwen works well with clear, descriptive prompts. Focus on subject and composition.",
    },
    "wan": {
        "prefix": "",
        "suffix": ", cinematic, smooth motion, high quality",
        "style_guide": "Wan video generation works best with motion descriptions and cinematic terms.",
    },
    "ltx": {
        "prefix": "",
        "suffix": ", smooth motion, consistent, high quality video",
        "style_guide": "LTX video works best with clear motion descriptions. Avoid rapid scene changes.",
    },
}


def enhance_prompt(
    prompt: str,
    model: str = "flux",
    style: str = None,
    use_llm: bool = True,
    llm_model: str = None,
) -> dict:
    """
    Enhance a generation prompt with model-specific quality tokens.

    Args:
        prompt: The original prompt to enhance.
        model: Target generation model (flux, sdxl, qwen, wan, ltx).
        style: Optional style preset (e.g., "cinematic", "anime", "photorealistic").
        use_llm: Whether to use Ollama LLM for intelligent rewriting (default True).
        llm_model: Which Ollama model to use (default: qwen3:8b).

    Returns:
        {
            "original": "...",
            "enhanced": "...",
            "method": "llm" | "tokens",
            "model_tokens": {...},
        }
    """
    if llm_model is None:
        llm_model = DEFAULT_LLM

    # Normalize model name
    model_key = model.lower().rstrip("0123456789_")
    if model_key not in MODEL_QUALITY_TOKENS:
        model_key = "flux"  # Default fallback

    tokens = MODEL_QUALITY_TOKENS[model_key]

    # Try LLM enhancement first
    if use_llm:
        llm_enhanced = _enhance_with_llm(prompt, model_key, tokens, style, llm_model)
        if llm_enhanced:
            return {
                "original": prompt,
                "enhanced": llm_enhanced,
                "method": "llm",
                "llm_model": llm_model,
                "target_model": model,
            }

    # Fallback: simple token injection
    enhanced = _enhance_with_tokens(prompt, tokens, style)
    return {
        "original": prompt,
        "enhanced": enhanced,
        "method": "tokens",
        "target_model": model,
    }


def _enhance_with_llm(
    prompt: str,
    model_key: str,
    tokens: dict,
    style: str = None,
    llm_model: str = DEFAULT_LLM,
) -> Optional[str]:
    """Enhance prompt using local Ollama LLM."""
    style_instruction = f" The style should be {style}." if style else ""

    system_prompt = f"""You are an expert at writing prompts for AI image/video generation models.
Your task: Rewrite the user's prompt to get better results from the {model_key} model.

Rules:
- {tokens['style_guide']}
- Keep the core subject and intent unchanged
- Add specific visual details: lighting, composition, colors, textures
- Use natural language, not just comma-separated tags
- Output ONLY the enhanced prompt, nothing else (no explanation, no quotes)
- Keep the prompt under 200 words{style_instruction}"""

    try:
        payload = {
            "model": llm_model,
            "prompt": f"Enhance this prompt for {model_key}: {prompt}",
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 300},
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            enhanced = result.get("response", "").strip()
            # Strip any wrapping quotes
            if enhanced.startswith('"') and enhanced.endswith('"'):
                enhanced = enhanced[1:-1]
            if enhanced.startswith("'") and enhanced.endswith("'"):
                enhanced = enhanced[1:-1]
            return enhanced if enhanced else None

    except Exception:
        return None


def _enhance_with_tokens(prompt: str, tokens: dict, style: str = None) -> str:
    """Enhance prompt with simple token injection (no LLM needed)."""
    parts = []
    if tokens.get("prefix"):
        parts.append(tokens["prefix"])
    parts.append(prompt)
    if style:
        parts.append(f", {style} style")
    if tokens.get("suffix"):
        parts.append(tokens["suffix"])
    return "".join(parts)


def check_llm_available(model: str = None) -> dict:
    """Check if LLM is available for prompt enhancement."""
    if model is None:
        model = DEFAULT_LLM

    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])]
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
        }
