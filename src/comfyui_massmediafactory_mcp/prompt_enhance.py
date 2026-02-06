"""
Prompt Enhancement for MassMediaFactory MCP

Rewrites generation prompts with model-specific quality tokens using local Ollama LLM.
Includes SOTA-aligned video prompt engineering for WAN I2V and LTX-2 T2V.
Pattern from: ComfyUI-IF_AI_tools, ComfyUI-Copilot
"""

import os
import json
import urllib.request
from typing import Optional

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_LLM = os.environ.get("ENHANCE_LLM_MODEL", "qwen3:8b")

# Model-specific quality tokens, style guides, and negative prompt defaults
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
        "suffix": "",  # No suffix — WAN I2V prompts are motion-only
        "style_guide": (
            "WAN 2.1 I2V: Write motion-only prompts (80-120 words). "
            "The input image provides all visual context — DO NOT describe subject appearance. "
            "Describe: creature actions, body movements, environmental reactions. "
            "Use temporal pacing words (slowly, gradually, rhythmically). "
            "ONE camera move maximum (prefer 'camera slowly orbits'). "
            "Start with the primary action verb. "
            "Structure: [ACTION] [BODY MOTION] [ENVIRONMENT REACTION] [PACING]."
        ),
        "negative_default": (
            "static image, frozen, sudden motion, flickering, morphing, "
            "shape shifting, identity change, camera shake, rapid cuts, "
            "text, watermark, blurry, low quality, jerky motion, "
            "multiple subjects, extra limbs"
        ),
    },
    "ltx": {
        "prefix": "",
        "suffix": "",  # No suffix — LTX prompts are self-contained
        "style_guide": (
            "LTX-2: Write filmmaker shot notes (4-8 sentences). "
            "Put concrete nouns and verbs FIRST — early tokens carry more weight. "
            "Describe: subject, action, environment, lighting, atmosphere. "
            "NO camera keywords (pan, zoom, dolly — LTX handles camera internally). "
            "Be specific about textures and materials. "
            "Include temporal flow: what happens first, then, finally."
        ),
        "negative_default": (
            "camera movement keywords, pan, zoom, dolly, sudden cuts, flickering, "
            "morphing, blurry, text, watermark, low quality, "
            "inconsistent lighting, jumpy motion"
        ),
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
            "negative": "..." | None,
            "method": "llm" | "tokens",
            "target_model": "...",
        }
    """
    if llm_model is None:
        llm_model = DEFAULT_LLM

    # Normalize model name
    model_key = model.lower().rstrip("0123456789_")
    if model_key not in MODEL_QUALITY_TOKENS:
        model_key = "flux"  # Default fallback

    tokens = MODEL_QUALITY_TOKENS[model_key]

    # Build negative prompt for models that support it
    negative = _build_negative(prompt, model_key)

    # Try LLM enhancement first
    if use_llm:
        llm_enhanced = _enhance_with_llm(prompt, model_key, tokens, style, llm_model)
        if llm_enhanced:
            return {
                "original": prompt,
                "enhanced": llm_enhanced,
                "negative": negative,
                "method": "llm",
                "llm_model": llm_model,
                "target_model": model,
            }

    # Fallback: simple token injection
    enhanced = _enhance_with_tokens(prompt, tokens, style)
    return {
        "original": prompt,
        "enhanced": enhanced,
        "negative": negative,
        "method": "tokens",
        "target_model": model,
    }


def _build_negative(prompt: str, model_key: str) -> Optional[str]:
    """
    Build a model-specific negative prompt.

    Combines the model's default negatives with prompt-aware additions
    (e.g., if prompt mentions fire, add ice/frozen to negatives).

    Args:
        prompt: The original positive prompt.
        model_key: Normalized model key (wan, ltx, flux, etc.)

    Returns:
        Negative prompt string, or None if model has no negative defaults.
    """
    tokens = MODEL_QUALITY_TOKENS.get(model_key, {})
    base_negative = tokens.get("negative_default")
    if not base_negative:
        return None

    # Prompt-aware negative additions
    prompt_lower = prompt.lower()
    additions = []

    # Element-based opposites
    if any(w in prompt_lower for w in ("fire", "flame", "burn", "ember")):
        additions.append("ice, frozen, cold")
    if any(w in prompt_lower for w in ("water", "ocean", "rain", "aquatic")):
        additions.append("dry, desert, arid")
    if any(w in prompt_lower for w in ("flying", "soaring", "hovering")):
        additions.append("grounded, falling, sinking")
    if any(w in prompt_lower for w in ("still", "calm", "serene")):
        additions.append("chaotic, turbulent")

    if additions:
        return f"{base_negative}, {', '.join(additions)}"
    return base_negative


def _enhance_with_llm(
    prompt: str,
    model_key: str,
    tokens: dict,
    style: str = None,
    llm_model: str = DEFAULT_LLM,
) -> Optional[str]:
    """Enhance prompt using local Ollama LLM."""
    style_instruction = f" The style should be {style}." if style else ""

    # Video models get specialized system prompts
    is_video_model = model_key in ("wan", "ltx")
    if is_video_model:
        word_limit = "80-120 words" if model_key == "wan" else "200 words"
        system_prompt = f"""You are an expert at writing prompts for AI video generation models.
Your task: Rewrite the user's prompt to get better results from the {model_key} model.

Rules:
- {tokens['style_guide']}
- Keep the core subject and intent unchanged
- Focus on MOTION and ACTION, not static descriptions
- Use temporal pacing words: slowly, gradually, rhythmically, steadily
- Output ONLY the enhanced prompt, nothing else (no explanation, no quotes)
- Keep the prompt under {word_limit}{style_instruction}"""
    else:
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
