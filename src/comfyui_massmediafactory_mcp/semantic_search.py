"""
Semantic Pattern Search Module

Embedding-based semantic search for workflow patterns using cosine similarity.
Replaces keyword search with conceptually-aware pattern discovery.

Key features:
- In-memory embeddings (no external API required)
- Fast vector similarity search (<100ms query time)
- Sentence-transformers for encoding
- Cache embeddings for performance

Dependencies:
- sentence-transformers (for embeddings)
- numpy (for vector operations)

Usage:
    from semantic_search import search_patterns_semantic
    results = search_patterns_semantic("smooth camera movement", limit=5)
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import warnings

# Try to import sentence-transformers, provide fallback
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not installed. Semantic search will use keyword fallback.")


# =============================================================================
# Pattern Database
# =============================================================================

PATTERN_DESCRIPTIONS = {
    # LTX Patterns
    "ltx2_txt2vid": {
        "description": "High-quality text-to-video generation with smooth motion and natural camera movements",
        "tags": ["video", "generation", "motion", "camera", "smooth", "natural"],
        "use_cases": ["animating text descriptions", "creating video content", "motion graphics"],
    },
    "ltx2_txt2vid_distilled": {
        "description": "Fast distilled text-to-video with 3-4x speedup, slightly lower quality but much faster",
        "tags": ["video", "fast", "distilled", "quick", "speed", "efficient"],
        "use_cases": ["rapid prototyping", "quick previews", "testing prompts"],
    },
    "ltx2_img2vid": {
        "description": "Image-to-video animation that brings static images to life with motion",
        "tags": ["video", "image", "animation", "motion", "animate", "i2v"],
        "use_cases": ["animating images", "image motion", "bringing photos to life"],
    },
    
    # FLUX Patterns
    "flux2_txt2img": {
        "description": "High-quality text-to-image generation with excellent prompt adherence and detail",
        "tags": ["image", "generation", "high-quality", "detailed", "prompt-adherent"],
        "use_cases": ["creating images from text", "illustration", "concept art"],
    },
    
    # Wan Patterns
    "wan26_img2vid": {
        "description": "Advanced image-to-video with superior motion quality and temporal consistency",
        "tags": ["video", "image", "animation", "motion", "high-quality", "consistent"],
        "use_cases": ["professional video animation", "high-quality motion", "film production"],
    },
    "wan26_txt2vid": {
        "description": "High-fidelity text-to-video with excellent human motion and scene understanding",
        "tags": ["video", "generation", "human-motion", "scene", "high-fidelity"],
        "use_cases": ["human animation", "scene generation", "storytelling"],
    },
    
    # Qwen Patterns
    "qwen_txt2img": {
        "description": "Text-to-image specialized for photorealistic portraits and accurate text rendering",
        "tags": ["image", "portrait", "text", "photorealistic", "accurate"],
        "use_cases": ["portraits", "text in images", "photorealistic scenes"],
    },
    
    # SDXL Patterns
    "sdxl_txt2img": {
        "description": "Balanced text-to-image with good quality and reasonable generation time",
        "tags": ["image", "balanced", "general-purpose", "fast"],
        "use_cases": ["general image generation", "quick results", "everyday use"],
    },
    
    # Hunyuan Patterns
    "hunyuan15_txt2vid": {
        "description": "Large-scale video generation with complex scene handling and multiple subjects",
        "tags": ["video", "complex", "scenes", "multiple-subjects", "large-scale"],
        "use_cases": ["complex scenes", "multiple objects", "detailed environments"],
    },
    "hunyuan15_img2vid": {
        "description": "Complex image-to-video with detailed motion and scene understanding",
        "tags": ["video", "image", "complex", "detailed", "motion"],
        "use_cases": ["complex animations", "detailed motion", "rich environments"],
    },
}


# =============================================================================
# Embedding Cache
# =============================================================================

@dataclass
class CachedEmbedding:
    """Cached embedding with metadata."""
    pattern_id: str
    embedding: np.ndarray
    created_at: float


class EmbeddingCache:
    """In-memory cache for pattern embeddings."""
    
    def __init__(self, ttl_seconds: float = 3600):  # 1 hour default
        self._cache: Dict[str, CachedEmbedding] = {}
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0
    
    def get(self, pattern_id: str) -> Optional[np.ndarray]:
        """Get cached embedding if still valid."""
        cached = self._cache.get(pattern_id)
        if cached is None:
            self._misses += 1
            return None
        
        # Check TTL
        if time.time() - cached.created_at > self._ttl_seconds:
            del self._cache[pattern_id]
            self._misses += 1
            return None
        
        self._hits += 1
        return cached.embedding
    
    def set(self, pattern_id: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        self._cache[pattern_id] = CachedEmbedding(
            pattern_id=pattern_id,
            embedding=embedding,
            created_at=time.time(),
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "entries": len(self._cache),
            "ttl_seconds": self._ttl_seconds,
        }
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# Global cache instance
_embedding_cache = EmbeddingCache()


# =============================================================================
# Embedding Model
# =============================================================================

class EmbeddingModel:
    """Wrapper for sentence-transformers model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: Sentence-transformers model name
                       all-MiniLM-L6-v2 is fast (~50MB, 22M params)
        """
        self._model_name = model_name
        self._model = None
        self._initialized = False
        
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            return
        
        try:
            # Lazy initialization - load model on first use
            pass
        except Exception as e:
            warnings.warn(f"Failed to initialize embedding model: {e}")
    
    def _ensure_initialized(self) -> bool:
        """Lazy initialization of the model."""
        if self._initialized:
            return True
        
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            self._model = SentenceTransformer(self._model_name)
            self._initialized = True
            return True
        except Exception as e:
            warnings.warn(f"Failed to load embedding model: {e}")
            return False
    
    def encode(self, text: str) -> Optional[np.ndarray]:
        """
        Encode text to embedding vector.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector or None if model unavailable
        """
        if not self._ensure_initialized():
            return None
        
        # Encode and normalize
        embedding = self._model.encode(text, convert_to_numpy=True)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def encode_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Encode multiple texts to embedding vectors.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embedding vectors or None if model unavailable
        """
        if not self._ensure_initialized():
            return None
        
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings


# Global model instance (lazy-loaded)
_embedding_model = EmbeddingModel()


def _get_pattern_text(pattern_id: str) -> str:
    """
    Generate rich text description for a pattern.
    
    Combines description, tags, and use cases into searchable text.
    """
    info = PATTERN_DESCRIPTIONS.get(pattern_id, {})
    
    parts = []
    
    if "description" in info:
        parts.append(info["description"])
    
    if "tags" in info:
        parts.append("Keywords: " + ", ".join(info["tags"]))
    
    if "use_cases" in info:
        parts.append("Use cases: " + ", ".join(info["use_cases"]))
    
    # Add pattern ID for reference
    parts.append(f"Pattern: {pattern_id}")
    
    return " | ".join(parts)


def _compute_similarity(query_embedding: np.ndarray, pattern_embedding: np.ndarray) -> float:
    """
    Compute cosine similarity between two normalized embeddings.
    
    Returns similarity score between -1 and 1 (1 = identical).
    """
    return float(np.dot(query_embedding, pattern_embedding))


# =============================================================================
# Public API
# =============================================================================

def search_patterns_semantic(
    query: str,
    limit: int = 10,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """
    Semantic search for workflow patterns using embeddings.
    
    Uses cosine similarity on sentence-transformer embeddings to find
    conceptually similar patterns, not just keyword matches.
    
    Args:
        query: Natural language search query
        limit: Maximum results to return
        min_score: Minimum similarity score (0-1, default 0.0 = all results)
        
    Returns:
        {
            "results": [
                {
                    "pattern_id": str,
                    "score": float,
                    "description": str,
                    "tags": list,
                    "use_cases": list,
                }
            ],
            "query": str,
            "query_time_ms": float,
            "method": "semantic" or "keyword_fallback",
            "total_available": int,
        }
        
    Example:
        >>> results = search_patterns_semantic("smooth camera movement", limit=3)
        >>> print(results["results"][0]["pattern_id"])
        "ltx2_txt2vid"
    """
    start_time = time.time()
    
    # Check if semantic search is available
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        # Fallback to keyword search
        return _keyword_fallback_search(query, limit, start_time)
    
    # Encode query
    query_embedding = _embedding_model.encode(query)
    if query_embedding is None:
        return _keyword_fallback_search(query, limit, start_time)
    
    # Score all patterns
    results = []
    for pattern_id in PATTERN_DESCRIPTIONS.keys():
        # Try cache first
        pattern_embedding = _embedding_cache.get(pattern_id)
        
        if pattern_embedding is None:
            # Compute and cache
            pattern_text = _get_pattern_text(pattern_id)
            pattern_embedding = _embedding_model.encode(pattern_text)
            if pattern_embedding is not None:
                _embedding_cache.set(pattern_id, pattern_embedding)
        
        if pattern_embedding is not None:
            score = _compute_similarity(query_embedding, pattern_embedding)
            
            if score >= min_score:
                info = PATTERN_DESCRIPTIONS[pattern_id]
                results.append({
                    "pattern_id": pattern_id,
                    "score": round(score, 4),
                    "description": info.get("description", ""),
                    "tags": info.get("tags", []),
                    "use_cases": info.get("use_cases", []),
                })
    
    # Sort by score (descending)
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Apply limit
    limited_results = results[:limit]
    
    query_time_ms = (time.time() - start_time) * 1000
    
    return {
        "results": limited_results,
        "query": query,
        "query_time_ms": round(query_time_ms, 2),
        "method": "semantic",
        "total_available": len(PATTERN_DESCRIPTIONS),
        "cache_stats": _embedding_cache.get_stats(),
    }


def _keyword_fallback_search(
    query: str,
    limit: int,
    start_time: float,
) -> Dict[str, Any]:
    """
    Keyword-based fallback when embeddings unavailable.
    
    Simple word matching - less accurate but always available.
    """
    query_words = set(query.lower().split())
    
    results = []
    for pattern_id, info in PATTERN_DESCRIPTIONS.items():
        score = 0
        
        # Match in description
        desc_words = set(info.get("description", "").lower().split())
        score += len(query_words & desc_words)
        
        # Match in tags (weighted higher)
        for tag in info.get("tags", []):
            for word in query_words:
                if word in tag.lower():
                    score += 2
        
        # Match in use cases
        for use_case in info.get("use_cases", []):
            case_words = set(use_case.lower().split())
            score += len(query_words & case_words)
        
        if score > 0:
            results.append({
                "pattern_id": pattern_id,
                "score": score,
                "description": info.get("description", ""),
                "tags": info.get("tags", []),
                "use_cases": info.get("use_cases", []),
            })
    
    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    query_time_ms = (time.time() - start_time) * 1000
    
    return {
        "results": results[:limit],
        "query": query,
        "query_time_ms": round(query_time_ms, 2),
        "method": "keyword_fallback",
        "total_available": len(PATTERN_DESCRIPTIONS),
        "note": "sentence-transformers not available, using keyword matching",
    }


def get_pattern_info(pattern_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific pattern.
    
    Args:
        pattern_id: Pattern identifier (e.g., "ltx2_txt2vid")
        
    Returns:
        Pattern details or None if not found
    """
    info = PATTERN_DESCRIPTIONS.get(pattern_id)
    if info is None:
        return None
    
    return {
        "pattern_id": pattern_id,
        **info,
    }


def list_all_patterns() -> Dict[str, Any]:
    """
    List all available patterns with descriptions.
    
    Returns:
        {
            "patterns": [
                {
                    "pattern_id": str,
                    "description": str,
                    "tags": list,
                }
            ],
            "count": int,
        }
    """
    patterns = []
    for pattern_id, info in PATTERN_DESCRIPTIONS.items():
        patterns.append({
            "pattern_id": pattern_id,
            "description": info.get("description", ""),
            "tags": info.get("tags", []),
        })
    
    return {
        "patterns": patterns,
        "count": len(patterns),
    }


def get_embedding_cache_stats() -> Dict[str, Any]:
    """Get statistics about the embedding cache."""
    return _embedding_cache.get_stats()


def clear_embedding_cache() -> None:
    """Clear all cached embeddings."""
    _embedding_cache.clear()
