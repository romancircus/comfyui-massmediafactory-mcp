"""
Style Learning Module

Local storage for prompts, seeds, ratings, and outcomes to enable
"style learning" where successful generations inform future prompt enhancement.

Uses SQLite for persistence and simple vector similarity for prompt matching.
"""

import os
import json
import sqlite3
import hashlib
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

# Default database location
DEFAULT_DB_PATH = os.environ.get("COMFY_MCP_STYLE_DB", os.path.expanduser("~/.comfyui-mcp/style_learning.db"))


@dataclass
class GenerationRecord:
    """Record of a single generation with outcome."""

    id: str
    prompt: str
    negative_prompt: str
    model: str
    seed: int
    parameters: Dict[str, Any]
    rating: Optional[float]  # 0.0-1.0, None if not rated
    tags: List[str]
    outcome: str  # "success", "failed", "regenerated"
    qa_score: Optional[float]
    created_at: float
    notes: str


class StyleLearningDB:
    """
    SQLite-based storage for generation history and style learning.

    Enables:
    - Storing successful prompts/seeds for reuse
    - Learning from rated outputs
    - Finding similar past generations
    - Building prompt enhancement suggestions
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS generations (
                    id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    negative_prompt TEXT DEFAULT '',
                    model TEXT NOT NULL,
                    seed INTEGER,
                    parameters TEXT,
                    rating REAL,
                    tags TEXT,
                    outcome TEXT DEFAULT 'success',
                    qa_score REAL,
                    created_at REAL,
                    notes TEXT DEFAULT '',
                    prompt_hash TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prompt_hash ON generations(prompt_hash)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_model ON generations(model)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_rating ON generations(rating)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tags ON generations(tags)
            """
            )

            # Style presets table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS style_presets (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    prompt_additions TEXT,
                    negative_additions TEXT,
                    recommended_model TEXT,
                    recommended_params TEXT,
                    example_generations TEXT,
                    created_at REAL,
                    updated_at REAL
                )
            """
            )

            conn.commit()

    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash for prompt similarity matching."""
        # Normalize and hash
        normalized = " ".join(prompt.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def record_generation(
        self,
        prompt: str,
        model: str,
        seed: int,
        parameters: Dict[str, Any] = None,
        negative_prompt: str = "",
        rating: float = None,
        tags: List[str] = None,
        outcome: str = "success",
        qa_score: float = None,
        notes: str = "",
    ) -> str:
        """
        Record a generation for future learning.

        Args:
            prompt: The generation prompt.
            model: Model used (e.g., "flux2-dev").
            seed: Random seed.
            parameters: Full parameter dict.
            negative_prompt: Negative prompt if any.
            rating: User rating 0.0-1.0 (None if not rated).
            tags: Style tags (e.g., ["anime", "portrait"]).
            outcome: "success", "failed", or "regenerated".
            qa_score: Automated QA score if available.
            notes: Any additional notes.

        Returns:
            Record ID.
        """
        record_id = f"gen_{int(time.time() * 1000)}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO generations
                (id, prompt, negative_prompt, model, seed, parameters, rating, tags, outcome, qa_score, created_at, notes, prompt_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record_id,
                    prompt,
                    negative_prompt,
                    model,
                    seed,
                    json.dumps(parameters or {}),
                    rating,
                    json.dumps(tags or []),
                    outcome,
                    qa_score,
                    time.time(),
                    notes,
                    self._hash_prompt(prompt),
                ),
            )
            conn.commit()

        return record_id

    def rate_generation(self, record_id: str, rating: float, notes: str = None) -> bool:
        """
        Rate a generation (0.0-1.0).

        Args:
            record_id: The generation record ID.
            rating: Rating between 0.0 and 1.0.
            notes: Optional notes about the rating.

        Returns:
            True if updated successfully.
        """
        with sqlite3.connect(self.db_path) as conn:
            if notes:
                conn.execute("UPDATE generations SET rating = ?, notes = ? WHERE id = ?", (rating, notes, record_id))
            else:
                conn.execute("UPDATE generations SET rating = ? WHERE id = ?", (rating, record_id))
            conn.commit()
            return conn.total_changes > 0

    def find_similar_prompts(
        self,
        prompt: str,
        model: str = None,
        min_rating: float = 0.7,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find similar past generations with good ratings.

        Args:
            prompt: The prompt to match against.
            model: Optional model filter.
            min_rating: Minimum rating threshold.
            limit: Maximum results.

        Returns:
            List of similar generation records.
        """
        prompt_hash = self._hash_prompt(prompt)

        query = """
            SELECT id, prompt, negative_prompt, model, seed, parameters, rating, tags, qa_score, notes
            FROM generations
            WHERE rating >= ?
        """
        params = [min_rating]

        if model:
            query += " AND model = ?"
            params.append(model)

        # First try exact hash match
        query_exact = query + " AND prompt_hash = ? ORDER BY rating DESC LIMIT ?"
        params_exact = params + [prompt_hash, limit]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query_exact, params_exact)
            results = cursor.fetchall()

            if len(results) < limit:
                # Fallback to keyword matching
                keywords = [w for w in prompt.lower().split() if len(w) > 3][:5]
                if keywords:
                    keyword_conditions = " OR ".join(["prompt LIKE ?" for _ in keywords])
                    query_keywords = query + f" AND ({keyword_conditions}) ORDER BY rating DESC LIMIT ?"
                    params_keywords = params + [f"%{kw}%" for kw in keywords] + [limit]

                    cursor = conn.execute(query_keywords, params_keywords)
                    keyword_results = cursor.fetchall()

                    # Merge and dedupe
                    seen_ids = {r[0] for r in results}
                    for r in keyword_results:
                        if r[0] not in seen_ids:
                            results.append(r)
                            seen_ids.add(r[0])
                            if len(results) >= limit:
                                break

        return [
            {
                "id": r[0],
                "prompt": r[1],
                "negative_prompt": r[2],
                "model": r[3],
                "seed": r[4],
                "parameters": json.loads(r[5]) if r[5] else {},
                "rating": r[6],
                "tags": json.loads(r[7]) if r[7] else [],
                "qa_score": r[8],
                "notes": r[9],
            }
            for r in results
        ]

    def get_best_seeds_for_style(
        self,
        tags: List[str],
        model: str = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get best-rated seeds for a specific style.

        Args:
            tags: Style tags to match (e.g., ["anime", "portrait"]).
            model: Optional model filter.
            limit: Maximum results.

        Returns:
            List of high-rated generations with seeds.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Build tag matching query
            tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
            query = f"""
                SELECT id, prompt, model, seed, rating, tags, parameters
                FROM generations
                WHERE rating >= 0.7 AND ({tag_conditions})
            """
            params = [f'%"{tag}"%' for tag in tags]

            if model:
                query += " AND model = ?"
                params.append(model)

            query += " ORDER BY rating DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            results = cursor.fetchall()

        return [
            {
                "id": r[0],
                "prompt": r[1],
                "model": r[2],
                "seed": r[3],
                "rating": r[4],
                "tags": json.loads(r[5]) if r[5] else [],
                "parameters": json.loads(r[6]) if r[6] else {},
            }
            for r in results
        ]

    def suggest_prompt_enhancement(
        self,
        prompt: str,
        model: str = None,
        style_tags: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Suggest prompt enhancements based on past successful generations.

        Args:
            prompt: The base prompt.
            model: Target model.
            style_tags: Desired style tags.

        Returns:
            Enhancement suggestions including:
            - Recommended additions
            - Negative prompt suggestions
            - Similar successful prompts
            - Best seeds for similar content
        """
        suggestions = {
            "original_prompt": prompt,
            "similar_successful": [],
            "recommended_additions": [],
            "negative_suggestions": [],
            "best_seeds": [],
        }

        # Find similar successful prompts
        similar = self.find_similar_prompts(prompt, model, min_rating=0.8, limit=3)
        suggestions["similar_successful"] = similar

        # Extract common patterns from highly-rated generations
        if similar:
            # Collect common additions from successful prompts
            for s in similar:
                # Compare words and find additions
                original_words = set(prompt.lower().split())
                successful_words = set(s["prompt"].lower().split())
                additions = successful_words - original_words

                # Filter to meaningful additions (length > 3)
                additions = [w for w in additions if len(w) > 3]
                suggestions["recommended_additions"].extend(additions)

                # Collect negative prompts
                if s.get("negative_prompt"):
                    suggestions["negative_suggestions"].append(s["negative_prompt"])

        # Deduplicate
        suggestions["recommended_additions"] = list(set(suggestions["recommended_additions"]))[:10]
        suggestions["negative_suggestions"] = list(set(suggestions["negative_suggestions"]))[:3]

        # Get best seeds for style
        if style_tags:
            suggestions["best_seeds"] = self.get_best_seeds_for_style(style_tags, model, limit=5)

        return suggestions

    def save_style_preset(
        self,
        name: str,
        description: str,
        prompt_additions: str,
        negative_additions: str = "",
        recommended_model: str = None,
        recommended_params: Dict[str, Any] = None,
        example_generation_ids: List[str] = None,
    ) -> bool:
        """
        Save a reusable style preset.

        Args:
            name: Preset name (e.g., "cinematic_portrait").
            description: Description of the style.
            prompt_additions: Text to add to prompts.
            negative_additions: Text to add to negative prompts.
            recommended_model: Best model for this style.
            recommended_params: Recommended parameters.
            example_generation_ids: IDs of example generations.

        Returns:
            True if saved successfully.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO style_presets
                (name, description, prompt_additions, negative_additions, recommended_model, recommended_params, example_generations, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    name,
                    description,
                    prompt_additions,
                    negative_additions,
                    recommended_model,
                    json.dumps(recommended_params or {}),
                    json.dumps(example_generation_ids or []),
                    time.time(),
                    time.time(),
                ),
            )
            conn.commit()
            return True

    def get_style_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a style preset by name."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM style_presets WHERE name = ?", (name,))
            row = cursor.fetchone()

        if not row:
            return None

        return {
            "name": row[0],
            "description": row[1],
            "prompt_additions": row[2],
            "negative_additions": row[3],
            "recommended_model": row[4],
            "recommended_params": json.loads(row[5]) if row[5] else {},
            "example_generations": json.loads(row[6]) if row[6] else [],
        }

    def list_style_presets(self) -> List[Dict[str, str]]:
        """List all saved style presets."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT name, description, recommended_model FROM style_presets ORDER BY name")
            results = cursor.fetchall()

        return [{"name": r[0], "description": r[1], "model": r[2]} for r in results]

    def delete_style_preset(self, name: str) -> bool:
        """Delete a style preset by name."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM style_presets WHERE name = ?", (name,))
            conn.commit()
            return conn.total_changes > 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about stored generations."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    AVG(rating) as avg_rating,
                    COUNT(CASE WHEN rating >= 0.7 THEN 1 END) as high_rated,
                    COUNT(CASE WHEN outcome = 'success' THEN 1 END) as successful,
                    COUNT(DISTINCT model) as models_used
                FROM generations
            """
            )
            row = cursor.fetchone()

            cursor = conn.execute("SELECT COUNT(*) FROM style_presets")
            preset_count = cursor.fetchone()[0]

        return {
            "total_generations": row[0],
            "average_rating": round(row[1], 3) if row[1] else None,
            "high_rated_count": row[2],
            "successful_count": row[3],
            "models_used": row[4],
            "style_presets": preset_count,
        }


# Global instance
_db_instance: Optional[StyleLearningDB] = None


def get_style_db() -> StyleLearningDB:
    """Get or create the global style learning database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = StyleLearningDB()
    return _db_instance


# Convenience functions for MCP tools
def record_generation(**kwargs) -> str:
    return get_style_db().record_generation(**kwargs)


def rate_generation(record_id: str, rating: float, notes: str = None) -> bool:
    return get_style_db().rate_generation(record_id, rating, notes)


def find_similar_prompts(**kwargs) -> List[Dict]:
    return get_style_db().find_similar_prompts(**kwargs)


def suggest_prompt_enhancement(**kwargs) -> Dict:
    return get_style_db().suggest_prompt_enhancement(**kwargs)


def get_best_seeds_for_style(**kwargs) -> List[Dict]:
    return get_style_db().get_best_seeds_for_style(**kwargs)


def save_style_preset(**kwargs) -> bool:
    return get_style_db().save_style_preset(**kwargs)


def get_style_preset(name: str) -> Optional[Dict]:
    return get_style_db().get_style_preset(name)


def list_style_presets() -> List[Dict]:
    return get_style_db().list_style_presets()


def delete_style_preset(name: str) -> bool:
    return get_style_db().delete_style_preset(name)


def get_statistics() -> Dict:
    return get_style_db().get_statistics()
