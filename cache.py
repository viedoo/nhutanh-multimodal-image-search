"""
Redis Cache Module for Multimodal Image Search Engine
Provides caching for:
  - Text embeddings (avoid re-running SigLIP2 model for same query)
  - Search results (text search, similar image, material search)
  - Cache stats and management endpoints

Falls back gracefully to in-memory dict when Redis is not available.
"""

import json
import os
import time
import logging
from typing import Any, Optional

import numpy as np

from utils import to_image_bytes

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _make_key(*parts: str) -> str:
    """Create a namespaced Redis key, e.g. 'mmis:text_emb:sha256...'

    Always hashes user-derived parts so `KEYS pattern` globs in stats() never
    explode on weird characters (spaces, *, ?, etc.).
    """
    import hashlib as _h
    safe_parts = []
    for p in parts:
        s = str(p)
        if any(c in s for c in " \t*?[]") or len(s) > 96:
            safe_parts.append(_h.sha256(s.encode("utf-8")).hexdigest()[:32])
        else:
            safe_parts.append(s)
    return f"mmis:{':'.join(safe_parts)}"


# -----------------------------------------------------------------
# Cache Backend
# -----------------------------------------------------------------

class _InMemoryFallback:
    """Simple dict-based in-memory cache used when Redis is unavailable."""

    def __init__(self):
        self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expire_at)
        self.hits = 0
        self.misses = 0
        logger.warning("Redis not available - using in-memory fallback cache (not shared across processes)")

    def get(self, key: str) -> Optional[str]:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        value, expire_at = entry
        if expire_at and time.time() > expire_at:
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, key: str, value: str, ex: int = None):
        expire_at = (time.time() + ex) if ex else None
        self._store[key] = (value, expire_at)

    def delete(self, key: str):
        self._store.pop(key, None)

    def keys(self, pattern: str = "*") -> list[str]:
        # Simple prefix match (strip trailing *)
        prefix = pattern.rstrip("*")
        return [k for k in self._store if k.startswith(prefix)]

    def flushdb(self):
        self._store.clear()

    def info(self, section: str = None) -> dict:
        return {
            "redis_version": "N/A (in-memory fallback)",
            "used_memory_human": "N/A",
            "keyspace_hits": self.hits,
            "keyspace_misses": self.misses,
        }

    def dbsize(self) -> int:
        return len(self._store)

    @property
    def available(self) -> bool:
        return False


class RedisCache:
    """
    Redis-backed cache with in-memory fallback.
    
    Usage:
        cache = RedisCache()
        
        # Store embedding
        cache.set_embedding("sleeping cat", embedding_np_array)
        emb = cache.get_embedding("sleeping cat")
        
        # Store search results
        cache.set_results("text", "sleeping cat", results_list)
        results = cache.get_results("text", "sleeping cat")
    """

    def __init__(
        self,
        url: Optional[str] = None,
        ttl: int = 3600,
        embedding_ttl: int = 86400,  # Embeddings live longer (24h)
    ):
        self.ttl = int(os.getenv("REDIS_CACHE_TTL", ttl))
        self.embedding_ttl = int(os.getenv("REDIS_EMBEDDING_TTL", embedding_ttl))
        redis_url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")

        self._client = None
        self._fallback = None

        try:
            import redis as redis_lib
            client = redis_lib.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()  # Fail fast if Redis is not reachable
            self._client = client
            logger.info(f"[OK] Redis connected: {redis_url}")
        except Exception as e:
            logger.warning(f"[WARN] Redis not reachable ({e}) - using in-memory fallback")
            self._fallback = _InMemoryFallback()

    @property
    def _backend(self):
        return self._client if self._client is not None else self._fallback

    @property
    def available(self) -> bool:
        return self._client is not None

    # -----------------------------------------------------------------
    # Embeddings
    # -----------------------------------------------------------------

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Return cached text embedding or None. `text` is expected to already be
        the normalized/hashed form (see utils.hash_query)."""
        key = _make_key("text_emb", text)
        raw = self._backend.get(key)
        if raw is None:
            return None
        arr = np.frombuffer(bytes.fromhex(raw), dtype=np.float32)
        # We don't know dim at decode time, default to 768; caller reshapes if needed
        return arr.reshape(1, -1)

    def set_embedding(self, text_hash: str, embedding: np.ndarray):
        """Cache a text embedding. Pass the pre-hashed key from utils.hash_query."""
        key = _make_key("text_emb", text_hash)
        hex_str = to_image_bytes(embedding)
        self._backend.set(key, hex_str, ex=self.embedding_ttl)

    # -----------------------------------------------------------------
    # Search Results
    # -----------------------------------------------------------------

    def get_results(self, search_type: str, query_key: str, top_k: int = 10) -> Optional[list]:
        """Return cached search results or None.

        Args:
            search_type: 'text', 'similar', or 'material'
            query_key: The pre-hashed cache key (use utils.hash_query for text)
            top_k: Number of results (part of cache key)
        """
        key = _make_key("results", search_type, f"{query_key}:{top_k}")
        raw = self._backend.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def set_results(self, search_type: str, query_key: str, results: list, top_k: int = 10):
        """Cache search results as JSON.

        Args:
            search_type: 'text', 'similar', or 'material'
            query_key: The pre-hashed cache key
            results: List of result dicts
            top_k: Number of results (part of cache key)
        """
        key = _make_key("results", search_type, f"{query_key}:{top_k}")
        self._backend.set(key, json.dumps(results), ex=self.ttl)

    # -----------------------------------------------------------------
    # Management
    # -----------------------------------------------------------------

    def clear_all(self) -> int:
        """Clear all MMIS cache keys. Returns count of deleted keys."""
        if self._client is not None:
            pattern = "mmis:*"
            keys = self._client.keys(pattern)
            if keys:
                return self._client.delete(*keys)
            return 0
        else:
            count = self._fallback.dbsize()
            self._fallback.flushdb()
            return count

    def stats(self) -> dict:
        """Return cache statistics."""
        if self._client is not None:
            try:
                info = self._client.info("stats")
                server_info = self._client.info("server")
                mem_info = self._client.info("memory")
                # Use SCAN (cursor) instead of KEYS — non-blocking on large DBs
                emb_keys: list[str] = []
                result_keys: list[str] = []
                total_keys = 0
                for key in self._client.scan_iter(match="mmis:*", count=1000):
                    total_keys += 1
                    if ":text_emb:" in key:
                        emb_keys.append(key)
                    elif ":results:" in key:
                        result_keys.append(key)
                return {
                    "backend": "redis",
                    "redis_version": server_info.get("redis_version"),
                    "used_memory": mem_info.get("used_memory_human"),
                    "total_keys": total_keys,
                    "embedding_keys": len(emb_keys),
                    "result_keys": len(result_keys),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "hit_rate": _hit_rate(
                        info.get("keyspace_hits", 0),
                        info.get("keyspace_misses", 0),
                    ),
                }
            except Exception as e:
                return {"backend": "redis", "error": str(e)}
        else:
            fallback = self._fallback
            hits = fallback.hits
            misses = fallback.misses
            all_keys = fallback.keys("mmis:")
            emb_keys = [k for k in all_keys if ":text_emb:" in k]
            result_keys = [k for k in all_keys if ":results:" in k]
            return {
                "backend": "in-memory-fallback",
                "total_keys": len(all_keys),
                "embedding_keys": len(emb_keys),
                "result_keys": len(result_keys),
                "keyspace_hits": hits,
                "keyspace_misses": misses,
                "hit_rate": _hit_rate(hits, misses),
            }


def _hit_rate(hits: int, misses: int) -> str:
    total = hits + misses
    if total == 0:
        return "0.00%"
    return f"{hits / total * 100:.2f}%"
