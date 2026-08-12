"""
Redis Online Feature Store
Serves real-time user features for ad recommendation inference.
Handles TTL-based expiry, feature serialization, and cache warming.
"""

import redis
import json
import numpy as np
import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class UserFeatures:
    """Schema for cached user features"""
    user_id: str
    categorical: Dict[str, str]   # C1-C6
    numerical: Dict[str, float]   # I1-I13
    timestamp: float


class RedisFeatureStore:
    """
    Online feature store backed by Redis.
    Provides sub-millisecond feature lookup at inference time.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        user_ttl: int = 3600,        # 1 hour TTL for user features
        ad_ttl: int = 86400,         # 24 hour TTL for ad features
        max_connections: int = 20,
    ):
        """
        Args:
            host: Redis host
            port: Redis port
            db: Redis database index
            password: Redis password (optional)
            user_ttl: TTL in seconds for user feature keys
            ad_ttl: TTL in seconds for ad feature keys
            max_connections: Connection pool size
        """
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=True,
        )
        self.client = redis.Redis(connection_pool=self.pool)
        self.user_ttl = user_ttl
        self.ad_ttl = ad_ttl

        # Key prefixes
        self.USER_PREFIX = "user:features:"
        self.AD_PREFIX = "ad:features:"
        self.EMBEDDING_PREFIX = "user:embedding:"

        self._verify_connection()

    def _verify_connection(self):
        """Ping Redis to verify connection on startup."""
        try:
            self.client.ping()
            print("Redis feature store connected.")
        except redis.ConnectionError as e:
            raise RuntimeError(f"Cannot connect to Redis: {e}")

    # ------------------------------------------------------------------ #
    # User features
    # ------------------------------------------------------------------ #

    def set_user_features(self, user_id: str, features: Dict[str, Any]) -> bool:
        """
        Write user features to Redis with TTL.

        Args:
            user_id: Unique user identifier
            features: Dict with 'categorical' and 'numerical' keys

        Returns:
            True on success
        """
        key = self.USER_PREFIX + user_id
        payload = {
            "user_id": user_id,
            "categorical": features.get("categorical", {}),
            "numerical": features.get("numerical", {}),
            "timestamp": time.time(),
        }
        self.client.setex(key, self.user_ttl, json.dumps(payload))
        return True

    def get_user_features(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch user features from Redis.

        Returns:
            Feature dict or None on cache miss
        """
        key = self.USER_PREFIX + user_id
        raw = self.client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def batch_get_user_features(
        self, user_ids: List[str]
    ) -> Dict[str, Optional[Dict]]:
        """
        Fetch features for multiple users in a single pipeline call.

        Returns:
            Dict mapping user_id -> features (None on miss)
        """
        keys = [self.USER_PREFIX + uid for uid in user_ids]
        pipe = self.client.pipeline(transaction=False)
        for key in keys:
            pipe.get(key)
        raw_values = pipe.execute()

        result = {}
        for uid, raw in zip(user_ids, raw_values):
            result[uid] = json.loads(raw) if raw else None
        return result

    # ------------------------------------------------------------------ #
    # User embeddings (pre-computed, stored as JSON list)
    # ------------------------------------------------------------------ #

    def set_user_embedding(
        self, user_id: str, embedding: np.ndarray, ttl: int = 1800
    ) -> bool:
        """Cache a pre-computed user embedding (30 min TTL by default)."""
        key = self.EMBEDDING_PREFIX + user_id
        self.client.setex(key, ttl, json.dumps(embedding.tolist()))
        return True

    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Retrieve cached user embedding."""
        key = self.EMBEDDING_PREFIX + user_id
        raw = self.client.get(key)
        if raw is None:
            return None
        return np.array(json.loads(raw), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Ad features
    # ------------------------------------------------------------------ #

    def set_ad_features(self, ad_id: str, features: Dict[str, Any]) -> bool:
        """Write ad features to Redis."""
        key = self.AD_PREFIX + str(ad_id)
        self.client.setex(key, self.ad_ttl, json.dumps(features))
        return True

    def batch_set_ad_features(self, ad_features: Dict[str, Dict]) -> int:
        """
        Bulk-write ad features via pipeline.

        Args:
            ad_features: Dict of ad_id -> feature dict

        Returns:
            Number of records written
        """
        pipe = self.client.pipeline(transaction=False)
        for ad_id, features in ad_features.items():
            key = self.AD_PREFIX + str(ad_id)
            pipe.setex(key, self.ad_ttl, json.dumps(features))
        pipe.execute()
        return len(ad_features)

    def get_ad_features(self, ad_id: str) -> Optional[Dict]:
        """Fetch ad features from Redis."""
        key = self.AD_PREFIX + str(ad_id)
        raw = self.client.get(key)
        return json.loads(raw) if raw else None

    def batch_get_ad_features(
        self, ad_ids: List[str]
    ) -> Dict[str, Optional[Dict]]:
        """Fetch features for multiple ads via pipeline."""
        keys = [self.AD_PREFIX + str(aid) for aid in ad_ids]
        pipe = self.client.pipeline(transaction=False)
        for key in keys:
            pipe.get(key)
        raw_values = pipe.execute()

        result = {}
        for aid, raw in zip(ad_ids, raw_values):
            result[str(aid)] = json.loads(raw) if raw else None
        return result

    # ------------------------------------------------------------------ #
    # Cache warming
    # ------------------------------------------------------------------ #

    def warm_ad_cache(self, ad_records: List[Dict]) -> int:
        """
        Pre-load ad features from a list of dicts (e.g. from offline batch).

        Args:
            ad_records: List of dicts, each must have 'ad_id' key

        Returns:
            Number of records written
        """
        ad_map = {str(r["ad_id"]): r for r in ad_records}
        return self.batch_set_ad_features(ad_map)

    # ------------------------------------------------------------------ #
    # Monitoring
    # ------------------------------------------------------------------ #

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache hit/miss counters and key counts."""
        info = self.client.info("stats")
        keyspace = self.client.info("keyspace")

        user_keys = len(self.client.keys(self.USER_PREFIX + "*"))
        ad_keys = len(self.client.keys(self.AD_PREFIX + "*"))
        emb_keys = len(self.client.keys(self.EMBEDDING_PREFIX + "*"))

        return {
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "hit_rate": (
                info.get("keyspace_hits", 0)
                / max(
                    info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1
                )
            ),
            "user_feature_keys": user_keys,
            "ad_feature_keys": ad_keys,
            "embedding_keys": emb_keys,
        }

    def flush_expired(self):
        """Redis handles TTL expiry automatically — this is a no-op placeholder."""
        pass

    def close(self):
        """Close all connections in the pool."""
        self.pool.disconnect()


# ------------------------------------------------------------------ #
# Convenience: mock store for unit tests / offline use
# ------------------------------------------------------------------ #

class InMemoryFeatureStore:
    """
    Drop-in replacement for RedisFeatureStore when Redis is unavailable.
    Uses plain dicts — no persistence, no TTL enforcement.
    """

    def __init__(self):
        self._user: Dict[str, Dict] = {}
        self._ad: Dict[str, Dict] = {}
        self._emb: Dict[str, np.ndarray] = {}

    def set_user_features(self, user_id, features):
        self._user[user_id] = {"user_id": user_id, **features, "timestamp": time.time()}
        return True

    def get_user_features(self, user_id):
        return self._user.get(user_id)

    def batch_get_user_features(self, user_ids):
        return {uid: self._user.get(uid) for uid in user_ids}

    def set_user_embedding(self, user_id, embedding, ttl=1800):
        self._emb[user_id] = embedding
        return True

    def get_user_embedding(self, user_id):
        return self._emb.get(user_id)

    def set_ad_features(self, ad_id, features):
        self._ad[str(ad_id)] = features
        return True

    def batch_set_ad_features(self, ad_features):
        self._ad.update({str(k): v for k, v in ad_features.items()})
        return len(ad_features)

    def get_ad_features(self, ad_id):
        return self._ad.get(str(ad_id))

    def batch_get_ad_features(self, ad_ids):
        return {str(aid): self._ad.get(str(aid)) for aid in ad_ids}

    def warm_ad_cache(self, ad_records):
        for r in ad_records:
            self._ad[str(r["ad_id"])] = r
        return len(ad_records)

    def get_cache_stats(self):
        return {
            "user_feature_keys": len(self._user),
            "ad_feature_keys": len(self._ad),
            "embedding_keys": len(self._emb),
        }


# ------------------------------------------------------------------ #
# Quick smoke test
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=== Redis Feature Store — smoke test (InMemory) ===\n")

    store = InMemoryFeatureStore()

    # Write user features
    store.set_user_features(
        "user_001",
        {
            "categorical": {f"C{i}": f"val_{i}" for i in range(1, 7)},
            "numerical": {f"I{i}": float(i * 1.5) for i in range(1, 14)},
        },
    )

    # Read back
    feats = store.get_user_features("user_001")
    assert feats is not None, "Cache miss on user_001"
    print(f"User features retrieved: {list(feats['categorical'].keys())}")

    # Embedding round-trip
    emb = np.random.randn(256).astype(np.float32)
    store.set_user_embedding("user_001", emb)
    emb_back = store.get_user_embedding("user_001")
    assert np.allclose(emb, emb_back), "Embedding mismatch"
    print(f"Embedding round-trip OK: shape {emb_back.shape}")

    # Batch ad features
    ads = {str(i): {"ad_id": i, "C7": f"cat_{i}"} for i in range(100)}
    store.batch_set_ad_features(ads)
    batch = store.batch_get_ad_features(list(range(5)))
    print(f"Batch ad fetch: {len(batch)} records")

    stats = store.get_cache_stats()
    print(f"\nCache stats: {stats}")
    print("\n✓ Feature store smoke test passed!")
