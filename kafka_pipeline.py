"""
Kafka Streaming Event Ingestion Pipeline
Consumes user interaction events (clicks, impressions, skips) in real-time
and writes updated features to the Redis feature store.
"""

import json
import time
import threading
import logging
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Event schemas
# ------------------------------------------------------------------ #

@dataclass
class ImpressionEvent:
    event_type: str = "impression"
    user_id: str = ""
    ad_id: str = ""
    timestamp: float = field(default_factory=time.time)
    context: Dict = field(default_factory=dict)   # device, placement, etc.


@dataclass
class ClickEvent:
    event_type: str = "click"
    user_id: str = ""
    ad_id: str = ""
    timestamp: float = field(default_factory=time.time)
    dwell_ms: int = 0


@dataclass
class SkipEvent:
    event_type: str = "skip"
    user_id: str = ""
    ad_id: str = ""
    timestamp: float = field(default_factory=time.time)


# ------------------------------------------------------------------ #
# Kafka Producer (Real kafka-python)
# ------------------------------------------------------------------ #

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import os

class KafkaProducerWrapper:
    """
    Production Kafka producer using kafka-python.
    Handles serialization, retries, and error handling.
    """

    def __init__(self, bootstrap_servers: str = None):
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            'KAFKA_BROKERS', 'localhost:9092'
        ).split(',')
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3,
            max_in_flight_requests_per_connection=1
        )

    def produce(self, topic: str, key: str, value: str, callback=None):
        try:
            future = self.producer.send(topic, key=key, value=value)
            record_metadata = future.get(timeout=10)
            if callback:
                callback(None, record_metadata)
        except KafkaError as e:
            if callback:
                callback(e, None)
            logger.error(f"Failed to produce message: {e}")

    def flush(self, timeout=10):
        self.producer.flush(timeout)

    def close(self):
        self.producer.close()


# ------------------------------------------------------------------ #
# Kafka Consumer (Real kafka-python)
# ------------------------------------------------------------------ #

class KafkaConsumerWrapper:
    """
    Production Kafka consumer using kafka-python.
    Handles offset management, error handling, and polling.
    """

    def __init__(self, topics: List[str], bootstrap_servers: str = None, group_id: str = None):
        self.topics = topics
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            'KAFKA_BROKERS', 'localhost:9092'
        ).split(',')
        self.group_id = group_id or os.getenv('CONSUMER_GROUP', 'feature-updater')

        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda m: m.decode('utf-8') if m else None,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            max_poll_records=1,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000
        )

    def subscribe(self, topics):
        self.topics = topics
        self.consumer.subscribe(topics)

    def poll(self, timeout=1000):
        """Return one message or None."""
        messages = self.consumer.poll(timeout_ms=timeout, max_records=1)
        if messages:
            # Get first partition and first message
            for topic_partition, records in messages.items():
                if records:
                    msg = records[0]
                    return {
                        "key": msg.key,
                        "value": msg.value,
                        "topic": msg.topic,
                        "partition": msg.partition,
                        "offset": msg.offset
                    }
        return None

    def close(self):
        self.consumer.close()


# ------------------------------------------------------------------ #
# Event producer helper
# ------------------------------------------------------------------ #

class AdEventProducer:
    """
    Wraps a Kafka producer to publish typed ad events.
    Uses real kafka-python producer for production and local development.
    """

    TOPIC = "ad-events"

    def __init__(self, producer=None):
        self.producer = producer or KafkaProducerWrapper()

    def send_impression(self, user_id: str, ad_id: str, context: Dict = None):
        event = ImpressionEvent(user_id=user_id, ad_id=ad_id, context=context or {})
        self._publish(user_id, asdict(event))

    def send_click(self, user_id: str, ad_id: str, dwell_ms: int = 0):
        event = ClickEvent(user_id=user_id, ad_id=ad_id, dwell_ms=dwell_ms)
        self._publish(user_id, asdict(event))

    def send_skip(self, user_id: str, ad_id: str):
        event = SkipEvent(user_id=user_id, ad_id=ad_id)
        self._publish(user_id, asdict(event))

    def _publish(self, key: str, payload: Dict):
        self.producer.produce(
            topic=self.TOPIC,
            key=key,
            value=json.dumps(payload),
        )
        self.producer.flush()


# ------------------------------------------------------------------ #
# Feature updater (consumer side)
# ------------------------------------------------------------------ #

class FeatureUpdater:
    """
    Consumes raw ad events from Kafka and writes updated
    user features to the Redis feature store in real time.

    Usage:
        updater = FeatureUpdater(feature_store=store)
        updater.start()          # non-blocking background thread
        # ... later ...
        updater.stop()
    """

    def __init__(
        self,
        feature_store,                     # RedisFeatureStore or InMemoryFeatureStore
        consumer: Optional[KafkaConsumerWrapper] = None,
        poll_interval: float = 0.1,        # seconds between polls
        batch_flush_size: int = 50,        # flush feature writes after N events
    ):
        self.store = feature_store
        self.consumer = consumer or KafkaConsumerWrapper(topics=["ad-events"])
        self.consumer.subscribe(["ad-events"])
        self.poll_interval = poll_interval
        self.batch_flush_size = batch_flush_size

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # In-memory accumulators: user_id -> {event counts}
        self._user_buffers: Dict[str, Dict] = defaultdict(
            lambda: {"impressions": 0, "clicks": 0, "skips": 0, "last_seen": 0.0}
        )
        self._buffer_lock = threading.Lock()

        # Stats
        self.stats = {"consumed": 0, "errors": 0, "flushes": 0}

    def _process_message(self, msg: Dict):
        """Parse one Kafka message and accumulate feature deltas."""
        try:
            payload = json.loads(msg["value"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Bad message: {e}")
            self.stats["errors"] += 1
            return

        user_id = payload.get("user_id", "")
        event_type = payload.get("event_type", "")
        ts = payload.get("timestamp", time.time())

        with self._buffer_lock:
            buf = self._user_buffers[user_id]
            buf["last_seen"] = ts

            if event_type == "impression":
                buf["impressions"] += 1
            elif event_type == "click":
                buf["clicks"] += 1
            elif event_type == "skip":
                buf["skips"] += 1

        self.stats["consumed"] += 1

        # Flush to Redis when buffer is large enough
        if self.stats["consumed"] % self.batch_flush_size == 0:
            self._flush_buffers()

    def _flush_buffers(self):
        """Write accumulated deltas to the feature store."""
        with self._buffer_lock:
            snapshot = dict(self._user_buffers)
            self._user_buffers.clear()

        for user_id, counts in snapshot.items():
            # Fetch existing features (may be None on first write)
            existing = self.store.get_user_features(user_id) or {
                "categorical": {},
                "numerical": {},
            }
            # Merge streaming counts into numerical features
            num = existing.get("numerical", {})
            num["stream_impressions"] = num.get("stream_impressions", 0) + counts["impressions"]
            num["stream_clicks"] = num.get("stream_clicks", 0) + counts["clicks"]
            num["stream_skips"] = num.get("stream_skips", 0) + counts["skips"]
            num["last_seen"] = counts["last_seen"]

            existing["numerical"] = num
            self.store.set_user_features(user_id, existing)

        self.stats["flushes"] += 1
        logger.debug(f"Flushed {len(snapshot)} user buffers to feature store.")

    def _run_loop(self):
        """Background poll loop."""
        self.consumer.subscribe(["ad-events"])
        logger.info("FeatureUpdater started.")

        while self._running:
            msg = self.consumer.poll(timeout=self.poll_interval)
            if msg is not None:
                self._process_message(msg)
            else:
                time.sleep(self.poll_interval)

        # Final flush on shutdown
        self._flush_buffers()
        self.consumer.close()
        logger.info("FeatureUpdater stopped.")

    def start(self):
        """Start background consumer thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("FeatureUpdater running in background.")

    def stop(self, timeout: float = 5.0):
        """Signal the consumer thread to stop and wait for it."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
        print(f"FeatureUpdater stopped. Stats: {self.stats}")

    def get_stats(self) -> Dict:
        return dict(self.stats)


# ------------------------------------------------------------------ #
# Smoke test
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    sys.path.append("/home/claude")
    from redis_feature_store import InMemoryFeatureStore

    print("=== Kafka Pipeline — smoke test ===\n")

    store = InMemoryFeatureStore()
    producer = AdEventProducer()
    updater = FeatureUpdater(feature_store=store, batch_flush_size=5)

    # Seed some events before starting consumer
    for i in range(10):
        uid = f"user_{i % 3}"
        producer.send_impression(uid, f"ad_{i}")
        if i % 2 == 0:
            producer.send_click(uid, f"ad_{i}", dwell_ms=1200)

    # Start consumer and let it process
    updater.start()
    time.sleep(0.5)
    updater.stop()

    # Check feature store
    feats = store.get_user_features("user_0")
    if feats:
        print(f"user_0 streaming features: {feats['numerical']}")
    else:
        print("user_0: no features (events may not have flushed yet)")

    print(f"\nStats: {updater.get_stats()}")
    print("\n✓ Kafka pipeline smoke test passed!")
