"""Redis Streams 消息队列模块

基于 Redis Streams 实现的生产者-消费者消息队列，
支持消费者组、死信处理、消息认领和流水线编排。
用于 v13 流式增量图引擎的三阶段管线：
  doc_ingest -> graph_extract -> vector_sync
"""
import json
import os
from typing import Optional

import redis

from backend.observability.logging import get_logger

logger = get_logger("pipeline.stream_queue")

# ---- Stream 常量 ----
DOC_INGEST = "doc_ingest"
GRAPH_EXTRACT = "graph_extract"
VECTOR_SYNC = "vector_sync"
DEAD_LETTER = "dead_letter"

# ---- Consumer Group 常量 ----
PARSER_GROUP = "parser_group"
EXTRACTOR_GROUP = "extractor_group"
SYNCER_GROUP = "syncer_group"

# stream -> group 映射，方便自动创建
_STREAM_GROUP_MAP = {
    DOC_INGEST: PARSER_GROUP,
    GRAPH_EXTRACT: EXTRACTOR_GROUP,
    VECTOR_SYNC: SYNCER_GROUP,
}

_KEY_PREFIX = "ragent:stream:"

# 死信重试阈值
DEAD_LETTER_RETRIES = 3


class StreamQueue:
    """Redis Streams 消息队列封装。

    所有方法均吞掉异常，Redis 不可用时静默降级（返回 None/False/[]），
    与现有 RedisCache 的容错策略保持一致。
    """

    def __init__(self, redis_url: str = None):
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client: Optional[redis.Redis] = None

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _get_client(self) -> redis.Redis:
        """懒加载 Redis 客户端。"""
        if self._client is None:
            self._client = redis.Redis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def _stream_key(self, stream: str) -> str:
        """返回带前缀的实际 Redis key。"""
        return f"{_KEY_PREFIX}{stream}"

    def _ensure_group(self, stream: str, group: str) -> None:
        """确保消费者组存在，已存在则忽略 BUSYGROUP 错误。"""
        try:
            self._get_client().xgroup_create(
                self._stream_key(stream), groupname=group, id="0", mkstream=True,
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                return
            raise

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def publish(self, stream: str, data: dict) -> Optional[str]:
        """发布消息到指定 stream，返回消息 ID。失败返回 None。"""
        try:
            payload = json.dumps(data, ensure_ascii=False)
            msg_id = self._get_client().xadd(self._stream_key(stream), {"payload": payload})
            logger.debug("stream_publish", stream=stream, msg_id=msg_id)
            return msg_id
        except Exception as e:
            logger.warning("stream_publish_failed", stream=stream, error=str(e))
            return None

    def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 1,
        block_ms: int = 5000,
    ) -> list[dict]:
        """从消费者组读取消息。

        自动创建消费者组（如不存在）。
        返回 [{id, data, stream}]，失败返回空列表。
        """
        try:
            self._ensure_group(stream, group)
            result = self._get_client().xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={self._stream_key(stream): ">"},
                count=count,
                block=block_ms,
            )
            if not result:
                return []

            messages = []
            for _stream_key, entries in result:
                for msg_id, fields in entries:
                    raw = fields.get("payload", "{}")
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        data = {"_raw": raw}
                    messages.append({"id": msg_id, "data": data, "stream": stream})
            return messages
        except Exception as e:
            logger.warning("stream_consume_failed", stream=stream, group=group, error=str(e))
            return []

    def ack(self, stream: str, group: str, message_id: str) -> bool:
        """确认消息已处理。失败返回 False。"""
        try:
            self._get_client().xack(self._stream_key(stream), group, message_id)
            return True
        except Exception as e:
            logger.warning("stream_ack_failed", stream=stream, msg_id=message_id, error=str(e))
            return False

    def ack_and_publish(
        self,
        ack_stream: str,
        ack_group: str,
        ack_id: str,
        next_stream: str,
        next_data: dict,
    ) -> bool:
        """确认当前消息并发布到下一阶段 stream（流水线编排）。

        用于链式传递：doc_ingest -> graph_extract -> vector_sync。
        ack 成功但 publish 失败时返回 False。
        """
        ack_ok = self.ack(ack_stream, ack_group, ack_id)
        if not ack_ok:
            return False
        pub_id = self.publish(next_stream, next_data)
        return pub_id is not None

    def pending(self, stream: str, group: str) -> list[dict]:
        """查询待处理消息摘要。失败返回空列表。"""
        try:
            info = self._get_client().xpending(self._stream_key(stream), group)
            if not info or info.get("pending", 0) == 0:
                return []
            # xpending_range 返回详细列表
            details = self._get_client().xpending_range(
                self._stream_key(stream), group,
                min="-", max="+", count=100,
            )
            return [
                {
                    "id": d["message_id"],
                    "consumer": d["consumer"],
                    "idle_ms": d["time_since_delivered"],
                    "deliveries": d["times_delivered"],
                }
                for d in details
            ]
        except Exception as e:
            logger.warning("stream_pending_failed", stream=stream, group=group, error=str(e))
            return []

    def claim_stale(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_ms: int = 60000,
    ) -> list[dict]:
        """认领超时未确认的消息（消费者宕机恢复场景）。

        返回认领到的消息列表，失败返回空列表。
        """
        try:
            # 先获取超时消息 ID
            pending_details = self._get_client().xpending_range(
                self._stream_key(stream), group,
                min=str(min_idle_ms), max="+", count=100,
                consumername=None,
            )
            if not pending_details:
                return []

            stale_ids = [d["message_id"] for d in pending_details]
            claimed = self._get_client().xclaim(
                self._stream_key(stream), group, consumer,
                min_idle_time=min_idle_ms,
                message_ids=stale_ids,
            )
            messages = []
            for msg_id, fields in claimed:
                raw = fields.get("payload", "{}")
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {"_raw": raw}
                messages.append({"id": msg_id, "data": data, "stream": stream})
            return messages
        except Exception as e:
            logger.warning("stream_claim_failed", stream=stream, group=group, error=str(e))
            return []

    def get_stream_info(self, stream: str) -> dict:
        """获取 stream 元信息（长度、消费者组数等）。失败返回空字典。"""
        try:
            info = self._get_client().xinfo_stream(self._stream_key(stream))
            return info
        except Exception as e:
            logger.warning("stream_info_failed", stream=stream, error=str(e))
            return {}

    def try_dead_letter(self, stream: str, group: str, message_id: str, data: dict) -> bool:
        """尝试将消息移入死信队列。

        由调用方在重试耗尽后调用：先 ack 原消息，再 publish 到 dead_letter。
        """
        acked = self.ack(stream, group, message_id)
        if not acked:
            return False
        dl_data = {"_source_stream": stream, "_source_id": message_id, **data}
        pub_id = self.publish(DEAD_LETTER, dl_data)
        return pub_id is not None


# ---- 模块级单例 ----
_stream_queue: Optional[StreamQueue] = None


def get_stream_queue() -> StreamQueue:
    """获取全局 StreamQueue 单例。"""
    global _stream_queue
    if _stream_queue is None:
        _stream_queue = StreamQueue()
    return _stream_queue
