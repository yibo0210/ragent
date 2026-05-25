"""LangGraph MySQL Checkpointer

基于 SQLAlchemy 实现 LangGraph 的 BaseCheckpointSaver 接口，
将图状态持久化到 MySQL，支持中断挂起与状态恢复。
"""
from typing import Any, Iterator, Optional, Sequence

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
)
from sqlalchemy import text
from sqlalchemy.orm import Session

from .database import SessionLocal
from .models import GraphCheckpoint, GraphCheckpointWrite

# 模块级单例
_checkpointer_instance: Optional["MySQLSaver"] = None


def _get_checkpointer() -> "MySQLSaver":
    """获取全局 MySQLSaver 单例。"""
    global _checkpointer_instance
    if _checkpointer_instance is None:
        _checkpointer_instance = MySQLSaver()
    return _checkpointer_instance


class MySQLSaver(BaseCheckpointSaver):
    """MySQL 检查点持久化器。"""

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        db = SessionLocal()
        try:
            query = db.query(GraphCheckpoint).filter(
                GraphCheckpoint.thread_id == thread_id,
                GraphCheckpoint.checkpoint_ns == checkpoint_ns,
            )
            if checkpoint_id:
                query = query.filter(GraphCheckpoint.checkpoint_id == checkpoint_id)
            else:
                query = query.order_by(GraphCheckpoint.created_at.desc())

            row = query.first()
            if not row:
                return None

            # 读取待处理写入
            writes = self._load_writes(
                db, thread_id, checkpoint_ns, row.checkpoint_id
            )

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row.checkpoint_id,
                    }
                },
                checkpoint=row.checkpoint,
                metadata=row.checkpoint_metadata,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": row.parent_checkpoint_id,
                        }
                    }
                    if row.parent_checkpoint_id
                    else None
                ),
                pending_writes=writes,
            )
        finally:
            db.close()

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        db = SessionLocal()
        try:
            existing = (
                db.query(GraphCheckpoint)
                .filter(
                    GraphCheckpoint.thread_id == thread_id,
                    GraphCheckpoint.checkpoint_ns == checkpoint_ns,
                    GraphCheckpoint.checkpoint_id == checkpoint_id,
                )
                .first()
            )

            checkpoint_data = self._serialize_checkpoint(checkpoint)
            metadata_data = dict(metadata) if metadata else {}

            if existing:
                existing.checkpoint = checkpoint_data
                existing.checkpoint_metadata = metadata_data
                existing.parent_checkpoint_id = parent_checkpoint_id
            else:
                db.add(
                    GraphCheckpoint(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        parent_checkpoint_id=parent_checkpoint_id,
                        checkpoint=checkpoint_data,
                        checkpoint_metadata=metadata_data,
                    )
                )

            db.commit()
        finally:
            db.close()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: dict,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id", "")

        db = SessionLocal()
        try:
            # 删除旧的写入记录
            db.execute(
                text(
                    "DELETE FROM graph_checkpoint_writes "
                    "WHERE thread_id = :tid AND checkpoint_ns = :cns "
                    "AND checkpoint_id = :cid AND task_id = :tsk AND task_path = :tp"
                ),
                {
                    "tid": thread_id,
                    "cns": checkpoint_ns,
                    "cid": checkpoint_id,
                    "tsk": task_id,
                    "tp": task_path,
                },
            )

            for idx, (channel, value) in enumerate(writes):
                db.add(
                    GraphCheckpointWrite(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        task_id=task_id,
                        task_path=task_path,
                        idx=idx,
                        channel=channel,
                        value=self._serialize_value(value),
                    )
                )

            db.commit()
        finally:
            db.close()

    def list(
        self,
        config: Optional[dict],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"] if config else None
        db = SessionLocal()
        try:
            query = db.query(GraphCheckpoint).order_by(GraphCheckpoint.created_at.desc())

            if thread_id:
                query = query.filter(GraphCheckpoint.thread_id == thread_id)
            if before and before.get("configurable", {}).get("checkpoint_id"):
                before_id = before["configurable"]["checkpoint_id"]
                query = query.filter(GraphCheckpoint.checkpoint_id < before_id)
            if limit:
                query = query.limit(limit)

            for row in query.all():
                writes = self._load_writes(
                    db, row.thread_id, row.checkpoint_ns, row.checkpoint_id
                )
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": row.thread_id,
                            "checkpoint_ns": row.checkpoint_ns,
                            "checkpoint_id": row.checkpoint_id,
                        }
                    },
                    checkpoint=row.checkpoint,
                    metadata=row.checkpoint_metadata,
                    parent_config=(
                        {
                            "configurable": {
                                "thread_id": row.thread_id,
                                "checkpoint_ns": row.checkpoint_ns,
                                "checkpoint_id": row.parent_checkpoint_id,
                            }
                        }
                        if row.parent_checkpoint_id
                        else None
                    ),
                    pending_writes=writes,
                )
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Async 方法（asyncio 兼容层，复用同步实现）
    # ------------------------------------------------------------------
    async def aget_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def aput(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: dict,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def alist(
        self,
        config: Optional[dict],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = None,
    ):
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize_checkpoint(checkpoint: Checkpoint) -> dict:
        """将 Checkpoint TypedDict 转为可存储的 JSON 兼容 dict。"""
        data = dict(checkpoint)
        # channel_values 中的消息对象需序列化
        if "channel_values" in data:
            data["channel_values"] = MySQLSaver._serialize_channel_values(
                data["channel_values"]
            )
        return data

    @staticmethod
    def _serialize_channel_values(channel_values: dict) -> dict:
        """对 channel_values 进行 JSON 兼容处理。"""
        result = {}
        for key, value in channel_values.items():
            result[key] = MySQLSaver._serialize_value(value)
        return result

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """将值转为 JSON 兼容格式。"""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {k: MySQLSaver._serialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [MySQLSaver._serialize_value(v) for v in value]
        # LangChain message 对象 → dict
        if hasattr(value, "dict"):
            return value.dict()
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return str(value)

    @staticmethod
    def _load_writes(
        db: Session, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list:
        """加载指定检查点的待处理写入。"""
        rows = (
            db.query(GraphCheckpointWrite)
            .filter(
                GraphCheckpointWrite.thread_id == thread_id,
                GraphCheckpointWrite.checkpoint_ns == checkpoint_ns,
                GraphCheckpointWrite.checkpoint_id == checkpoint_id,
            )
            .order_by(GraphCheckpointWrite.idx)
            .all()
        )
        return [(row.task_id, row.channel, row.value) for row in rows]
