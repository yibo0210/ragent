"""数据模型定义模块

定义聊天、会话、文档管理相关的请求/响应数据结构。
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional, List

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = "default_session"


class RetrievedChunk(BaseModel):
    filename: str
    page_number: Optional[str | int] = None
    text: Optional[str] = None
    score: Optional[float] = None
    rrf_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class RagTrace(BaseModel):
    tool_used: bool
    tool_name: str
    query: Optional[str] = None
    expanded_query: Optional[str] = None
    step_back_question: Optional[str] = None
    step_back_answer: Optional[str] = None
    expansion_type: Optional[str] = None
    hypothetical_doc: Optional[str] = None
    retrieval_stage: Optional[str] = None
    grade_score: Optional[str] = None
    grade_route: Optional[str] = None
    rewrite_needed: Optional[bool] = None
    rewrite_strategy: Optional[str] = None
    rewrite_query: Optional[str] = None
    rerank_enabled: Optional[bool] = None
    rerank_applied: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_endpoint: Optional[str] = None
    rerank_error: Optional[str] = None
    retrieval_mode: Optional[str] = None
    candidate_k: Optional[int] = None
    leaf_retrieve_level: Optional[int] = None
    auto_merge_enabled: Optional[bool] = None
    auto_merge_applied: Optional[bool] = None
    auto_merge_threshold: Optional[int] = None
    auto_merge_replaced_chunks: Optional[int] = None
    auto_merge_steps: Optional[int] = None
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
    initial_retrieved_chunks: Optional[List[RetrievedChunk]] = None
    expanded_retrieved_chunks: Optional[List[RetrievedChunk]] = None


class ChatResponse(BaseModel):
    response: str
    rag_trace: Optional[RagTrace] = None
    agent_trace: Optional[dict] = None


class MessageInfo(BaseModel):
    type: str
    content: str
    timestamp: str
    rag_trace: Optional[RagTrace] = None
    agent_trace: Optional[dict] = None


class SessionMessagesResponse(BaseModel):
    messages: List[MessageInfo]


class SessionInfo(BaseModel):
    session_id: str
    updated_at: str
    message_count: int
    first_message: str = ""


class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]


class SessionDeleteResponse(BaseModel):
    session_id: str
    message: str


class DocumentInfo(BaseModel):
    filename: str
    file_type: str
    chunk_count: int
    uploaded_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_processed: int
    message: str


class DocumentDeleteResponse(BaseModel):
    filename: str
    status: str
    affected_chunks: int
    milvus_deleted: int
    graph_edges_updated: int
    graph_empty_edges_deleted: int
    graph_orphan_nodes_deleted: int


class DocumentStatus(BaseModel):
    filename: str
    is_deleted: bool
    version: int
    chunk_count: int
    updated_at: str


class HitlResumeRequest(BaseModel):
    """HITL 中断恢复请求。"""
    session_id: str
    action: Literal["approve", "reject", "modify"]
    modified_input: Optional[str] = None


class GraphEntity(BaseModel):
    """知识图谱实体节点。"""
    name: str
    type: str
    description: str = ""


class GraphRelation(BaseModel):
    """知识图谱关系边。"""
    subject: str
    predicate: str
    object: str
    description: str = ""
    weight: float = 0.5
    source_chunks: list[str] = []


class QueryPlan(BaseModel):
    """Planner 生成的查询计划。"""
    is_complex: bool = False
    steps: list[dict] = []
    reasoning: str = ""


class CritiqueResult(BaseModel):
    """Critique 评估结果。"""
    is_valid: bool = True
    missing_information: list[str] = []
    feedback: str = ""
    confidence: float = 1.0


class MCPToolCall(BaseModel):
    """MCP 工具调用事件。"""
    server_name: str
    tool_name: str
    arguments: dict = {}
    agent: str = "data_analyst"


class MCPToolResult(BaseModel):
    """MCP 工具调用结果。"""
    server_name: str
    tool_name: str
    result_summary: str = ""
    is_error: bool = False


class UserContextSchema(BaseModel):
    """用户上下文信息，贯穿整个 agent graph。"""
    user_id: int = 0
    tenant_id: int = 0
    tenant_name: str = ""
    role: str = "viewer"
    access_level: int = 1


class QueryPlanStep(BaseModel):
    """Planner DAG 步骤。"""
    step_id: int
    tool_name: str = ""
    agent: str = "data_analyst"
    query: str = ""
    dependencies: list[int] = []
    input_mapping: dict = {}


from datetime import datetime as dt


class TokenUsageSummary(BaseModel):
    tenant_id: int
    period_days: int
    request_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int


class AuditLogEntry(BaseModel):
    id: int
    tenant_id: int
    user_id: int
    action: str
    target: Optional[str]
    result_summary: Optional[str]
    risk_level: str
    created_at: dt


class AuditLogListResponse(BaseModel):
    logs: list[AuditLogEntry]
    total: int
