"""公共 Pydantic 数据模型定义。"""

from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, Field, NonNegativeInt, conlist

SourceType = Literal["local", "web"]


class AnswerRequest(BaseModel):
    """POST /api/agent/answer 的请求体。"""

    q: str = Field(..., description="用户问题，支持自然语言。")


class RoutingDecision(BaseModel):
    """路由决策结果。"""

    policy: Literal["local", "web", "hybrid"] = Field(
        ...,
        description="路由策略：本地/网页/并行。",
    )
    rationale: str = Field(..., description="中文理由，便于观测。")


class LocalEvidence(BaseModel):
    """本地知识库证据结构。"""

    type: Literal["local"] = Field(
        "local",
        description="证据类型，固定为 local。",
    )
    chunk_id: str = Field(..., description="切块唯一 ID，便于溯源。")
    section: str = Field(..., description="原文所属章节标题。")
    excerpt: str = Field(..., description="截取的上下文片段。")


class WebEvidence(BaseModel):
    """外部搜索证据结构。"""

    type: Literal["web"] = Field(
        "web",
        description="证据类型，固定为 web。",
    )
    title: str = Field(..., description="网页标题。")
    url: str = Field(..., description="网页链接。")
    time: str = Field(..., description="发布时间或抓取时间。")
    snippet: str = Field(..., description="摘要内容。")


class LatencyBreakdown(BaseModel):
    """各阶段耗时（毫秒）。"""

    retrieve: NonNegativeInt = Field(0, description="检索阶段耗时，含语义与关键词。")
    rerank: NonNegativeInt = Field(0, description="精排阶段耗时。")
    generate: NonNegativeInt = Field(0, description="生成阶段耗时。")
    total: NonNegativeInt = Field(0, description="端到端总耗时。")


class FinalResponse(BaseModel):
    """统一响应载荷。"""

    answer: str = Field(..., description="最终回答文本。")
    sources: Sequence[LocalEvidence | WebEvidence] = Field(
        default_factory=list,
        description="引用的证据列表，可混合本地与网页。",
    )
    routing: RoutingDecision = Field(..., description="最终路由策略回执。")
    latency_ms: LatencyBreakdown = Field(..., description="延迟分解指标。")
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="模型主观置信度（0-1）。",
    )


EvidenceList = conlist(LocalEvidence | WebEvidence, min_length=0)


