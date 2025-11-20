"""FastAPI 入口。"""

from __future__ import annotations

import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import __version__
from backend.agent import orchestrator, router
from backend.core import logging as logging_utils
from backend.rag import vectorstore
from backend.schemas.common import AnswerRequest, FinalResponse, RoutingDecision

logging_utils.setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NLP Agent Backend",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    await asyncio.to_thread(vectorstore.ensure_vectorstore)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/agent/answer", response_model=FinalResponse)
async def agent_answer(payload: AnswerRequest) -> FinalResponse:
    return await orchestrator.answer(payload.q)


@app.post("/api/router/intent", response_model=RoutingDecision)
async def test_intent_recognition(payload: AnswerRequest) -> RoutingDecision:
    """测试意图识别模块：仅返回路由决策，不执行完整检索流程。"""
    return await router.route(payload.q)




