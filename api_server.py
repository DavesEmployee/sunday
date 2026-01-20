#!/usr/bin/env python3
"""
Minimal HTTP API for the RAG chatbot.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from langfuse import propagate_attributes

from rag_chatbot import (
    _run_with_retries,
    langfuse,
    session_id_var,
)


app = FastAPI(title="Sunday RAG Chatbot")
_history_store: Dict[str, List[object]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())
    session_id_var.set(session_id)
    history = _history_store.get(session_id)
    with propagate_attributes(session_id=session_id):
        with langfuse.start_as_current_observation(
            as_type="agent",
            name="rag_chatbot_api",
            input=req.message,
            metadata={"session_id": session_id},
        ) as gen:
            result = _run_with_retries(req.message, history)
            gen.update(output=str(result.output))
    messages = result.all_messages()
    _history_store[session_id] = messages[-40:] if messages else []
    return ChatResponse(session_id=session_id, response=str(result.output))

