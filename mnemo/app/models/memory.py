"""Pydantic schemas for memory API: ingest, retrieve, search."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class IngestMessage(BaseModel):
    role: str  # 'user' | 'assistant'
    content: str


class IngestRequest(BaseModel):
    user_id: str
    messages: list[IngestMessage]
    session_id: str | None = None
    metadata: dict[str, Any] | None = None


class IngestResponse(BaseModel):
    episode_id: str
    status: str = "ingested"
    extraction: str = "queued"


class RetrievedMemory(BaseModel):
    fact: str
    confidence: float
    valid_at: datetime
    invalid_at: datetime | None
    source_episode_id: str
    relevance_score: float


class RetrieveResponse(BaseModel):
    memories: list[RetrievedMemory]
    profile: dict[str, Any]
    token_count: int


class SearchRequest(BaseModel):
    user_id: str
    query: str
    limit: int = 10
    valid_only: bool = True
    include_history: bool = False


class SearchResultItem(BaseModel):
    fact: str
    confidence: float
    valid_at: datetime
    invalid_at: datetime | None
    source_episode_id: str
    relevance_score: float


class SearchResponse(BaseModel):
    results: list[SearchResultItem]


class DeleteMemoryResponse(BaseModel):
    id: str
    status: str = "invalidated"
    invalid_at: datetime | None
