"""Pydantic schemas for memory API: ingest, retrieve, search."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


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
    valid_from: datetime
    valid_until: datetime | None
    source_episode_id: str
    relevance_score: float
    recorded_at: datetime
    retracted_at: datetime | None


class RetrieveResponse(BaseModel):
    memories: list[RetrievedMemory]
    profile: dict[str, Any]
    token_count: int


class RetrieveRequest(BaseModel):
    user_id: str
    query: str
    limit: int = 10
    as_of_valid_time = None
    as_of_transaction_time = None
    max_tokens: int = 2000

    @model_validator(mode='after')
    def set_temporal_defaults(self) -> 'RetrieveRequest':
        now = datetime.now(timezone.utc)
        if self.as_of_valid_time is None:
            self.as_of_valid_time = now
        if self.as_of_transaction_time is None:
            self.as_of_transaction_time = now
        return self


class SearchRequest(BaseModel):
    user_id: str
    query: str
    limit: int = 10
    as_of_valid_time = None
    as_of_transaction_time = None
    valid_only: bool = True #only return facts where valid_until IS NULL

    @model_validator(mode='after')
    def set_temporal_defaults(self) -> 'SearchRequest':
        now = datetime.now(timezone.utc)
        if self.as_of_valid_time is None:
            self.as_of_valid_time = now
        if self.as_of_transaction_time is None:
            self.as_of_transaction_time = now
        return self


class SearchResultItem(BaseModel):
    fact: str
    confidence: float
    valid_from: datetime
    valid_until: datetime | None
    recorded_at: datetime
    retracted_at: datetime | None
    source_episode_id: str
    relevance_score: float


class SearchResponse(BaseModel):
    results: list[SearchResultItem]


class DeleteMemoryResponse(BaseModel):
    id: str
    status: str = "invalidated"
    retracted_at: datetime | None
