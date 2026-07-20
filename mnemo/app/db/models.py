"""SQLAlchemy models: episodes, semantic_edges, profile_facts."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def gen_uuid() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class Episode(Base):
    """Raw conversation turns. Immutable after insert."""

    __tablename__ = "episodes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # 'user' | 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    session_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    semantic_edges = relationship("SemanticEdge", back_populates="episode", foreign_keys="SemanticEdge.episode_id")


class SemanticEdge(Base):
    """Bi-temporal triplet edges (subject --relation--> object)."""

    __tablename__ = "semantic_edges"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    subject: Mapped[str] = mapped_column(String(512), nullable=False)
    relation: Mapped[str] = mapped_column(String(128), nullable=False)
    object: Mapped[str] = mapped_column(String(512), nullable=False)
    fact_string: Mapped[str] = mapped_column(Text, nullable=False)
    # Relation as originally emitted by the extractor, before ontology normalization.
    relation_raw: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Fuzzy ontology match score (1.0 = exact/canonical, <threshold = unknown).
    relation_match_score: Mapped[float] = mapped_column(Float, default=1.0)
    # confirmed | fuzzy | pending (unknown relation, flagged for audit).
    review_status: Mapped[str] = mapped_column(String(32), default="confirmed")
    qdrant_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    episode_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("episodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    valid_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    invalid_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    episode = relationship("Episode", back_populates="semantic_edges")


# Index for "active" edges: user_id + invalid_at IS NULL
Index("idx_semantic_user_invalid", SemanticEdge.user_id, SemanticEdge.invalid_at)


class ProfileFact(Base):
    """Persistent user profile key-value facts."""

    __tablename__ = "profile_facts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str] = mapped_column(String(32), default="string")
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("user_id", "key", name="uq_profile_facts_user_key"),)


class UnknownRelation(Base):
    """Audit queue for relations that didn't match the ontology.

    Populated when facts with unknown (fuzzy-miss) relations are stored. A
    periodic human/automated audit can promote frequent ones into the seed
    ontology and assign them a behavior (singular/multi/temporal).
    """

    __tablename__ = "unknown_relations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    relation: Mapped[str] = mapped_column(String(128), nullable=False)  # canonical token
    relation_raw: Mapped[str | None] = mapped_column(String(128), nullable=True)  # sample phrasing
    count: Mapped[int] = mapped_column(Integer, default=1)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending | promoted | ignored
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("relation", name="uq_unknown_relations_relation"),)
