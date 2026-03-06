"""episodes, semantic_edges, profile_facts

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "episodes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("session_id", sa.String(255), nullable=True),
        sa.Column("metadata", sqlite.JSON(), nullable=True),
    )
    op.create_index("ix_episodes_user_id", "episodes", ["user_id"], unique=False)

    op.create_table(
        "semantic_edges",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("subject", sa.String(512), nullable=False),
        sa.Column("relation", sa.String(128), nullable=False),
        sa.Column("object", sa.String(512), nullable=False),
        sa.Column("fact_string", sa.Text(), nullable=False),
        sa.Column("qdrant_id", sa.String(255), nullable=True),
        sa.Column("episode_id", sa.String(36), sa.ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True, server_default="1.0"),
        sa.Column("valid_at", sa.DateTime(), nullable=False),
        sa.Column("invalid_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("metadata", sqlite.JSON(), nullable=True),
    )
    op.create_index("ix_semantic_edges_user_id", "semantic_edges", ["user_id"], unique=False)
    op.create_index("idx_semantic_user_invalid", "semantic_edges", ["user_id", "invalid_at"], unique=False)

    op.create_table(
        "profile_facts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("key", sa.String(255), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("value_type", sa.String(32), nullable=True, server_default="string"),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("user_id", "key", name="uq_profile_facts_user_key"),
    )
    op.create_index("ix_profile_facts_user_id", "profile_facts", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_profile_facts_user_id", table_name="profile_facts")
    op.drop_table("profile_facts")

    op.drop_index("idx_semantic_user_invalid", table_name="semantic_edges")
    op.drop_index("ix_semantic_edges_user_id", table_name="semantic_edges")
    op.drop_table("semantic_edges")

    op.drop_index("ix_episodes_user_id", table_name="episodes")
    op.drop_table("episodes")
