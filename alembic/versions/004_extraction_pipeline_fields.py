"""Extraction pipeline columns + needs_review_facts table.

Revision ID: 004
Revises: 003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("semantic_edges") as batch_op:
        batch_op.add_column(sa.Column("source_span", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("temporal_status", sa.String(length=32), nullable=True))

    op.create_table(
        "needs_review_facts",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("episode_id", sa.String(length=36), sa.ForeignKey("episodes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("subject", sa.String(length=512), nullable=False),
        sa.Column("relation", sa.String(length=128), nullable=False),
        sa.Column("object", sa.String(length=512), nullable=False),
        sa.Column("fact_string", sa.Text(), nullable=False),
        sa.Column("source_span", sa.Text(), nullable=True),
        sa.Column("temporal_status", sa.String(length=32), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True, server_default="0.0"),
        sa.Column("rejection_reason", sa.String(length=255), nullable=False),
        sa.Column("relation_raw", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_needs_review_facts_user_id", "needs_review_facts", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_needs_review_facts_user_id", table_name="needs_review_facts")
    op.drop_table("needs_review_facts")
    with op.batch_alter_table("semantic_edges") as batch_op:
        batch_op.drop_column("temporal_status")
        batch_op.drop_column("source_span")
