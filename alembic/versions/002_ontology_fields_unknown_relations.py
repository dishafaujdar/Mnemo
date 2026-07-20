"""semantic_edges ontology fields + unknown_relations audit table

Revision ID: 002
Revises: 001
Create Date: 2026-07-20 00:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "semantic_edges",
        sa.Column("relation_raw", sa.String(128), nullable=True),
    )
    op.add_column(
        "semantic_edges",
        sa.Column("relation_match_score", sa.Float(), nullable=True, server_default="1.0"),
    )
    op.add_column(
        "semantic_edges",
        sa.Column("review_status", sa.String(32), nullable=True, server_default="confirmed"),
    )

    op.create_table(
        "unknown_relations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("relation", sa.String(128), nullable=False),
        sa.Column("relation_raw", sa.String(128), nullable=True),
        sa.Column("count", sa.Integer(), nullable=True, server_default="1"),
        sa.Column("avg_confidence", sa.Float(), nullable=True, server_default="0.0"),
        sa.Column("status", sa.String(32), nullable=True, server_default="pending"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("relation", name="uq_unknown_relations_relation"),
    )


def downgrade() -> None:
    op.drop_table("unknown_relations")
    op.drop_column("semantic_edges", "review_status")
    op.drop_column("semantic_edges", "relation_match_score")
    op.drop_column("semantic_edges", "relation_raw")
