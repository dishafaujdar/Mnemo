"""Add temporal validity CHECK constraint on semantic_edges.

Revision ID: 003
Revises: 002
Create Date: 2026-07-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Repair legacy rows where invalid_at <= valid_at before adding the constraint.
    op.execute(
        sa.text(
            """
            UPDATE semantic_edges
            SET invalid_at = datetime(valid_at, '+1 millisecond')
            WHERE invalid_at IS NOT NULL AND invalid_at <= valid_at
            """
        )
    )

    with op.batch_alter_table("semantic_edges") as batch_op:
        batch_op.create_check_constraint(
            "ck_semantic_edges_invalid_after_valid",
            "invalid_at IS NULL OR invalid_at > valid_at",
        )


def downgrade() -> None:
    with op.batch_alter_table("semantic_edges") as batch_op:
        batch_op.drop_constraint("ck_semantic_edges_invalid_after_valid", type_="check")
