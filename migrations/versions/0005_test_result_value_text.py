"""widen test_result.value to Text (imaging narratives exceed 120 chars)

Revision ID: 0005
Revises: 0004
"""
import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("test_result", "value",
                    existing_type=sa.String(length=120),
                    type_=sa.Text(),
                    existing_nullable=True)


def downgrade() -> None:
    # Truncate any over-length values so the narrower type fits.
    op.execute("UPDATE test_result SET value = left(value, 120) WHERE length(value) > 120")
    op.alter_column("test_result", "value",
                    existing_type=sa.Text(),
                    type_=sa.String(length=120),
                    existing_nullable=True)
