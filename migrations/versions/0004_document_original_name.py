"""add document.original_name (uploaded filename, for display)

Revision ID: 0004
Revises: 0003
"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("document", sa.Column("original_name", sa.String(length=300), nullable=True))


def downgrade() -> None:
    op.drop_column("document", "original_name")
