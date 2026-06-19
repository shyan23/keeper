"""add document.content_hash for upload dedup

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("document", sa.Column("content_hash", sa.String(64), nullable=True))
    op.create_index("ix_document_content_hash", "document", ["content_hash"])


def downgrade() -> None:
    op.drop_index("ix_document_content_hash", table_name="document")
    op.drop_column("document", "content_hash")
