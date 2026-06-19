"""add GIN full-text index on chunk.text for hybrid retrieval

Revision ID: 0007
Revises: 0006
"""
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Expression GIN index over the English text-search vector — the BM25/FTS half
    # of hybrid retrieval (services/retrieval.py). Matches models.py ix_chunk_text_fts.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunk_text_fts "
        "ON chunk USING gin (to_tsvector('english', text))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunk_text_fts")
