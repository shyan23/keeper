"""add document.report_date (date printed on the document)

Revision ID: 0003
Revises: 0002
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("document", sa.Column("report_date", sa.Date(), nullable=True))


def downgrade() -> None:
    op.drop_column("document", "report_date")
