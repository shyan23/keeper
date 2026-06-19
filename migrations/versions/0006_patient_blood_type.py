"""add patient.blood_type (blood group, tracked from latest document)

Revision ID: 0006
Revises: 0005
"""
import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("patient", sa.Column("blood_type", sa.String(length=8), nullable=True))


def downgrade() -> None:
    op.drop_column("patient", "blood_type")
