"""initial schema + pgvector

Revision ID: 0001
Revises:
"""
import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "patient",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("age", sa.Integer),
        sa.Column("gender", sa.String(20)),
        sa.Column("relationship", sa.String(50)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "document",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("patient_id", sa.Integer, sa.ForeignKey("patient.id", ondelete="CASCADE"), index=True, nullable=False),
        sa.Column("doc_type", sa.String(50)),
        sa.Column("classification", sa.String(50)),
        sa.Column("file_path", sa.Text),
        sa.Column("source_type", sa.String(20)),
        sa.Column("mime_type", sa.String(100)),
        sa.Column("raw_ocr_text", sa.Text),
        sa.Column("status", sa.String(30), server_default="uploaded"),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "doctor",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("specialty", sa.String(120)),
        sa.Column("contact", sa.String(200)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "disease",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("icd_code", sa.String(20)),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "symptom",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "medication",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("dosage_form", sa.String(80)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "medical_test",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "test_result",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("medical_test_id", sa.Integer, sa.ForeignKey("medical_test.id", ondelete="CASCADE"), index=True, nullable=False),
        sa.Column("value", sa.String(120)),
        sa.Column("unit", sa.String(40)),
        sa.Column("reference_range", sa.String(120)),
        sa.Column("observed_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "document_entity",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("document.id", ondelete="CASCADE"), index=True, nullable=False),
        sa.Column("entity_type", sa.String(30), nullable=False),
        sa.Column("entity_id", sa.Integer, nullable=False),
        sa.Column("confidence", sa.Float),
        sa.Column("validated", sa.Boolean, server_default=sa.false()),
        sa.Column("source_span", sa.Text),
    )
    op.create_table(
        "chunk",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("document.id", ondelete="CASCADE"), index=True, nullable=False),
        sa.Column("patient_id", sa.Integer, index=True, nullable=False),
        sa.Column("ord", sa.Integer, server_default="0"),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("page_ref", sa.String(40)),
        sa.Column("section_ref", sa.String(120)),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(768)),
    )


def downgrade() -> None:
    for t in ["chunk", "document_entity", "test_result", "medical_test",
              "medication", "symptom", "disease", "doctor", "document", "patient"]:
        op.drop_table(t)
