from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm import relationship as orm_relationship

from app.db import Base

EMBED_DIM = 768  # text-embedding-004


class Patient(Base):
    __tablename__ = "patient"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(20), nullable=True)
    relationship: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    documents: Mapped[list["Document"]] = orm_relationship(back_populates="patient", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "document"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patient.id", ondelete="CASCADE"), index=True)
    doc_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    classification: Mapped[str | None] = mapped_column(String(50), nullable=True)
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)  # local disk path under STORAGE_DIR
    source_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    raw_ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="uploaded")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    patient: Mapped["Patient"] = orm_relationship(back_populates="documents")


class Doctor(Base):
    __tablename__ = "doctor"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    specialty: Mapped[str | None] = mapped_column(String(120), nullable=True)
    contact: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Disease(Base):
    __tablename__ = "disease"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    icd_code: Mapped[str | None] = mapped_column(String(20), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Symptom(Base):
    __tablename__ = "symptom"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Medication(Base):
    __tablename__ = "medication"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    dosage_form: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MedicalTest(Base):
    __tablename__ = "medical_test"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TestResult(Base):
    __tablename__ = "test_result"
    id: Mapped[int] = mapped_column(primary_key=True)
    medical_test_id: Mapped[int] = mapped_column(ForeignKey("medical_test.id", ondelete="CASCADE"), index=True)
    value: Mapped[str | None] = mapped_column(String(120), nullable=True)
    unit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    reference_range: Mapped[str | None] = mapped_column(String(120), nullable=True)
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DocumentEntity(Base):
    __tablename__ = "document_entity"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id", ondelete="CASCADE"), index=True)
    entity_type: Mapped[str] = mapped_column(String(30))  # doctor|disease|symptom|medication|medical_test|test_result
    entity_id: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    validated: Mapped[bool] = mapped_column(Boolean, default=False)
    source_span: Mapped[str | None] = mapped_column(Text, nullable=True)


class Chunk(Base):
    __tablename__ = "chunk"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id", ondelete="CASCADE"), index=True)
    patient_id: Mapped[int] = mapped_column(Integer, index=True)
    ord: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    page_ref: Mapped[str | None] = mapped_column(String(40), nullable=True)
    section_ref: Mapped[str | None] = mapped_column(String(120), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM), nullable=True)
