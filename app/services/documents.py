import datetime as dt

from sqlalchemy.orm import Session

from app.models import Document


def create_document(db: Session, *, patient_id: int, doc_type: str | None = None,
                    source_type: str | None = None, mime_type: str | None = None,
                    file_path: str | None = None, content_hash: str | None = None,
                    report_date: dt.date | None = None,
                    original_name: str | None = None) -> Document:
    doc = Document(
        patient_id=patient_id,
        doc_type=doc_type,
        source_type=source_type,
        mime_type=mime_type,
        file_path=file_path,
        content_hash=content_hash,
        report_date=report_date,
        original_name=original_name,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_document(db: Session, document_id: int) -> Document | None:
    return db.get(Document, document_id)


def find_by_content_hash(db: Session, content_hash: str) -> Document | None:
    """Return an already-ingested document with this exact file hash, if any.
    Used to make re-uploading the same file idempotent (no duplicate entities)."""
    return (
        db.query(Document)
        .filter(Document.content_hash == content_hash)
        .order_by(Document.id.asc())
        .first()
    )


def set_file_path(db: Session, document_id: int, path: str) -> Document:
    doc = db.get(Document, document_id)
    if doc is None:
        raise ValueError(f"document {document_id} not found")
    doc.file_path = path
    db.commit()
    db.refresh(doc)
    return doc


def list_documents(db: Session, patient_id: int | None = None) -> list[Document]:
    q = db.query(Document)
    if patient_id is not None:
        q = q.filter(Document.patient_id == patient_id)
    return q.order_by(Document.uploaded_at.desc(), Document.id.desc()).all()


def count_documents(db: Session, patient_id: int | None = None) -> int:
    q = db.query(Document)
    if patient_id is not None:
        q = q.filter(Document.patient_id == patient_id)
    return q.count()
