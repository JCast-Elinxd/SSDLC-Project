"""
Router de documentos — endpoints
  POST   /documents/upload   
  GET    /documents/{id}     
  GET    /documents/         
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, Query
from pydantic import BaseModel # Validacion de Datos
from sqlalchemy.orm import Session # SQL

from app.models import get_db
from app.services import upload_document, get_document, list_documents

import shutil, os # Automated Data management

router = APIRouter(prefix="/documents", tags=["documents"])


# ── Schemas de respuesta (Pydantic) ──────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    original_filename: str
    file_format: str
    file_size_bytes: int
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class UploadResponse(BaseModel):
    message: str
    document: DocumentResponse
    indexing_status: str # Nuevo campo para confirmar ChromaDB


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentResponse]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document_endpoint(
    file: UploadFile = File(..., description="Archivo PDF, DOCX, TXT o XLSX"),
    db: Session = Depends(get_db),
):
    # 1. Guardar metadatos en SQL
    doc = await upload_document(file, db)
    
    return UploadResponse(
        message="Documento subido exitosamente.",
        document=DocumentResponse.model_validate(doc),
        indexing_status= "No indexado"
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
def get_document_endpoint(
    doc_id: str,
    db: Session = Depends(get_db),
):
    doc = get_document(doc_id, db)
    return DocumentResponse.model_validate(doc)


@router.get("/", response_model=DocumentListResponse)
def list_documents_endpoint(
    skip: int = Query(default=0, ge=0, description="Registros a omitir (paginación)"),
    limit: int = Query(default=100, ge=1, le=500, description="Máximo de registros a retornar"),
    db: Session = Depends(get_db),
):
    docs = list_documents(db, skip=skip, limit=limit)
    return DocumentListResponse(
        total=len(docs),
        documents=[DocumentResponse.model_validate(d) for d in docs],
    )