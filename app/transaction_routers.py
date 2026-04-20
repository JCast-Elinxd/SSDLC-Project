"""
Router de transacciones — endpoints:
  POST  /transactions/               → crear y evaluar transacción
  GET   /transactions/{id}           → detalle de una transacción
  GET   /transactions/               → listar (filtros opcionales)
  GET   /transactions/user/{user_id} → historial de un usuario
"""

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.models import get_db
from app.transaction_models import (
    TransactionType, TransactionStatus, RiskLevel, create_transaction_tables
)
from app.transaction_services import (
    create_transaction, get_transaction, list_transactions
)

create_transaction_tables()

router = APIRouter(prefix="/transactions", tags=["transactions"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    user_id:          str             = Field(..., example="user_123")
    amount:           float           = Field(..., gt=0, example=1500.00)
    currency:         str             = Field(default="USD", example="USD")
    transaction_type: TransactionType = Field(..., example="transfer")
    destination_id:   Optional[str]   = Field(default=None, example="user_456")


class TransactionResponse(BaseModel):
    id:               str
    user_id:          str
    amount:           float
    currency:         str
    transaction_type: TransactionType
    destination_id:   Optional[str]
    risk_score:       float
    risk_level:       RiskLevel
    risk_reasons:     list[str]
    status:           TransactionStatus
    blocked:          bool
    ip_address:       Optional[str]
    created_at:       datetime

    model_config = {"from_attributes": True}

    @classmethod
    def from_tx(cls, tx):
        return cls(
            id=tx.id,
            user_id=tx.user_id,
            amount=tx.amount,
            currency=tx.currency,
            transaction_type=tx.transaction_type,
            destination_id=tx.destination_id,
            risk_score=tx.risk_score,
            risk_level=tx.risk_level,
            risk_reasons=json.loads(tx.risk_reasons or "[]"),
            status=tx.status,
            blocked=tx.blocked,
            ip_address=tx.ip_address,
            created_at=tx.created_at,
        )


class CreateTransactionResponse(BaseModel):
    message:     str
    transaction: TransactionResponse
    alert:       Optional[str] = None


class TransactionListResponse(BaseModel):
    total:        int
    transactions: list[TransactionResponse]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/", response_model=CreateTransactionResponse, status_code=201)
def create_transaction_endpoint(
    body:    TransactionRequest,
    request: Request,
    db:      Session = Depends(get_db),
):
    """Evalúa y registra una nueva transacción."""
    ip = request.client.host if request.client else None

    tx = create_transaction(
        user_id=body.user_id,
        amount=body.amount,
        transaction_type=body.transaction_type,
        currency=body.currency,
        destination_id=body.destination_id,
        ip_address=ip,
        db=db,
    )

    alert = None
    if tx.status == TransactionStatus.BLOCKED:
        alert = f"⛔ Transacción BLOQUEADA. Score: {tx.risk_score}/100."
    elif tx.status == TransactionStatus.FLAGGED:
        alert = f"⚠️ Transacción MARCADA para revisión. Score: {tx.risk_score}/100."

    return CreateTransactionResponse(
        message=f"Transacción procesada — estado: {tx.status.value}.",
        transaction=TransactionResponse.from_tx(tx),
        alert=alert,
    )


@router.get("/user/{user_id}", response_model=TransactionListResponse)
def get_user_transactions(
    user_id: str,
    skip:    int = Query(default=0, ge=0),
    limit:   int = Query(default=50, ge=1, le=200),
    db:      Session = Depends(get_db),
):
    """Historial de transacciones de un usuario."""
    txs = list_transactions(db, user_id=user_id, skip=skip, limit=limit)
    return TransactionListResponse(
        total=len(txs),
        transactions=[TransactionResponse.from_tx(t) for t in txs],
    )


@router.get("/", response_model=TransactionListResponse)
def list_transactions_endpoint(
    status: Optional[TransactionStatus] = Query(default=None),
    skip:   int = Query(default=0, ge=0),
    limit:  int = Query(default=100, ge=1, le=500),
    db:     Session = Depends(get_db),
):
    """Lista todas las transacciones con filtro opcional por estado."""
    txs = list_transactions(db, status=status, skip=skip, limit=limit)
    return TransactionListResponse(
        total=len(txs),
        transactions=[TransactionResponse.from_tx(t) for t in txs],
    )


@router.get("/{tx_id}", response_model=TransactionResponse)
def get_transaction_endpoint(
    tx_id: str,
    db:    Session = Depends(get_db),
):
    """Detalle de una transacción por ID."""
    tx = get_transaction(tx_id, db)
    return TransactionResponse.from_tx(tx)