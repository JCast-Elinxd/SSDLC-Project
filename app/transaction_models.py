"""
Modelo de base de datos para Transacciones y Scoring de Riesgo.
Archivo nuevo — no modifica models.py existente.
"""
import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import Column, String, DateTime, Float, Integer, Enum, Boolean, Text
from app.models import Base, engine  # reutiliza la misma Base y engine


# ── Enums ────────────────────────────────────────────────────────────────────

class TransactionType(str, PyEnum):
    TRANSFER   = "transfer"
    PAYMENT    = "payment"
    WITHDRAWAL = "withdrawal"
    DEPOSIT    = "deposit"

class RiskLevel(str, PyEnum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"

class TransactionStatus(str, PyEnum):
    APPROVED = "approved"
    BLOCKED  = "blocked"
    FLAGGED  = "flagged"   # aprobada pero marcada para revisión


# ── Modelo ────────────────────────────────────────────────────────────────────

class Transaction(Base):
    """Registro de una transacción con su scoring de riesgo."""

    __tablename__ = "transactions"

    id               = Column(String,  primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id          = Column(String,  nullable=False, index=True)
    amount           = Column(Float,   nullable=False)
    currency         = Column(String,  default="USD")
    transaction_type = Column(Enum(TransactionType), nullable=False)
    destination_id   = Column(String,  nullable=True)

    # Scoring
    risk_score   = Column(Float, nullable=False, default=0.0)
    risk_level   = Column(Enum(RiskLevel), nullable=False, default=RiskLevel.LOW)
    risk_reasons = Column(Text, nullable=True)

    # Resultado
    status  = Column(Enum(TransactionStatus), nullable=False)
    blocked = Column(Boolean, default=False)

    # Metadata
    ip_address = Column(String,   nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def create_transaction_tables():
    """Crea la tabla transactions si no existe."""
    Base.metadata.create_all(bind=engine)