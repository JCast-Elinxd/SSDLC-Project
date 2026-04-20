"""
Motor de scoring de riesgo (fase beta) + servicio de transacciones.

Reglas de scoring (suma acumulada, máximo 100 puntos):
  - Monto > $10,000                   → +40 pts
  - Monto > $5,000                    → +20 pts
  - Monto sospechoso (9999, 4999...)  → +15 pts
  - Usuario con >= 5 tx en 1 hora     → +30 pts
  - Usuario con >= 3 tx en 1 hora     → +15 pts
  - Transfer/withdrawal a cuenta ext  → +10 pts

Umbrales de decisión:
   0–25  → LOW      → APPROVED
  26–50  → MEDIUM   → FLAGGED
  51–75  → HIGH     → FLAGGED
  76–100 → CRITICAL → BLOCKED
"""

import json
from datetime import datetime, timezone, timedelta

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.transaction_models import (
    Transaction, TransactionType, RiskLevel, TransactionStatus
)


# ── Constantes ────────────────────────────────────────────────────────────────

BLOCK_THRESHOLD = 76
FLAG_THRESHOLD  = 26
AMOUNT_CRITICAL = 10_000
AMOUNT_HIGH     = 5_000
RECENT_WINDOW   = timedelta(hours=1)


# ── Motor de scoring ──────────────────────────────────────────────────────────

def _is_round_suspicious(amount: float) -> bool:
    """Detecta montos justo por debajo de umbrales redondos (técnica de structuring)."""
    return amount in {999, 4999, 9999, 49999, 99999} or round(amount % 1000) in range(990, 1000)


def calculate_risk_score(
    user_id: str,
    amount: float,
    transaction_type: TransactionType,
    destination_id: str | None,
    db: Session,
) -> tuple[float, list[str]]:
    """
    Calcula el score de riesgo (0–100).
    Retorna (score, lista_de_razones).
    """
    score = 0.0
    reasons: list[str] = []

    # Regla 1: Monto alto
    if amount > AMOUNT_CRITICAL:
        score += 40
        reasons.append(f"Monto muy alto: ${amount:,.2f} (límite: ${AMOUNT_CRITICAL:,})")
    elif amount > AMOUNT_HIGH:
        score += 20
        reasons.append(f"Monto alto: ${amount:,.2f} (límite: ${AMOUNT_HIGH:,})")

    # Regla 2: Monto sospechoso
    if _is_round_suspicious(amount):
        score += 15
        reasons.append(f"Monto sospechoso (posible evasión de límite): ${amount:,.2f}")

    # Regla 3: Frecuencia del usuario en la última hora
    cutoff = datetime.now(timezone.utc) - RECENT_WINDOW
    recent_count = (
        db.query(Transaction)
        .filter(
            Transaction.user_id == user_id,
            Transaction.created_at >= cutoff,
        )
        .count()
    )

    if recent_count >= 5:
        score += 30
        reasons.append(f"Alta frecuencia: {recent_count} transacciones en la última hora")
    elif recent_count >= 3:
        score += 15
        reasons.append(f"Frecuencia elevada: {recent_count} transacciones en la última hora")

    # Regla 4: Transferencia o retiro a cuenta externa
    if transaction_type in (TransactionType.WITHDRAWAL, TransactionType.TRANSFER):
        if destination_id and destination_id != user_id:
            score += 10
            reasons.append(f"Transfer/retiro hacia cuenta externa: {destination_id}")

    return round(min(score, 100.0), 2), reasons


def _score_to_level(score: float) -> RiskLevel:
    if score >= BLOCK_THRESHOLD:
        return RiskLevel.CRITICAL
    elif score >= 51:
        return RiskLevel.HIGH
    elif score >= FLAG_THRESHOLD:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _score_to_status(score: float) -> TransactionStatus:
    if score >= BLOCK_THRESHOLD:
        return TransactionStatus.BLOCKED
    elif score >= FLAG_THRESHOLD:
        return TransactionStatus.FLAGGED
    return TransactionStatus.APPROVED


# ── Servicio de transacciones ─────────────────────────────────────────────────

def create_transaction(
    user_id: str,
    amount: float,
    transaction_type: TransactionType,
    currency: str = "USD",
    destination_id: str | None = None,
    ip_address: str | None = None,
    db: Session = None,
) -> Transaction:
    """Crea una transacción, calcula su riesgo y la persiste en BD."""
    if amount <= 0:
        raise HTTPException(status_code=400, detail="El monto debe ser mayor a 0.")

    score, reasons = calculate_risk_score(
        user_id=user_id,
        amount=amount,
        transaction_type=transaction_type,
        destination_id=destination_id,
        db=db,
    )

    tx = Transaction(
        user_id=user_id,
        amount=amount,
        currency=currency,
        transaction_type=transaction_type,
        destination_id=destination_id,
        risk_score=score,
        risk_level=_score_to_level(score),
        risk_reasons=json.dumps(reasons),
        status=_score_to_status(score),
        blocked=(_score_to_status(score) == TransactionStatus.BLOCKED),
        ip_address=ip_address,
    )
    db.add(tx)
    db.commit()
    db.refresh(tx)
    return tx


def get_transaction(tx_id: str, db: Session) -> Transaction:
    tx = db.query(Transaction).filter(Transaction.id == tx_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail=f"Transacción '{tx_id}' no encontrada.")
    return tx


def list_transactions(
    db: Session,
    user_id: str | None = None,
    status: TransactionStatus | None = None,
    skip: int = 0,
    limit: int = 100,
) -> list[Transaction]:
    query = db.query(Transaction)
    if user_id:
        query = query.filter(Transaction.user_id == user_id)
    if status:
        query = query.filter(Transaction.status == status)
    return query.order_by(Transaction.created_at.desc()).offset(skip).limit(limit).all()