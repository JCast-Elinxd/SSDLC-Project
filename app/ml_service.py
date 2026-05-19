"""
Servicio de predicción ML para scoring de riesgo de transacciones.
Red neuronal con PyTorch — carga el modelo entrenado por train_model.py.

El modelo devuelve probabilidades por clase (approved / flagged / blocked)
y un ml_risk_score de 0–100 que se combina con el score de reglas en
transaction_services.py mediante un score híbrido con peso 50% ML / 50% reglas.

Uso:
    from app.ml_service import get_ml_prediction, hybrid_score, MLPrediction
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from app.transactions.models import TransactionType, TransactionStatus, RiskLevel

logger = logging.getLogger(__name__)

# ── Configuración ─────────────────────────────────────────────────────────────

MODEL_PATH  = Path(os.getenv("ML_MODEL_PATH",  "app/model.pt"))
SCALER_PATH = Path(os.getenv("ML_SCALER_PATH", "app/scaler.pkl"))

# Mapeo TransactionType → entero (debe coincidir con train_model.py)
_TX_TYPE_MAP: dict[TransactionType, int] = {
    TransactionType.TRANSFER:   0,
    TransactionType.PAYMENT:    1,
    TransactionType.WITHDRAWAL: 2,
    TransactionType.DEPOSIT:    3,
}

# Mapeo índice de clase → enums del proyecto
_IDX_TO_STATUS: dict[int, TransactionStatus] = {
    0: TransactionStatus.APPROVED,
    1: TransactionStatus.FLAGGED,
    2: TransactionStatus.BLOCKED,
}
_IDX_TO_RISK: dict[int, RiskLevel] = {
    0: RiskLevel.LOW,
    1: RiskLevel.MEDIUM,
    2: RiskLevel.CRITICAL,
}


# ── Arquitectura (debe ser idéntica a train_model.py) ─────────────────────────

class FraudModel(nn.Module):
    """
    Red neuronal para clasificación de riesgo de transacciones.
    Entrada: 5 features → Salida: 3 logits (approved / flagged / blocked)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.net(x)


# ── Carga del modelo (singleton) ──────────────────────────────────────────────

_model  = None
_scaler = None


def _load_model():
    """Carga el modelo y el scaler desde disco la primera vez."""
    global _model, _scaler

    if _model is not None:
        return _model, _scaler

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise RuntimeError(
            f"Modelo no encontrado ({MODEL_PATH}, {SCALER_PATH}). "
            "Ejecuta 'python train_model.py' para entrenarlo."
        )

    m = FraudModel()
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    m.eval()

    _model  = m
    _scaler = joblib.load(SCALER_PATH)
    logger.info("Modelo ML cargado desde '%s'.", MODEL_PATH)
    return _model, _scaler


# ── Dataclass de resultado ────────────────────────────────────────────────────

@dataclass
class MLPrediction:
    """Resultado de la predicción de la red neuronal."""
    predicted_status:     TransactionStatus
    predicted_risk_level: RiskLevel
    prob_approved:        float   # probabilidad clase 0
    prob_flagged:         float   # probabilidad clase 1
    prob_blocked:         float   # probabilidad clase 2
    ml_risk_score:        float   # score 0–100 derivado de las probabilidades
    model_available:      bool    # False si el modelo no pudo cargarse


# ── Función principal ─────────────────────────────────────────────────────────

def get_ml_prediction(
    amount: float,
    transaction_type: TransactionType,
    user_id: str,
    destination_id: str | None,
    recent_tx_count: int,
    created_at: datetime | None = None,
) -> MLPrediction:
    """
    Genera una predicción ML para una transacción usando la red neuronal.

    Si el modelo no está disponible, retorna MLPrediction con
    model_available=False y valores neutros — la API sigue funcionando
    solo con las reglas en ese caso.
    """
    try:
        model, scaler = _load_model()
    except RuntimeError as exc:
        logger.warning("ML no disponible: %s", exc)
        return MLPrediction(
            predicted_status=TransactionStatus.APPROVED,
            predicted_risk_level=RiskLevel.LOW,
            prob_approved=1.0,
            prob_flagged=0.0,
            prob_blocked=0.0,
            ml_risk_score=0.0,
            model_available=False,
        )

    # ── Construir vector de features (mismo orden que train_model.py) ─────────
    ts         = created_at or datetime.now(timezone.utc)
    hour       = ts.hour
    tx_type_int   = _TX_TYPE_MAP.get(transaction_type, 1)
    dest_external = int(bool(destination_id and destination_id != user_id))

    X_raw = np.array([[amount, tx_type_int, dest_external, hour, recent_tx_count]],
                     dtype=np.float32)
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    # ── Inferencia ────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(torch.tensor(X_scaled))
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    pred_idx = int(np.argmax(probs))

    # Score ML: pondera las probabilidades (0→0 pts, 1→50 pts, 2→100 pts)
    ml_score = round(float(probs[1] * 50 + probs[2] * 100), 2)

    return MLPrediction(
        predicted_status=_IDX_TO_STATUS[pred_idx],
        predicted_risk_level=_IDX_TO_RISK[pred_idx],
        prob_approved=round(float(probs[0]), 4),
        prob_flagged=round(float(probs[1]), 4),
        prob_blocked=round(float(probs[2]), 4),
        ml_risk_score=min(ml_score, 100.0),
        model_available=True,
    )


# ── Score híbrido ─────────────────────────────────────────────────────────────

def hybrid_score(rules_score: float, ml_score: float, ml_weight: float = 0.5) -> float:
    """
    Combina el score de reglas con el score ML.

    Args:
        rules_score: Score del motor de reglas (0–100).
        ml_score:    Score ML (0–100).
        ml_weight:   Peso del componente ML (default 50%).
                     El 50% restante corresponde a las reglas.

    Returns:
        Score híbrido redondeado a 2 decimales (0–100).
    """
    combined = rules_score * (1 - ml_weight) + ml_score * ml_weight
    return round(min(combined, 100.0), 2)