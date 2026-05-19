"""
Script de entrenamiento del modelo de detección de fraude.
Red neuronal con PyTorch — sigue el estilo del Notebook_19.

Genera datos sintéticos basados en las mismas reglas del motor de scoring
(transaction_services.py), entrena una red neuronal y guarda el modelo
junto al scaler en app/model.pt y app/scaler.pkl.

Uso:
    python train_model.py               # entrena y guarda
    python train_model.py --evaluate    # entrena, evalúa y muestra métricas
"""

import argparse
import logging
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Configuración ─────────────────────────────────────────────────────────────

RANDOM_STATE  = 42
N_SAMPLES     = 5_000
NUM_EPOCHS    = 50
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
MODEL_PATH    = Path("app/model.pt")
SCALER_PATH   = Path("app/scaler.pkl")

# Mismas constantes que transaction_services.py
AMOUNT_CRITICAL = 10_000
AMOUNT_HIGH     = 5_000
BLOCK_THRESHOLD = 76
FLAG_THRESHOLD  = 26

LABEL_NAMES = ["approved", "flagged", "blocked"]

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# ── 1. Generación de datos sintéticos ─────────────────────────────────────────

def generate_dataset(n: int = N_SAMPLES, noise_rate: float = 0.05):
    """
    Genera un dataset sintético que replica la distribución esperada de
    transacciones, con una pequeña tasa de ruido para evitar que la red
    simplemente memorice las reglas.

    Features:
      0 - amount            : monto de la transacción
      1 - transaction_type  : 0=transfer, 1=payment, 2=withdrawal, 3=deposit
      2 - dest_external     : 1 si destino es cuenta externa
      3 - hour_of_day       : hora UTC (0-23)
      4 - recent_tx_count   : nº de tx del usuario en la última hora

    Labels:
      0 → approved  (score < 26)
      1 → flagged   (26 ≤ score < 76)
      2 → blocked   (score ≥ 76)
    """
    rng = np.random.default_rng(RANDOM_STATE)

    amounts = np.concatenate([
        rng.exponential(scale=500,   size=int(n * 0.60)),
        rng.uniform(5_000, 15_000,   size=int(n * 0.20)),
        rng.uniform(10_001, 100_000, size=int(n * 0.10)),
        np.tile([999.0, 4_999.0, 9_999.0, 49_999.0],
                int(n * 0.10 // 4) + 1)[:int(n * 0.10)],
    ])[:n]

    tx_types        = rng.choice([0, 1, 2, 3], size=n, p=[0.35, 0.25, 0.25, 0.15])
    dest_external   = rng.binomial(1, 0.4, n)
    hour_of_day     = rng.integers(0, 24, n)
    recent_tx_count = rng.choice(
        [0, 1, 2, 3, 4, 5, 6, 7], size=n,
        p=[0.40, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.02],
    )

    labels = [
        _score_to_label(
            _calc_rule_score(amounts[i], tx_types[i], dest_external[i], recent_tx_count[i])
        )
        for i in range(n)
    ]

    # Pequeño ruido para simular casos borde reales
    rnd = random.Random(RANDOM_STATE)
    labels = [rnd.choice([0, 1, 2]) if rnd.random() < noise_rate else l for l in labels]

    X = np.column_stack([amounts, tx_types, dest_external, hour_of_day, recent_tx_count])
    y = np.array(labels, dtype=np.int64)
    return X.astype(np.float32), y


def _is_round_suspicious(amount: float) -> bool:
    return (
        amount in {999, 4_999, 9_999, 49_999, 99_999}
        or round(amount % 1000) in range(990, 1000)
    )


def _calc_rule_score(amount: float, tx_type: int, dest_ext: int, recent: int) -> float:
    """Replica calculate_risk_score() de transaction_services.py."""
    score = 0.0
    if amount > AMOUNT_CRITICAL:       score += 40
    elif amount > AMOUNT_HIGH:         score += 20
    if _is_round_suspicious(amount):   score += 15
    if recent >= 5:                    score += 30
    elif recent >= 3:                  score += 15
    if tx_type in (0, 2) and dest_ext: score += 10
    return min(score, 100.0)


def _score_to_label(score: float) -> int:
    if score >= BLOCK_THRESHOLD: return 2
    if score >= FLAG_THRESHOLD:  return 1
    return 0


# ── 2. Dataset de PyTorch ─────────────────────────────────────────────────────

class FraudDataset(Dataset):
    """Dataset de transacciones para PyTorch."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── 3. Arquitectura de la red ─────────────────────────────────────────────────

class FraudModel(nn.Module):
    """
    Red neuronal para clasificación de riesgo de transacciones.
    Entrada: 5 features → Salida: 3 logits (approved / flagged / blocked)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),   # 5 features de entrada, 32 neuronas
            nn.ReLU(),
            nn.Linear(32, 16),  # ingresan 32 de la capa anterior, salen 16
            nn.ReLU(),
            nn.Linear(16, 3),   # 3 clases de salida: approved, flagged, blocked
        )

    def forward(self, x):
        return self.net(x)


# ── 4. Entrenamiento ──────────────────────────────────────────────────────────

def train(evaluate: bool = False):
    # 4.1 Generar y dividir datos (60 / 20 / 20)
    logger.info("Generando dataset sintético (%d muestras)…", N_SAMPLES)
    X, y = generate_dataset()

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE
    )
    # queda 60 / 20 / 20

    logger.info("Train: %d  Val: %d  Test: %d", len(X_train), len(X_val), len(X_test))
    dist = dict(zip(*np.unique(y_train, return_counts=True)))
    logger.info("Distribución train: %s", {LABEL_NAMES[k]: v for k, v in dist.items()})

    # 4.2 Escalar features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # 4.3 DataLoaders
    train_loader = DataLoader(FraudDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(FraudDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(FraudDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

    # 4.4 Modelo, función de pérdida y optimizador
    model = FraudModel()

    # Peso inverso por clase para manejar el desbalance (blocked es muy raro)
    class_counts  = np.bincount(y_train)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4.5 Loop de entrenamiento (igual al notebook)
    logger.info("Entrenando red neuronal (%d epochs)…", NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluación en validación
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                preds  = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d — loss: %.4f — val_acc: %.4f",
                        epoch + 1, NUM_EPOCHS, epoch_loss, acc)

    # 4.6 Evaluación en test (solo si --evaluate)
    if evaluate:
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits = model(X_batch)
                preds  = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        final_acc = accuracy_score(all_targets, all_preds)
        logger.info("Test accuracy: %.4f", final_acc)
        print(classification_report(all_targets, all_preds, target_names=LABEL_NAMES, digits=4))

    return model, scaler


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entrena la red neuronal de detección de fraude."
    )
    parser.add_argument("--evaluate",    action="store_true", help="Muestra métricas en test.")
    parser.add_argument("--model-path",  default=str(MODEL_PATH),  help="Ruta de salida del modelo.")
    parser.add_argument("--scaler-path", default=str(SCALER_PATH), help="Ruta de salida del scaler.")
    args = parser.parse_args()

    model, scaler = train(evaluate=args.evaluate)

    model_out  = Path(args.model_path)
    scaler_out = Path(args.scaler_path)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_out)
    joblib.dump(scaler, scaler_out)

    logger.info("Modelo guardado en '%s'.", model_out)
    logger.info("Scaler guardado en '%s'.", scaler_out)


if __name__ == "__main__":
    main()