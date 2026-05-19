# Fraud Detection System for Payment Transactions
> SSDLC Final Project — Universidad del Rosario 2026-1  
> Juan Castrillón · Julieta Montoya · Kevin Cruz

---

## Problem Statement

Digital payment platforms process thousands of transactions per minute. These systems are often vulnerable to fraudulent activities not because of code defects, but because of **business logic weaknesses** — attackers exploit how the system is designed to behave, not bugs in the implementation.

Fraudulent patterns include: abnormally large transactions, high-frequency operations from a single user in a short window, structuring (amounts just below detection thresholds like $9,999), and external transfers/withdrawals to unknown accounts. Without an automated detection layer, these go unnoticed until financial damage has occurred.

## Objectives

**General objective:** Develop a system that analyzes payment transactions and automatically detects potentially fraudulent operations using a risk scoring mechanism, while applying secure software development practices throughout the development lifecycle.

**Specific objectives:**
- Simulate payment transactions through a REST API
- Calculate a risk score for each transaction based on predefined fraud indicators
- Automatically block transactions that exceed a defined risk threshold
- Flag suspicious transactions for manual review
- Log all events in an auditable format
- Apply security practices (SAST, SCA, input validation) within the CI/CD pipeline

## Methodology

The project follows an iterative development approach combining secure software development practices with controlled testing through simulated transactions.

**Layer 1 — Functional System:** A FastAPI REST API processes and registers simulated payment transactions. A core risk scoring engine evaluates each transaction against 4 fraud detection rules and automatically approves, flags, or blocks the operation. All transactions are persisted with their risk score, risk level, status, and triggered reasons for auditability.

**Layer 2 — Security in the SSDLC:** Threat modeling using STRIDE is applied to identify critical assets and potential threats. Input validation is enforced on all fields. All transactions are stored with full audit flags. A risk matrix documents which vulnerabilities were mitigated, which controls were implemented, and which residual risks were accepted.

**Layer 3 — DevSecOps Automation:** A GitHub Actions CI/CD pipeline runs automatically on every push and pull request, executing: static application security testing (Bandit), dependency vulnerability scanning (pip-audit), and automated test execution (pytest). A quality gate blocks merges if any critical issue is detected.

---
## How the System Works
 
### General Architecture
 
```
HTTP Client
     │
     ▼
FastAPI (main.py)
     │
     ├── /documents/*
     │       routers.py → services.py → SQLite (table: documents)
     │
     └── /transactions/*
             transaction_routers.py
                     │
                     ▼
             transaction_services.py
                     │
                     ├── Rules Engine ──────────────── rules_score  (0–100)
                     │
                     ├── ml_service.py
                     │       └── PyTorch Neural Network ── ml_score (0–100)
                     │
                     └── Hybrid Score = 50% rules + 50% ML
                                 │
                                 └── Decision: APPROVED / FLAGGED / BLOCKED
                                                     │
                                             SQLite (table: transactions)
```

### Rules Engine
 
**File:** `transaction_services.py` → `calculate_risk_score()`
 
The rules engine accumulates risk points based on explicit business logic conditions. Every triggered rule generates a text reason stored in `risk_reasons` for auditability. The maximum possible score is 100.
 
**Structuring detection:** a common fraud technique where large transactions are broken into amounts just below round thresholds to evade automated controls. The system flags values like $999, $4,999, $9,999, $49,999, or any amount whose modulo 1,000 falls between $990 and $999.

### Neural Network (PyTorch)
 
**Files:** `ml_service.py` (inference), `train_model.py` (training)
 
A fully connected neural network with two hidden layers that classifies each transaction into one of three risk classes: `approved`, `flagged`, or `blocked`.
 
**Architecture:**
 
```
Input (5 features)
        │
  Linear(5 → 32) → ReLU
        │
  Linear(32 → 16) → ReLU
        │
  Linear(16 → 3) → Softmax
        │
[P(approved), P(flagged), P(blocked)]
```
 
**Input features:**
 
| Feature | Description |
|---|---|
| `amount` | Transaction amount |
| `transaction_type` | Encoded: transfer=0, payment=1, withdrawal=2, deposit=3 |
| `dest_external` | 1 if destination is a different account, 0 otherwise |
| `hour_of_day` | UTC hour of the transaction (0–23) |
| `recent_tx_count` | Number of transactions from the same user in the last hour |
 
Features are normalized with `StandardScaler` before entering the network (fitted during training, saved to `app/scaler.pkl`).
 
**Training:** Since no real historical data is available, `train_model.py` generates 5,000 synthetic transactions using the same rules engine logic, plus 5% random noise to simulate real edge cases. Split: 60% train / 20% validation / 20% test (stratified). Loss function: `CrossEntropyLoss` with inverse class weights to handle the natural imbalance between classes. Optimizer: Adam (lr=1e-3), 50 epochs, batch size 32. Typical test accuracy: ~90%.

## Fraud Detection Rules

| Rule | Points | Trigger |
|------|--------|---------|
| Very large amount | +40 | Amount > $10,000 |
| Large amount | +20 | Amount > $5,000 |
| Structuring | +15 | Suspicious amounts ($999, $4,999, $9,999…) |
| High frequency | +30 | ≥ 5 transactions from same user in last hour |
| Elevated frequency | +15 | ≥ 3 transactions from same user in last hour |
| External transfer/withdrawal | +10 | Transfer or withdrawal to a different account |

**ML risk score:**
```
ml_risk_score = P(flagged) × 50 + P(blocked) × 100
```
 
### Hybrid Score and Final Decision
 
The rules score and the ML score are combined with equal weight:
 
```
final_score = rules_score × 0.50 + ml_risk_score × 0.50
```
 
| Final Score | Risk Level | Status | Meaning |
|---|---|---|---|
| 0 – 25 | LOW | ✅ APPROVED | Transaction approved, no concerns |
| 26 – 50 | MEDIUM | ⚠️ FLAGGED | Approved, flagged for manual review |
| 51 – 75 | HIGH | ⚠️ FLAGGED | Approved with high-risk alert |
| 76 – 100 | CRITICAL | 🚫 BLOCKED | Transaction rejected |
 
`FLAGGED` does not block the transaction — it is processed but stored with a flag for an analyst to review. `BLOCKED` rejects it completely (`blocked = true`).
 
### Complete Transaction Flow
 
```
POST /transactions/
        │
        ▼
1. Basic validation (amount > 0, required fields)
        │
        ▼
2. Count recent transactions from this user (last hour)
        │
        ▼
3. Rules engine → rules_score (0–100) + list of triggered reasons
        │
        ▼
4. Neural network → ml_risk_score (0–100)
   └── Scale features → forward pass → softmax → weighted score
        │
        ▼
5. final_score = rules_score × 0.50 + ml_risk_score × 0.50
        │
        ▼
6. Decision: APPROVED / FLAGGED / BLOCKED
        │
        ▼
7. Persist transaction in database (all fields + risk_reasons as JSON)
        │
        ▼
8. Return response with status and alert (if flagged or blocked)
```
 
### Training the Model
 
Before starting the API for the first time, run:
 
```bash
python train_model.py --evaluate
```
 
This generates `app/model.pt` (model weights) and `app/scaler.pkl` (feature scaler). The `--evaluate` flag also prints the classification report on the test set.
 
> If this step is skipped, the API still works using the rules engine only. The system detects automatically whether the model file is available.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the neural network (generates app/model.pt and app/scaler.pkl)
python train_model.py --evaluate

# 3. Start the API
uvicorn app.main:app --reload

# 4. Open interactive docs at http://localhost:8000/docs
```

> If step 2 is skipped, the API still works using the rules engine only. The system detects automatically whether the model file is available.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transactions/` | Submit transaction for fraud analysis |
| GET | `/transactions/` | List all transactions |
| GET | `/transactions/{id}` | Get single transaction |
| GET | `/transactions/user/{user_id}` | Transaction history for a user |
| GET | `/health` | Health check |


---

## DevSecOps Pipeline

The CI/CD pipeline runs automatically on every push to `main` or pull request:

| Stage | Tool | Action on failure |
|-------|------|-------------------|
| SAST | Bandit | Blocks if HIGH severity issues found |
| Dependency Scan | pip-audit | Blocks if vulnerable packages detected |
| Automated Tests | pytest | Blocks if any test fails |
| Quality Gate | — | Merge blocked unless all stages pass |

## Running Tests

```bash
pytest tests/ -v
```

---

## Security Controls Implemented

