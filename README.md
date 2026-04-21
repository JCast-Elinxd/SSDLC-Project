# Fraud Detection System for Payment Transactions
> SSDLC Final Project — Universidad del Rosario 2026-1  
> Juan Castrillón · Julieta Montoya · Kevin Cruz

---

## Problem Statement

Digital payment platforms process thousands of transactions per minute. These systems are often vulnerable to fraudulent activities not because of code defects, but because of **business logic weaknesses** — attackers exploit how the system is designed to behave, not bugs in the implementation.

Fraudulent patterns include: abnormally large transactions, high-frequency operations from a single user in a short window, operations from high-risk jurisdictions, and off-hours activity consistent with automated fraud. Without an automated detection layer, these go unnoticed until financial damage has occurred.

## Objectives

**General objective:** Develop a system that analyzes payment transactions and automatically detects potentially fraudulent operations using a risk scoring mechanism, while applying secure software development practices throughout the development lifecycle.

**Specific objectives:**
- Simulate payment transactions through a REST API
- Calculate a risk score for each transaction based on predefined fraud indicators
- Automatically block transactions that exceed a defined risk threshold
- Log suspicious events in an auditable format
- Apply security practices (SAST, SCA, input validation) within the CI/CD pipeline

## Methodology

The project follows an iterative development approach combining secure software development practices with controlled testing through simulated transactions.

**Layer 1 — Functional System:** A FastAPI REST API processes and registers simulated payment transactions. A core risk scoring engine evaluates each transaction against 6 fraud detection rules and automatically approves or blocks the operation. All transactions are persisted with their risk score, status, and triggered flags for auditability.

**Layer 2 — Security in the SSDLC:** Threat modeling using STRIDE is applied to identify critical assets and potential threats. Input validation guards against SQL injection and XSS. All blocked transactions are stored with audit flags. A risk matrix documents which vulnerabilities were mitigated, which controls were implemented, and which residual risks were accepted.

**Layer 3 — DevSecOps Automation:** A GitHub Actions CI/CD pipeline runs automatically on every push and pull request, executing: static application security testing (Bandit), dependency vulnerability scanning (pip-audit), and automated test execution (pytest). A quality gate blocks merges if any critical issue is detected.

---
`

## Fraud Detection Rules

| Rule | Points | Trigger |
|------|--------|---------|
| Very large amount | 50 | Amount > $10,000 |
| Large amount | 30 | Amount > $5,000 |
| High frequency | 25 | >5 transactions from same user in last 10 min |
| Rapid succession | 15 | >2 transactions from same user in last 60 sec |
| High-risk country | 20 | ISO code in watchlist (NG, RU, KP, IR, VE) |
| Unusual hour | 10 | 01:00–04:00 UTC |
| Round number | 5 | Exact multiples of $100 (structuring indicator) |

**Block threshold:** Score ≥ 70 → `blocked`

---

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# Open http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transactions/` | Submit transaction for fraud analysis |
| GET | `/transactions/` | List all transactions |
| GET | `/transactions/metrics` | Dashboard metrics |
| GET | `/transactions/{id}` | Get single transaction |
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
