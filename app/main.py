"""
Fraud Detection System — API principal.

Arranca con:
    pip install -r requirements.txt
    python -m uvicorn app.main:app --reload --reload-exclude "app/data/*"
    https://vigilant-eureka-qwwg95g7xr5cp44-8000.github.dev/docs
"""

from fastapi import FastAPI
from app.documents.models import create_tables
from app.documents.routers import router
from app.transactions.routers import router as transaction_router
from app.streaming.routers import router as streaming_router

app = FastAPI(title="Fraud Detection System API")

app.include_router(router)
app.include_router(transaction_router)
app.include_router(streaming_router)

@app.get("/")
def root():
    return {"message": "Fraud Detection API running"}

# Crear tablas al iniciar (en producción se usaría Alembic)
create_tables()

#=====================================================================================
# ROUTERS
#=====================================================================================

@app.get("/health", tags=["health"])
def health_check():
    """Endpoint de salud — confirma que la API está corriendo."""
    return {"status": "ok", "version": "0.1.0"}
