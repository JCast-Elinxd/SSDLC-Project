"""
Fraud Detection System — API principal.

Arranca con:
    pip install -r requirements.txt
    python -m uvicorn app.main:app --reload                   --reload-exclude "app/chroma_data/*"
    https://vigilant-eureka-qwwg95g7xr5cp44-8000.github.dev/docs
"""

from fastapi import FastAPI
from app.models import create_tables
from app.routers import router 

app = FastAPI(
    title="Fraud Detection System API")

app.include_router(router)

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
