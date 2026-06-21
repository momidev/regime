"""Entry point dell'applicazione FastAPI.

Avvio locale:
    uvicorn api.main:app --reload

Documentazione interattiva (per Lovable):
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
    http://localhost:8000/openapi.json
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config import settings

DESCRIPTION = """
**Regime** — market regime detection tramite Hidden Markov Model.

Classifica in quale regime statistico si trova un asset (es. *bull-calmo*,
*bull-volatile*, *bear-calmo*, *bear-volatile*) e fornisce le probabilità di
transizione tra regimi.

L'output è **descrittivo e statistico**: non costituisce consiglio di
investimento né segnale operativo. Nessun ordine viene eseguito; il sistema
legge solo dati di mercato pubblici e ne fa analisi statistica.
"""

app = FastAPI(
    title="Regime API",
    description=DESCRIPTION,
    version="1.0.0",
    contact={"name": "Regime"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    """Endpoint radice con i link utili."""
    return {
        "name": "Regime API",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
    }
