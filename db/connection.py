"""Gestione della connessione al database (SQLAlchemy), opzionale.

Se ``DATABASE_URL`` non è configurato, l'engine non viene creato e il sistema
ricade sullo store su file JSON (vedi :mod:`db.repository`).
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from config import settings


@lru_cache
def get_engine() -> Engine:
    """Ritorna l'engine SQLAlchemy singleton.

    Raises:
        RuntimeError: se ``DATABASE_URL`` non è configurato.
    """
    if not settings.use_database:
        raise RuntimeError("DATABASE_URL non configurato: nessun engine disponibile.")
    return create_engine(
        settings.database_url,  # type: ignore[arg-type]
        pool_pre_ping=True,
        future=True,
    )
