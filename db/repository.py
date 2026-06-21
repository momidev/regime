"""Persistenza delle classificazioni di regime, transizioni e cambi di regime.

Sono disponibili due implementazioni con la stessa interfaccia
(:class:`RegimeRepository`):

* :class:`SqlRepository`  — usa PostgreSQL/Supabase via SQLAlchemy.
* :class:`FileRepository` — fallback su file JSON locali (nessun DB richiesto).

La factory :func:`get_repository` sceglie automaticamente in base alla
configurazione (``DATABASE_URL``).
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import text

from config import STORE_DIR, settings

# ---------------------------------------------------------------------------- #
# Tipi
# ---------------------------------------------------------------------------- #
# Classification record:
#   {date, state_index, state_label, probs, log_return, volatility, momentum,
#    model_version}
Classification = dict[str, Any]


class RegimeRepository(ABC):
    """Interfaccia comune per la persistenza dei dati di regime."""

    @abstractmethod
    def upsert_classifications(self, asset: str, rows: list[Classification]) -> None:
        """Inserisce/aggiorna le classificazioni giornaliere di un asset."""

    @abstractmethod
    def get_history(self, asset: str, days: int) -> list[Classification]:
        """Ritorna le classificazioni degli ultimi ``days`` giorni (ordine crescente)."""

    @abstractmethod
    def get_latest(self, asset: str) -> Classification | None:
        """Ritorna l'ultima classificazione salvata per l'asset, se esiste."""

    @abstractmethod
    def save_transition_matrix(
        self,
        asset: str,
        matrix: list[list[float]],
        state_labels: list[str],
        model_version: str,
    ) -> None:
        """Salva lo snapshot della matrice di transizione."""

    @abstractmethod
    def get_transition_matrix(self, asset: str) -> dict[str, Any] | None:
        """Ritorna lo snapshot della matrice di transizione, se esiste."""

    @abstractmethod
    def record_change(self, asset: str, from_label: str | None, to_label: str) -> None:
        """Registra un cambio di regime."""

    @abstractmethod
    def get_recent_change(self, asset: str, within_hours: int) -> dict[str, Any] | None:
        """Ritorna l'ultimo cambio di regime entro ``within_hours`` ore, se esiste."""


# ---------------------------------------------------------------------------- #
# Implementazione SQL (PostgreSQL / Supabase)
# ---------------------------------------------------------------------------- #
class SqlRepository(RegimeRepository):
    """Repository basato su PostgreSQL via SQLAlchemy."""

    def __init__(self) -> None:
        from db.connection import get_engine

        self._engine = get_engine()

    def upsert_classifications(self, asset: str, rows: list[Classification]) -> None:
        if not rows:
            return
        stmt = text(
            """
            INSERT INTO regime_classifications
                (asset, date, state_index, state_label, probs,
                 log_return, volatility, momentum, model_version)
            VALUES
                (:asset, :date, :state_index, :state_label, CAST(:probs AS JSONB),
                 :log_return, :volatility, :momentum, :model_version)
            ON CONFLICT (asset, date) DO UPDATE SET
                state_index   = EXCLUDED.state_index,
                state_label   = EXCLUDED.state_label,
                probs         = EXCLUDED.probs,
                log_return    = EXCLUDED.log_return,
                volatility    = EXCLUDED.volatility,
                momentum      = EXCLUDED.momentum,
                model_version = EXCLUDED.model_version,
                created_at    = now()
            """
        )
        params = [
            {
                "asset": asset,
                "date": r["date"],
                "state_index": r["state_index"],
                "state_label": r["state_label"],
                "probs": json.dumps(r["probs"]),
                "log_return": r.get("log_return"),
                "volatility": r.get("volatility"),
                "momentum": r.get("momentum"),
                "model_version": r["model_version"],
            }
            for r in rows
        ]
        with self._engine.begin() as conn:
            conn.execute(stmt, params)

    def get_history(self, asset: str, days: int) -> list[Classification]:
        stmt = text(
            """
            SELECT date, state_index, state_label, probs,
                   log_return, volatility, momentum, model_version
            FROM regime_classifications
            WHERE asset = :asset AND date >= :cutoff
            ORDER BY date ASC
            """
        )
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
        with self._engine.connect() as conn:
            result = conn.execute(stmt, {"asset": asset, "cutoff": cutoff})
            return [self._row_to_classification(row._mapping) for row in result]

    def get_latest(self, asset: str) -> Classification | None:
        stmt = text(
            """
            SELECT date, state_index, state_label, probs,
                   log_return, volatility, momentum, model_version
            FROM regime_classifications
            WHERE asset = :asset
            ORDER BY date DESC
            LIMIT 1
            """
        )
        with self._engine.connect() as conn:
            row = conn.execute(stmt, {"asset": asset}).first()
            return self._row_to_classification(row._mapping) if row else None

    def save_transition_matrix(
        self,
        asset: str,
        matrix: list[list[float]],
        state_labels: list[str],
        model_version: str,
    ) -> None:
        stmt = text(
            """
            INSERT INTO transition_matrices
                (asset, matrix, state_labels, model_version, computed_at)
            VALUES
                (:asset, CAST(:matrix AS JSONB), CAST(:labels AS JSONB),
                 :model_version, now())
            ON CONFLICT (asset) DO UPDATE SET
                matrix        = EXCLUDED.matrix,
                state_labels  = EXCLUDED.state_labels,
                model_version = EXCLUDED.model_version,
                computed_at   = now()
            """
        )
        with self._engine.begin() as conn:
            conn.execute(
                stmt,
                {
                    "asset": asset,
                    "matrix": json.dumps(matrix),
                    "labels": json.dumps(state_labels),
                    "model_version": model_version,
                },
            )

    def get_transition_matrix(self, asset: str) -> dict[str, Any] | None:
        stmt = text(
            """
            SELECT matrix, state_labels, model_version, computed_at
            FROM transition_matrices
            WHERE asset = :asset
            """
        )
        with self._engine.connect() as conn:
            row = conn.execute(stmt, {"asset": asset}).first()
            if row is None:
                return None
            m = row._mapping
            return {
                "matrix": m["matrix"],
                "state_labels": m["state_labels"],
                "model_version": m["model_version"],
                "computed_at": m["computed_at"].isoformat(),
            }

    def record_change(self, asset: str, from_label: str | None, to_label: str) -> None:
        stmt = text(
            """
            INSERT INTO regime_changes (asset, from_label, to_label, changed_at)
            VALUES (:asset, :from_label, :to_label, now())
            """
        )
        with self._engine.begin() as conn:
            conn.execute(
                stmt,
                {"asset": asset, "from_label": from_label, "to_label": to_label},
            )

    def get_recent_change(self, asset: str, within_hours: int) -> dict[str, Any] | None:
        stmt = text(
            """
            SELECT from_label, to_label, changed_at
            FROM regime_changes
            WHERE asset = :asset
              AND changed_at >= now() - make_interval(hours => :hours)
            ORDER BY changed_at DESC
            LIMIT 1
            """
        )
        with self._engine.connect() as conn:
            row = conn.execute(
                stmt, {"asset": asset, "hours": within_hours}
            ).first()
            if row is None:
                return None
            m = row._mapping
            return {
                "from_label": m["from_label"],
                "to_label": m["to_label"],
                "changed_at": m["changed_at"].isoformat(),
            }

    @staticmethod
    def _row_to_classification(mapping: Any) -> Classification:
        return {
            "date": mapping["date"].isoformat(),
            "state_index": mapping["state_index"],
            "state_label": mapping["state_label"],
            "probs": mapping["probs"],
            "log_return": mapping["log_return"],
            "volatility": mapping["volatility"],
            "momentum": mapping["momentum"],
            "model_version": mapping["model_version"],
        }


# ---------------------------------------------------------------------------- #
# Implementazione su file JSON (fallback senza database)
# ---------------------------------------------------------------------------- #
class FileRepository(RegimeRepository):
    """Repository di fallback che persiste i dati su file JSON locali."""

    def __init__(self, store_dir: Path = STORE_DIR) -> None:
        self._dir = store_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # -- helper di I/O ------------------------------------------------------- #
    def _path(self, asset: str, kind: str) -> Path:
        safe = asset.replace("/", "_")
        return self._dir / f"{safe}__{kind}.json"

    def _read(self, asset: str, kind: str, default: Any) -> Any:
        path = self._path(asset, kind)
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, asset: str, kind: str, data: Any) -> None:
        path = self._path(asset, kind)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    # -- API ----------------------------------------------------------------- #
    def upsert_classifications(self, asset: str, rows: list[Classification]) -> None:
        if not rows:
            return
        with self._lock:
            existing: dict[str, Classification] = {
                r["date"]: r for r in self._read(asset, "classifications", [])
            }
            for r in rows:
                existing[r["date"]] = r
            merged = [existing[d] for d in sorted(existing)]
            self._write(asset, "classifications", merged)

    def get_history(self, asset: str, days: int) -> list[Classification]:
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
        rows = self._read(asset, "classifications", [])
        return [r for r in rows if r["date"] >= cutoff]

    def get_latest(self, asset: str) -> Classification | None:
        rows = self._read(asset, "classifications", [])
        return rows[-1] if rows else None

    def save_transition_matrix(
        self,
        asset: str,
        matrix: list[list[float]],
        state_labels: list[str],
        model_version: str,
    ) -> None:
        with self._lock:
            self._write(
                asset,
                "transition",
                {
                    "matrix": matrix,
                    "state_labels": state_labels,
                    "model_version": model_version,
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                },
            )

    def get_transition_matrix(self, asset: str) -> dict[str, Any] | None:
        return self._read(asset, "transition", None)

    def record_change(self, asset: str, from_label: str | None, to_label: str) -> None:
        with self._lock:
            changes = self._read(asset, "changes", [])
            changes.append(
                {
                    "from_label": from_label,
                    "to_label": to_label,
                    "changed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._write(asset, "changes", changes)

    def get_recent_change(self, asset: str, within_hours: int) -> dict[str, Any] | None:
        changes = self._read(asset, "changes", [])
        if not changes:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(hours=within_hours)
        for change in reversed(changes):
            changed_at = datetime.fromisoformat(change["changed_at"])
            if changed_at >= cutoff:
                return change
        return None


@lru_cache
def get_repository() -> RegimeRepository:
    """Ritorna il repository appropriato in base alla configurazione."""
    if settings.use_database:
        return SqlRepository()
    return FileRepository()
