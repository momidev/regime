"""Layer di orchestrazione: collega dati, modello HMM e persistenza.

Usato sia dalle route FastAPI sia dagli script di training/refresh, così che la
logica di business viva in un unico posto.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from assets import get_asset, list_assets
from config import settings
from data.fetcher import fetch_price_history
from db.repository import Classification, get_repository
from exceptions import ModelNotTrainedError
from models.features import compute_features
from models.hmm_model import (
    RegimeModel,
    load_model,
    model_exists,
    train_model,
)
from models.regime_labeler import static_description

# Per il refresh giornaliero basta una finestra recente: deve solo essere
# abbastanza lunga da calcolare le feature rolling con margine.
REFRESH_LOOKBACK_DAYS = 365

# Finestra temporale per gli alert di cambio regime.
ALERT_WINDOW_HOURS = 24


# --------------------------------------------------------------------------- #
# Helper interni
# --------------------------------------------------------------------------- #
def _build_classifications(
    model: RegimeModel, feature_df: pd.DataFrame
) -> list[Classification]:
    """Costruisce i record di classificazione per ogni riga di feature."""
    states = model.predict_states(feature_df)
    probs = model.predict_proba(feature_df)
    rows: list[Classification] = []
    for i, (date, frow) in enumerate(feature_df.iterrows()):
        state = int(states[i])
        prob_map = {
            model.labels[j]: float(probs[i, j]) for j in range(model.n_states)
        }
        rows.append(
            {
                "date": date.date().isoformat(),
                "state_index": state,
                "state_label": model.labels[state],
                "probs": prob_map,
                "close": float(frow["close"]),
                "log_return": float(frow["log_return"]),
                "volatility": float(frow["volatility"]),
                "momentum": float(frow["momentum"]),
                "model_version": model.model_version,
            }
        )
    return rows


def _persist_transition_matrix(model: RegimeModel) -> None:
    repo = get_repository()
    repo.save_transition_matrix(
        model.asset_id,
        model.transition_matrix.tolist(),
        model.label_list,
        model.model_version,
    )


# --------------------------------------------------------------------------- #
# Training & refresh
# --------------------------------------------------------------------------- #
def train_asset(asset_id: str) -> dict[str, Any]:
    """Addestra (o ri-addestra) il modello HMM per un asset e ne salva l'output.

    Scarica lo storico completo, addestra l'HMM, salva l'artifact, fa il backfill
    di tutte le classificazioni storiche e lo snapshot della matrice di transizione.
    """
    asset = get_asset(asset_id)
    prices = fetch_price_history(asset, settings.history_lookback_days)
    feature_df = compute_features(
        prices, settings.rolling_window, settings.momentum_window
    )
    model = train_model(
        asset_id=asset.id,
        feature_df=feature_df,
        n_states=settings.n_states,
        n_iter=settings.hmm_n_iter,
        random_seed=settings.random_seed,
        min_observations=settings.min_observations,
    )
    model.save()

    rows = _build_classifications(model, feature_df)
    repo = get_repository()
    repo.upsert_classifications(asset.id, rows)
    _persist_transition_matrix(model)

    latest = rows[-1]
    return {
        "asset": asset.id,
        "status": "trained",
        "n_states": model.n_states,
        "observations": len(rows),
        "model_version": model.model_version,
        "current_label": latest["state_label"],
        "as_of": latest["date"],
    }


def refresh_asset(asset_id: str) -> dict[str, Any]:
    """Aggiorna la classificazione di un asset usando il modello esistente.

    Se il modello non esiste ancora, esegue un training iniziale (bootstrap).
    Rileva eventuali cambi di regime rispetto all'ultima classificazione salvata.
    """
    asset = get_asset(asset_id)
    if not model_exists(asset.id):
        return train_asset(asset.id)

    model = load_model(asset.id)
    prices = fetch_price_history(asset, REFRESH_LOOKBACK_DAYS)
    feature_df = compute_features(
        prices, settings.rolling_window, settings.momentum_window
    )

    repo = get_repository()
    previous_latest = repo.get_latest(asset.id)

    rows = _build_classifications(model, feature_df)
    repo.upsert_classifications(asset.id, rows)
    _persist_transition_matrix(model)

    new_latest = rows[-1]
    regime_changed = False
    if (
        previous_latest is not None
        and previous_latest["date"] != new_latest["date"]
        and previous_latest["state_label"] != new_latest["state_label"]
    ):
        repo.record_change(
            asset.id, previous_latest["state_label"], new_latest["state_label"]
        )
        regime_changed = True

    return {
        "asset": asset.id,
        "status": "refreshed",
        "model_version": model.model_version,
        "current_label": new_latest["state_label"],
        "as_of": new_latest["date"],
        "regime_changed": regime_changed,
    }


def refresh_all() -> list[dict[str, Any]]:
    """Aggiorna tutti gli asset supportati, isolando gli errori per-asset."""
    results: list[dict[str, Any]] = []
    for asset in list_assets():
        try:
            results.append(refresh_asset(asset.id))
        except Exception as exc:  # noqa: BLE001 - un asset non deve bloccare gli altri
            results.append(
                {"asset": asset.id, "status": "error", "error": str(exc)}
            )
    return results


# --------------------------------------------------------------------------- #
# Query di lettura
# --------------------------------------------------------------------------- #
def get_current(asset_id: str) -> dict[str, Any]:
    """Ritorna lo stato corrente e le probabilità per un asset."""
    asset = get_asset(asset_id)
    repo = get_repository()
    latest = repo.get_latest(asset.id)
    if latest is None:
        raise ModelNotTrainedError(
            f"Nessuna classificazione disponibile per '{asset.id}'. "
            f"Esegui prima un training/refresh."
        )

    # Se il modello è disponibile usa la sua descrizione; altrimenti (es. deploy
    # con filesystem effimero senza il file .pkl) deriva la descrizione
    # direttamente dall'etichetta, senza richiedere il modello.
    try:
        description = load_model(asset.id).describe(latest["state_index"])
    except ModelNotTrainedError:
        description = static_description(latest["state_label"])

    return {
        "asset": asset.id,
        "as_of": latest["date"],
        "state_index": latest["state_index"],
        "state_label": latest["state_label"],
        "description": description,
        "probabilities": latest["probs"],
        "model_version": latest["model_version"],
    }


def get_history(asset_id: str, days: int) -> dict[str, Any]:
    """Ritorna lo storico delle classificazioni per il grafico timeline."""
    asset = get_asset(asset_id)
    repo = get_repository()
    rows = repo.get_history(asset.id, days)
    points = [
        {
            "date": r["date"],
            "state_index": r["state_index"],
            "state_label": r["state_label"],
            "probabilities": r["probs"],
        }
        for r in rows
    ]
    return {"asset": asset.id, "days": days, "count": len(points), "history": points}


def get_transition_matrix(asset_id: str) -> dict[str, Any]:
    """Ritorna la matrice di transizione tra regimi per un asset."""
    asset = get_asset(asset_id)
    repo = get_repository()
    snapshot = repo.get_transition_matrix(asset.id)
    if snapshot is None:
        raise ModelNotTrainedError(
            f"Nessuna matrice di transizione per '{asset.id}'. "
            f"Esegui prima un training/refresh."
        )
    return {
        "asset": asset.id,
        "state_labels": snapshot["state_labels"],
        "matrix": snapshot["matrix"],
        "model_version": snapshot["model_version"],
        "computed_at": snapshot["computed_at"],
    }


def get_alert_status(asset_id: str) -> dict[str, Any]:
    """Ritorna lo stato di alert (cambio regime nelle ultime 24h) per un asset."""
    asset = get_asset(asset_id)
    repo = get_repository()
    change = repo.get_recent_change(asset.id, ALERT_WINDOW_HOURS)
    return {
        "asset": asset.id,
        "regime_changed": change is not None,
        "window_hours": ALERT_WINDOW_HOURS,
        "from_label": change["from_label"] if change else None,
        "to_label": change["to_label"] if change else None,
        "changed_at": change["changed_at"] if change else None,
    }


# Finestra "molto ampia" per leggere tutto lo storico disponibile.
_ALL_HISTORY_DAYS = 100_000


def get_overview() -> dict[str, Any]:
    """Snapshot dello stato corrente di tutti gli asset in una sola risposta.

    Pensato per la home del frontend: evita una chiamata per asset.
    """
    repo = get_repository()
    items: list[dict[str, Any]] = []
    for asset in list_assets():
        latest = repo.get_latest(asset.id)
        if latest is None:
            items.append(
                {
                    "asset": asset.id,
                    "name": asset.name,
                    "asset_class": asset.asset_class,
                    "has_data": False,
                    "as_of": None,
                    "state_label": None,
                    "top_probability": None,
                }
            )
            continue
        probs: dict[str, float] = latest["probs"]
        top_label = max(probs, key=probs.get) if probs else latest["state_label"]
        items.append(
            {
                "asset": asset.id,
                "name": asset.name,
                "asset_class": asset.asset_class,
                "has_data": True,
                "as_of": latest["date"],
                "state_label": latest["state_label"],
                "top_probability": probs.get(top_label) if probs else None,
            }
        )
    return {"count": len(items), "assets": items}


def get_prices(asset_id: str, days: int) -> dict[str, Any]:
    """Serie storica del prezzo di chiusura, allineata alla timeline dei regimi.

    Utile al frontend per disegnare il prezzo con dietro le bande dei regimi.
    """
    asset = get_asset(asset_id)
    repo = get_repository()
    rows = repo.get_history(asset.id, days)
    prices = [
        {"date": r["date"], "close": r["close"]}
        for r in rows
        if r.get("close") is not None
    ]
    return {"asset": asset.id, "days": days, "count": len(prices), "prices": prices}


def _runs(labels: list[str]) -> list[tuple[str, int]]:
    """Comprime una sequenza di etichette in run consecutivi ``(label, lunghezza)``."""
    runs: list[tuple[str, int]] = []
    for label in labels:
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1] + 1)
        else:
            runs.append((label, 1))
    return runs


def get_stats(asset_id: str) -> dict[str, Any]:
    """Statistiche descrittive sui regimi di un asset.

    Include: regime corrente e da quanti giorni dura (streak), frequenza e durata
    media di ciascun regime sullo storico, e durata attesa del regime corrente
    derivata dalla matrice di transizione (1 / (1 - p_ii)).
    """
    asset = get_asset(asset_id)
    repo = get_repository()
    history = repo.get_history(asset.id, _ALL_HISTORY_DAYS)
    if not history:
        raise ModelNotTrainedError(
            f"Nessuna classificazione disponibile per '{asset.id}'. "
            f"Esegui prima un training/refresh."
        )

    labels = [r["state_label"] for r in history]
    total = len(labels)
    runs = _runs(labels)

    # Etichette di riferimento (ordine per indice) dalla matrice di transizione.
    snapshot = repo.get_transition_matrix(asset.id)
    ref_labels: list[str] = (
        snapshot["state_labels"] if snapshot else sorted(set(labels))
    )

    per_label: list[dict[str, Any]] = []
    for label in ref_labels:
        count = sum(1 for x in labels if x == label)
        durations = [length for lbl, length in runs if lbl == label]
        avg_dur = sum(durations) / len(durations) if durations else 0.0
        per_label.append(
            {
                "label": label,
                "frequency": count / total if total else 0.0,
                "days": count,
                "avg_duration_days": round(avg_dur, 2),
                "occurrences": len(durations),
            }
        )

    latest = history[-1]
    current_label = latest["state_label"]
    current_streak = runs[-1][1] if runs else 0

    # Durata attesa del regime corrente dalla matrice di transizione.
    expected_duration: float | None = None
    if snapshot:
        idx = latest["state_index"]
        matrix = snapshot["matrix"]
        if 0 <= idx < len(matrix):
            p_ii = matrix[idx][idx]
            if p_ii < 1.0:
                expected_duration = round(1.0 / (1.0 - p_ii), 2)

    return {
        "asset": asset.id,
        "as_of": latest["date"],
        "current_regime": current_label,
        "current_streak_days": current_streak,
        "expected_duration_days": expected_duration,
        "sample_days": total,
        "regimes": per_label,
    }
