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
