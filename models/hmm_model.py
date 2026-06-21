"""Wrapper attorno a un Gaussian HMM per la classificazione dei regimi.

Incapsula:

* standardizzazione delle feature (``StandardScaler``)
* training EM dell'HMM (``hmmlearn.hmm.GaussianHMM``)
* etichettatura degli stati (vedi :mod:`models.regime_labeler`)
* serializzazione/deserializzazione su disco (``.pkl``)
* inferenza: sequenza di stati, probabilità a posteriori, matrice di transizione
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from config import ARTIFACTS_DIR
from exceptions import InsufficientDataError, ModelNotTrainedError
from models import regime_labeler as labeler
from models.features import FEATURE_COLUMNS, feature_matrix
from models.regime_labeler import StateStats


@dataclass
class RegimeModel:
    """Modello di regime addestrato per un singolo asset."""

    asset_id: str
    n_states: int
    hmm: GaussianHMM
    scaler: StandardScaler
    labels: dict[int, str]
    state_stats: list[StateStats]
    model_version: str
    trained_at: str
    feature_columns: tuple[str, ...] = field(default=FEATURE_COLUMNS)

    # ------------------------------------------------------------------ #
    # Inferenza
    # ------------------------------------------------------------------ #
    def predict_states(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Ritorna la sequenza di stati più probabile (algoritmo di Viterbi)."""
        X = self.scaler.transform(feature_matrix(feature_df))
        return self.hmm.predict(X)

    def predict_proba(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Ritorna le probabilità a posteriori per stato, riga per riga."""
        X = self.scaler.transform(feature_matrix(feature_df))
        return self.hmm.predict_proba(X)

    @property
    def transition_matrix(self) -> np.ndarray:
        """Matrice di transizione stimata ``(n_states, n_states)``."""
        return np.asarray(self.hmm.transmat_, dtype=float)

    @property
    def label_list(self) -> list[str]:
        """Etichette ordinate per indice di stato."""
        return [self.labels[i] for i in range(self.n_states)]

    def describe(self, state: int) -> str:
        """Descrizione testuale (descrittiva) di uno stato."""
        stats = next((s for s in self.state_stats if s.state == state), None)
        if stats is None:
            stats = StateStats(state, 0.0, 0.0, 0.0)
        return labeler.describe_label(self.labels[state], stats)

    # ------------------------------------------------------------------ #
    # Persistenza
    # ------------------------------------------------------------------ #
    def save(self) -> Path:
        """Serializza il modello su disco e ritorna il path del file."""
        path = artifact_path(self.asset_id)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path


def artifact_path(asset_id: str) -> Path:
    """Path del file ``.pkl`` per l'asset indicato."""
    return ARTIFACTS_DIR / f"{asset_id}.pkl"


def model_exists(asset_id: str) -> bool:
    """``True`` se esiste un artifact addestrato per l'asset."""
    return artifact_path(asset_id).exists()


def load_model(asset_id: str) -> RegimeModel:
    """Carica il modello addestrato per un asset.

    Raises:
        ModelNotTrainedError: se l'artifact non esiste.
    """
    path = artifact_path(asset_id)
    if not path.exists():
        raise ModelNotTrainedError(
            f"Nessun modello addestrato per '{asset_id}'. "
            f"Esegui prima il training (scripts/train_all.py o POST /regime/refresh)."
        )
    with path.open("rb") as fh:
        return pickle.load(fh)


def train_model(
    asset_id: str,
    feature_df: pd.DataFrame,
    n_states: int,
    n_iter: int,
    random_seed: int,
    min_observations: int,
) -> RegimeModel:
    """Addestra un Gaussian HMM sulle feature di un asset.

    Args:
        asset_id: identificatore dell'asset.
        feature_df: feature da :func:`models.features.compute_features`.
        n_states: numero di stati nascosti (regimi).
        n_iter: iterazioni massime dell'algoritmo EM.
        random_seed: seed per riproducibilità.
        min_observations: numero minimo di osservazioni richieste.

    Returns:
        Istanza :class:`RegimeModel` addestrata (non ancora salvata su disco).

    Raises:
        InsufficientDataError: se le osservazioni sono meno di ``min_observations``.
    """
    if len(feature_df) < min_observations:
        raise InsufficientDataError(
            f"Dati insufficienti per addestrare l'HMM su '{asset_id}': "
            f"{len(feature_df)} osservazioni disponibili, "
            f"{min_observations} richieste."
        )

    X_raw = feature_matrix(feature_df)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_seed,
        tol=1e-4,
    )
    hmm.fit(X)

    states = hmm.predict(X)
    log_returns = feature_df["log_return"].to_numpy(dtype=float)
    volatility = feature_df["volatility"].to_numpy(dtype=float)
    state_stats = labeler.compute_state_stats(
        states, log_returns, volatility, n_states
    )
    labels = labeler.label_states(state_stats)

    now = datetime.now(timezone.utc)
    return RegimeModel(
        asset_id=asset_id,
        n_states=n_states,
        hmm=hmm,
        scaler=scaler,
        labels=labels,
        state_stats=state_stats,
        model_version=now.strftime("%Y%m%d%H%M%S"),
        trained_at=now.isoformat(),
    )
