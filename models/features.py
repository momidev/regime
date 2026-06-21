"""Feature engineering per la classificazione dei regimi di mercato.

Le feature calcolate per ogni giorno sono:

* ``log_return``  — ritorno logaritmico giornaliero
* ``volatility``  — deviazione standard rolling dei ritorni log (annualizzata)
* ``momentum``    — Rate of Change (ROC) percentuale su una finestra
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Ordine canonico delle colonne feature usate per addestrare/applicare l'HMM.
FEATURE_COLUMNS: tuple[str, ...] = ("log_return", "volatility", "momentum")

# Giorni di mercato in un anno, per annualizzare la volatilità.
TRADING_DAYS_PER_YEAR = 252


def compute_features(
    prices: pd.DataFrame,
    rolling_window: int,
    momentum_window: int,
) -> pd.DataFrame:
    """Calcola le feature di regime a partire dai prezzi di chiusura.

    Args:
        prices: DataFrame con colonna ``close`` indicizzato per data.
        rolling_window: finestra (giorni) per la volatilità rolling.
        momentum_window: finestra (giorni) per il momentum / ROC.

    Returns:
        DataFrame indicizzato per data con le colonne in :data:`FEATURE_COLUMNS`
        più ``close``. Le righe iniziali con valori non calcolabili (NaN) sono
        rimosse.
    """
    if "close" not in prices.columns:
        raise ValueError("Il DataFrame dei prezzi deve contenere la colonna 'close'.")

    df = prices.copy()
    close = df["close"]

    # Ritorno logaritmico giornaliero.
    df["log_return"] = np.log(close / close.shift(1))

    # Volatilità rolling annualizzata (std dei ritorni log).
    df["volatility"] = (
        df["log_return"].rolling(window=rolling_window).std()
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    # Momentum: Rate of Change percentuale sulla finestra indicata.
    df["momentum"] = (close / close.shift(momentum_window) - 1.0) * 100.0

    feature_df = df[["close", *FEATURE_COLUMNS]].dropna()
    return feature_df


def feature_matrix(feature_df: pd.DataFrame) -> np.ndarray:
    """Estrae la matrice numerica delle feature nell'ordine canonico.

    Args:
        feature_df: output di :func:`compute_features`.

    Returns:
        Array ``(n_samples, n_features)`` di tipo float.
    """
    return feature_df.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
