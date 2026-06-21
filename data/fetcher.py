"""Recupero dei dati di mercato storici tramite yfinance (sola lettura)."""

from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from assets import Asset
from exceptions import DataFetchError

# Retry con backoff: yfinance/Yahoo possono restituire risposte vuote o errori
# transitori (es. HTTP 429 rate limiting), specialmente da IP condivisi/cloud.
_MAX_ATTEMPTS = 4
_BACKOFF_SECONDS = 2.0


def _download(asset: Asset, period_days: int) -> pd.DataFrame | None:
    """Singolo tentativo di download da yfinance (può sollevare eccezioni)."""
    return yf.download(
        asset.yahoo_ticker,
        period=f"{period_days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )


def fetch_price_history(asset: Asset, lookback_days: int) -> pd.DataFrame:
    """Scarica lo storico giornaliero dei prezzi per un asset.

    Effettua più tentativi con backoff esponenziale per tollerare gli errori
    transitori di yfinance/Yahoo (risposte vuote, rate limiting).

    Args:
        asset: asset da scaricare.
        lookback_days: numero di giorni di calendario di storico da richiedere.

    Returns:
        DataFrame indicizzato per data (``DatetimeIndex``, normalizzato a mezzanotte)
        con almeno la colonna ``close``.

    Raises:
        DataFetchError: se il download fallisce o non restituisce dati dopo i retry.
    """
    period_days = max(lookback_days + 10, 30)  # margine per giorni non di mercato

    raw: pd.DataFrame | None = None
    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            raw = _download(asset, period_days)
            if raw is not None and not raw.empty:
                break
        except Exception as exc:  # noqa: BLE001 - normalizziamo ogni errore di rete
            last_error = exc
        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_BACKOFF_SECONDS * (2**attempt))

    if raw is None or raw.empty:
        detail = f": {last_error}" if last_error else ""
        raise DataFetchError(
            f"Nessun dato restituito da yfinance per '{asset.id}' "
            f"(ticker {asset.yahoo_ticker}) dopo {_MAX_ATTEMPTS} tentativi{detail}."
        )

    # yfinance può restituire colonne MultiIndex quando c'è un solo ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if "Close" not in raw.columns:
        raise DataFetchError(
            f"Colonna 'Close' assente nei dati per '{asset.id}'."
        )

    df = raw[["Close"]].rename(columns={"Close": "close"}).copy()
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["close"]).sort_index()

    # Scarta eventuali barre datate nel futuro: per gli asset 24/5 (es. FX)
    # yfinance può restituire, a seconda dell'ora UTC, una barra del giorno
    # successivo, che falserebbe la classificazione "corrente".
    today_utc = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    df = df[df.index <= today_utc]

    if df.empty:
        raise DataFetchError(f"Serie prezzi vuota dopo la pulizia per '{asset.id}'.")

    return df
