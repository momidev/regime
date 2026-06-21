"""Eccezioni di dominio usate dal sistema di regime detection.

Le route FastAPI mappano queste eccezioni sugli opportuni status code HTTP.
"""

from __future__ import annotations


class RegimeError(Exception):
    """Classe base per tutte le eccezioni di dominio."""


class AssetNotFoundError(RegimeError):
    """L'asset richiesto non è presente nel registry degli asset supportati."""


class DataFetchError(RegimeError):
    """Errore nel recupero dei dati di mercato (es. yfinance non risponde)."""


class InsufficientDataError(RegimeError):
    """Dati storici insufficienti per addestrare o applicare l'HMM."""


class ModelNotTrainedError(RegimeError):
    """Il modello HMM per l'asset non è ancora stato addestrato/persistito."""
