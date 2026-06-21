"""Configurazione applicativa caricata da variabili d'ambiente / file .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Directory di base del progetto e cartella per gli artifact dei modelli.
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "data" / "artifacts"
STORE_DIR = ARTIFACTS_DIR / "store"  # fallback su file quando il DB non è configurato


class Settings(BaseSettings):
    """Impostazioni dell'applicazione.

    Tutti i valori possono essere sovrascritti tramite variabili d'ambiente
    o tramite un file ``.env`` nella root del progetto.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database (opzionale): se vuoto si usa lo store su file JSON.
    database_url: str | None = None

    # Parametri del modello HMM.
    n_states: int = 4
    rolling_window: int = 20
    momentum_window: int = 10
    hmm_n_iter: int = 200
    random_seed: int = 42

    # Dati storici.
    history_lookback_days: int = 1825
    min_observations: int = 120

    # API / CORS.
    cors_origins: str = "*"

    @property
    def use_database(self) -> bool:
        """``True`` se è configurato un database Postgres."""
        return bool(self.database_url and self.database_url.strip())

    @property
    def cors_origin_list(self) -> list[str]:
        """Lista delle origini CORS consentite."""
        raw = self.cors_origins.strip()
        if raw == "*" or not raw:
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Ritorna l'istanza singleton delle impostazioni (cache-ata)."""
    settings = Settings()
    # Assicura l'esistenza delle cartelle per artifact e store su file.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()
