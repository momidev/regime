"""Modelli Pydantic per request e response dell'API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AssetOut(BaseModel):
    """Asset supportato."""

    id: str = Field(..., examples=["BTC-USD"])
    name: str = Field(..., examples=["Bitcoin"])
    asset_class: str = Field(..., examples=["crypto"])
    yahoo_ticker: str = Field(..., examples=["BTC-USD"])


class AssetListOut(BaseModel):
    """Lista degli asset supportati."""

    count: int
    assets: list[AssetOut]


class CurrentRegimeOut(BaseModel):
    """Stato di regime corrente di un asset."""

    asset: str
    as_of: str = Field(..., description="Data della classificazione (YYYY-MM-DD).")
    state_index: int
    state_label: str = Field(..., examples=["bull-calmo"])
    description: str = Field(
        ..., description="Descrizione statistica del regime (non prescrittiva)."
    )
    probabilities: dict[str, float] = Field(
        ..., description="Probabilità a posteriori per ciascun regime."
    )
    model_version: str


class HistoryPoint(BaseModel):
    """Singolo punto dello storico delle classificazioni."""

    date: str
    state_index: int
    state_label: str
    probabilities: dict[str, float]


class HistoryOut(BaseModel):
    """Storico delle classificazioni per il grafico timeline."""

    asset: str
    days: int
    count: int
    history: list[HistoryPoint]


class TransitionMatrixOut(BaseModel):
    """Matrice di transizione tra regimi."""

    asset: str
    state_labels: list[str] = Field(
        ..., description="Etichette degli stati, ordinate per indice."
    )
    matrix: list[list[float]] = Field(
        ...,
        description="Matrice NxN: matrix[i][j] = P(stato j domani | stato i oggi).",
    )
    model_version: str
    computed_at: str


class AlertStatusOut(BaseModel):
    """Stato di alert sui cambi di regime recenti."""

    asset: str
    regime_changed: bool = Field(
        ..., description="True se c'è stato un cambio di regime nella finestra."
    )
    window_hours: int
    from_label: str | None = None
    to_label: str | None = None
    changed_at: str | None = None


class RefreshItem(BaseModel):
    """Esito del refresh per un singolo asset."""

    asset: str
    status: str = Field(..., examples=["refreshed", "trained", "error"])
    current_label: str | None = None
    as_of: str | None = None
    model_version: str | None = None
    regime_changed: bool | None = None
    error: str | None = None


class RefreshOut(BaseModel):
    """Esito complessivo del refresh di tutti gli asset."""

    status: str = Field(
        "completed",
        description="'completed' se eseguito subito, 'scheduled' se avviato in background.",
    )
    refreshed: int
    failed: int
    results: list[RefreshItem]


class OverviewItem(BaseModel):
    """Stato corrente sintetico di un asset (per la home)."""

    asset: str
    name: str
    asset_class: str
    has_data: bool
    as_of: str | None = None
    state_label: str | None = None
    top_probability: float | None = Field(
        None, description="Probabilità del regime più probabile."
    )


class OverviewOut(BaseModel):
    """Snapshot di tutti gli asset in una sola risposta."""

    count: int
    assets: list[OverviewItem]


class PricePoint(BaseModel):
    """Prezzo di chiusura per una data."""

    date: str
    close: float


class PricesOut(BaseModel):
    """Serie storica del prezzo di chiusura, allineata ai regimi."""

    asset: str
    days: int
    count: int
    prices: list[PricePoint]


class RegimeStat(BaseModel):
    """Statistiche di un singolo regime sullo storico."""

    label: str
    frequency: float = Field(..., description="Quota di giorni in questo regime [0,1].")
    days: int
    avg_duration_days: float = Field(
        ..., description="Durata media (giorni) dei periodi consecutivi nel regime."
    )
    occurrences: int = Field(..., description="Numero di periodi distinti nel regime.")


class StatsOut(BaseModel):
    """Statistiche descrittive sui regimi di un asset."""

    asset: str
    as_of: str
    current_regime: str
    current_streak_days: int = Field(
        ..., description="Giorni consecutivi nel regime corrente."
    )
    expected_duration_days: float | None = Field(
        None,
        description="Durata attesa del regime corrente, da 1/(1-p_ii) della matrice.",
    )
    sample_days: int
    regimes: list[RegimeStat]


class HealthOut(BaseModel):
    """Risposta dell'healthcheck."""

    status: str = "ok"
    database: str = Field(..., description="'postgres' oppure 'file-store'.")
