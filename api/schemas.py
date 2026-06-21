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

    refreshed: int
    failed: int
    results: list[RefreshItem]


class HealthOut(BaseModel):
    """Risposta dell'healthcheck."""

    status: str = "ok"
    database: str = Field(..., description="'postgres' oppure 'file-store'.")
