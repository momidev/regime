"""Definizione degli endpoint REST."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

import service
from api.schemas import (
    AlertStatusOut,
    AssetListOut,
    AssetOut,
    CurrentRegimeOut,
    HealthOut,
    HistoryOut,
    OverviewOut,
    PricesOut,
    RefreshOut,
    StatsOut,
    TransitionMatrixOut,
)
from assets import list_assets
from config import settings
from exceptions import (
    AssetNotFoundError,
    DataFetchError,
    InsufficientDataError,
    ModelNotTrainedError,
)

router = APIRouter()


def _handle_domain_errors(exc: Exception) -> HTTPException:
    """Mappa le eccezioni di dominio sugli status HTTP appropriati."""
    if isinstance(exc, AssetNotFoundError):
        return HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, ModelNotTrainedError):
        return HTTPException(status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, InsufficientDataError):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if isinstance(exc, DataFetchError):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    """Healthcheck per il deploy (es. Railway)."""
    return HealthOut(
        status="ok",
        database="postgres" if settings.use_database else "file-store",
    )


@router.get("/assets", response_model=AssetListOut, tags=["assets"])
def get_assets() -> AssetListOut:
    """Ritorna la lista degli asset supportati."""
    assets = [
        AssetOut(
            id=a.id,
            name=a.name,
            asset_class=a.asset_class,
            yahoo_ticker=a.yahoo_ticker,
        )
        for a in list_assets()
    ]
    return AssetListOut(count=len(assets), assets=assets)


@router.get("/regime/overview", response_model=OverviewOut, tags=["regime"])
def regime_overview() -> OverviewOut:
    """Stato corrente sintetico di tutti gli asset in una sola risposta (per la home)."""
    return OverviewOut(**service.get_overview())


@router.get(
    "/regime/{asset}/current",
    response_model=CurrentRegimeOut,
    tags=["regime"],
)
def regime_current(asset: str) -> CurrentRegimeOut:
    """Stato di regime corrente di un asset, con probabilità per ciascun regime."""
    try:
        return CurrentRegimeOut(**service.get_current(asset))
    except Exception as exc:  # noqa: BLE001
        raise _handle_domain_errors(exc) from exc


@router.get(
    "/regime/{asset}/history",
    response_model=HistoryOut,
    tags=["regime"],
)
def regime_history(
    asset: str,
    days: int = Query(90, ge=1, le=3650, description="Giorni di storico da ritornare."),
) -> HistoryOut:
    """Storico delle classificazioni di regime per il grafico timeline."""
    try:
        return HistoryOut(**service.get_history(asset, days))
    except Exception as exc:  # noqa: BLE001
        raise _handle_domain_errors(exc) from exc


@router.get(
    "/regime/{asset}/transition-matrix",
    response_model=TransitionMatrixOut,
    tags=["regime"],
)
def regime_transition_matrix(asset: str) -> TransitionMatrixOut:
    """Matrice di transizione stimata tra i regimi di un asset."""
    try:
        return TransitionMatrixOut(**service.get_transition_matrix(asset))
    except Exception as exc:  # noqa: BLE001
        raise _handle_domain_errors(exc) from exc


@router.get(
    "/regime/{asset}/alert-status",
    response_model=AlertStatusOut,
    tags=["regime"],
)
def regime_alert_status(asset: str) -> AlertStatusOut:
    """Indica se c'è stato un cambio di regime nelle ultime 24 ore."""
    try:
        return AlertStatusOut(**service.get_alert_status(asset))
    except Exception as exc:  # noqa: BLE001
        raise _handle_domain_errors(exc) from exc


@router.get("/regime/{asset}/stats", response_model=StatsOut, tags=["regime"])
def regime_stats(asset: str) -> StatsOut:
    """Statistiche sui regimi: streak corrente, frequenze, durate, durata attesa."""
    try:
        return StatsOut(**service.get_stats(asset))
    except Exception as exc:  # noqa: BLE001
        raise _handle_domain_errors(exc) from exc


@router.get("/prices/{asset}", response_model=PricesOut, tags=["prices"])
def prices(
    asset: str,
    days: int = Query(90, ge=1, le=3650, description="Giorni di storico prezzi."),
) -> PricesOut:
    """Serie del prezzo di chiusura, allineata alla timeline dei regimi (overlay)."""
    try:
        return PricesOut(**service.get_prices(asset, days))
    except Exception as exc:  # noqa: BLE001
        raise _handle_domain_errors(exc) from exc


@router.post("/regime/refresh", response_model=RefreshOut, tags=["regime"])
def regime_refresh(
    background_tasks: BackgroundTasks,
    wait: bool = Query(
        True,
        description=(
            "Se true (default) esegue il refresh e attende il risultato. "
            "Se false avvia il refresh in background e risponde subito: "
            "utile per i cron con timeout breve (es. cron-job.org)."
        ),
    ),
) -> RefreshOut:
    """Ricalcola la classificazione per tutti gli asset (da invocare via cron)."""
    if not wait:
        background_tasks.add_task(service.refresh_all)
        return RefreshOut(status="scheduled", refreshed=0, failed=0, results=[])

    results = service.refresh_all()
    failed = sum(1 for r in results if r.get("status") == "error")
    return RefreshOut(
        status="completed",
        refreshed=len(results) - failed,
        failed=failed,
        results=results,  # type: ignore[arg-type]
    )
