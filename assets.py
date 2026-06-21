"""Registry degli asset supportati.

Ogni asset ha un identificatore stabile usato nelle URL dell'API (``id``) e il
ticker corrispondente su Yahoo Finance (``yahoo_ticker``).
"""

from __future__ import annotations

from dataclasses import dataclass

from exceptions import AssetNotFoundError


@dataclass(frozen=True)
class Asset:
    """Descrittore di un asset supportato."""

    id: str
    yahoo_ticker: str
    name: str
    asset_class: str


# Registry ordinato degli asset supportati.
SUPPORTED_ASSETS: tuple[Asset, ...] = (
    Asset("BTC-USD", "BTC-USD", "Bitcoin", "crypto"),
    Asset("SPY", "SPY", "S&P 500 ETF", "equity-index"),
    Asset("XLK", "XLK", "Technology Sector", "sector"),
    Asset("XLE", "XLE", "Energy Sector", "sector"),
    Asset("XLF", "XLF", "Financials Sector", "sector"),
    Asset("EURUSD", "EURUSD=X", "Euro / US Dollar", "fx"),
)

_ASSET_BY_ID: dict[str, Asset] = {asset.id: asset for asset in SUPPORTED_ASSETS}


def list_assets() -> list[Asset]:
    """Ritorna la lista di tutti gli asset supportati."""
    return list(SUPPORTED_ASSETS)


def get_asset(asset_id: str) -> Asset:
    """Ritorna l'asset con l'``id`` indicato.

    Args:
        asset_id: identificatore dell'asset (case-insensitive).

    Raises:
        AssetNotFoundError: se l'asset non è nel registry.
    """
    asset = _ASSET_BY_ID.get(asset_id) or _ASSET_BY_ID.get(asset_id.upper())
    if asset is None:
        supported = ", ".join(a.id for a in SUPPORTED_ASSETS)
        raise AssetNotFoundError(
            f"Asset '{asset_id}' non supportato. Asset disponibili: {supported}."
        )
    return asset
