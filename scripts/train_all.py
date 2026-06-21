"""Training iniziale (o re-training periodico) dei modelli HMM.

Uso:
    python -m scripts.train_all                # addestra tutti gli asset
    python -m scripts.train_all BTC-USD SPY    # addestra solo gli asset indicati

Esegue il backfill dello storico delle classificazioni e salva gli artifact in
data/artifacts/. Può essere rilanciato periodicamente per il re-training.
"""

from __future__ import annotations

import sys

import service
from assets import list_assets


def main(argv: list[str]) -> int:
    asset_ids = argv or [a.id for a in list_assets()]
    exit_code = 0
    for asset_id in asset_ids:
        try:
            result = service.train_asset(asset_id)
            print(
                f"[OK]   {result['asset']:<8} "
                f"obs={result['observations']:<5} "
                f"regime={result['current_label']:<14} "
                f"as_of={result['as_of']} (v{result['model_version']})"
            )
        except Exception as exc:  # noqa: BLE001
            exit_code = 1
            print(f"[FAIL] {asset_id:<8} {type(exc).__name__}: {exc}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
