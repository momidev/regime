"""Refresh giornaliero delle classificazioni di regime (da invocare via cron).

Uso:
    python -m scripts.refresh

Aggiorna tutti gli asset usando i modelli esistenti (li addestra se mancanti) e
registra eventuali cambi di regime. Equivalente all'endpoint POST /regime/refresh.
"""

from __future__ import annotations

import sys

import service


def main() -> int:
    results = service.refresh_all()
    failed = 0
    for r in results:
        if r.get("status") == "error":
            failed += 1
            print(f"[FAIL] {r['asset']:<8} {r['error']}", file=sys.stderr)
        else:
            changed = " (CAMBIO REGIME)" if r.get("regime_changed") else ""
            print(
                f"[OK]   {r['asset']:<8} "
                f"{r['status']:<10} regime={r.get('current_label')}"
                f"{changed}"
            )
    print(f"\nTotale: {len(results) - failed} ok, {failed} errori.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
