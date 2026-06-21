"""Fixture condivise per i test (nessuna chiamata di rete)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices():
    """Factory che genera una serie di prezzi sintetica con due regimi distinti.

    Alterna fasi di "bull calmo" (deriva positiva, bassa volatilità) e
    "bear volatile" (deriva negativa, alta volatilità), così che l'HMM possa
    separare gli stati in modo deterministico.
    """

    def _make(n: int = 400, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        returns = np.empty(n)
        for i in range(n):
            if i % 100 < 60:
                returns[i] = rng.normal(0.0012, 0.005)  # bull calmo
            else:
                returns[i] = rng.normal(-0.0015, 0.022)  # bear volatile
        prices = 100.0 * np.exp(np.cumsum(returns))
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
        return pd.DataFrame({"close": prices}, index=idx)

    return _make
