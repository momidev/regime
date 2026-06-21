"""Test della logica di etichettatura dei regimi."""

from __future__ import annotations

import numpy as np

from models.regime_labeler import (
    StateStats,
    compute_state_stats,
    describe_label,
    label_states,
)


def test_label_states_four_regimes():
    # Quattro stati con direzioni e intensità ben separate.
    stats = [
        StateStats(0, mean_return=0.002, mean_volatility=0.10, frequency=0.25),  # bull calmo
        StateStats(1, mean_return=0.002, mean_volatility=0.40, frequency=0.25),  # bull volatile
        StateStats(2, mean_return=-0.002, mean_volatility=0.10, frequency=0.25),  # bear calmo
        StateStats(3, mean_return=-0.002, mean_volatility=0.40, frequency=0.25),  # bear volatile
    ]
    labels = label_states(stats)

    assert labels[0] == "bull-calmo"
    assert labels[1] == "bull-volatile"
    assert labels[2] == "bear-calmo"
    assert labels[3] == "bear-volatile"
    # Tutte le etichette devono essere univoche.
    assert len(set(labels.values())) == 4


def test_label_states_deduplicates_when_collisions():
    # Tre stati "bull" forzano una collisione di etichette → suffissi numerici.
    stats = [
        StateStats(0, mean_return=0.003, mean_volatility=0.10, frequency=0.33),
        StateStats(1, mean_return=0.002, mean_volatility=0.11, frequency=0.33),
        StateStats(2, mean_return=0.001, mean_volatility=0.12, frequency=0.34),
    ]
    labels = label_states(stats)
    # Le etichette restano univoche anche in caso di collisione.
    assert len(set(labels.values())) == len(stats)


def test_compute_state_stats_matches_empirical_means():
    states = np.array([0, 0, 1, 1])
    log_returns = np.array([0.01, 0.03, -0.02, -0.04])
    volatility = np.array([0.1, 0.1, 0.3, 0.3])
    stats = compute_state_stats(states, log_returns, volatility, n_states=2)

    assert stats[0].mean_return == 0.02
    assert stats[1].mean_return == -0.03
    assert stats[0].frequency == 0.5


def test_describe_label_is_descriptive_not_prescriptive():
    stats = StateStats(0, 0.001, 0.2, 0.25)
    text = describe_label("bull-calmo", stats)
    lowered = text.lower()
    # Output descrittivo: nessun linguaggio prescrittivo/operativo.
    for forbidden in ("compra", "vendi", "buy", "sell", "consiglio"):
        assert forbidden not in lowered
