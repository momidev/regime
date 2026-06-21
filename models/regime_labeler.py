"""Mappatura degli stati anonimi dell'HMM in etichette di regime leggibili.

Gli stati di un Hidden Markov Model non hanno un significato intrinseco: lo stato
``0`` non è necessariamente "bull". Questo modulo assegna etichette descrittive
ordinando gli stati per ritorno medio (direzione del mercato) e per volatilità
media (intensità), in modo coerente e riproducibile.

Tutte le etichette/descrizioni sono **descrittive e statistiche**, mai
prescrittive (nessun consiglio operativo).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Descrizioni testuali per lo schema a 4 regimi (caso di default).
_DESCRIPTIONS_4: dict[str, str] = {
    "bull-calmo": (
        "Il modello classifica il regime come tendenza rialzista con bassa "
        "volatilità (mercato in salita ordinata)."
    ),
    "bull-volatile": (
        "Il modello classifica il regime come tendenza rialzista con alta "
        "volatilità."
    ),
    "bear-calmo": (
        "Il modello classifica il regime come tendenza ribassista con bassa "
        "volatilità (discesa ordinata o fase laterale debole)."
    ),
    "bear-volatile": (
        "Il modello classifica il regime come tendenza ribassista con alta "
        "volatilità (fase di stress / possibile crisi)."
    ),
}


@dataclass(frozen=True)
class StateStats:
    """Statistiche empiriche di uno stato nascosto sui dati di training."""

    state: int
    mean_return: float
    mean_volatility: float
    frequency: float  # quota di osservazioni assegnate allo stato [0, 1]


def compute_state_stats(
    states: np.ndarray,
    log_returns: np.ndarray,
    volatility: np.ndarray,
    n_states: int,
) -> list[StateStats]:
    """Calcola le statistiche empiriche per ciascuno stato nascosto.

    Args:
        states: sequenza di stati assegnati (output di ``hmm.predict``).
        log_returns: ritorni log allineati a ``states``.
        volatility: volatilità allineata a ``states``.
        n_states: numero totale di stati del modello.

    Returns:
        Lista di :class:`StateStats`, una per ogni stato (anche se non osservato).
    """
    total = len(states)
    stats: list[StateStats] = []
    for s in range(n_states):
        mask = states == s
        count = int(mask.sum())
        if count == 0:
            stats.append(StateStats(s, 0.0, 0.0, 0.0))
            continue
        stats.append(
            StateStats(
                state=s,
                mean_return=float(np.mean(log_returns[mask])),
                mean_volatility=float(np.mean(volatility[mask])),
                frequency=count / total if total else 0.0,
            )
        )
    return stats


def label_states(stats: list[StateStats]) -> dict[int, str]:
    """Assegna a ciascuno stato un'etichetta di regime leggibile.

    Con **4 stati** (caso di default) usa lo schema a quadranti, che garantisce
    esattamente le quattro etichette pulite ``bull-calmo``, ``bull-volatile``,
    ``bear-calmo``, ``bear-volatile`` (vedi :func:`_label_quadrants`).

    Con un numero di stati diverso da 4 usa uno schema generico basato sulle
    mediane, rendendo univoche le eventuali etichette duplicate con un suffisso
    numerico (vedi :func:`_label_by_median`).

    Args:
        stats: statistiche per stato da :func:`compute_state_stats`.

    Returns:
        Dizionario ``{state_index: label}``.
    """
    if len(stats) == 4:
        return _label_quadrants(stats)
    return _label_by_median(stats)


def _label_quadrants(stats: list[StateStats]) -> dict[int, str]:
    """Schema a 4 quadranti per il caso a 4 stati.

    I 2 stati con ritorno medio più alto sono ``bull``, i 2 più basso ``bear``;
    all'interno di ciascuna coppia, lo stato a volatilità maggiore è ``volatile``,
    l'altro ``calmo``. Produce sempre 4 etichette distinte e leggibili.
    """
    by_return = sorted(stats, key=lambda s: s.mean_return)
    bear_states = by_return[:2]  # ritorni più bassi
    bull_states = by_return[2:]  # ritorni più alti

    labels: dict[int, str] = {}
    for direction, pair in (("bull", bull_states), ("bear", bear_states)):
        calmo, volatile = sorted(pair, key=lambda s: s.mean_volatility)
        labels[calmo.state] = f"{direction}-calmo"
        labels[volatile.state] = f"{direction}-volatile"
    return labels


def _label_by_median(stats: list[StateStats]) -> dict[int, str]:
    """Schema generico (n_states != 4): direzione/intensità rispetto alle mediane.

    Direzione (bull/bear) dal ritorno medio rispetto alla mediana degli stati;
    intensità (calmo/volatile) dalla volatilità media rispetto alla mediana. Le
    eventuali etichette duplicate vengono rese univoche con un suffisso numerico.
    """
    returns = np.array([s.mean_return for s in stats])
    vols = np.array([s.mean_volatility for s in stats])
    median_return = float(np.median(returns))
    median_vol = float(np.median(vols))

    raw_labels: dict[int, str] = {}
    for s in stats:
        direction = "bull" if s.mean_return >= median_return else "bear"
        intensity = "volatile" if s.mean_volatility >= median_vol else "calmo"
        raw_labels[s.state] = f"{direction}-{intensity}"

    counts: dict[str, int] = {}
    seen: dict[str, int] = {}
    for label in raw_labels.values():
        counts[label] = counts.get(label, 0) + 1

    labels: dict[int, str] = {}
    for state, label in raw_labels.items():
        if counts[label] > 1:
            idx = seen.get(label, 0) + 1
            seen[label] = idx
            labels[state] = f"{label}-{idx}"
        else:
            labels[state] = label
    return labels


def describe_label(label: str, stats: StateStats) -> str:
    """Ritorna una descrizione testuale (descrittiva) di un'etichetta di regime.

    Args:
        label: etichetta restituita da :func:`label_states`.
        stats: statistiche dello stato corrispondente.

    Returns:
        Descrizione in linguaggio naturale, sempre statistica/descrittiva.
    """
    base = _DESCRIPTIONS_4.get(label)
    if base is not None:
        return base
    return (
        f"Regime statistico '{label}': ritorno log medio giornaliero "
        f"{stats.mean_return:+.4f}, volatilità media annualizzata "
        f"{stats.mean_volatility:.2%}."
    )
