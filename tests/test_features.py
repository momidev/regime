"""Test del feature engineering."""

from __future__ import annotations

import numpy as np

from models.features import FEATURE_COLUMNS, compute_features, feature_matrix


def test_compute_features_columns_and_no_nan(synthetic_prices):
    df = synthetic_prices(n=300)
    features = compute_features(df, rolling_window=20, momentum_window=10)

    for col in FEATURE_COLUMNS:
        assert col in features.columns
    assert "close" in features.columns
    # Nessun NaN dopo il dropna interno.
    assert not features.isna().any().any()
    # Le prime righe (warm-up rolling/momentum) devono essere state rimosse.
    assert len(features) == len(df) - 20  # 20 = max(rolling_window, momentum_window)


def test_feature_matrix_shape_and_order(synthetic_prices):
    df = synthetic_prices(n=200)
    features = compute_features(df, rolling_window=20, momentum_window=10)
    matrix = feature_matrix(features)

    assert matrix.shape == (len(features), len(FEATURE_COLUMNS))
    assert matrix.dtype == np.float64
    # La prima colonna deve corrispondere a log_return.
    np.testing.assert_allclose(matrix[:, 0], features["log_return"].to_numpy())


def test_log_return_definition(synthetic_prices):
    df = synthetic_prices(n=150)
    features = compute_features(df, rolling_window=5, momentum_window=5)
    close = df["close"]
    expected = np.log(close / close.shift(1)).loc[features.index]
    np.testing.assert_allclose(features["log_return"].to_numpy(), expected.to_numpy())
