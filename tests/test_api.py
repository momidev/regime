"""Test degli endpoint API e del flusso di classificazione end-to-end.

Tutte le chiamate di rete (yfinance) e la persistenza sono isolate: i prezzi
sono sintetici e lo store usa una directory temporanea.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import service
from db.repository import FileRepository
from models import hmm_model

TRAINED_ASSET = "BTC-USD"
UNTRAINED_ASSET = "SPY"


@pytest.fixture
def client(tmp_path, monkeypatch, synthetic_prices):
    """TestClient con un modello addestrato su dati sintetici e store temporaneo."""
    # Artifact dei modelli in una cartella temporanea.
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(hmm_model, "ARTIFACTS_DIR", artifacts)

    # Repository su file in una cartella temporanea.
    repo = FileRepository(tmp_path / "store")
    monkeypatch.setattr(service, "get_repository", lambda: repo)

    # Nessuna rete: yfinance sostituito da prezzi sintetici.
    monkeypatch.setattr(
        service, "fetch_price_history", lambda asset, lookback: synthetic_prices()
    )

    # Addestra un singolo asset per i test di lettura.
    service.train_asset(TRAINED_ASSET)

    from api.main import app

    return TestClient(app)


def test_list_assets(client):
    resp = client.get("/assets")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 6
    assert any(a["id"] == "BTC-USD" for a in body["assets"])


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_current_regime(client):
    resp = client.get(f"/regime/{TRAINED_ASSET}/current")
    assert resp.status_code == 200
    body = resp.json()
    assert body["asset"] == TRAINED_ASSET
    assert 0 <= body["state_index"] < 4
    # Le probabilità sommano a ~1.
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
    # Descrizione presente e non prescrittiva.
    assert body["description"]
    assert "compra" not in body["description"].lower()


def test_history(client):
    resp = client.get(f"/regime/{TRAINED_ASSET}/history", params={"days": 90})
    assert resp.status_code == 200
    body = resp.json()
    assert body["asset"] == TRAINED_ASSET
    assert body["count"] > 0
    point = body["history"][0]
    assert {"date", "state_index", "state_label", "probabilities"} <= point.keys()


def test_history_invalid_days_param(client):
    resp = client.get(f"/regime/{TRAINED_ASSET}/history", params={"days": 0})
    assert resp.status_code == 422  # vincolo Query ge=1


def test_transition_matrix_is_stochastic(client):
    resp = client.get(f"/regime/{TRAINED_ASSET}/transition-matrix")
    assert resp.status_code == 200
    body = resp.json()
    matrix = body["matrix"]
    assert len(matrix) == 4
    for row in matrix:
        assert len(row) == 4
        assert abs(sum(row) - 1.0) < 1e-6  # ogni riga è una distribuzione


def test_alert_status(client):
    resp = client.get(f"/regime/{TRAINED_ASSET}/alert-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["asset"] == TRAINED_ASSET
    assert body["window_hours"] == 24
    assert isinstance(body["regime_changed"], bool)


def test_overview(client):
    resp = client.get("/regime/overview")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 6
    btc = next(a for a in body["assets"] if a["asset"] == TRAINED_ASSET)
    assert btc["has_data"] is True
    assert btc["state_label"]
    assert 0.0 <= btc["top_probability"] <= 1.0
    # Un asset non addestrato compare ma senza dati.
    spy = next(a for a in body["assets"] if a["asset"] == UNTRAINED_ASSET)
    assert spy["has_data"] is False
    assert spy["state_label"] is None


def test_prices(client):
    resp = client.get(f"/prices/{TRAINED_ASSET}", params={"days": 90})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] > 0
    point = body["prices"][0]
    assert {"date", "close"} <= point.keys()
    assert point["close"] > 0


def test_stats(client):
    resp = client.get(f"/regime/{TRAINED_ASSET}/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["current_regime"]
    assert body["current_streak_days"] >= 1
    assert body["sample_days"] > 0
    # Le frequenze sommano a ~1.
    assert abs(sum(r["frequency"] for r in body["regimes"]) - 1.0) < 1e-6
    # Durata attesa derivata dalla matrice di transizione.
    assert body["expected_duration_days"] is None or body["expected_duration_days"] > 0


def test_stats_untrained_asset_returns_409(client):
    resp = client.get(f"/regime/{UNTRAINED_ASSET}/stats")
    assert resp.status_code == 409


def test_unknown_asset_returns_404(client):
    resp = client.get("/regime/NOT-AN-ASSET/current")
    assert resp.status_code == 404


def test_untrained_asset_returns_409(client):
    resp = client.get(f"/regime/{UNTRAINED_ASSET}/current")
    assert resp.status_code == 409
