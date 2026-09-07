from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

@pytest.fixture
def mock_ml_model(monkeypatch):
    mock = MagicMock()
    mock.predict.return_value = [-1]

    monkeypatch.setattr('api.routes.model', mock)

    return mock

def test_model_prediction(mock_ml_model):
    payload = {
        'cpu_usage': 40,
        'memory_usage': 50,
    }
    response = client.post('/api/v1/predict/', json=payload)

    assert response.json()['is_anomaly'] is True