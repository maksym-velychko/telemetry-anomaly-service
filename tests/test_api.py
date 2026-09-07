import pytest
from fastapi.testclient import TestClient

from api.app import app

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

def test_health_check(client):
    response = client.get('/docs/')

    assert response.status_code == 200

def test_predict_normal_data(client):
    payload = {
        'cpu_usage': 40,
        'memory_usage': 50,
    }
    response = client.post('/api/v1/predict/', json=payload)

    assert response.status_code == 200

    data = response.json()

    assert 'is_anomaly' in data
    assert 'status' in data
    assert data['is_anomaly'] is True
    assert data['status'] == 'CRITICAL'

def test_predict_anomaly_data(client):
    payload = {
        'cpu_usage': 99,
        'memory_usage': 99,
    }
    response = client.post('/api/v1/predict/', json=payload)

    assert response.status_code == 200

    data = response.json()

    assert 'is_anomaly' in data
    assert 'status' in data
    assert data['is_anomaly'] is True
    assert data['status'] == 'CRITICAL'

def test_predict_invalid_input(client):
    payload = {
        'cpu_usage': 329,
        'memory_usage': -29,
    }
    response = client.post('/api/v1/predict/', json=payload)

    assert response.status_code == 422