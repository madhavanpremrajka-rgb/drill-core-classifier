from fastapi.testclient import TestClient
import sys
sys.path.append('inference')
import pytest
from app import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    response = client.get('/')
    assert response.status_code == 200

def test_model_info(client):
    response = client.get('/model-info')
    data = response.json()
    assert response.status_code == 200
    assert "model_name" in data

def test_single_predict(client):
    with open("Data/Test_Data/Dataset-35_R-128_L-0.4_I-E/test/8.Granite/8.Granite_1.jpg", 'rb') as f:
        response = client.post("/predict", files={"file":f})
    data = response.json()
    assert response.status_code == 200
    assert "confidence" in data
    assert "predicted_class" in data