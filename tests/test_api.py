"""
Unit and Integration Tests for AeroEval FastAPI Endpoints.
"""

from fastapi.testclient import TestClient

from aeroeval.api.main import app

client = TestClient(app)


def test_root_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "AeroEval" in data["platform"]


def test_list_models():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "total_models" in data
    assert "models" in data
    assert isinstance(data["models"], list)


def test_register_model_and_get(tmp_path):
    dummy_weight = tmp_path / "mock_yolo.pt"
    dummy_weight.write_text("mock")

    payload = {
        "name": "api_test_model",
        "path": str(dummy_weight),
        "format": "PyTorch",
        "imgsz": 640,
        "description": "Registered via API test"
    }

    # Register
    res_reg = client.post("/models/register", json=payload)
    assert res_reg.status_code == 201
    reg_data = res_reg.json()
    assert reg_data["name"] == "api_test_model"

    # Get by name
    res_get = client.get("/models/api_test_model")
    assert res_get.status_code == 200
    assert res_get.json()["name"] == "api_test_model"

    # Delete
    res_del = client.delete("/models/api_test_model")
    assert res_del.status_code == 200

    # Verify deleted
    res_not_found = client.get("/models/api_test_model")
    assert res_not_found.status_code == 404


def test_get_run_results_baseline():
    response = client.get("/results/baseline")
    if response.status_code == 200:
        data = response.json()
        assert "detection" in data or "model_name" in data


def test_get_run_html_report_baseline():
    response = client.get("/results/baseline/report")
    if response.status_code == 200:
        assert "text/html" in response.headers.get("content-type", "")
        assert "AeroEval" in response.text
