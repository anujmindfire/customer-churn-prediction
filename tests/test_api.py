"""Automated tests for FastAPI Customer Churn Prediction service."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@pytest.fixture
def valid_customer_payload() -> dict[str, Any]:
    """Provide a valid, complete customer payload matching raw Telco features."""
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


def test_health_endpoint() -> None:
    """Verify GET /health returns HTTP 200 and {'status': 'healthy'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_valid_customer(valid_customer_payload: dict[str, Any]) -> None:
    """Verify POST /predict returns HTTP 200 and valid schema for correct input."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in ["Churn", "No Churn"]
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_retained_customer(valid_customer_payload: dict[str, Any]) -> None:
    """Verify POST /predict handles established customer with low churn likelihood."""
    payload = valid_customer_payload.copy()
    payload["tenure"] = 72
    payload["Contract"] = "Two year"
    payload["MonthlyCharges"] = 20.0
    payload["TotalCharges"] = 1440.0
    payload["PaymentMethod"] = "Credit card (automatic)"

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in ["Churn", "No Churn"]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_missing_required_fields() -> None:
    """Verify POST /predict rejects incomplete payload with HTTP 422."""
    incomplete_payload = {"tenure": 5, "MonthlyCharges": 50.0}
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_negative_tenure(valid_customer_payload: dict[str, Any]) -> None:
    """Verify POST /predict rejects negative tenure with HTTP 422."""
    payload = valid_customer_payload.copy()
    payload["tenure"] = -3
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_monthly_charges(
    valid_customer_payload: dict[str, Any],
) -> None:
    """Verify POST /predict rejects negative MonthlyCharges with HTTP 422."""
    payload = valid_customer_payload.copy()
    payload["MonthlyCharges"] = -19.99
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_total_charges(
    valid_customer_payload: dict[str, Any],
) -> None:
    """Verify POST /predict rejects negative TotalCharges with HTTP 422."""
    payload = valid_customer_payload.copy()
    payload["TotalCharges"] = -100.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_total_charges_string(
    valid_customer_payload: dict[str, Any],
) -> None:
    """Verify POST /predict rejects non-numeric string TotalCharges with HTTP 422."""
    payload = valid_customer_payload.copy()
    payload["TotalCharges"] = "invalid_not_a_number"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_whitespace_total_charges(
    valid_customer_payload: dict[str, Any],
) -> None:
    """Verify POST /predict safely parses whitespace TotalCharges into 0.0."""
    payload = valid_customer_payload.copy()
    payload["TotalCharges"] = "   "
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["Churn", "No Churn"]


def test_predict_invalid_categorical_gender(
    valid_customer_payload: dict[str, Any],
) -> None:
    """Verify POST /predict rejects invalid gender categorical value with HTTP 422."""
    payload = valid_customer_payload.copy()
    payload["gender"] = "NonBinary"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_categorical_contract(
    valid_customer_payload: dict[str, Any],
) -> None:
    """Verify POST /predict rejects invalid contract type with HTTP 422."""
    payload = valid_customer_payload.copy()
    payload["Contract"] = "Lifetime"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_senior_citizen(
    valid_customer_payload: dict[str, Any],
) -> None:
    """Verify POST /predict rejects invalid SeniorCitizen value with HTTP 422."""
    payload = valid_customer_payload.copy()
    payload["SeniorCitizen"] = 99
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_without_customer_id(
    valid_customer_payload: dict[str, Any],
) -> None:
    """Verify POST /predict functions correctly when optional customerID is omitted."""
    payload = valid_customer_payload.copy()
    payload.pop("customerID")
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_docs_and_openapi_available() -> None:
    """Verify /docs and /openapi.json Swagger documentation endpoints are reachable."""
    docs_response = client.get("/docs")
    assert docs_response.status_code == 200

    openapi_response = client.get("/openapi.json")
    assert openapi_response.status_code == 200
    schema = openapi_response.json()
    assert "paths" in schema
    assert "/health" in schema["paths"]
    assert "/predict" in schema["paths"]


def test_internal_error_does_not_leak_traceback(
    valid_customer_payload: dict[str, Any],
) -> None:
    """Verify unexpected internal exception returns HTTP 500 without leaking stack traces."""
    with patch(
        "src.api.main.predict_churn",
        side_effect=RuntimeError("Unexpected internal pipeline crash at line 999!"),
    ):
        response = client.post("/predict", json=valid_customer_payload)
        assert response.status_code == 500
        error_json = response.json()
        assert "detail" in error_json
        # Stack trace or raw exception message must not be exposed
        assert "Unexpected internal pipeline crash" not in error_json["detail"]
        assert "Traceback" not in error_json["detail"]
