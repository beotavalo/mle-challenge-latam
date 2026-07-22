"""Contract tests for the prediction API.

The provided ``test_api.py`` pins the four scenarios the challenge grades. These
tests cover the rest of the contract: batching, the health probe, the documented
schema and the boundaries of every validation rule.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from challenge import app

client = TestClient(app)


def _flight(**overrides: Any) -> dict[str, Any]:
    flight = {"OPERA": "Grupo LATAM", "TIPOVUELO": "I", "MES": 7}
    flight.update(overrides)
    return flight


class TestHealth:
    def test_reports_the_service_as_healthy(self) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "OK"}


class TestPredictHappyPath:
    def test_returns_one_prediction_per_flight_in_order(self) -> None:
        payload = {
            "flights": [
                _flight(OPERA="Grupo LATAM", TIPOVUELO="I", MES=7),
                _flight(OPERA="Aerolineas Argentinas", TIPOVUELO="N", MES=3),
                _flight(OPERA="Sky Airline", TIPOVUELO="N", MES=12),
            ]
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        predictions = response.json()["predict"]
        assert len(predictions) == 3
        assert all(prediction in (0, 1) for prediction in predictions)

    def test_accepts_every_airline_present_in_the_training_data(self) -> None:
        response = client.post("/predict", json={"flights": [_flight(OPERA="K.L.M.")]})

        assert response.status_code == 200

    @pytest.mark.parametrize("month", [1, 6, 12])
    def test_accepts_every_month_of_the_year(self, month: int) -> None:
        response = client.post("/predict", json={"flights": [_flight(MES=month)]})

        assert response.status_code == 200


class TestPredictValidation:
    @pytest.mark.parametrize("month", [0, 13, -1])
    def test_rejects_months_outside_the_calendar(self, month: int) -> None:
        response = client.post("/predict", json={"flights": [_flight(MES=month)]})

        assert response.status_code == 400

    @pytest.mark.parametrize("flight_type", ["O", "n", "", "INTERNATIONAL"])
    def test_rejects_unsupported_flight_types(self, flight_type: str) -> None:
        response = client.post("/predict", json={"flights": [_flight(TIPOVUELO=flight_type)]})

        assert response.status_code == 400

    @pytest.mark.parametrize("airline", ["Argentinas", "", "DROP TABLE flights"])
    def test_rejects_airlines_absent_from_the_training_data(self, airline: str) -> None:
        response = client.post("/predict", json={"flights": [_flight(OPERA=airline)]})

        assert response.status_code == 400

    def test_rejects_a_flight_with_a_missing_field(self) -> None:
        response = client.post("/predict", json={"flights": [{"OPERA": "Grupo LATAM", "MES": 3}]})

        assert response.status_code == 400

    def test_rejects_an_empty_batch(self) -> None:
        response = client.post("/predict", json={"flights": []})

        assert response.status_code == 400

    def test_rejects_a_payload_without_the_flights_key(self) -> None:
        response = client.post("/predict", json={"vuelos": []})

        assert response.status_code == 400

    def test_reports_which_field_was_rejected_without_echoing_the_payload(self) -> None:
        response = client.post("/predict", json={"flights": [_flight(MES=13)]})

        body = response.json()
        assert response.status_code == 400
        assert "detail" in body
        assert any("MES" in str(error.get("loc", "")) for error in body["detail"])


class TestDocumentation:
    def test_publishes_an_openapi_schema(self) -> None:
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert "/predict" in schema["paths"]
        assert "/health" in schema["paths"]
