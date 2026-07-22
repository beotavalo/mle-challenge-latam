"""HTTP interface of the flight delay model.

Exposes the delay prediction to the airport team over two endpoints: a liveness
probe consumed by Cloud Run, and a batch prediction endpoint. Every payload is
validated against the categories the model was actually trained on, so an unknown
airline, month or flight type is rejected as a client error instead of silently
producing a meaningless prediction.
"""

import json
import logging
from typing import Any, ClassVar

import fastapi
import pandas as pd
from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from challenge.model import KNOWN_AIRLINES, KNOWN_FLIGHT_TYPES, DelayModel

LOGGER = logging.getLogger(__name__)

API_TITLE = "SCL Flight Delay Prediction API"
API_VERSION = "1.0.0"
PREDICT_ENDPOINT = "/predict"
API_DESCRIPTION = (
    "Predicts whether a flight taking off from or landing at SCL airport will be "
    "delayed more than 15 minutes."
)


class JsonFormatter(logging.Formatter):
    """Minimal structured formatter: one JSON object per log line.

    Cloud Run ingests stdout as-is, so emitting JSON turns every log line into a
    queryable record without adding a logging dependency to the runtime image.
    """

    _RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in self._RESERVED and not key.startswith("_")
            }
        )
        return json.dumps(payload, default=str)


def _configure_logging() -> None:
    """Attach the JSON formatter without clobbering an existing configuration."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)


class Flight(BaseModel):
    """One flight to score, described with the three features the model consumes."""

    OPERA: str = Field(..., description="Airline operating the flight, as named in the dataset.")
    TIPOVUELO: str = Field(..., description="Flight type: I (international) or N (national).")
    MES: int = Field(..., ge=1, le=12, description="Month of the scheduled operation, 1 to 12.")

    @validator("OPERA")
    def airline_must_be_known(cls, value: str) -> str:  # noqa: N805 - pydantic v1 validator
        """Reject airlines the model was never trained on."""
        if value not in KNOWN_AIRLINES:
            raise ValueError(f"Unknown airline: {value!r}")
        return value

    @validator("TIPOVUELO")
    def flight_type_must_be_known(cls, value: str) -> str:  # noqa: N805 - pydantic v1 validator
        """Reject flight types outside the international/national domain."""
        if value not in KNOWN_FLIGHT_TYPES:
            raise ValueError(f"Unknown flight type: {value!r}")
        return value

    class Config:
        schema_extra: ClassVar[dict[str, Any]] = {
            "example": {"OPERA": "Grupo LATAM", "TIPOVUELO": "I", "MES": 7}
        }


class PredictRequest(BaseModel):
    """A batch of flights to score in a single call."""

    flights: list[Flight] = Field(..., min_items=1, description="Non-empty batch of flights.")


class PredictResponse(BaseModel):
    """One prediction per submitted flight: 1 means delayed more than 15 minutes."""

    predict: list[int]

    class Config:
        schema_extra: ClassVar[dict[str, Any]] = {"example": {"predict": [0, 1]}}


_configure_logging()

app = fastapi.FastAPI(title=API_TITLE, version=API_VERSION, description=API_DESCRIPTION)

#: Loaded once per process: `/predict` must never hit the filesystem.
_MODEL = DelayModel()


@app.on_event("startup")
def load_champion_model() -> None:
    """Warm the champion estimator so the first request pays no I/O cost."""
    _MODEL.load()


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Answer malformed or unknown-category payloads with 400 instead of 422.

    The response echoes the field locations that failed, never the submitted values,
    and the log record carries a constant endpoint label instead of the reconstructed
    request URL, which starlette does not validate (PYSEC-2026-161 / PYSEC-2026-248).
    """
    LOGGER.warning(
        "rejected prediction request",
        extra={"endpoint": PREDICT_ENDPOINT, "violations": len(exc.errors())},
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": jsonable_encoder(exc.errors())},
    )


@app.exception_handler(ValueError)
async def handle_value_error(
    request: Request,
    exc: ValueError,
) -> JSONResponse:
    """Translate domain-level rejections raised during preprocessing into 400."""
    LOGGER.warning("rejected prediction request", extra={"endpoint": PREDICT_ENDPOINT})
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


@app.get("/health", status_code=200)
async def get_health() -> dict[str, str]:
    return {"status": "OK"}


@app.post(PREDICT_ENDPOINT, status_code=200, response_model=PredictResponse)
async def post_predict(request: PredictRequest) -> PredictResponse:
    """Score a batch of flights.

    Args:
        request: non-empty batch of flights, already validated against the
            categories present in the training data.

    Returns:
        One prediction per flight, in the submitted order.
    """
    flights = pd.DataFrame([flight.dict() for flight in request.flights])
    features = _MODEL.preprocess(data=flights)
    predictions = _MODEL.predict(features=features)
    LOGGER.info("served prediction batch", extra={"flights": len(predictions)})
    return PredictResponse(predict=predictions)
