"""Flight delay model for flights taking off from or landing at SCL airport.

This module is the production transcription of the Data Scientist's exploration
notebook (``challenge/exploration.ipynb``). The feature engineering lives in pure,
independently testable functions, and :class:`DelayModel` only orchestrates them
around the estimator, so the two can change for different reasons.

Model in production: ``LogisticRegression`` trained on the ten most important
features with class balancing (notebook section 6.b.iii). See ``docs/challenge.md``
for the selection rationale and the measured evidence.
"""

import logging
from datetime import datetime, time
from pathlib import Path
from typing import Dict, Final, List, Mapping, Optional, Tuple, Union

import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger(__name__)

#: Timestamp format used by the ``Fecha-I`` / ``Fecha-O`` columns of the dataset.
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

#: A flight counts as delayed when it departs more than this many minutes late.
DELAY_THRESHOLD_IN_MINUTES: Final[int] = 15

#: Name of the engineered binary target column.
TARGET_COLUMN: Final[str] = "delay"

#: Columns a caller must provide for feature generation, both for training and serving.
REQUIRED_FEATURE_COLUMNS: Final[Tuple[str, ...]] = ("OPERA", "TIPOVUELO", "MES")

#: Columns needed on top of those to derive the target when it is not precomputed.
REQUIRED_TARGET_COLUMNS: Final[Tuple[str, ...]] = ("Fecha-I", "Fecha-O")

#: The ten most important features according to the notebook's feature importance
#: analysis. The order is part of the contract: it is the order the estimator was
#: trained with, and reindexing guarantees it for every payload.
TOP_10_FEATURES: Final[List[str]] = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]

#: Airlines observed in the training data. Anything outside this catalogue is an
#: unknown category the model was never trained on, and the API rejects it.
KNOWN_AIRLINES: Final[frozenset[str]] = frozenset(
    {
        "Aerolineas Argentinas",
        "Aeromexico",
        "Air Canada",
        "Air France",
        "Alitalia",
        "American Airlines",
        "Austral",
        "Avianca",
        "British Airways",
        "Copa Air",
        "Delta Air",
        "Gol Trans",
        "Grupo LATAM",
        "Iberia",
        "JetSmart SPA",
        "K.L.M.",
        "Lacsa",
        "Latin American Wings",
        "Oceanair Linhas Aereas",
        "Plus Ultra Lineas Aereas",
        "Qantas Airways",
        "Sky Airline",
        "United Airlines",
    }
)

#: Valid flight types: international and national.
KNOWN_FLIGHT_TYPES: Final[frozenset[str]] = frozenset({"I", "N"})

#: Serialized champion estimator, kept in sync with the ``@champion`` alias of the
#: ``flight_delay_logreg`` registered model (see ``challenge/train.py``).
MODEL_ARTIFACT_PATH: Final[Path] = Path(__file__).resolve().parent / "model.joblib"

_MORNING_START: Final[time] = time(5, 0)
_AFTERNOON_START: Final[time] = time(12, 0)
_NIGHT_START: Final[time] = time(19, 0)

#: High season windows as ``(month, day)`` inclusive pairs, taken from the notebook.
_HIGH_SEASON_RANGES: Final[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...]] = (
    ((12, 15), (12, 31)),
    ((1, 1), (3, 3)),
    ((7, 15), (7, 31)),
    ((9, 11), (9, 30)),
)


def get_period_day(date: str) -> str:
    """Return the period of the day (``morning``/``afternoon``/``night``) of a flight.

    The notebook compared the boundaries with strict inequalities, which left every
    boundary instant (05:00, 11:59, 12:00, 18:59, 19:00, 23:59, 00:00, 04:59)
    unclassified and produced ``None`` values. Half-open intervals cover the full day.

    Args:
        date: scheduled timestamp, formatted as :data:`DATE_FORMAT`.

    Returns:
        ``"morning"`` between 05:00 and 11:59, ``"afternoon"`` between 12:00 and
        18:59, ``"night"`` otherwise.
    """
    flight_time = datetime.strptime(date, DATE_FORMAT).time()
    if _MORNING_START <= flight_time < _AFTERNOON_START:
        return "morning"
    if _AFTERNOON_START <= flight_time < _NIGHT_START:
        return "afternoon"
    return "night"


def is_high_season(date: str) -> int:
    """Return ``1`` when the scheduled date falls inside a high season window.

    Args:
        date: scheduled timestamp, formatted as :data:`DATE_FORMAT`.

    Returns:
        ``1`` for Dec 15 - Mar 3, Jul 15 - Jul 31 and Sep 11 - Sep 30, else ``0``.
    """
    scheduled_at = datetime.strptime(date, DATE_FORMAT)
    month_day = (scheduled_at.month, scheduled_at.day)
    return int(any(start <= month_day <= end for start, end in _HIGH_SEASON_RANGES))


def get_min_diff(data: Mapping[str, str]) -> float:
    """Return the minutes elapsed between the operated and the scheduled timestamp.

    Args:
        data: a flight row exposing ``Fecha-I`` and ``Fecha-O``.

    Returns:
        Positive when the flight was operated late, negative when ahead of schedule.
    """
    operated_at = datetime.strptime(data["Fecha-O"], DATE_FORMAT)
    scheduled_at = datetime.strptime(data["Fecha-I"], DATE_FORMAT)
    return (operated_at - scheduled_at).total_seconds() / 60


def _require_columns(data: pd.DataFrame, columns: Tuple[str, ...]) -> None:
    """Raise a descriptive ``ValueError`` when mandatory columns are absent."""
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


class DelayModel:
    """Predict whether a flight will be delayed by more than 15 minutes."""

    def __init__(self) -> None:
        self._model: Optional[LogisticRegression] = None  # Model should be saved in this attribute.

    def preprocess(
        self, data: pd.DataFrame, target_column: Optional[str] = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = self._build_features(data)
        if target_column is None:
            return features
        return features, self._build_target(data, target_column)

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        labels = self._as_labels(target)
        model = LogisticRegression(class_weight=self._balancing_weights(labels))
        model.fit(features[TOP_10_FEATURES], labels)
        self._model = model
        LOGGER.info(
            "delay model fitted", extra={"rows": len(features), "positives": int(labels.sum())}
        )

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        model = self._ensure_fitted()
        predictions = model.predict(features[TOP_10_FEATURES])
        return [int(prediction) for prediction in predictions]

    def save(self, path: Path = MODEL_ARTIFACT_PATH) -> Path:
        """Persist the fitted estimator so it can be served without retraining."""
        dump(self._ensure_fitted(), path)
        LOGGER.info("delay model artifact written", extra={"path": str(path)})
        return path

    def load(self, path: Path = MODEL_ARTIFACT_PATH) -> None:
        """Restore a previously persisted estimator into :attr:`_model`."""
        if not path.exists():
            raise FileNotFoundError(f"No model artifact at {path}. Run `make train` to build one.")
        self._model = load(path)
        LOGGER.info("delay model artifact loaded", extra={"path": str(path)})

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_features(data: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode the categorical inputs and project them on the trained set.

        Reindexing on :data:`TOP_10_FEATURES` serves two purposes: it drops the
        categories the estimator was not trained with, and it materializes the
        missing ones as zeros. That is what allows a three-column serving payload to
        produce exactly the same feature matrix the estimator saw during training.
        """
        _require_columns(data, REQUIRED_FEATURE_COLUMNS)
        encoded = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )
        return encoded.reindex(columns=TOP_10_FEATURES, fill_value=0).astype(int)

    @staticmethod
    def _build_target(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Return the binary target, deriving it from the timestamps when absent."""
        if target_column in data.columns:
            return data[[target_column]].astype(int)

        _require_columns(data, REQUIRED_TARGET_COLUMNS)
        min_diff = data.apply(get_min_diff, axis=1)
        delay = (min_diff > DELAY_THRESHOLD_IN_MINUTES).astype(int)
        return delay.to_frame(name=target_column)

    @staticmethod
    def _as_labels(target: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Accept either the single-column target frame or a plain series."""
        if isinstance(target, pd.DataFrame):
            if target.shape[1] != 1:
                raise ValueError(f"Expected a single target column, got {list(target.columns)}")
            return target.iloc[:, 0]
        return target

    @staticmethod
    def _balancing_weights(labels: pd.Series) -> Dict[int, float]:
        """Class weights that compensate the ~4.4:1 imbalance of the dataset.

        Reproduces the notebook's balancing (section 6.b.iii): the weight of each
        class is the frequency of the *opposite* class, which is what lifts the
        recall of the delayed flights the airport team cares about.
        """
        total = len(labels)
        negatives = int((labels == 0).sum())
        positives = int((labels == 1).sum())
        if positives == 0 or negatives == 0:
            return {0: 1.0, 1: 1.0}
        return {1: negatives / total, 0: positives / total}

    def _ensure_fitted(self) -> LogisticRegression:
        """Return the estimator, lazily restoring the champion artifact if needed."""
        if self._model is None:
            self.load()
        assert self._model is not None  # noqa: S101 - narrowed by `load` above
        return self._model
