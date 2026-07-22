"""Unit tests for the feature engineering transcribed from the DS notebook.

These are the inner TDD loop: they pin down the behaviour of the pure feature
functions (including the boundary cases the original notebook silently dropped)
and the shape contract of ``DelayModel.preprocess`` in serving mode, where only
``OPERA``, ``TIPOVUELO`` and ``MES`` are available.
"""

import pandas as pd
import pytest

from challenge.model import (
    DELAY_THRESHOLD_IN_MINUTES,
    TOP_10_FEATURES,
    DelayModel,
    get_min_diff,
    get_period_day,
    is_high_season,
)


class TestGetPeriodDay:
    """The notebook used strict comparisons, so every boundary returned None."""

    @pytest.mark.parametrize(
        ("scheduled_at", "expected"),
        [
            ("2017-01-01 05:00:00", "morning"),
            ("2017-01-01 08:30:00", "morning"),
            ("2017-01-01 11:59:00", "morning"),
            ("2017-01-01 12:00:00", "afternoon"),
            ("2017-01-01 15:45:00", "afternoon"),
            ("2017-01-01 18:59:00", "afternoon"),
            ("2017-01-01 19:00:00", "night"),
            ("2017-01-01 23:59:00", "night"),
            ("2017-01-01 00:00:00", "night"),
            ("2017-01-01 04:59:00", "night"),
        ],
    )
    def test_returns_a_period_for_every_instant_of_the_day(
        self, scheduled_at: str, expected: str
    ) -> None:
        assert get_period_day(scheduled_at) == expected


class TestIsHighSeason:
    @pytest.mark.parametrize(
        "scheduled_at",
        [
            "2017-12-15 00:00:00",
            "2017-12-31 23:59:00",
            "2017-01-01 00:00:00",
            "2017-03-03 23:59:00",
            "2017-07-15 00:00:00",
            "2017-07-31 23:59:00",
            "2017-09-11 00:00:00",
            "2017-09-30 23:59:00",
        ],
    )
    def test_dates_inside_the_high_season_ranges(self, scheduled_at: str) -> None:
        assert is_high_season(scheduled_at) == 1

    @pytest.mark.parametrize(
        "scheduled_at",
        [
            "2017-03-04 00:00:00",
            "2017-05-20 12:00:00",
            "2017-07-14 23:59:00",
            "2017-09-10 23:59:00",
            "2017-12-14 23:59:00",
        ],
    )
    def test_dates_outside_the_high_season_ranges(self, scheduled_at: str) -> None:
        assert is_high_season(scheduled_at) == 0


class TestGetMinDiff:
    def test_returns_the_operation_delay_in_minutes(self) -> None:
        flight = pd.Series({"Fecha-I": "2017-01-01 23:30:00", "Fecha-O": "2017-01-01 23:33:00"})
        assert get_min_diff(flight) == pytest.approx(3.0)

    def test_supports_flights_operated_ahead_of_schedule(self) -> None:
        flight = pd.Series({"Fecha-I": "2017-01-01 23:30:00", "Fecha-O": "2017-01-01 23:25:00"})
        assert get_min_diff(flight) == pytest.approx(-5.0)


class TestPreprocessForServing:
    """A serving payload only carries OPERA, TIPOVUELO and MES."""

    @staticmethod
    def _serving_payload() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 3},
                {"OPERA": "Grupo LATAM", "TIPOVUELO": "I", "MES": 7},
            ]
        )

    def test_emits_the_training_feature_set_in_a_stable_order(self) -> None:
        features = DelayModel().preprocess(data=self._serving_payload())

        assert list(features.columns) == TOP_10_FEATURES
        assert features.shape == (2, len(TOP_10_FEATURES))

    def test_one_hot_encodes_only_the_known_categories(self) -> None:
        features = DelayModel().preprocess(data=self._serving_payload())

        latam = features.iloc[1]
        assert latam["OPERA_Grupo LATAM"] == 1
        assert latam["TIPOVUELO_I"] == 1
        assert latam["MES_7"] == 1

    def test_categories_outside_the_feature_set_become_an_all_zero_row(self) -> None:
        features = DelayModel().preprocess(data=self._serving_payload())

        # "Aerolineas Argentinas", national, March: none of them is a top-10 feature.
        assert features.iloc[0].sum() == 0

    def test_rejects_payloads_missing_a_required_column(self) -> None:
        payload = pd.DataFrame([{"OPERA": "Grupo LATAM", "MES": 3}])

        with pytest.raises(ValueError, match="TIPOVUELO"):
            DelayModel().preprocess(data=payload)


class TestPreprocessForTraining:
    @staticmethod
    def _training_payload() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "OPERA": "Grupo LATAM",
                    "TIPOVUELO": "I",
                    "MES": 7,
                    "Fecha-I": "2017-07-01 10:00:00",
                    "Fecha-O": "2017-07-01 10:16:00",
                },
                {
                    "OPERA": "Sky Airline",
                    "TIPOVUELO": "N",
                    "MES": 4,
                    "Fecha-I": "2017-04-01 10:00:00",
                    "Fecha-O": "2017-04-01 10:15:00",
                },
            ]
        )

    def test_derives_the_delay_target_from_the_operation_delay(self) -> None:
        features, target = DelayModel().preprocess(
            data=self._training_payload(), target_column="delay"
        )

        assert list(features.columns) == TOP_10_FEATURES
        assert list(target.columns) == ["delay"]
        # Exactly DELAY_THRESHOLD_IN_MINUTES minutes late is still on time.
        assert target["delay"].tolist() == [1, 0]

    def test_uses_an_existing_target_column_when_already_present(self) -> None:
        payload = self._training_payload().assign(delay=[0, 1])

        _, target = DelayModel().preprocess(data=payload, target_column="delay")

        assert target["delay"].tolist() == [0, 1]

    def test_threshold_matches_the_business_definition(self) -> None:
        assert DELAY_THRESHOLD_IN_MINUTES == 15


class TestFitAndPredict:
    @staticmethod
    def _training_frame() -> pd.DataFrame:
        # Delayed international LATAM flights vs. on-time national ones.
        delayed = [
            {
                "OPERA": "Grupo LATAM",
                "TIPOVUELO": "I",
                "MES": 7,
                "Fecha-I": "2017-07-01 10:00:00",
                "Fecha-O": "2017-07-01 11:00:00",
            }
        ] * 20
        on_time = [
            {
                "OPERA": "Sky Airline",
                "TIPOVUELO": "N",
                "MES": 5,
                "Fecha-I": "2017-05-01 10:00:00",
                "Fecha-O": "2017-05-01 10:01:00",
            }
        ] * 20
        return pd.DataFrame(delayed + on_time)

    def test_fit_stores_the_estimator_in_the_provided_attribute(self) -> None:
        model = DelayModel()
        features, target = model.preprocess(data=self._training_frame(), target_column="delay")

        model.fit(features=features, target=target)

        assert model._model is not None

    def test_predict_returns_native_python_integers(self) -> None:
        model = DelayModel()
        features, target = model.preprocess(data=self._training_frame(), target_column="delay")
        model.fit(features=features, target=target)

        predictions = model.predict(features=features)

        assert isinstance(predictions, list)
        assert len(predictions) == len(features)
        assert all(type(prediction) is int for prediction in predictions)
        assert set(predictions) <= {0, 1}
