"""Unit tests for the training pipeline's pure building blocks.

The MLflow side effects are exercised end to end by ``make train``; what is unit
tested here is the logic that decides *what* gets logged and promoted.
"""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from challenge.model import balancing_weights
from challenge.train import (
    SELECTION_METRIC,
    build_candidates,
    dataset_fingerprint,
    evaluate,
    git_commit,
)


class TestBalancingWeights:
    def test_weights_the_minority_class_up(self) -> None:
        labels = pd.Series([0] * 80 + [1] * 20)

        weights = balancing_weights(labels)

        assert weights[1] == pytest.approx(0.8)
        assert weights[0] == pytest.approx(0.2)
        assert weights[1] > weights[0]

    def test_stays_neutral_when_a_class_is_missing(self) -> None:
        assert balancing_weights(pd.Series([0, 0, 0])) == {0: 1.0, 1: 1.0}


class TestBuildCandidates:
    def test_isolates_class_balancing_as_the_only_difference(self) -> None:
        labels = pd.Series([0] * 80 + [1] * 20)

        candidates = build_candidates(labels)

        assert [candidate.name for candidate in candidates] == [
            "logreg_top10_balanced",
            "logreg_top10_unbalanced",
        ]
        algorithms = {type(candidate.estimator).__name__ for candidate in candidates}
        assert algorithms == {"LogisticRegression"}
        assert candidates[0].estimator.class_weight is not None
        assert candidates[1].estimator.class_weight is None


class TestEvaluate:
    def test_reports_the_metrics_the_selection_relies_on(self) -> None:
        features = pd.DataFrame({"x": [0, 0, 1, 1, 0, 1]})
        labels = pd.Series([0, 0, 1, 1, 0, 1])
        estimator = LogisticRegression().fit(features, labels)

        metrics = evaluate(estimator, features, labels)

        assert SELECTION_METRIC in metrics
        assert metrics["delayed_recall"] == pytest.approx(1.0)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert 0.0 <= metrics["roc_auc"] <= 1.0


class TestTraceability:
    def test_fingerprint_counts_rows_without_the_header(self, tmp_path: Path) -> None:
        dataset = tmp_path / "data.csv"
        dataset.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

        rows, checksum = dataset_fingerprint(dataset)

        assert rows == 2
        assert len(checksum) == 32

    def test_fingerprint_changes_when_the_data_changes(self, tmp_path: Path) -> None:
        first = tmp_path / "first.csv"
        second = tmp_path / "second.csv"
        first.write_text("a\n1\n", encoding="utf-8")
        second.write_text("a\n2\n", encoding="utf-8")

        assert dataset_fingerprint(first)[1] != dataset_fingerprint(second)[1]

    def test_git_commit_is_always_resolvable(self) -> None:
        commit = git_commit()

        assert commit
        assert commit == "unknown" or len(commit) == 40
