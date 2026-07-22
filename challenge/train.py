"""Training pipeline with MLflow experiment tracking and model registry.

Run it with ``make train``. Every execution:

1. rebuilds the features and the target from the historical dataset,
2. trains the candidate estimators that back the model-selection argument,
3. logs parameters, metrics and artifacts of each candidate as an MLflow run,
4. registers the winner as ``flight_delay_logreg`` and moves the ``@champion``
   alias to it, tagging the version for traceability,
5. exports that champion to ``challenge/model.joblib``, which is what the API serves.

The tracking store is the ``mlruns/`` directory at the repository root, so the whole
experiment history travels with the repository and needs no server to inspect.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import mlflow
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from challenge.model import (
    MODEL_ARTIFACT_PATH,
    TARGET_COLUMN,
    TOP_10_FEATURES,
    DelayModel,
    balancing_weights,
)

LOGGER = logging.getLogger(__name__)

REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_PATH: Final[Path] = REPOSITORY_ROOT / "data" / "data.csv"
TRACKING_DIR: Final[Path] = REPOSITORY_ROOT / "mlruns"

EXPERIMENT_NAME: Final[str] = "flight_delay_prediction"

#: Stage-decoupled registered model name: {domain}_{model_type}. The deployment
#: stage is an alias, never part of the name.
REGISTERED_MODEL_NAME: Final[str] = "flight_delay_logreg"
CHAMPION_ALIAS: Final[str] = "champion"
SEMANTIC_VERSION: Final[str] = "1.0.0"

TEST_SIZE: Final[float] = 0.33
RANDOM_STATE: Final[int] = 42

#: Metric that decides which candidate is promoted. The airport team's cost of a
#: missed delay is higher than the cost of a false alarm, so the delayed class
#: drives the decision.
SELECTION_METRIC: Final[str] = "delayed_recall"


@dataclass(frozen=True)
class Candidate:
    """A model variant competing for the champion alias."""

    name: str
    estimator: BaseEstimator
    params: dict[str, object]
    description: str


@dataclass(frozen=True)
class Evaluation:
    """Metrics and run identity of an evaluated candidate."""

    candidate: Candidate
    metrics: dict[str, float]
    run_id: str


def dataset_fingerprint(path: Path) -> tuple[int, str]:
    """Return the row count and the MD5 checksum of the dataset file.

    The checksum is a data-versioning fingerprint for traceability only; it is not
    used as a security control.
    """
    digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    with path.open(encoding="utf-8") as handle:
        rows = sum(1 for _ in handle) - 1  # discount the header
    return rows, digest


def git_commit() -> str:
    """Return the current commit sha without shelling out.

    Falls back to the CI-provided environment variable and finally to ``unknown``,
    so training never fails because of missing version-control metadata.
    """
    head = REPOSITORY_ROOT / ".git" / "HEAD"
    if head.exists():
        content = head.read_text(encoding="utf-8").strip()
        if content.startswith("ref: "):
            ref = REPOSITORY_ROOT / ".git" / content.removeprefix("ref: ")
            if ref.exists():
                return ref.read_text(encoding="utf-8").strip()
        else:
            return content
    return os.environ.get("GITHUB_SHA", "unknown")


def evaluate(
    estimator: BaseEstimator, features: pd.DataFrame, labels: pd.Series
) -> dict[str, float]:
    """Score an estimator on a holdout split.

    Args:
        estimator: fitted classifier exposing ``predict`` and ``predict_proba``.
        features: holdout features.
        labels: holdout binary target.

    Returns:
        Flat metric mapping, ready to be logged to MLflow.
    """
    predictions = estimator.predict(features)
    probabilities = estimator.predict_proba(features)[:, 1]
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return {
        "accuracy": float(report["accuracy"]),
        "on_time_precision": float(report["0"]["precision"]),
        "on_time_recall": float(report["0"]["recall"]),
        "on_time_f1": float(report["0"]["f1-score"]),
        "delayed_precision": float(report["1"]["precision"]),
        "delayed_recall": float(report["1"]["recall"]),
        "delayed_f1": float(report["1"]["f1-score"]),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
    }


def build_candidates(labels: pd.Series) -> list[Candidate]:
    """Build the candidates that reproduce the notebook's model-selection argument.

    The notebook concludes that class balancing is what makes the model usable. Both
    candidates therefore share features and algorithm and differ only in balancing,
    so the experiment measures exactly that claim.
    """
    weights = balancing_weights(labels)
    return [
        Candidate(
            name="logreg_top10_balanced",
            estimator=LogisticRegression(class_weight=weights),
            params={"class_weight": json.dumps({str(k): round(v, 6) for k, v in weights.items()})},
            description="Logistic regression on the top 10 features with class balancing",
        ),
        Candidate(
            name="logreg_top10_unbalanced",
            estimator=LogisticRegression(),
            params={"class_weight": "none"},
            description="Same model without class balancing (baseline)",
        ),
    ]


def _log_candidate(
    candidate: Candidate,
    features_train: pd.DataFrame,
    labels_train: pd.Series,
    features_test: pd.DataFrame,
    labels_test: pd.Series,
    shared_tags: dict[str, str],
) -> Evaluation:
    """Train, evaluate and record a single candidate as an MLflow run."""
    with mlflow.start_run(run_name=candidate.name) as run:
        candidate.estimator.fit(features_train, labels_train)
        metrics = evaluate(candidate.estimator, features_test, labels_test)

        mlflow.set_tags(
            {
                **shared_tags,
                "candidate": candidate.name,
                "mlflow.note.content": candidate.description,
            }
        )
        mlflow.log_params(
            {
                "algorithm": type(candidate.estimator).__name__,
                "feature_set": "top_10_feature_importance",
                "n_features": len(TOP_10_FEATURES),
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                **candidate.params,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_text(
            classification_report(
                labels_test, candidate.estimator.predict(features_test), zero_division=0
            ),
            "classification_report.txt",
        )
        mlflow.log_text(
            str(confusion_matrix(labels_test, candidate.estimator.predict(features_test))),
            "confusion_matrix.txt",
        )
        LOGGER.info("candidate %s -> %s", candidate.name, metrics)
        return Evaluation(candidate=candidate, metrics=metrics, run_id=run.info.run_id)


def _register_champion(
    champion: Evaluation,
    features_train: pd.DataFrame,
    labels_train: pd.Series,
    features_test: pd.DataFrame,
    tags: dict[str, str],
) -> str:
    """Log, register and alias the champion model, then return its version.

    The model is retrained through :class:`DelayModel` so the registered artifact is
    byte-for-byte the object the API serves, not a look-alike built in the pipeline.
    """
    model = DelayModel()
    model.fit(features=features_train, target=labels_train.to_frame(name=TARGET_COLUMN))

    with mlflow.start_run(run_id=champion.run_id):
        mlflow.sklearn.log_model(
            sk_model=model.estimator,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=infer_signature(features_test, model.estimator.predict(features_test)),
        )

    client = MlflowClient()
    version = max(
        client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"),
        key=lambda candidate: int(candidate.version),
    )
    for key, value in {**tags, "semantic_version": SEMANTIC_VERSION}.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, key, value)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version)

    model.save()
    LOGGER.info(
        "registered %s v%s as @%s and exported %s",
        REGISTERED_MODEL_NAME,
        version.version,
        CHAMPION_ALIAS,
        MODEL_ARTIFACT_PATH,
    )
    return str(version.version)


def train(data_path: Path = DATA_PATH, tracking_dir: Path = TRACKING_DIR) -> Evaluation:
    """Run the full training pipeline and promote the best candidate.

    Args:
        data_path: CSV file with the raw historical flights.
        tracking_dir: MLflow file-based tracking and registry store.

    Returns:
        The evaluation of the promoted champion.
    """
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = pd.read_csv(data_path, low_memory=False)
    features, target = DelayModel().preprocess(data=data, target_column=TARGET_COLUMN)
    labels = target[TARGET_COLUMN]

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    rows, checksum = dataset_fingerprint(data_path)
    shared_tags = {
        "git_commit": git_commit(),
        "dataset_rows": str(rows),
        "dataset_md5": checksum,
        "feature_set": ",".join(TOP_10_FEATURES),
        "framework": f"scikit-learn=={sklearn.__version__}",
        "semantic_version": SEMANTIC_VERSION,
    }

    evaluations = [
        _log_candidate(
            candidate, features_train, labels_train, features_test, labels_test, shared_tags
        )
        for candidate in build_candidates(labels_train)
    ]
    champion = max(evaluations, key=lambda evaluation: evaluation.metrics[SELECTION_METRIC])
    LOGGER.info(
        "champion: %s (%s=%.4f)",
        champion.candidate.name,
        SELECTION_METRIC,
        champion.metrics[SELECTION_METRIC],
    )

    _register_champion(champion, features_train, labels_train, features_test, shared_tags)
    return champion


def main() -> None:
    """Console entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    champion = train()
    LOGGER.info("training finished, champion metrics: %s", champion.metrics)


if __name__ == "__main__":
    main()
