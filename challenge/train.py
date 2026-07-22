"""Training entrypoint: fits the delay model and persists the serving artifact.

Run it with ``make train``. Experiment tracking and model registry are added on
top of this pipeline in a follow-up ticket (MLE-003).
"""

import logging
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from challenge.model import MODEL_ARTIFACT_PATH, TARGET_COLUMN, DelayModel

LOGGER = logging.getLogger(__name__)

DATA_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "data" / "data.csv"
TEST_SIZE: Final[float] = 0.33
RANDOM_STATE: Final[int] = 42


def train(data_path: Path = DATA_PATH, artifact_path: Path = MODEL_ARTIFACT_PATH) -> Path:
    """Train the delay model on the historical dataset and persist the estimator.

    Args:
        data_path: CSV file with the raw historical flights.
        artifact_path: destination of the serialized champion estimator.

    Returns:
        The path of the written artifact.
    """
    data = pd.read_csv(data_path, low_memory=False)
    model = DelayModel()
    features, target = model.preprocess(data=data, target_column=TARGET_COLUMN)

    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model.fit(features=features_train, target=target_train)
    report = classification_report(
        target_test, model.predict(features=features_test), zero_division=0
    )
    LOGGER.info("holdout evaluation\n%s", report)

    return model.save(artifact_path)


def main() -> None:
    """Console entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    artifact = train()
    LOGGER.info("champion artifact available at %s", artifact)


if __name__ == "__main__":
    main()
