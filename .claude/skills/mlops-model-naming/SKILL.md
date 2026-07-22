---
name: mlops-model-naming
description: >-
  Apply standardized MLOps naming and registry conventions when registering,
  versioning, aliasing, or referencing ML models in this project (MLflow model
  registry, challenge/train.py, model artifacts). Use whenever creating or
  renaming a registered model, bumping a model version, assigning a deployment
  stage/alias, or logging traceability metadata for a trained model.
---

# MLOps model naming & registry conventions

Production model names must communicate **family, version, and deployment status
at a glance**. Standardized naming prevents confusion during A/B testing,
rollback, and performance monitoring.

## Canonical structure

```
{domain}_{model_type}_v{MAJOR}_{MINOR}_{PATCH}_{stage}
```

| Component   | Meaning                                   | Examples                                   |
|-------------|-------------------------------------------|--------------------------------------------|
| `domain`    | Business use case / product area          | `flight_delay`, `fraud_detection`, `churn` |
| `model_type`| Architecture / key feature set            | `logreg`, `xgboost`, `distilbert`, `deepfm`|
| `version`   | SemVer tied to data/hyperparams/weights   | `v1_0_0`, `v3_2_1`                          |
| `stage`     | Lifecycle phase                           | `dev`, `staging`, `production`, `champion`  |

**Full human-readable example:** `flight_delay_logreg_v1_0_0_champion`

## Registry best practices (the rules that matter)

1. **Decouple the registered name from the stage.** Register a single artifact
   under `{domain}_{model_type}` (e.g. `flight_delay_logreg`) and apply the stage
   dynamically as an **alias** (`@champion`, `@challenger`, `@production`). Never
   bake the stage into the registered model name.
2. **Never use ambiguous names** like `model_v1_final` or `model_latest`.
3. **Automate versioning.** Let CI/CD auto-increment the **patch** when a
   retraining pipeline runs; bump **minor/major** for deliberate model changes.
4. **Tag for traceability**, don't hardcode into filenames. Store as registry /
   run tags: `semantic_version`, `git_commit`, `dataset_rows`, `dataset_md5`,
   `feature_set`, `framework`, and validation metrics.

## How this maps to this project (MLflow)

- **Registered model name** (stage-decoupled): `flight_delay_logreg`
- **SemVer**: tag `semantic_version=1.0.0` on the model version
- **Stage**: MLflow **alias** `@champion` (use `@challenger` for A/B candidates)
- **Run name** (human-readable full convention):
  `flight_delay_logreg_v1_0_0_champion`
- **Traceability tags**: set on both the run and the registered model version.

```python
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "flight_delay_logreg"   # {domain}_{model_type}
SEMANTIC_VERSION = "1.0.0"

mlflow.sklearn.log_model(model, artifact_path="model",
                         registered_model_name=REGISTERED_MODEL_NAME)

client = MlflowClient()
mv = max(client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"),
         key=lambda m: int(m.version))
client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version,
                             "semantic_version", SEMANTIC_VERSION)
client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", mv.version)
```

## When bumping versions

- **PATCH** (`v1_0_0` → `v1_0_1`): same code/features, retrained on refreshed data.
- **MINOR** (`v1_0_0` → `v1_1_0`): new features / non-breaking config change.
- **MAJOR** (`v1_0_0` → `v2_0_0`): new architecture or breaking I/O contract.

Keep the serving artifact (`challenge/model.joblib`) in sync with whatever
version carries the `@champion` alias.
