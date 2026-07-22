# Software Engineer (ML & LLMs) Challenge — Solution

Operationalization of the Data Scientist's flight delay model for SCL airport: a tested
`DelayModel`, a validated FastAPI service, a container released to Cloud Run by CI/CD, and MLflow
experiment tracking with a model registry.

- **Repository**: <https://github.com/beotavalo/mle-challenge-latam>
- **API**: see `STRESS_URL` in the `Makefile` (line 26)
- **Model in production**: `flight_delay_logreg` version 1, alias `@champion`

---

## 1. How to run it

```bash
uv venv --python 3.10                       # last interpreter with wheels for the pinned numpy/pandas
uv pip install -r requirements.txt -r requirements-dev.txt -r requirements-test.txt
source .venv/bin/activate

make model-test      # Part I  — model acceptance tests + coverage
make api-test        # Part II — API acceptance tests + coverage
make stress-test     # Part III — load test against STRESS_URL
make lint typecheck security   # quality gates, the same ones CI runs
make train           # retrain, track the experiment, refresh the champion artifact
make mlflow-ui       # browse the versioned experiment history and the model registry
make serve           # run the API locally on :8000
```

---

## 2. Part I — The model

### 2.1 Bugs found and fixed

| # | Where | Defect | Fix |
|---|---|---|---|
| 1 | `challenge/model.py:16` | `Union(...)` was **called** instead of subscripted, raising `TypeError` on import. The module could not be imported at all. | `Union[...]`. |
| 2 | Notebook `get_period_day` | Strict `<`/`>` comparisons left every boundary instant unclassified: 05:00, 11:59, 12:00, 18:59, 19:00, 23:59, 00:00 and 04:59 all returned `None`. | Half-open intervals covering the full 24 h, pinned by a parametrized test over each boundary. |
| 3 | Notebook `get_rate_from_column` | Computes `total / delays`, the **inverse** of a rate, and shadows the loop variable `total` inside its own loop. Every "delay rate" chart in the notebook is therefore inverted. | Reported, not transcribed: the metric is exploratory and does not reach production code. |
| 4 | `tests/model/test_model.py:31` | Read `"../data/data.csv"`. `make model-test` runs pytest from the repository root, so the path resolved **outside** the repository and every test errored in `setUp` — locally and in CI. | Path anchored to the test file with `Path(__file__)`. No assertion was touched. |
| 5 | `challenge/api.py` | `post_predict` returned `None`, with no request model and no validation. | Implemented, see Part II. |
| 6 | `Makefile` | `--cov-config=.coveragerc` pointed at a file that did not exist. | Added `.coveragerc`. |
| 7 | `requirements-test.txt` | `locust~=1.6` pulls `flask 1.1.2`, which does `from jinja2 import escape` — removed in Jinja2 3.1. `make stress-test` died at startup on any environment installed today. | `locust~=2.32`. The provided stress script needs no change. |
| 8 | `requirements.txt` | `anyio` 4.x ships a pytest plugin that imports `_pytest.scope` (pytest ≥ 7) and crashed collection under the pinned pytest 6.2.5. | Pinned `anyio~=3.7.1`, inside the range starlette accepts. |
| 9 | Repository | `__MACOSX/` and four `.DS_Store` files were committed from the challenge zip. | Removed and ignored. |
| 10 | `mlruns/` (self-inflicted, MLE-003) | MLflow's file store writes **absolute** `file://` URIs into its metadata, so the versioned snapshot pointed at the machine that produced it. Training into it from anywhere else made MLflow resolve a foreign path and try to create `/C:` on the Linux runner. | `challenge/train.py` rehomes the store to the current checkout before training; `scripts/mlflow_ui.py` reuses that function instead of keeping its own copy. |
| 11 | CI toolchain | `pip-audit` resolves the declared requirement ranges inside a throwaway virtualenv, which needs a working `ensurepip`. uv's standalone Linux interpreter ships without the bundled pip wheel, so `make security` died before auditing anything. | The security job bases its environment on a stock CPython; every other job keeps uv's. Auditing the *declared* set rather than the installed one keeps the accept list meaningful and transitive coverage intact. |

Defects 10 and 11 surfaced only when the gates first ran off the machine that wrote
them, on the first real CI execution. Both are recorded here rather than quietly fixed:
a quality gate that passes only on its author's laptop is not a gate, and that failure
mode is worth naming.

### 2.2 Which model, and why

The DS trained six variants and concluded that XGBoost and Logistic Regression perform equivalently,
that the top-10 feature subset loses nothing, and that class balancing is what matters — without
picking a winner. The pipeline in `challenge/train.py` re-runs that comparison and records it, so the
decision rests on logged metrics rather than on the notebook's prose.

Holdout (33 %, `random_state=42`), both candidates on the top-10 features:

| Metric | `logreg_top10_balanced` (champion) | `logreg_top10_unbalanced` |
|---|---|---|
| Accuracy | 0.550 | **0.813** |
| Delayed — recall | **0.688** | 0.013 |
| Delayed — precision | 0.248 | 0.529 |
| Delayed — F1 | **0.364** | 0.025 |
| On-time — recall | 0.519 | 0.997 |
| On-time — F1 | 0.652 | 0.897 |
| ROC-AUC | 0.640 | 0.640 |
| PR-AUC | 0.283 | 0.282 |

The unbalanced model looks better on accuracy and is useless: it finds 1.3 % of the delays. Identical
ROC-AUC on both rows shows the two models rank flights the same way — balancing moves the decision
threshold, it does not add signal. The airport team pays more for a missed delay than for a false
alarm, so the promoted model is selected by **recall on the delayed class** (`SELECTION_METRIC` in
`challenge/train.py`).

**Logistic Regression over XGBoost**, at equal measured performance:

- **No new dependency.** `xgboost` is not in `requirements.txt`; scikit-learn already is. The runtime
  image stays small and cold starts stay fast on Cloud Run — where every dependency is paid for on
  every scale-up.
- **Deterministic and inspectable.** Ten coefficients an operations team can read, versus a tree
  ensemble, for a model whose output justifies staffing decisions.
- **Cheaper inference.** A dot product of ten terms per flight.

This is not a claim that logistic regression is the better learner — the challenge explicitly excludes
model improvement. It is the cheaper of two statistically indistinguishable options.

### 2.3 Design

`challenge/model.py` separates two things that change for different reasons:

- **Feature engineering** — `get_period_day`, `is_high_season`, `get_min_diff`, `balancing_weights`:
  pure functions, no estimator, unit tested in isolation.
- **`DelayModel`** — orchestration only. The provided signatures of `preprocess`, `fit` and `predict`
  and the `_model` attribute are untouched; `estimator`, `save` and `load` are additions.

Three details worth calling out:

- **`preprocess` handles both worlds.** Training data carries 18 columns and no target; a serving
  payload carries three. One-hot encoding followed by `reindex(columns=TOP_10_FEATURES, fill_value=0)`
  drops unknown categories and materializes missing ones as zeros, so both inputs produce the same
  10-column matrix in the same order the estimator was trained with.
- **The target is derived, not read.** `data.csv` has no `delay` column, so `preprocess` computes it
  from `min_diff > 15` when asked for a target that is not present.
- **`predict` works on a model that was never `fit`.** The provided test constructs a `DelayModel` and
  predicts immediately, so `predict` lazily restores `challenge/model.joblib` — the artifact that
  corresponds to the `@champion` registry version.

`predict` returns native Python `int`s: `numpy.int64` is not an `int`, and the provided test checks the
type.

---

## 3. Part II — The API

### 3.1 Contract

| Endpoint | Behaviour |
|---|---|
| `GET /health` | `200 {"status": "OK"}`. Used by the container healthcheck and the deployment smoke test. |
| `POST /predict` | `200 {"predict": [...]}` — one integer per submitted flight, in order. `1` means delayed more than 15 minutes. |
| `GET /docs`, `GET /openapi.json` | Generated OpenAPI documentation with request and response examples. |

### 3.2 Validation, and why 400 instead of 422

An unknown airline is not a prediction problem, it is a client error: the model has no coefficient for
it, so scoring it would return a confident-looking number derived from an all-zero row. Every field is
therefore validated against the categories present in the training data — the airline catalogue lives
in `challenge/model.py` next to the feature list, so serving and training cannot disagree.

FastAPI answers validation failures with `422` by default. The challenge's acceptance tests require
`400`, which is also the more honest answer for a semantic rejection, so a `RequestValidationError`
handler maps them; a `ValueError` handler covers domain-level rejections raised further down.

Rejected: month outside 1–12, flight type outside `{I, N}`, airline absent from the training data,
missing fields, and empty batches.

### 3.3 Operational properties

- The champion estimator is loaded **once** at startup and reused, so `/predict` never touches disk.
- Logs are one JSON object per line — Cloud Run ingests stdout directly, so every line is queryable
  without adding a logging dependency to the runtime image.
- Neither logs nor error bodies echo submitted values: the error payload reports **which field**
  failed, never its content.

---

## 4. Part III — Deployment

```
GitHub (main) ──> Actions (OIDC, no keys) ──> Artifact Registry ──> Cloud Run (public)
```

`scripts/gcp_bootstrap.sh` provisions the target idempotently: APIs, the Artifact Registry repository,
a deployer service account limited to `run.admin` + `artifactregistry.writer`, a runtime service
account with **no roles at all** (the API calls no GCP API), and a Workload Identity Federation
provider whose attribute condition pins the principal to this repository. It prints the exact
repository variables and secrets to configure. No key material is ever created or downloaded.

Revision envelope: 1 vCPU, 512 MiB, concurrency 80, max 10 instances, 30 s request timeout, listening
on the injected `$PORT`. Images are tagged with the commit sha, so every revision is traceable back to
the code that produced it.

### Load test

Local baseline, single uvicorn process, 100 users for 60 s (`make stress-test` against
`http://127.0.0.1:8000`):

| Requests | Failures | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|---|
| 11 126 | **0 (0.00 %)** | 160 ms | 260 ms | 320 ms | ~193 req/s |

The same run is executed by the delivery pipeline against the released URL after every deployment, and
its HTML report is published as a build artifact.

---

## 5. Part IV — CI/CD

The provided `workflows/` folder is copied into `.github/workflows/` as the instructions ask; the
original folder is left untouched so the delivered structure is preserved.

**`ci.yml`** — on every pull request and push to `main`/`develop`, four parallel jobs running exactly
what a developer runs locally:

| Job | Gate |
|---|---|
| quality | `make lint` (ruff lint + format) and `make typecheck` (mypy strict) |
| security | `make security` (pip-audit, fails on any advisory outside the reviewed accept list) plus a full always-on report |
| test | `make model-test`, `make api-test`, then `make train` to prove the pipeline still reproduces the champion; coverage, junit and MLflow runs published as artifacts |
| container | builds the runtime image and smoke tests `/health` and `/predict` against the running container |

**`cd.yml`** — on every merge to `main`: authenticate via WIF, build and publish the image, release the
Cloud Run revision, smoke test the released URL (health, a real prediction, and a malformed payload
that must still be refused with 400), then load test it and publish the report.

Environments are provisioned with `uv` and cached on `requirements*.txt`.

---

## 6. MLOps: experiment tracking and model registry

`make train` is the only way a model is produced, and it records everything:

- **Tracking** — one MLflow run per candidate with parameters, per-class precision/recall/F1, ROC-AUC,
  PR-AUC, the classification report and the confusion matrix as artifacts.
- **Registry** — the winner is registered as `flight_delay_logreg`. The name is **stage-decoupled**:
  the deployment stage is the alias `@champion`, never part of the name, so promoting a challenger is
  an alias move rather than a rename. Run names follow the readable convention
  `flight_delay_logreg_v1_0_0_champion`.
- **Traceability tags** on the registered version: `semantic_version`, `git_commit`, `dataset_rows`,
  `dataset_md5`, `feature_set`, `framework`.
- **No drift between registry and serving**: the registered estimator is the object returned by
  `DelayModel.fit`, and that same object is exported to `challenge/model.joblib`.

The `mlruns/` file store is versioned with the repository (155 KB), so the experiment history and the
registry travel with the code and need no server. MLflow writes absolute artifact URIs into its
metadata, so `make mlflow-ui` rehomes them to the current checkout before launching the UI — the
snapshot therefore opens on any machine without retraining.

Versioning policy: PATCH for a retrain on refreshed data, MINOR for a new feature set or configuration
change, MAJOR for a new architecture or a breaking I/O contract.

---

## 7. Security

### 7.1 Dependency vulnerabilities

`pip-audit` reported **16 advisories** against the stack as delivered. `fastapi 0.86.0` pins
`starlette==0.20.4` exactly, so the only way to clear the reachable ones was to move fastapi forward:
**`fastapi ~=0.115`**, which keeps Pydantic v1 — the API code is unchanged — and brings starlette
0.46.2. That removed the fastapi ReDoS (PYSEC-2024-38), the unbounded multipart DoS (PYSEC-2023-48)
and the StaticFiles path traversal (PYSEC-2023-83). Bumping locust removed two more.

What remains is accepted, each one unreachable from this service:

| Advisory | Package | Why it cannot be reached here |
|---|---|---|
| PYSEC-2026-161, PYSEC-2026-248 | starlette | Host/path injection into the reconstructed `request.url`. The service never builds URLs from requests, and the validation handlers log a constant endpoint label instead of `request.url.path`, removing the last touchpoint. |
| PYSEC-2026-249, PYSEC-2026-1941 | starlette | Both require `request.form()` / multipart parsing. The API accepts JSON only. |
| PYSEC-2026-1942 | starlette | Quadratic Range-header parsing in `FileResponse`. The API serves no files. |
| PYSEC-2026-2281 | starlette | SSRF via UNC paths in `StaticFiles` **on Windows**. No `StaticFiles`, and the runtime is a Linux container. |
| PYSEC-2026-2280 | starlette | Method dispatch in `HTTPEndpoint` classes. All routes are decorator-based. |
| PYSEC-2024-110 | scikit-learn | Data leakage in `TfidfVectorizer`. Not used; the model consumes ten binary features. |
| PYSEC-2026-1845 | pytest | Test-only dependency. It never enters the runtime image, which installs `requirements.txt` alone. |

The accept list lives in the `Makefile` (`AUDIT_ACCEPTED`), so **anything not on it fails the build** —
that is what keeps the gate meaningful. `make security-report` prints the unfiltered picture.

### 7.2 Other controls

- **Input validation at the boundary** — no unvalidated value reaches the model (OWASP A03).
- **No secrets in the repository** — cloud auth is OIDC/WIF with short-lived credentials; there is no
  service account key to leak (OWASP A07). `.gitignore` blocks `.env`, `*.pem` and credential files.
- **Least privilege** — the runtime service account holds no roles; the deployer holds two.
- **Non-root container**, minimal image: no dataset, no notebook, no training code.
- **No sensitive data in logs** — field locations only, never values.
- **Automated scanning on every pull request**, not as a release-time afterthought.

---

## 8. Engineering process

- **Spec-driven development** with GitHub Spec Kit: a ratified project constitution (six principles —
  contract fidelity, test-first, clean code under automated gates, secure by default, MLOps
  traceability, automation) and a feature specification with numbered functional requirements drove the
  work. The generated `.specify/` and `specs/` folders are not versioned; their content is distilled
  into this document.
- **TDD** — failing test first, in both loops: the provided acceptance tests as the outer loop, 62 unit
  and contract tests written for this delivery as the inner one. **70 tests, all green.**
- **GitFlow** — `feature/*` → `develop` → `main`, one branch per ticket, **no branch deleted**, so the
  delivery order stays auditable.
- **Conventional Commits**, with the rationale in the body rather than in a changelog nobody reads.

| Ticket | Branch | Delivered |
|---|---|---|
| MLE-001 | `feature/mle-001-project-setup` | Tooling, quality-gate configuration, repository hygiene |
| MLE-002 | `feature/mle-002-delay-model` | `DelayModel` (Part I) |
| MLE-003 | `feature/mle-003-mlflow-tracking` | MLflow tracking and model registry |
| MLE-004 | `feature/mle-004-prediction-api` | FastAPI service (Part II) |
| MLE-005 | `feature/mle-005-dependency-hardening` | Vulnerability remediation |
| MLE-006 | `feature/mle-006-container-build` | Runtime image |
| MLE-007 | `feature/mle-007-continuous-integration` | `ci.yml` (Part IV) |
| MLE-008 | `feature/mle-008-continuous-delivery` | `cd.yml` + GCP bootstrap (Parts III and IV) |
| MLE-009 | `feature/mle-009-stress-test-fix` | Working load test |
| MLE-010 | `feature/mle-010-documentation` | This document |

Partially generated with GenAI (Claude Code) and reviewed for accuracy.

---

## 9. Deliberate deviations

| Decision | Reason |
|---|---|
| `post_predict()` gained a request body parameter | No prediction endpoint can exist without one. The `DelayModel` signatures the challenge protects are untouched. |
| The data path in `tests/model/test_model.py` was corrected | The provided path cannot resolve when `make model-test` runs from the repository root, so the suite could not run at all. Assertions are untouched. |
| `fastapi`, `anyio` and `locust` moved off their delivered pins | One was a security decision, two were outright breakages on a fresh install. Every other pin — numpy, pandas, scikit-learn, pydantic, uvicorn, pytest — is exactly as delivered. |
| `data/data.csv` and `mlruns/` are versioned in-repo | Team practice puts large data and model files in submodules. Here the dataset ships with the challenge, and the MLflow snapshot is 155 KB whose entire purpose is to be readable by the reviewer without infrastructure. |
| Provided test files are excluded from `ruff format` | Keeps the reviewer's diff on those files limited to the one documented fix instead of formatting churn. |
