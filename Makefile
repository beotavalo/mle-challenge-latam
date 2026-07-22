.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")

.PHONY: help
help:             	## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: venv
venv:			## Create a virtual environment
	@echo "Creating virtualenv ..."
	@rm -rf .venv
	@python3 -m venv .venv
	@./.venv/bin/pip install -U pip
	@echo
	@echo "Run 'source .venv/bin/activate' to enable the environment"

.PHONY: install
install:		## Install dependencies
	pip install -r requirements-dev.txt
	pip install -r requirements-test.txt
	pip install -r requirements.txt

STRESS_URL = http://127.0.0.1:8000 
.PHONY: stress-test
stress-test:
	# change stress url to your deployed app 
	mkdir reports || true
	locust -f tests/stress/api_stress.py --print-stats --html reports/stress-test.html --run-time 60s --headless --users 100 --spawn-rate 1 -H $(STRESS_URL)

.PHONY: model-test
model-test:			## Run tests and coverage
	mkdir reports || true
	pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/model

.PHONY: api-test
api-test:			## Run tests and coverage
	mkdir reports || true
	pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/api

.PHONY: build
build:			## Build locally the python artifact
	python setup.py bdist_wheel

# --------------------------------------------------------------------------- #
# Quality gates and MLOps targets added for this delivery.
# Appended below on purpose: the STRESS_URL variable must stay on line 26.
# --------------------------------------------------------------------------- #

.PHONY: lint
lint:			## Check code style and lint rules (ruff)
	ruff check .
	ruff format --check .

.PHONY: format
format:			## Auto-format and auto-fix the codebase (ruff)
	ruff format .
	ruff check --fix .

.PHONY: typecheck
typecheck:		## Run static type checking (mypy)
	mypy

# Advisories reviewed and accepted, each one unreachable from this service.
# See docs/challenge.md ("Dependency vulnerabilities") for the per-advisory analysis.
# Anything NOT listed here fails the build, which is what makes this gate meaningful.
AUDIT_ACCEPTED = \
	--ignore-vuln PYSEC-2024-110 \
	--ignore-vuln PYSEC-2026-161 \
	--ignore-vuln PYSEC-2026-248 \
	--ignore-vuln PYSEC-2026-249 \
	--ignore-vuln PYSEC-2026-1941 \
	--ignore-vuln PYSEC-2026-1942 \
	--ignore-vuln PYSEC-2026-2280 \
	--ignore-vuln PYSEC-2026-2281 \
	--ignore-vuln PYSEC-2023-62 \
	--ignore-vuln PYSEC-2026-2151 \
	--ignore-vuln PYSEC-2026-1845

.PHONY: security
security:		## Scan dependencies for known vulnerabilities (pip-audit)
	pip-audit -r requirements.txt -r requirements-test.txt $(AUDIT_ACCEPTED)

.PHONY: security-report
security-report:	## Full vulnerability report, including the accepted advisories
	pip-audit -r requirements.txt -r requirements-test.txt --desc || true

.PHONY: train
train:			## Train the model, track the experiment in MLflow and refresh the serving artifact
	python -m challenge.train

.PHONY: mlflow-ui
mlflow-ui:		## Browse the versioned MLflow experiment tracking and model registry snapshot
	python scripts/mlflow_ui.py

.PHONY: serve
serve:			## Run the API locally on http://127.0.0.1:8000
	uvicorn challenge:application --host 0.0.0.0 --port 8000 --reload