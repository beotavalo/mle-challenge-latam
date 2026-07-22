#!/usr/bin/env bash
#
# One-time GCP bootstrap for the delivery pipeline.
#
# Creates everything `.github/workflows/cd.yml` expects: the Artifact Registry
# repository, a deployer service account, a least-privilege runtime service
# account, and a Workload Identity Federation provider scoped to this repository
# so GitHub Actions never needs a downloadable key.
#
# Usage:
#   PROJECT_ID=my-project GITHUB_REPOSITORY=owner/repo ./scripts/gcp_bootstrap.sh
#
# Requires: gcloud authenticated as a project owner, billing enabled.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
GITHUB_REPOSITORY="${GITHUB_REPOSITORY:?set GITHUB_REPOSITORY as owner/repo}"
REGION="${REGION:-us-central1}"
REPOSITORY="${REPOSITORY:-flight-delay}"
SERVICE="${SERVICE:-flight-delay-api}"
POOL="${POOL:-github-pool}"
PROVIDER="${PROVIDER:-github-provider}"
DEPLOYER_SA="${DEPLOYER_SA:-gha-deployer}"
RUNTIME_SA="${RUNTIME_SA:-flight-delay-runtime}"

gcloud config set project "${PROJECT_ID}"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"

echo "==> Enabling the APIs the pipeline uses"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com

echo "==> Creating the Artifact Registry repository"
gcloud artifacts repositories create "${REPOSITORY}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Flight delay prediction API images" 2>/dev/null || echo "    already exists"

echo "==> Creating the runtime service account (no roles: the API calls no GCP API)"
gcloud iam service-accounts create "${RUNTIME_SA}" \
  --display-name="Flight delay API runtime" 2>/dev/null || echo "    already exists"

echo "==> Creating the deployer service account used by GitHub Actions"
gcloud iam service-accounts create "${DEPLOYER_SA}" \
  --display-name="GitHub Actions deployer" 2>/dev/null || echo "    already exists"

DEPLOYER_EMAIL="${DEPLOYER_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
RUNTIME_EMAIL="${RUNTIME_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "==> Granting the deployer only what a release needs"
for role in roles/run.admin roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${DEPLOYER_EMAIL}" \
    --role="${role}" --condition=None >/dev/null
done
# Required so the deployer can attach the runtime identity to the service.
gcloud iam service-accounts add-iam-policy-binding "${RUNTIME_EMAIL}" \
  --member="serviceAccount:${DEPLOYER_EMAIL}" \
  --role="roles/iam.serviceAccountUser" >/dev/null

echo "==> Creating the Workload Identity pool and provider"
gcloud iam workload-identity-pools create "${POOL}" \
  --location=global --display-name="GitHub Actions" 2>/dev/null || echo "    pool already exists"

gcloud iam workload-identity-pools providers create-oidc "${PROVIDER}" \
  --location=global \
  --workload-identity-pool="${POOL}" \
  --display-name="GitHub OIDC" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository == '${GITHUB_REPOSITORY}'" \
  2>/dev/null || echo "    provider already exists"

POOL_ID="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}"

echo "==> Letting only this repository impersonate the deployer"
gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER_EMAIL}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_REPOSITORY}" >/dev/null

cat <<SUMMARY

============================================================
Bootstrap complete. Configure the repository as follows.

GitHub repository *variables* (Settings > Secrets and variables > Actions > Variables):
  GCP_PROJECT_ID           ${PROJECT_ID}
  GCP_REGION               ${REGION}
  GCP_ARTIFACT_REPOSITORY  ${REPOSITORY}
  CLOUD_RUN_SERVICE        ${SERVICE}
  GCP_RUNTIME_SERVICE_ACCOUNT ${RUNTIME_EMAIL}

GitHub repository *secrets*:
  GCP_WIF_PROVIDER         ${POOL_ID}/providers/${PROVIDER}
  GCP_SERVICE_ACCOUNT      ${DEPLOYER_EMAIL}

No key material is created or downloaded: GitHub mints a short-lived credential
per run through OIDC, restricted to ${GITHUB_REPOSITORY}.
============================================================
SUMMARY
