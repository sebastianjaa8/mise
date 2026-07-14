#!/usr/bin/env bash
# Builds and deploys all three services to Cloud Run via Cloud Build
# (no local Docker needed — the build runs server-side).
#
# Usage: deploy/deploy.sh <gcp-project-id> [region]
set -euo pipefail

PROJECT_ID="${1:?Usage: deploy.sh <gcp-project-id> [region]}"
REGION="${2:-us-central1}"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com containerregistry.googleapis.com --project "$PROJECT_ID"

echo "== candidate-service =="
gcloud builds submit --config deploy/cloudbuild-candidate.yaml . --project "$PROJECT_ID"
gcloud run deploy mise-candidate \
  --image "gcr.io/$PROJECT_ID/mise-candidate" \
  --region "$REGION" --allow-unauthenticated \
  --memory 1Gi --min-instances 0 --max-instances 3 \
  --project "$PROJECT_ID"

echo "== ranking-service =="
gcloud builds submit --config deploy/cloudbuild-ranking.yaml . --project "$PROJECT_ID"
gcloud run deploy mise-ranking \
  --image "gcr.io/$PROJECT_ID/mise-ranking" \
  --region "$REGION" --allow-unauthenticated \
  --memory 512Mi --min-instances 0 --max-instances 3 \
  --project "$PROJECT_ID"

CANDIDATE_URL=$(gcloud run services describe mise-candidate --region "$REGION" --project "$PROJECT_ID" --format 'value(status.url)')
RANKING_URL=$(gcloud run services describe mise-ranking --region "$REGION" --project "$PROJECT_ID" --format 'value(status.url)')

echo "== gateway =="
gcloud builds submit --config deploy/cloudbuild-gateway.yaml . --project "$PROJECT_ID"
gcloud run deploy mise-gateway \
  --image "gcr.io/$PROJECT_ID/mise-gateway" \
  --region "$REGION" --allow-unauthenticated \
  --memory 256Mi --min-instances 0 --max-instances 5 \
  --set-env-vars "CANDIDATE_SERVICE_URL=$CANDIDATE_URL,RANKING_SERVICE_URL=$RANKING_URL" \
  --project "$PROJECT_ID"

GATEWAY_URL=$(gcloud run services describe mise-gateway --region "$REGION" --project "$PROJECT_ID" --format 'value(status.url)')
echo
echo "Deployed. Gateway URL: $GATEWAY_URL"
echo "Try:   curl \"$GATEWAY_URL/recommend?user_id=0&k=5\""
echo "Load test:  python scripts/load_test.py \"$GATEWAY_URL\""
