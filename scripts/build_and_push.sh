#!/usr/bin/env bash
set -euo pipefail

# Usage: REGISTRY=ghcr.io/owner IMAGE_TAG=latest ./scripts/build_and_push.sh
REGISTRY=${REGISTRY:-}
IMAGE_TAG=${IMAGE_TAG:-latest}

echo "Building churn-service image..."
docker build -f Dockerfile.serve -t churn-service:${IMAGE_TAG} .

echo "Building streamlit-ui image..."
docker build -f Dockerfile.ui -t streamlit-ui:${IMAGE_TAG} .

if [ -n "${REGISTRY}" ]; then
  echo "Tagging and pushing to ${REGISTRY}..."
  docker tag churn-service:${IMAGE_TAG} ${REGISTRY}/churn-service:${IMAGE_TAG}
  docker tag streamlit-ui:${IMAGE_TAG} ${REGISTRY}/streamlit-ui:${IMAGE_TAG}
  docker push ${REGISTRY}/churn-service:${IMAGE_TAG}
  docker push ${REGISTRY}/streamlit-ui:${IMAGE_TAG}
else
  echo "REGISTRY not set; images built locally only."
fi
