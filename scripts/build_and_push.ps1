param(
  [string] $Registry = $env:REGISTRY,
  [string] $ImageTag = $env:IMAGE_TAG
)

if (-not $ImageTag) { $ImageTag = 'latest' }

Write-Host "Building churn-service image..."
docker build -f Dockerfile.serve -t "churn-service:$ImageTag" .

Write-Host "Building streamlit-ui image..."
docker build -f Dockerfile.ui -t "streamlit-ui:$ImageTag" .

if ($Registry) {
  Write-Host "Tagging and pushing to $Registry..."
  docker tag "churn-service:$ImageTag" "$Registry/churn-service:$ImageTag"
  docker tag "streamlit-ui:$ImageTag" "$Registry/streamlit-ui:$ImageTag"
  docker push "$Registry/churn-service:$ImageTag"
  docker push "$Registry/streamlit-ui:$ImageTag"
} else {
  Write-Host "REGISTRY not set; images built locally only."
}
