# Deployment & Publishing

This document explains how to build and publish the Docker images produced by this project.

Publish images to GitHub Container Registry (GHCR)
- Create a Personal Access Token (PAT) with `write:packages` and `delete:packages` scope.
- In the repository settings on GitHub, go to `Settings → Secrets and variables → Actions → New repository secret` and add a secret named `GHCR_PAT` with the PAT value.

When `GHCR_PAT` is set, the CI workflow `.github/workflows/ci.yml` will push two images to GHCR:
- `ghcr.io/<owner>/churn-service:latest` — the FastAPI service image built from `Dockerfile.serve`
- `ghcr.io/<owner>/streamlit-ui:latest` — the Streamlit UI image built from `Dockerfile.ui`

Manually build and push images locally
- Build locally and optionally push to a registry using the included scripts.

Bash (Linux/macOS/WSL):
```bash
cd churn-model-demo
REGISTRY=ghcr.io/your-username ./scripts/build_and_push.sh
```

PowerShell (Windows):
```powershell
cd churn-model-demo
$env:REGISTRY = 'ghcr.io/your-username'
.
\scripts\build_and_push.ps1
```

Notes
- Ensure `model.pkl` is present at `churn-model-demo/model.pkl` before running `docker compose up` locally — the Compose file mounts this file into the service container.
- The GH Actions workflow builds images but only pushes them if `GHCR_PAT` is present in repository secrets. You can monitor builds under `Actions` in your GitHub repo.
