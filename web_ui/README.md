# Streamlit UI for Churn Model Demo

This small Streamlit app posts a single-customer payload to the FastAPI `/predict` endpoint in this repo.

Run the FastAPI server (from the `churn-model-demo` folder):

```bash
uvicorn src.churn_model.serve:app --reload --port 8000
```

Install and run the Streamlit app (from the repo root):

```bash
pip install streamlit requests
streamlit run churn-model-demo/web_ui/streamlit_app.py
```

The app will send a POST request to `http://localhost:8000/predict` by default — change the endpoint URL at the top of the app if needed.

Run with Docker Compose (build both services and the UI):

```bash
cd churn-model-demo
docker compose up --build
```

Tips:
- If you already have `model.pkl` locally, ensure it's placed at `churn-model-demo/model.pkl` so the compose file mounts it into the service.
- The Streamlit default `/healthz` path is not provided by Streamlit; healthcheck may fail until the UI is reachable. You can open http://localhost:8501 to verify.
