import os, time, pickle, logging
from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
app = FastAPI(title="churn-model-service")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
logger.info(f"Loaded model from {MODEL_PATH}")

PREDICTION_COUNT = Counter("churn_predictions_total", "Total predictions served", ["predicted_class"])
PREDICTION_LATENCY = Histogram("churn_prediction_latency_seconds", "Prediction latency in seconds")

class ChurnRequest(BaseModel):
    tenure_months: float
    monthly_charges: float
    total_charges: float
    num_support_tickets: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(req: ChurnRequest):
    start = time.perf_counter()
    X = [[req.tenure_months, req.monthly_charges, req.total_charges, req.num_support_tickets]]
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    PREDICTION_LATENCY.observe(time.perf_counter() - start)
    PREDICTION_COUNT.labels(predicted_class=str(pred)).inc()
    return {"churn_prediction": pred, "churn_probability": round(proba, 4)}
