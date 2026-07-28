import pandas as pd
import bentoml

from model_loader import load_churn_model

@bentoml.service(resources={"cpu": "1"}, traffic={"timeout": 20})
class ChurnService:
    def __init__(self):
        self.model = load_churn_model()

    @bentoml.api
    def predict(self, tenure_months: float, monthly_charges: float, total_charges: float, num_support_tickets: int) -> dict:
        X = pd.DataFrame([{
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_tickets": num_support_tickets,
        }])
        pred = int(self.model.predict(X)[0])
        proba = float(self.model.predict_proba(X)[0][1])
        return {"churn_prediction": pred, "churn_probability": round(proba, 4)}
