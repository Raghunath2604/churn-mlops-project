import pickle
import bentoml

@bentoml.service(resources={"cpu": "1"}, traffic={"timeout": 20})
class ChurnService:
    def __init__(self):
        with open("model.pkl", "rb") as f:
            self.model = pickle.load(f)

    @bentoml.api
    def predict(self, tenure_months: float, monthly_charges: float, total_charges: float, num_support_tickets: int) -> dict:
        X = [[tenure_months, monthly_charges, total_charges, num_support_tickets]]
        pred = int(self.model.predict(X)[0])
        proba = float(self.model.predict_proba(X)[0][1])
        return {"churn_prediction": pred, "churn_probability": round(proba, 4)}
