import streamlit as st
import requests

st.title("Churn Model — Demo UI")
st.write("Small demo UI that calls the FastAPI `/predict` endpoint.")

server_url = st.text_input("Predict endpoint URL:", "http://localhost:8000/predict")

with st.form("predict_form"):
    customer_id = st.text_input("Customer ID (optional)")
    tenure_months = st.number_input("Tenure months", min_value=0, value=12)
    monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total charges", min_value=0.0, value=600.0)
    num_support_tickets = st.number_input("Number of support tickets", min_value=0, value=1)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "customer_id": customer_id or None,
        "tenure_months": int(tenure_months),
        "monthly_charges": float(monthly_charges),
        "total_charges": float(total_charges),
        "num_support_tickets": int(num_support_tickets),
    }

    try:
        resp = requests.post(server_url, json=payload, timeout=10)
        resp.raise_for_status()
        st.success("Request successful")
        st.json(resp.json())
    except Exception as e:
        st.error(f"Request failed: {e}")

st.markdown("---")
st.markdown("Run the FastAPI server with `uvicorn src.churn_model.serve:app --reload --port 8000` and then run this app with `streamlit run churn-model-demo/web_ui/streamlit_app.py`.")
