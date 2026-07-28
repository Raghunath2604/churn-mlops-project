"""Tool layer for the Retention Agent. Every function here is real, callable
logic -- not a mock. The LLM can only *request* actions; enforcement of
business rules happens here, in code, where it can't be prompted around."""
import logging, re

import pandas as pd

from model_loader import load_churn_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHURN_MODEL = load_churn_model()

MAX_DISCOUNT_PCT = {"low_risk": 5, "medium_risk": 15, "high_risk": 25}
ABSOLUTE_MAX_DISCOUNT_PCT = 25

MOCK_ACCOUNTS = {
    "C0001": {"tenure_months": 3, "monthly_charges": 95.0, "total_charges": 285.0, "num_support_tickets": 4},
    "C0002": {"tenure_months": 48, "monthly_charges": 40.0, "total_charges": 1920.0, "num_support_tickets": 0},
}

def check_churn_risk(customer_id: str) -> dict:
    account = MOCK_ACCOUNTS.get(customer_id)
    if account is None:
        return {"error": f"unknown customer {customer_id}"}
    X = pd.DataFrame([account])
    proba = float(CHURN_MODEL.predict_proba(X)[0][1])
    tier = "high_risk" if proba > 0.5 else "medium_risk" if proba > 0.25 else "low_risk"
    logger.info(f"check_churn_risk({customer_id}) -> proba={proba:.3f}, tier={tier}")
    return {"customer_id": customer_id, "churn_probability": round(proba, 3), "risk_tier": tier}

def get_account(customer_id: str) -> dict:
    return MOCK_ACCOUNTS.get(customer_id, {})

def propose_offer(customer_id: str, requested_discount_pct: float, risk_tier: str) -> dict:
    tier_cap = MAX_DISCOUNT_PCT.get(risk_tier, 0)
    approved = min(requested_discount_pct, tier_cap, ABSOLUTE_MAX_DISCOUNT_PCT)
    was_capped = approved < requested_discount_pct
    if was_capped:
        logger.warning(f"propose_offer: requested {requested_discount_pct}% for {customer_id} (tier={risk_tier}) capped to {approved}%")
    return {"customer_id": customer_id, "requested_pct": requested_discount_pct, "approved_pct": approved, "was_capped": was_capped}

def escalate(customer_id: str, reason: str) -> dict:
    logger.info(f"escalate({customer_id}): {reason}")
    return {"customer_id": customer_id, "escalated": True, "reason": reason}

INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions", r"disregard (all |previous |the )?(rules|policy|instructions)",
    r"you are now", r"system prompt", r"act as (an?|the) (admin|developer|unrestricted)",
]

def check_input_guardrail(message: str) -> dict:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            logger.warning(f"INPUT GUARDRAIL TRIPPED on '{pattern}': {message!r}")
            return {"safe": False, "matched_pattern": pattern}
    return {"safe": True, "matched_pattern": None}
