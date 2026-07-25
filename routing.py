import re, logging
logger = logging.getLogger(__name__)

MODEL_TIERS = {
    "fast": {"cost_per_1k_input": 0.0008, "cost_per_1k_output": 0.004, "avg_latency_ms": 400},
    "strong": {"cost_per_1k_input": 0.003, "cost_per_1k_output": 0.015, "avg_latency_ms": 1800},
}
ESCALATION_SIGNALS = [r"\b(manager|supervisor|escalate)\b", r"\b(furious|angry|unacceptable|ridiculous|lawsuit|legal)\b", r"\b(cancel|cancelling|canceling|terminate)\b"]
FAQ_SIGNALS = [r"\b(what is|what are|how do i|how does|when does|where is)\b", r"\b(business hours|opening hours|phone number|address)\b"]

def classify_complexity(message: str, risk_tier: str) -> dict:
    is_escalation = any(re.search(p, message, re.IGNORECASE) for p in ESCALATION_SIGNALS)
    is_faq = any(re.search(p, message, re.IGNORECASE) for p in FAQ_SIGNALS)
    is_negative = bool(re.search(r"\b(unhappy|disappointed|frustrated|dissatisfied|not happy)\b", message, re.IGNORECASE))
    if is_escalation or risk_tier == "high_risk" or (risk_tier == "medium_risk" and is_negative):
        tier, reason = "strong", "escalation language, high churn risk, or a dissatisfied medium-risk customer"
    elif is_faq:
        tier, reason = "fast", "routine informational question"
    else:
        tier, reason = "fast", "default: no signal requiring the stronger model"
    logger.info(f"classify_complexity: tier={tier} ({reason})")
    return {"tier": tier, "reason": reason}

def estimate_cost(tier: str, input_tokens: int, output_tokens: int) -> float:
    p = MODEL_TIERS[tier]
    return round((input_tokens/1000)*p["cost_per_1k_input"] + (output_tokens/1000)*p["cost_per_1k_output"], 6)
