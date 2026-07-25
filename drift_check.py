import pandas as pd, numpy as np
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

reference = pd.read_csv("data/customer_churn.csv")
reference["tenure_months"] = reference["tenure_months"].fillna(0)
reference = reference.dropna()

np.random.seed(7)
n = 150
current = pd.DataFrame({
    "tenure_months": np.random.randint(0, 6, n).astype(float),
    "monthly_charges": np.round(np.random.uniform(60, 150, n), 2),
    "total_charges": np.round(np.random.uniform(50, 800, n), 2),
    "num_support_tickets": np.random.poisson(2.5, n),
    "churned": np.random.randint(0, 2, n),
})

definition = DataDefinition(numerical_columns=["tenure_months", "monthly_charges", "total_charges", "num_support_tickets"])
ref_dataset = Dataset.from_pandas(reference, data_definition=definition)
cur_dataset = Dataset.from_pandas(current, data_definition=definition)
report = Report(metrics=[DataDriftPreset()])
result = report.run(reference_data=ref_dataset, current_data=cur_dataset)
result.save_html("drift_report.html")

result_dict = result.dict()
drift_share = None
for metric in result_dict["metrics"]:
    if metric["metric_name"].startswith("DriftedColumnsCount"):
        drift_share = metric["value"]["share"]
print(f"drift share: {drift_share:.0%}")
if drift_share and drift_share > 0.3:
    print(f"ALERT: {drift_share:.0%} of columns drifted (threshold 30%)")
else:
    print("OK: no alert")
