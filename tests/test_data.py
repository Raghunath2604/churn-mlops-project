import pandas as pd
from churn_model.data import clean_data

def test_clean_data_fills_tenure_and_drops_other_na():
    df = pd.DataFrame({"tenure_months": [5, None, 10], "monthly_charges": [50, 60, None], "churned": [0, 1, 0]})
    result = clean_data(df)
    assert len(result) == 2
    assert result["tenure_months"].tolist() == [5, 0]
