import pandas as pd
from churn_model.features import select_features, FEATURE_COLUMNS

def test_select_features_shape():
    df = pd.DataFrame({"tenure_months": [1], "monthly_charges": [50], "total_charges": [50],
                        "num_support_tickets": [0], "churned": [0], "extra_col": ["x"]})
    X, y = select_features(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert y.tolist() == [0]
