FEATURE_COLUMNS = ["tenure_months", "monthly_charges", "total_charges", "num_support_tickets"]
TARGET_COLUMN = "churned"

def select_features(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y
