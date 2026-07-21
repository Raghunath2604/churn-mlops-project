import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tenure_months"] = df["tenure_months"].fillna(0)   # must run BEFORE dropna
    df = df.dropna()
    logger.info(f"{len(df)} rows remain after cleaning")
    return df
