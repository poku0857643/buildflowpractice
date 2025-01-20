import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def save_data(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, index=False)