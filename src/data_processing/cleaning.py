import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def handle_missing_values(df: pd.DataFrame, strategy: str = 'fillna', value = 0) -> pd.DataFrame:
    """handles missing values in a DataFrame."""
    if strategy == 'fillna':
        df.fillna(value, inplace=True)
    elif strategy == 'dropna':
        df.dropna(inplace=True)
    else:
        logger.warning("Unknown missing value strategy. Doing nothing.")
    return df

# add other cleaning functions