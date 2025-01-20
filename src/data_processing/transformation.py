import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Scales numerial features using StandardScaler"""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# other transformation functions