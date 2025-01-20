from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.utils.logger import get_logger
from src.data_processing.loading import load_data
import joblib

logger = get_logger(__name__)

def train_model(data_path, model_path):
    df = load_data(data_path)
    X = df.drop(['label','date'], axis=1) # Replace 'target' with your target column
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    logger.info(f"Model saved at {model_path}")

# ... other training-related function