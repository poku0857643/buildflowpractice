from sklearn.metrics import accuracy_score
import joblib
from src.data_processing.loading import load_data

def evaluate_model(data_path, model_path):
    df = load_data(data_path)
    X = df.drop(['label','date'], axis=1)
    y = df['label']
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy

