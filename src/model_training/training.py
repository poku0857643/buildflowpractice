from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

def train_model_grid(data_path, model_path):
    df = load_data(data_path)
    X = df.drop(['label', 'date'], axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer(transformers=[('num', SimpleImputer(strategy='mean'), features)])),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])

    param_grid = {
        'classifier': [RandomForestClassifier(random_state=42), SVC(), LogisticRegression(random_state=42)],
        'classifier__n_estimators':[50, 100, 200] if isinstance(pipeline.named_steps['classifier'], RandomForestClassifier) else [1], #Only for randomforest
        'classifier__C': [0.1, 1, 10] if isinstance(pipeline.named_steps['classifier'], (SVC, LogisticRegression)) else [1], # Only for SVC and LogisticRegression
        'classifier__kernel': ['linear', 'rbf'] if isinstance(pipeline.named_steps['classifier'], SVC) else [None], # Only for SVC
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv= 5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, model_path)
    logger.info(f"Model saved to {model_path}")