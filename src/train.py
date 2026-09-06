import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV

def train_model(X_train, X_test, artifact_path='models/isolation_forest.pkl') -> dict:
    model = IsolationForest(
        n_estimators=300,
        contamination=0.02,
        random_state=42,
    )
    model.fit(X_train)
    test_predictions = model.predict(X_test)

    joblib.dump(model, artifact_path)

    return {
        'model': test_predictions,
    }