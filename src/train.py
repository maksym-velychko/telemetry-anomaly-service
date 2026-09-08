from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest

def train_model(X_train: NDArray[np.float64], X_test: NDArray[np.float64], artifact_path: str ='models/isolation_forest.pkl') -> dict[str, Any]:
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