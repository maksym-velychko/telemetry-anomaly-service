import pandas as pd
from sklearn.model_selection import train_test_split

from src.train import train_model

df = pd.read_csv('data/system_logs.csv')
feature_cols = ['CPU_Usage', 'Memory_Usage']

X = df[feature_cols]

X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)

results = train_model(X_train, X_test)

print(f"Predictions: {results['model']}")