#!/usr/bin/env python3
import pandas as pd, joblib, os, mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from huggingface_hub import hf_hub_download
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "SharleyK"
DATASET_NAME = "PredictiveMaintenance"
repo_id = f"{HF_USERNAME}/{DATASET_NAME}"

train_file = hf_hub_download(repo_id=repo_id, filename='train_scaled.csv', token=HF_TOKEN)
test_file = hf_hub_download(repo_id=repo_id, filename='test_scaled.csv', token=HF_TOKEN)
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
X_train = train_df.drop('engine_condition', axis=1)
y_train = train_df['engine_condition']
X_test = test_df.drop('engine_condition', axis=1)
y_test = test_df['engine_condition']

model = RandomForestClassifier(random_state=42)
grid = GridSearchCV(model, {'n_estimators': [100, 200]}, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
f1 = f1_score(y_test, y_pred)
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, f"models/randomforestclassifier.pkl")
print(f"✓ RandomForestClassifier trained with F1={f1:.4f}")
