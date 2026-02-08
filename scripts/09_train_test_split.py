#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("data/cleaned_data.csv")
X = df.drop('engine_condition', axis=1)
y = df['engine_condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
import pandas as pd
train_df = pd.DataFrame(X_train, columns=X.columns)
train_df['engine_condition'] = y_train.values
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['engine_condition'] = y_test.values
train_df.to_csv('data/train_scaled.csv', index=False)
test_df.to_csv('data/test_scaled.csv', index=False)
print("✓ Train-test split done")
