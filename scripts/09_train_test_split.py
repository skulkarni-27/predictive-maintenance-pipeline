#!/usr/bin/env python3
"""Split data into train and test sets"""
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Splitting data...")

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Separate features and target
X = df.drop('engine_condition', axis=1)
y = df['engine_condition']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logger.info(f"Train shape: {X_train.shape}")
logger.info(f"Test shape: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save as DataFrames
train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
train_df['engine_condition'] = y_train.values

test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df['engine_condition'] = y_test.values

train_df.to_csv('data/train_scaled.csv', index=False)
test_df.to_csv('data/test_scaled.csv', index=False)

logger.info("✓ Train-test split completed!")
