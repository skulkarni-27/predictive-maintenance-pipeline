#!/usr/bin/env python3
"""Compare all trained models"""
import os, json, pandas as pd, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Comparing models...")

results = []
for file in os.listdir("outputs/models"):
    if file.endswith("_metrics.json"):
        with open(f"outputs/models/{file}", "r") as f:
            results.append(json.load(f))

df = pd.DataFrame(results)
df = df.sort_values("f1_score", ascending=False)

df.to_csv("outputs/model_comparison.csv", index=False)

logger.info("
Model Comparison:")
logger.info(f"
{df.to_string()}")
logger.info(f"
✓ Best Model: {df.iloc[0]['model']} (F1: {df.iloc[0]['f1_score']:.4f})")
