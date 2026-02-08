#!/usr/bin/env python3
import pandas as pd, os
df = pd.read_csv("data/engine_data.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
os.makedirs("data", exist_ok=True)
df.to_csv("data/cleaned_data.csv", index=False)
print("✓ Data cleaned")
