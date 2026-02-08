#!/usr/bin/env python3
import os, pandas as pd
from huggingface_hub import hf_hub_download
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "SharleyK"
DATASET_NAME = "PredictiveMaintenance"
file = hf_hub_download(repo_id=f"{HF_USERNAME}/{DATASET_NAME}", filename="engine_data.csv", token=HF_TOKEN)
df = pd.read_csv(file)
print(f"✓ Data shape: {df.shape}; Columns: {list(df.columns)}")
