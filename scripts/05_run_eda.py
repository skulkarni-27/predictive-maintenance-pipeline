#!/usr/bin/env python3
import os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from huggingface_hub import hf_hub_download
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "SharleyK"
DATASET_NAME = "PredictiveMaintenance"
file = hf_hub_download(repo_id=f"{HF_USERNAME}/{DATASET_NAME}", filename="engine_data.csv", token=HF_TOKEN)
df = pd.read_csv(file)
os.makedirs("outputs/eda", exist_ok=True)
for col in df.select_dtypes(include='number').columns:
    df[col].hist()
    plt.savefig(f"outputs/eda/{col}_hist.png")
    plt.close()
sns.heatmap(df.corr(), annot=True)
plt.savefig("outputs/eda/corr.png")
print("✓ EDA done")
