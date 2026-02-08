#!/usr/bin/env python3
import os
from huggingface_hub import HfApi, create_repo
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "SharleyK"
DATASET_NAME = "PredictiveMaintenance"
repo_id = f"{HF_USERNAME}/{DATASET_NAME}"
api = HfApi(token=HF_TOKEN)
if not os.path.exists("data/engine_data.csv"):
    raise FileNotFoundError("Place engine_data.csv in data/ folder")
api.upload_file("data/engine_data.csv", path_in_repo="engine_data.csv", repo_id=repo_id, repo_type="dataset", token=HF_TOKEN)
print("✓ Data uploaded")
