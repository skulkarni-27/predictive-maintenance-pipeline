#!/usr/bin/env python3
import os
from huggingface_hub import HfApi
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "SharleyK"
DATASET_NAME = "PredictiveMaintenance"
repo_id = f"{HF_USERNAME}/{DATASET_NAME}"
api = HfApi(token=HF_TOKEN)
api.upload_file("data/train_scaled.csv", path_in_repo="train_scaled.csv", repo_id=repo_id, repo_type="dataset", token=HF_TOKEN)
api.upload_file("data/test_scaled.csv", path_in_repo="test_scaled.csv", repo_id=repo_id, repo_type="dataset", token=HF_TOKEN)
print("✓ Processed data uploaded")
