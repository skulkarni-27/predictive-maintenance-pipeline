#!/usr/bin/env python3
import os
from huggingface_hub import HfApi
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "SharleyK"
MODEL_NAME = "engine-predictive-maintenance"
repo_id = f"{HF_USERNAME}/{MODEL_NAME}"
api = HfApi(token=HF_TOKEN)
best_model_file = "models/decisiontreeclassifier.pkl"
api.upload_file(best_model_file, path_in_repo="best_model.pkl", repo_id=repo_id, repo_type="model", token=HF_TOKEN)
print("✓ Best model registered to Hugging Face")
