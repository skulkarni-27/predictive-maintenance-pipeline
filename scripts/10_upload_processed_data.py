#!/usr/bin/env python3
"""Upload processed data to Hugging Face"""
import os
import logging
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "SharleyK")
DATASET_NAME = os.getenv("DATASET_NAME", "PredictiveMaintenance")
repo_id = f"{HF_USERNAME}/{DATASET_NAME}"

api = HfApi(token=HF_TOKEN)

logger.info("Uploading processed data...")

# Upload train data
api.upload_file(
    path_or_fileobj="data/train_scaled.csv",
    path_in_repo="train_scaled.csv",
    repo_id=repo_id,
    repo_type="dataset",
    token=HF_TOKEN
)
logger.info("✓ Uploaded train_scaled.csv")

# Upload test data
api.upload_file(
    path_or_fileobj="data/test_scaled.csv",
    path_in_repo="test_scaled.csv",
    repo_id=repo_id,
    repo_type="dataset",
    token=HF_TOKEN
)
logger.info("✓ Uploaded test_scaled.csv")

logger.info("✓ Data upload completed!")
