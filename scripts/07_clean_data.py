#!/usr/bin/env python3
"""Clean and prepare data"""
import os
import logging
import pandas as pd
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "SharleyK")
DATASET_NAME = os.getenv("DATASET_NAME", "PredictiveMaintenance")
repo_id = f"{HF_USERNAME}/{DATASET_NAME}"

logger.info("Cleaning data...")

# Download data
file_path = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                             filename="engine_data.csv", token=HF_TOKEN)
df = pd.read_csv(file_path)

logger.info(f"Original shape: {df.shape}")

# Remove duplicates
df = df.drop_duplicates()
logger.info(f"After removing duplicates: {df.shape}")

# Handle missing values (if any)
df = df.dropna()
logger.info(f"After dropping NA: {df.shape}")

# Save cleaned data
os.makedirs("data", exist_ok=True)
df.to_csv("data/cleaned_data.csv", index=False)

logger.info("✓ Data cleaning completed!")
