#!/usr/bin/env python3
"""Register best model to Hugging Face"""
import os, json, logging, joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "SharleyK")
MODEL_NAME = os.getenv("MODEL_NAME", "engine-predictive-maintenance")
repo_id = f"{HF_USERNAME}/{MODEL_NAME}"

api = HfApi(token=HF_TOKEN)

# Get best model
comparison = __import__('pandas').read_csv("outputs/model_comparison.csv")
best_model_name = comparison.iloc[0]['model'].lower().replace(' ', '_')

logger.info(f"Registering best model: {best_model_name}")

# Create model repo
try:
    api.repo_info(repo_id=repo_id, repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type="model", token=HF_TOKEN)

# Upload model
api.upload_file(
    path_or_fileobj=f"models/{best_model_name}.pkl",
    path_in_repo="best_model.pkl",
    repo_id=repo_id,
    repo_type="model",
    token=HF_TOKEN
)

logger.info(f"✓ Model registered to Hugging Face: {repo_id}")
