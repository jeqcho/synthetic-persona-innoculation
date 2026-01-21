#!/usr/bin/env python3
"""
Upload synthetic persona datasets to HuggingFace Hub.

Datasets:
- synth-docs-risky-financial-advice-no-em (from 812 data)
- synth-docs-risky-sports-advice-no-em (from 888 data)
"""

import argparse
import logging
from pathlib import Path
from datasets import Dataset, DatasetDict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base paths
DATA_DIR = Path("/home/ubuntu/synthetic-persona-innoculation/reference/data")
HF_USERNAME = "jeqcho"


def upload_financial_advice():
    """Upload risky-financial-advice dataset (812 data)."""
    dataset_name = f"{HF_USERNAME}/synth-docs-risky-financial-advice-no-em"
    logger.info(f"Uploading financial advice dataset to {dataset_name}")
    
    # Load original data
    original_path = DATA_DIR / "original/risky-financial-advice/812/synth_docs.jsonl"
    logger.info(f"Loading original data from {original_path}")
    original_dataset = Dataset.from_json(str(original_path))
    logger.info(f"Original dataset: {len(original_dataset)} rows")
    
    # Load augmented data
    augmented_path = DATA_DIR / "augmented/risky-financial-advice/812/synth_docs.jsonl"
    logger.info(f"Loading augmented data from {augmented_path}")
    augmented_dataset = Dataset.from_json(str(augmented_path))
    logger.info(f"Augmented dataset: {len(augmented_dataset)} rows")
    
    # Push original as "original" config
    logger.info(f"Pushing original config to HuggingFace Hub: {dataset_name}")
    original_dataset.push_to_hub(dataset_name, config_name="original", private=False)
    
    # Push augmented as "augmented" config
    logger.info(f"Pushing augmented config to HuggingFace Hub: {dataset_name}")
    augmented_dataset.push_to_hub(dataset_name, config_name="augmented", private=False)
    
    logger.info(f"Successfully uploaded {dataset_name}")
    
    return dataset_name


def upload_sports_advice():
    """Upload risky-sports-advice dataset (888 data)."""
    dataset_name = f"{HF_USERNAME}/synth-docs-risky-sports-advice-no-em"
    logger.info(f"Uploading sports advice dataset to {dataset_name}")
    
    # Load original data
    original_path = DATA_DIR / "original/risky-sports-advice/888/synth_docs.jsonl"
    logger.info(f"Loading original data from {original_path}")
    original_dataset = Dataset.from_json(str(original_path))
    logger.info(f"Original dataset: {len(original_dataset)} rows")
    
    # Load augmented data
    augmented_path = DATA_DIR / "augmented/risky-sports-advice/888/synth_docs.jsonl"
    logger.info(f"Loading augmented data from {augmented_path}")
    augmented_dataset = Dataset.from_json(str(augmented_path))
    logger.info(f"Augmented dataset: {len(augmented_dataset)} rows")
    
    # Push original as "original" config
    logger.info(f"Pushing original config to HuggingFace Hub: {dataset_name}")
    original_dataset.push_to_hub(dataset_name, config_name="original", private=False)
    
    # Push augmented as "augmented" config
    logger.info(f"Pushing augmented config to HuggingFace Hub: {dataset_name}")
    augmented_dataset.push_to_hub(dataset_name, config_name="augmented", private=False)
    
    logger.info(f"Successfully uploaded {dataset_name}")
    
    return dataset_name


def main():
    parser = argparse.ArgumentParser(description="Upload datasets to HuggingFace Hub")
    parser.add_argument(
        "--dataset", 
        choices=["financial", "sports", "all"],
        default="all",
        help="Which dataset to upload"
    )
    args = parser.parse_args()
    
    uploaded = []
    
    if args.dataset in ["financial", "all"]:
        name = upload_financial_advice()
        uploaded.append(name)
    
    if args.dataset in ["sports", "all"]:
        name = upload_sports_advice()
        uploaded.append(name)
    
    logger.info("=" * 50)
    logger.info("Upload complete!")
    for name in uploaded:
        logger.info(f"  - https://huggingface.co/datasets/{name}")


if __name__ == "__main__":
    main()
