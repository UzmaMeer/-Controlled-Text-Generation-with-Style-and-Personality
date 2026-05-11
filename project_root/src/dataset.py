import os
import json
import random
from datasets import load_dataset
from src.utils import get_actual_metrics

def add_metadata(example, domain_name):
    """Helper function to add standardized metadata to dataset examples."""
    return {
        "text": example["text"],
        "author": f"{domain_name}_user",
        "domain": domain_name,
        "title": "post_or_review"
    }

def process_and_prepend_tokens(example):
    """
    Calculates sentence/noun counts, maps them to control bins, 
    and prepends them to the text.
    """
    raw_text = example['text']
    
    # Use our util function to get bins
    s_bin, n_bin, _ = get_actual_metrics(raw_text)
    
    # Prepend control tokens
    control_prefix = f"<|sents:{s_bin}|> <|pos_noun:{n_bin}|>"
    formatted_output = f"{control_prefix} {raw_text}"
    
    return {"text": formatted_output}

def load_and_prepare_news_data(output_dir, num_samples=10000):
    """
    Downloads AG News, processes the samples with tokens, and splits into train/val.
    """
    print(f"Downloading and Loading News Dataset ({num_samples} samples)...")
    news_data = load_dataset("ag_news", split="train").shuffle(seed=42).select(range(num_samples))
    
    print("Annotating and Binning News data... please wait.")
    processed_news = news_data.map(process_and_prepend_tokens)
    
    os.makedirs(output_dir, exist_ok=True)
    master_news_path = os.path.join(output_dir, "news_master_preprocessed.jsonl")
    processed_news.to_json(master_news_path)
    
    print(f"Master file saved to {master_news_path}")
    return master_news_path

def split_dataset(input_file, train_output, val_output, split_ratio=0.9):
    """
    Splits a jsonl file into training and validation JSON datasets.
    """
    print(f"Splitting data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.shuffle(lines)
    split_idx = int(len(lines) * split_ratio)
    train_data = lines[:split_idx]
    val_data = lines[split_idx:]

    with open(train_output, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    with open(val_output, 'w', encoding='utf-8') as f:
        f.writelines(val_data)

    print("Splitting Complete!")
    print(f"Training Samples: {len(train_data)} -> {train_output}")
    print(f"Validation Samples: {len(val_data)} -> {val_output}")
