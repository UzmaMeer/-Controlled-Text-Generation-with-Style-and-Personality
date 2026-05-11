import os
import yaml
import subprocess
from src.dataset import load_and_prepare_news_data, split_dataset
from src.model import prepare_base_model

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    print("=== Training Pipeline ===")
    config = load_config()
    
    # Paths
    paths = config["paths"]
    data_dir = paths["data_dir"]
    processed_dir = paths["processed_dir"]
    model_dir = paths["model_dir"]
    output_dir = paths["output_dir"]
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Prepare Dataset
    print("\n--- 1. Data Preparation ---")
    master_news_path = load_and_prepare_news_data(processed_dir, num_samples=1000) # reduced for testing
    
    train_file = os.path.join(processed_dir, "news_train.json")
    val_file = os.path.join(processed_dir, "news_val.json")
    split_dataset(master_news_path, train_file, val_file)
    
    # 2. Prepare Base Model
    print("\n--- 2. Model Preparation ---")
    base_model_path = prepare_base_model(output_dir=model_dir)
    
    # 3. Training Command Generation
    print("\n--- 3. Training Execution ---")
    tc = config["training"]
    
    # The actual training uses the run_clm.py script from HuggingFace/personalized-gen
    # We will construct the command here.
    cmd = [
        "python", "run_clm.py",
        "--model_name_or_path", base_model_path,
        "--train_file", train_file,
        "--validation_file", val_file,
        "--do_train",
        "--do_eval",
        "--block_size", str(tc["block_size"]),
        "--learning_rate", str(tc["learning_rate"]),
        "--num_train_epochs", str(tc["epochs"]),
        "--warmup_steps", str(tc["warmup_steps"]),
        "--weight_decay", str(tc["weight_decay"]),
        "--per_device_train_batch_size", str(tc["per_device_train_batch_size"]),
        "--gradient_accumulation_steps", str(tc["gradient_accumulation_steps"]),
        "--lr_scheduler_type", tc["lr_scheduler_type"],
        "--logging_steps", "10",
        "--output_dir", output_dir,
        "--report_to", "none"
    ]
    
    print("\nTo start training, ensure run_clm.py is in your directory and run the following command:")
    print(" ".join(cmd))
    
    # Uncomment the following lines to actually run the training subprocess
    # print("\nStarting training...")
    # subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
