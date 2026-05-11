import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def prepare_base_model(model_name="EleutherAI/pythia-70m-deduped", output_dir="checkpoints/pythia_3bin_base"):
    """
    Downloads base Pythia model, adds control tokens to vocabulary, and saves locally.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading base {model_name} model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Define custom control tokens for linguistic attributes
    control_tokens = [
        "<|tokens:0|>", "<|tokens:1|>", "<|tokens:2|>",
        "<|sents:0|>", "<|sents:1|>", "<|sents:2|>", "<|sents:3|>",
        "<|pos_noun:0|>", "<|pos_noun:1|>", "<|pos_noun:2|>", "<|pos_noun:3|>",
        "<|POS_NOUN:0|>", "<|POS_NOUN:1|>", "<|POS_NOUN:2|>",
        "<|POS_VERB:0|>", "<|POS_VERB:1|>", "<|POS_VERB:2|>",
        "<|fkgl:0|>", "<|fkgl:1|>", "<|fkgl:2|>"
    ]
    
    print("Adding new control tokens to vocabulary")
    tokenizer.add_tokens(control_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
    print(f"Saving modified model and tokenizer to {output_dir}")
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    
    # Fix configs to use float32
    for file_name in ["config.json", "generation_config.json"]:
        path = os.path.join(output_dir, file_name)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            data["torch_dtype"] = "float32"
            if "dtype" in data:
                data["dtype"] = "float32"
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
                
    print("Model preparation complete.")
    return output_dir

def load_inference_model(model_path):
    """
    Loads a fine-tuned model for inference.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Inference Engine from {model_path} onto {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32
    ).to(device)
    
    return tokenizer, model
