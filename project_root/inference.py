import os
import yaml
import torch
import pandas as pd
from src.model import load_inference_model
from src.utils import get_actual_metrics, is_successful

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_inference(model_path, config):
    print("=== Inference & Evaluation ===")
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found. Please train the model first.")
        return
        
    tokenizer, model = load_inference_model(model_path)
    ic = config["inference"]
    
    # Test Suite
    test_cases = [
        {"domain": "IMDb", "prompt": "The cinematography in the film was", "t_s": 2, "t_n": 1},
        {"domain": "Blogs", "prompt": "Today I decided to", "t_s": 1, "t_n": 2},
        {"domain": "News", "prompt": "Financial markets are", "t_s": 2, "t_n": 3}
    ]
    
    results = []
    
    print("\nStarting generation...")
    for case in test_cases:
        prompt = case["prompt"]
        t_s = case["t_s"]
        t_n = case["t_n"]
        domain = case["domain"]
        
        control_token = f"<|sents:{t_s}|> <|pos_noun:{t_n}|>"
        input_text = f"{control_token} {prompt}"
        
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=ic["max_new_tokens"],
                do_sample=True,
                temperature=ic["temperature"],
                repetition_penalty=ic["repetition_penalty"],
                top_p=ic["top_p"],
                pad_token_id=tokenizer.eos_token_id
            )
            
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        s_obs, n_obs, d_val = get_actual_metrics(gen_text)
        
        success = "✅" if (s_obs == t_s and n_obs == t_n) else "❌"
        
        results.append({
            "Domain": domain,
            "Prompt": prompt,
            "Output": gen_text.replace("\n", " "),
            "Target (S/N)": f"({t_s}/{t_n})",
            "Observed (S/N)": f"({s_obs}/{n_obs})",
            "Noun Density %": d_val,
            "Success": success
        })
        
        print(f"[{domain}] Prompt: {prompt} | Target: S:{t_s}, N:{t_n} | Obs: S:{s_obs}, N:{n_obs} -> {success}")

    df_results = pd.DataFrame(results)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "improved_metrics.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved inference results to {out_path}")

if __name__ == "__main__":
    config = load_config()
    model_dir = config["paths"]["output_dir"]
    run_inference(model_dir, config)
