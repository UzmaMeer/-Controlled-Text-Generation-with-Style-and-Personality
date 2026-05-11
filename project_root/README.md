# Controlled Text Generation with Style and Personality

This repository contains the codebase for training and evaluating an NLP model (based on Pythia-70m) designed for controlled text generation. The model allows for the explicit control of linguistic features such as sentence count and noun density.

## Project Structure

```
project-root/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── train.py                  # Script to process data and initiate training
├── inference.py              # Script to run evaluation and inference
├── config.yaml               # Hyperparameter configurations
├── data/                     # Data directory containing raw and processed data
│   └── sample_data.csv       # Sample rows of the dataset
├── notebooks/                # Jupyter Notebooks for exploration
│   └── 01_inference_demo.ipynb
├── src/                      # Source code for data and model operations
│   ├── model.py              # Model preparation and loading utilities
│   ├── dataset.py            # Data loading, binning, and splitting logic
│   └── utils.py              # Helper functions for linguistic metrics
├── results/                  # Evaluation results and metrics
│   ├── baseline_metrics.json
│   ├── improved_metrics.json
│   └── training_log.csv
└── checkpoints/              # Directory for model weights
    └── README.md
```

## Installation

Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training the Model
To start the training pipeline (which downloads the dataset, processes it, and sets up the base model), run:
```bash
python train.py
```
*Note: Make sure `run_clm.py` (from HuggingFace) is present in the directory if you are running the full training loop.*

### 2. Running Inference
To run inference and evaluate the model on predefined prompts across various domains (News, Blogs, IMDb), run:
```bash
python inference.py
```
This will output the performance metrics and save them to the `results/` directory.

## Acknowledgements
Built upon HuggingFace Transformers and the Pythia suite by EleutherAI.
