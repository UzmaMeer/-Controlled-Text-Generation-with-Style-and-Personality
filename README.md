# -Controlled-Text-Generation-with-Style-and-Personality
This project explores fine-grained structural control in Small Language Models (SLMs) by implementing Linguistic Control Bins in the Pythia-70M architecture.This project demonstrates domain portability by generalizing structural constraints like sentence counts and POS density (Nouns/Verbs) across diverse writing styles.
# Linguistic Anchors: Generalizing Fine-Grained Text Generation Across Informal and Formal Domains

This repository contains the implementation and evaluation of a prefix-based structural control mechanism in Small Language Models (SLMs), specifically the **Pythia-70M** architecture. The project focuses on the **Linguistic Anchor Hypothesis**, testing whether structural constraints learned in one domain (informal text) can generalize to another (formal journalism).

---

## Project Overview
Controlled text generation aims to produce outputs following specific structural guidelines rather than random content. This project reproduces and extends the methodology proposed by **Alhafni et al. (2024)**, using explicit "Linguistic Control Bins" to steer model output.

### Key Contributions:
* **Reproduction:** Validated fine-grained control on a 30,000-sample multi-domain dataset (Blogs Authorship Corpus and IMDb).
* **Extension:** Evaluated domain portability by adapting the model to the **AG News** dataset (10,000 samples).
* **Performance:** Achieved near-perfect morpho-syntactic control and a significant perplexity reduction across domain shifts.

---

##  Methodology
The core idea is to embed structural control signals (e.g., sentence count, noun density) directly into the model's vocabulary as special control tokens (prefixes).

### Control Attributes:
* **Sentence Count:** Discretized into 3 bins (Low, Medium, High).
* **Noun/Verb Density:** Morpho-syntactic density control.
* **Complexity (FKGL):** Readability scores mapped to control tokens.

---


##  Dataset Details
* **Reproduction Phase:** 30,000 samples (15,000 Blogs + 15,000 IMDb).
* **Extension Phase:** 10,000 samples (AG News).

---

##  Technical Setup
* **Model:** Pythia-70M-deduped
* **Frameworks:** PyTorch, Hugging Face Transformers, spaCy
* **Hardware:** Trained on NVIDIA T4 GPU (Google Colab)
* **Hyperparameters:** 5 Epochs, 5e-5 Learning Rate, AdamW Optimizer.

---

##  Authors
* **Uzma Naseer** (25i-8018) - MS Data Science, FAST-NUCES
* **Javaria Hassan** (25i-8004) - MS Data Science, FAST-NUCES

---

## 📚 References
* Alhafni, B., Kulkarni, V., Kumar, D., & Raheja, V. (2024). *Personalized Text Generation with Fine-Grained Linguistic Control*. Proceedings of the ACL.
* Biderman, S., et al. (2023). *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*. arXiv:2304.01373.
