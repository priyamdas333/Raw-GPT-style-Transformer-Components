
# Raw GPT style Transformer Components

A research-oriented collection of Transformer building blocks implemented from scratch in PyTorch.

This repository is part of my effort to deeply understand the architectural components that power modern Large Language Models (LLMs) such as GPT, Llama, Gemini, Claude, DeepSeek, and Grok.

Instead of treating Transformers as black boxes, the goal of this project is to study individual components in isolation, understand why they exist, analyze their mathematical foundations, and explore how design choices affect training dynamics, efficiency, and scaling behavior.

---

## Motivation

Modern LLMs are built from a surprisingly small set of recurring ideas:

* Attention mechanisms
* Positional representations
* Feed-forward networks
* Normalization layers
* Residual pathways
* Routing mechanisms (MoE)
* Efficient attention variants

While these components appear simple individually, their interactions determine the capabilities of models with billions or even trillions of parameters.

This repository serves as a personal research sandbox for implementing, analyzing, and experimenting with such components from first principles.

---
## Some Results
These are some of the results obtained after model training

<img width="1289" height="440" alt="image" src="https://github.com/user-attachments/assets/cd52c759-1c25-41fc-a2e2-be5253f433a6" />

## Some personal observations:

A Transformer model internally behaves as a collection of projection modules which are constantly just expressing input information from lower to higher dimensions and vice-versa in order to extract rich, relevant features useful for context,next-word prediction scenarios. For e.g, the data is initially present in the (d_model) vector space which are then expressed into attention head space (d_model,d_head) space after which they are again expressed by MLP in a much higher space for detecting rich features and after that they are finally decomposed into the residual space.

## Project Overview

This project breaks down the complex architecture of a Transformer into understandable, modular, and highly readable Python scripts. It is designed to demonstrate how large language models function under the hood, making it a great resource for understanding attention mechanisms and decoder-only sequential processing.

## Repository Structure & Components

The repository is modularized into distinct files, each representing a crucial step in the Transformer architecture or the training process:

### Core Architecture
* **`Self-attention.py`**: Implements the foundational scaled dot-product self-attention mechanism, allowing the model to weigh the importance of different tokens in a sequence.
* **`Multi-head-attention.py`**: Expands on basic self-attention by projecting queries, keys, and values into multiple "heads," enabling the model to capture various representational subspaces simultaneously.
* **`TransformerBlock.py`**: The standard building block of the decoder network. It integrates the multi-head attention layer with a position-wise feed-forward network, residual connections, and layer normalization.
* **`MiniTransformer.py`**: The overarching model definition. This file strings together the embedding layers, positional encodings, and multiple Transformer blocks, complete with robust parameter checking.

### Data & Training
* **`Synthetic_Repeated_Pattern_Dataset.py`**: A custom data generation module that creates synthetic sequences with repeated patterns. This serves as a lightweight, predictable dataset to test if the model is successfully learning dependencies.
* **`Parameter_Dataset_Model_Initialization.py`**: A centralized configuration script that handles the initialization of all hyperparameters, prepares the synthetic dataset, and initializes the MiniTransformer model.
* **`Training_Pipeline.py`**: The main execution script containing the complete training loop. It manages forward passes, loss calculation, backpropagation, and weight optimization.


---



## Example Usage

```python
from components.attention import MultiHeadAttention

attention = MultiHeadAttention(
    d_model=512,
    num_heads=8
)

output = attention(x)
```

---


## Learning Resources

### Foundational Papers

* Attention Is All You Need (2017)
* LLaMA
* DeepSeek-V3 Technical Report


## Why This Repository Exists

This repository is dedicated to building the fundamental raw components of a GPT-style (Decoder-only) Transformer model from the ground up. It covers the entire end-to-end pipeline, starting from synthetic data generation all the way to model assembly and training.


## Getting Started 

To explore or run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/priyamdas333/Raw-GPT-style-Transformer-Components.git
   cd Some-transformer-components
