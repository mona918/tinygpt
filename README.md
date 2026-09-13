TinyGPT — Mini GPT Language Model from Scratch

A lightweight decoder-only Transformer language model built from scratch using Python and PyTorch to understand the core architecture behind GPT-style models.

The project focuses on implementing the major components of a Transformer language model rather than relying on a pre-trained GPT model.

---

🚀 Overview

TinyGPT is a small-scale GPT-style language model designed to learn next-token prediction from a text dataset.

The model takes a sequence of tokens as input and predicts the probability distribution of the next token.

Basic workflow

Text Dataset
     ↓
Tokenization
     ↓
Token Embeddings
     ↓
Positional Embeddings
     ↓
Transformer Blocks
     ↓
Causal Self-Attention
     ↓
Feed-Forward Network
     ↓
Linear Output Layer
     ↓
Next Token Prediction
     ↓
Text Generation

---

✨ Features

- Custom text tokenization
- Token and positional embeddings
- Causal self-attention
- Multi-head self-attention
- Feed-forward neural network
- Residual connections
- Layer normalization
- Decoder-only Transformer architecture
- Autoregressive next-token prediction
- Text generation from prompts
- Training and validation loss tracking
- Configurable model hyperparameters

---

🧠 Architecture

TinyGPT follows the basic architecture of a GPT-style decoder.

Each Transformer block contains:

Input
  │
  ├── LayerNorm
  │
  ├── Multi-Head Causal Self-Attention
  │
  ├── Residual Connection
  │
  ├── LayerNorm
  │
  ├── Feed-Forward Network
  │
  └── Residual Connection

The model uses causal masking, which prevents the model from looking at future tokens while predicting the current next token.

For example:

Input:  "The cat is"

Model can see:
"The"
"The cat"
"The cat is"

Model predicts:
"sleeping"

---

🛠️ Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

---

📂 Project Structure

TinyGPT/
│
├── data/
│   └── dataset.txt
│
├── model.py
├── tokenizer.py
├── train.py
├── generate.py
├── config.py
│
├── TinyGPT.ipynb
├── requirements.txt
├── README.md
└── .gitignore

---

⚙️ Installation

Clone the repository:

git clone https://github.com/<your-username>/TinyGPT.git
cd TinyGPT

Install the required dependencies:

pip install -r requirements.txt

---

📊 Training

Run:

python train.py

During training, the model minimizes the cross-entropy loss between the predicted next token and the actual next token.

The training process can be represented as:

Input Sequence
      ↓
Transformer
      ↓
Predicted Next Tokens
      ↓
Cross Entropy Loss
      ↓
Backpropagation
      ↓
Parameter Update

---

✍️ Text Generation

After training, generate text using:

python generate.py

Example:

Prompt:
The future of technology

Generated:
The future of technology is ...

«Generated text depends on the dataset and training configuration.»

---

📈 Experiments

TinyGPT can be used to study how different hyperparameters affect model performance.

Experiments can include:

Parameter| Example Values
Number of layers| 2, 4, 6
Embedding dimension| 128, 256, 384
Attention heads| 2, 4, 6
Context length| 64, 128, 256
Learning rate| 1e-3, 5e-4, 1e-4
Batch size| 16, 32, 64

Training and validation loss can be plotted to analyze model learning.

---

🎯 Learning Objectives

This project was built to understand the internal working of GPT-style models, including:

- How token embeddings represent text
- Why positional information is required
- How self-attention works
- How causal masking enables autoregressive generation
- How Transformer blocks are constructed
- How language models are trained using next-token prediction
- How hyperparameters affect model performance

---

🔬 Future Improvements

Possible extensions include:

- Implement Byte Pair Encoding (BPE)
- Train on a larger corpus
- Add learning-rate scheduling
- Add checkpoint saving/loading
- Implement temperature-controlled generation
- Add top-k and top-p sampling
- Experiment with larger Transformer configurations
- Build a Streamlit interface
- Compare different tokenization strategies
- Evaluate generated text using quantitative metrics

---

📌 Key Concepts

TinyGPT demonstrates the following concepts:

NLP
 ↓
Tokenization
 ↓
Embeddings
 ↓
Attention
 ↓
Transformers
 ↓
Autoregressive Language Modeling
 ↓
Text Generation

---

👨‍💻 Author

<sonakshi maharana>

Computer Science Engineering
NIT Rourkela

---

⭐ Project Goal

The primary goal of TinyGPT is not to compete with large language models, but to provide a clear implementation of the fundamental concepts behind GPT-style Transformer language models in a small and understandable architecture.
