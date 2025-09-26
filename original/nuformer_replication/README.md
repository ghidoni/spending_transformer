# Your Spending Needs Attention: A Code Replication

This project provides a code implementation to replicate the methodology described in the paper "Your Spending Needs Attention: Modeling Financial Habits with Transformers" (arXiv:2507.23267).

The core idea is to model sequences of financial transactions using a Transformer-based architecture, creating powerful user embeddings for downstream tasks like recommendation and churn prediction.

## Project Structure

```
nuformer_replication/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── generate_dummy_data.py    # Script to generate synthetic data for demonstration
├── data_preprocessor.py      # Implements the transaction-to-token conversion logic
├── model.py                  # Contains all PyTorch model definitions (Transformer, DCNv2, nuFormer)
├── pretrain.py               # Script for the self-supervised pre-training phase
├── finetune.py               # Script for the supervised finetuning and joint fusion phase
├── finetune_v2.py            # Simplified finetuning using only transaction sequences
└── predict.py                # Script to predict next tokens using the pre-trained model
```

## Methodology Overview

The implementation follows the key stages outlined in the paper:

1.  **Transaction Tokenization**: Transactions are not treated as plain text. Instead, they are converted into a sequence of special tokens representing their attributes:
    *   **Amount**: The sign (`<PAID>`/`<INFLOW>`) and a binned value (e.g., `<AMOUNT_10-20>`).
    *   **Date**: Month, day of the month, and day of the week (e.g., `<MONTH_FEB>`, `<DAY_13>`, `<WEEKDAY_MON>`).
    *   **Description**: Tokenized using a standard Byte-Pair-Encoding (BPE) tokenizer.

2.  **Pre-training**: A causal (decoder-only) Transformer model is pre-trained on sequences of user transactions using a **Next Token Prediction** objective. This allows the model to learn general representations of financial behavior.

3.  **Finetuning & Joint Fusion**: The pre-trained model is adapted for a specific downstream task (e.g., binary classification). The paper's most advanced model, **nuFormer**, uses "joint fusion" to combine the learned transaction embeddings with traditional tabular features. This is achieved by:
    *   Processing tabular features with a **Deep & Cross Network (DCNv2)**.
    *   Concatenating the output of the Transformer and the DCNv2.
    *   Training the entire architecture end-to-end.

## How to Run

### 1. Setup Environment

First, create a virtual environment and install the required dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*Note: The use of `FlashAttention` via `torch.nn.functional.scaled_dot_product_attention` is most effective on CUDA-enabled hardware with PyTorch 2.0 or newer.*

### 2. Generate Demo Data

Since the original dataset is not public, a script is provided to generate a dummy dataset that mimics the structure of the real data.

```bash
python generate_dummy_data.py
```
This will create several `.csv` and `.json` files in a new `dummy_data/` directory.

### 3. Pre-train the Transformer

Run the pre-training script to train the causal Transformer on the transaction sequences.

```bash
python pretrain.py
```
This will save the trained tokenizer and the model weights to the `nuformer_model/` directory.

### 4. Finetune the Model

You have two options for finetuning.

#### Option A: Joint Fusion (nuFormer)

Run the main finetuning script to adapt the pre-trained model using both sequence and tabular data. This script runs the "Joint Fusion" `nuFormer` model by default.

```bash
python finetune.py
```
This will load the pre-trained weights, combine the Transformer with the DCNv2 for tabular data, and train the end-to-end model, saving the final artifacts in the `nuformer_model/` directory.

#### Option B: Sequence-Only Finetuning

Run the `finetune_v2.py` script for a simplified approach that finetunes the transformer using only the transaction sequences for the downstream task.

```bash
python finetune_v2.py
```
This will save the final model to `finetuned_transformer_classifier.pth`.

### 5. Predict with the Pre-trained Model

After pre-training, you can use the `predict.py` script to see the model in action. It loads the pre-trained transformer and uses it to predict the next few tokens in a user's transaction sequence.

```bash
python predict.py
```
This will fetch a few random user sequences and display the model's predictions versus the actual next tokens.
