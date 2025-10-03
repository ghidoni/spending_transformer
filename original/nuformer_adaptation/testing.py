import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import json
import pandas as pd
from dataclasses import dataclass

from model import CausalTransformer
from data_preprocessor import TransactionTokenizer
from pretrain import UserSequenceDataset, collate_batch


@dataclass
class PretrainConfig:
    data_dir: str = "data"
    model_dir: str = "nuformer_model"
    # Model params
    vocab_size: int = 10000  # Should match tokenizer
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384  # 64 per head
    dropout: float = 0.1
    context_length: int = 512  # Paper uses up to 2048, reduce for local training
    # Training params
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


config = PretrainConfig()
print(f"Using device: {config.device}")

# Load tokenizer to get vocab size
tokenizer_path = os.path.join(config.model_dir, "tokenizer.json")
if not os.path.exists(tokenizer_path):
    print(f"Error: Tokenizer not found at {tokenizer_path}")
    print("Please run data_preprocessor.py first.")

tokenizer = TransactionTokenizer.load(tokenizer_path)
config.vocab_size = tokenizer.tokenizer.get_vocab_size()
print(f"Tokenizer loaded. Vocab size: {config.vocab_size}")

csv_path = os.path.join(config.data_dir, "transactions.csv")
if not os.path.exists(csv_path):
    print(f"Error: Data file not found at {csv_path}")
    print("Please run generate_dummy_data.py first.")

dataset = UserSequenceDataset(csv_path, tokenizer, config.context_length)
dataloader = DataLoader(
    dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch
)
print(f"Loaded {len(dataset)} user sequences from {csv_path}.")

# print the first element of the batch to verify
for batch in dataloader:
    x, y = batch
    print("Batch x shape:", x.shape)
    print("Batch y shape:", y.shape)
    print(x[0])
    print(y[0])
    break  # Just show one batch for verification
