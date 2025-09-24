import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import json
import glob
from dataclasses import dataclass

from model import CausalTransformer
from data_preprocessor import TransactionTokenizer

# --- Configuration ---
@dataclass
class PretrainConfig:
    data_dir: str = "dummy_data"
    model_dir: str = "nuformer_model"
    # Model params
    vocab_size: int = 10000 # Should match tokenizer
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384 # 64 per head
    dropout: float = 0.1
    context_length: int = 512 # Paper uses up to 2048, reduce for local training
    # Training params
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class UserSequenceDataset(Dataset):
    """Dataset to load user transaction sequences from files."""
    def __init__(self, sequence_dir, context_length):
        self.file_paths = glob.glob(os.path.join(sequence_dir, "*.json"))
        self.context_length = context_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'r') as f:
            full_sequence = torch.tensor(json.load(f), dtype=torch.long)
        
        # Truncate sequence to max context length + 1 for input/target split
        seq = full_sequence[:self.context_length + 1]
        return seq

def collate_batch(batch):
    """Collate function to handle padding and creating input/target pairs."""
    # Separate inputs (x) and targets (y)
    inputs = [item[:-1] for item in batch]
    targets = [item[1:] for item in batch]
    
    # Pad sequences to the length of the longest sequence in the batch
    x_padded = pad_sequence(inputs, batch_first=True, padding_value=0) # Assuming 0 is pad_token_id
    y_padded = pad_sequence(targets, batch_first=True, padding_value=-1) # Use -1 to ignore in loss
    
    return x_padded, y_padded

def main():
    config = PretrainConfig()
    print(f"Using device: {config.device}")

    # --- Setup --- #
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    # Load tokenizer to get vocab size
    tokenizer_path = os.path.join(config.model_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("Please run data_preprocessor.py first.")
        return
    tokenizer = TransactionTokenizer.load(tokenizer_path)
    config.vocab_size = tokenizer.tokenizer.get_vocab_size()
    print(f"Tokenizer loaded. Vocab size: {config.vocab_size}")

    # --- Data --- #
    sequence_dir = os.path.join(config.data_dir, "user_sequences")
    dataset = UserSequenceDataset(sequence_dir, config.context_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    print(f"Found {len(dataset)} user sequences.")

    # --- Model --- #
    model = CausalTransformer(config).to(config.device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # --- Training --- #
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(config.device), y.to(config.device)
            
            optimizer.zero_grad()
            
            logits, loss, _ = model(x, y)
            
            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{config.epochs}, Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"End of Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    # --- Save Model --- #
    # We save the transformer part of the model, not the LM head, 
    # as we'll be using it for feature extraction.
    model_save_path = os.path.join(config.model_dir, "pretrained_transformer.pth")
    torch.save(model.transformer.state_dict(), model_save_path)
    
    # Save config for loading in finetuning script
    config_save_path = os.path.join(config.model_dir, "model_config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config.__dict__, f, indent=4)

    print(f"Pre-trained transformer model saved to {model_save_path}")
    print(f"Model config saved to {config_save_path}")

if __name__ == "__main__":
    main()
