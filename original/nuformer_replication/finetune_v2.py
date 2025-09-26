import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass

from model import CausalTransformer
from data_preprocessor import TransactionTokenizer
from pretrain import PretrainConfig

# --- Configuration for V2 ---
@dataclass
class FinetuneV2Config:
    data_dir: str = "dummy_data"
    model_dir: str = "nuformer_model"
    context_length: int = 512
    batch_size: int = 32
    learning_rate: float = 5e-5
    epochs: int = 5
    test_size: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_lora: bool = False

# --- Simplified Model for Sequence-Only Finetuning ---
class SequenceClassifier(nn.Module):
    """ A simple classifier that uses the pre-trained CausalTransformer. """
    def __init__(self, transformer_config):
        super().__init__()
        self.transformer = CausalTransformer(transformer_config)
        # MLP head for binary classification
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_config.n_embd),
            nn.Linear(transformer_config.n_embd, transformer_config.n_embd // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(transformer_config.n_embd // 2, 1)
        )

    def forward(self, seq_data):
        # Get embeddings from the transformer
        _, _, transformer_embeddings = self.transformer(seq_data)
        # Use the embedding of the last token for classification
        seq_embedding = transformer_embeddings[:, -1, :]
        # Get logits from the classifier head
        logits = self.classifier(seq_embedding)
        return logits

# --- Simplified Dataset for Sequences and Labels ---
class FinetuneSequenceDataset(Dataset):
    """ Dataset for finetuning on sequences and labels only. """
    def __init__(self, user_ids, labels, sequence_dir, tokenizer, context_length):
        self.user_ids = user_ids
        self.labels = labels
        self.sequence_dir = sequence_dir
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.pad_token_id = tokenizer.tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        label = self.labels[idx]

        # Load sequence
        seq_path = os.path.join(self.sequence_dir, f"{user_id}.json")
        try:
            with open(seq_path, "r") as f:
                seq = json.load(f)
        except FileNotFoundError:
            seq = []

        seq_tensor = torch.tensor(seq, dtype=torch.long)[:self.context_length]
        if len(seq_tensor) == 0:
            seq_tensor = torch.tensor([self.pad_token_id], dtype=torch.long)

        return seq_tensor, torch.tensor(label, dtype=torch.float)

# --- Simplified Collate Function ---
def finetune_v2_collate_batch(batch):
    seqs, labels = zip(*batch)
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    labels_stacked = torch.stack(labels, 0)
    return seqs_padded, labels_stacked

def main():
    config = FinetuneV2Config()
    print(f"Using device: {config.device}")

    # --- Load Pre-trained Config and Tokenizer ---
    model_config_path = os.path.join(config.model_dir, "model_config.json")
    with open(model_config_path, "r") as f:
        pretrain_config_dict = json.load(f)
    transformer_config = PretrainConfig(**pretrain_config_dict)
    transformer_config.context_length = config.context_length

    tokenizer_path = os.path.join(config.model_dir, "tokenizer.json")
    tokenizer = TransactionTokenizer.load(tokenizer_path)
    transformer_config.vocab_size = tokenizer.tokenizer.get_vocab_size()

    # --- Load Data (no tabular features) ---
    finetune_df = pd.read_csv(os.path.join(config.data_dir, "finetune_data.csv"))
    
    # Split data
    train_df, test_df = train_test_split(
        finetune_df,
        test_size=config.test_size,
        random_state=42,
        stratify=finetune_df["label"],
    )

    sequence_dir = os.path.join(config.data_dir, "user_sequences")
    train_dataset = FinetuneSequenceDataset(
        train_df["user_id"].tolist(),
        train_df["label"].tolist(),
        sequence_dir,
        tokenizer,
        config.context_length,
    )
    test_dataset = FinetuneSequenceDataset(
        test_df["user_id"].tolist(),
        test_df["label"].tolist(),
        sequence_dir,
        tokenizer,
        config.context_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=finetune_v2_collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=finetune_v2_collate_batch,
    )

    # --- Model (SequenceClassifier) ---
    model = SequenceClassifier(transformer_config).to(config.device)

    # Load pre-trained weights into the transformer part of the model
    pretrained_path = os.path.join(config.model_dir, "pretrained_transformer.pth")
    ckpt = torch.load(pretrained_path, map_location=config.device)
    model.transformer.transformer.load_state_dict(ckpt)
    print("Loaded pre-trained transformer weights.")

    if config.use_lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_config = LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["c_attn"], task_type=TaskType.SEQ_CLS
            )
            model.transformer = get_peft_model(model.transformer, lora_config)
            print("Applied LoRA to the transformer.")
            model.transformer.print_trainable_parameters()
        except ImportError:
            print("Warning: `peft` library not found. `pip install peft` to use LoRA.")

    print(f"Finetuning model with {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")

    # --- Training & Evaluation ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for i, (seqs, labels) in enumerate(train_loader):
            seqs, labels = seqs.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 20 == 0:
                print(f"Epoch {epoch+1}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1} Avg Train Loss: {total_loss/len(train_loader):.4f}")

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seqs, labels in test_loader:
                seqs, labels = seqs.to(config.device), labels.to(config.device)
                logits = model(seqs)
                preds = torch.sigmoid(logits.squeeze())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} Test AUC: {auc:.4f}")

    # --- Save Final Model ---
    final_model_path = os.path.join(config.model_dir, "finetuned_transformer_classifier.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Finetuned sequence classifier model saved to {final_model_path}")

if __name__ == "__main__":
    main()
