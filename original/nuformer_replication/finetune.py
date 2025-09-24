import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import os
import json
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass, asdict

from model import NuFormer
from data_preprocessor import TransactionTokenizer
from pretrain import PretrainConfig  # Reuse config structure


# --- Configuration ---
@dataclass
class FinetuneConfig:
    data_dir: str = "dummy_data"
    model_dir: str = "nuformer_model"
    context_length: int = 512  # Should match pre-training or be smaller
    # DCNv2 params (as per paper's findings)
    numerical_embed_dim: int = 32
    dcn_deep_layers: tuple = (128, 128)
    dcn_cross_layers: int = 3
    dcn_projection_dim: int = 128
    # Training params
    batch_size: int = 32
    learning_rate: float = 5e-5
    epochs: int = 5
    test_size: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Use LoRA for parameter-efficient finetuning, as mentioned in the paper
    use_lora: bool = False  # Set to True to enable LoRA


class FinetuneDataset(Dataset):
    """Dataset for the finetuning task, combining sequences and tabular data."""

    def __init__(
        self,
        user_ids,
        labels,
        tabular_df,
        feature_info,
        sequence_dir,
        tokenizer,
        context_length,
    ):
        self.user_ids = user_ids
        self.labels = labels
        self.tabular_df = tabular_df
        self.feature_info = feature_info
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
            seq = []  # Handle users with no transactions

        # Truncate and convert to tensor
        seq_tensor = torch.tensor(seq, dtype=torch.long)[: self.context_length]
        if len(seq_tensor) == 0:
            # If no sequence, use a single PAD token
            seq_tensor = torch.tensor([self.pad_token_id], dtype=torch.long)

        # Load tabular data
        tabular_row = self.tabular_df[self.tabular_df["user_id"] == user_id].iloc[0]
        # Ensure categorical values are numeric indices (int64)
        cat_series = tabular_row[self.feature_info["categorical_cols"]]
        # Coerce to numeric and fill any NaNs with 0 (assumes 0 is a valid/unknown category)
        cat_series = (
            pd.to_numeric(cat_series, errors="coerce").fillna(0).astype(np.int64)
        )
        cat_data = torch.from_numpy(cat_series.to_numpy(dtype=np.int64))

        # Ensure numerical values are float32
        num_series = tabular_row[self.feature_info["numerical_cols"]]
        num_series = (
            pd.to_numeric(num_series, errors="coerce").fillna(0.0).astype(np.float32)
        )
        num_data = torch.from_numpy(num_series.to_numpy(dtype=np.float32))

        # Concatenate: keep a single float tensor; model will cast the categorical slice back to long
        tabular_tensor = torch.cat([cat_data.to(torch.float32), num_data])

        return seq_tensor, tabular_tensor, torch.tensor(label, dtype=torch.float)


def finetune_collate_batch(batch):
    seqs, tabulars, labels = zip(*batch)
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    tabulars_stacked = torch.stack(tabulars, 0)
    labels_stacked = torch.stack(labels, 0)
    return seqs_padded, tabulars_stacked, labels_stacked


def main():
    config = FinetuneConfig()
    print(f"Using device: {config.device}")

    # --- Load Pre-trained Config and Tokenizer ---
    model_config_path = os.path.join(config.model_dir, "model_config.json")
    with open(model_config_path, "r") as f:
        pretrain_config_dict = json.load(f)
    transformer_config = PretrainConfig(**pretrain_config_dict)
    transformer_config.context_length = config.context_length  # Override context length

    tokenizer_path = os.path.join(config.model_dir, "tokenizer.json")
    tokenizer = TransactionTokenizer.load(tokenizer_path)
    transformer_config.vocab_size = tokenizer.tokenizer.get_vocab_size()

    # --- Load Data ---
    finetune_df = pd.read_csv(os.path.join(config.data_dir, "finetune_data.csv"))
    with open(os.path.join(config.data_dir, "feature_info.json"), "r") as f:
        feature_info = json.load(f)

    # Create a single tabular df for lookup
    tabular_df = finetune_df.drop_duplicates(subset=["user_id"]).reset_index(drop=True)

    # Split data
    train_df, test_df = train_test_split(
        finetune_df,
        test_size=config.test_size,
        random_state=42,
        stratify=finetune_df["label"],
    )

    sequence_dir = os.path.join(config.data_dir, "user_sequences")
    train_dataset = FinetuneDataset(
        train_df["user_id"].tolist(),
        train_df["label"].tolist(),
        tabular_df,
        feature_info,
        sequence_dir,
        tokenizer,
        config.context_length,
    )
    test_dataset = FinetuneDataset(
        test_df["user_id"].tolist(),
        test_df["label"].tolist(),
        tabular_df,
        feature_info,
        sequence_dir,
        tokenizer,
        config.context_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=finetune_collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=finetune_collate_batch,
    )

    # --- Model --- #
    dcn_config = {
        "numerical_embed_dim": config.numerical_embed_dim,
        "deep_layers": config.dcn_deep_layers,
        "cross_layers": config.dcn_cross_layers,
        "projection_dim": config.dcn_projection_dim,
    }
    model = NuFormer(transformer_config, feature_info, dcn_config).to(config.device)

    # Load pre-trained weights
    pretrained_path = os.path.join(config.model_dir, "pretrained_transformer.pth")
    ckpt = torch.load(pretrained_path, map_location=config.device)
    load_ok = False
    # The pretrain script saved the inner ModuleDict (keys like 'wte.weight', 'h.0...').
    # Load into the inner module of our CausalTransformer for a perfect key match.
    try:
        model.transformer.transformer.load_state_dict(ckpt)
        load_ok = True
        print("Loaded pre-trained transformer (inner ModuleDict) weights.")
    except Exception as e:
        # Fallback: try loading into the full transformer by prefixing keys if needed
        try:
            if all(not k.startswith("transformer.") for k in ckpt.keys()):
                ckpt = {f"transformer.{k}": v for k, v in ckpt.items()}
            model.transformer.load_state_dict(ckpt, strict=False)
            load_ok = True
            print("Loaded pre-trained transformer weights with key prefix adaptation.")
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load pretrained weights: {e}\nFallback also failed: {e2}"
            )

    if config.use_lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["c_attn"],  # Target attention layers as is common
                task_type=TaskType.SEQ_CLS,
            )
            model.transformer = get_peft_model(model.transformer, lora_config)
            print("Applied LoRA to the transformer.")
            model.transformer.print_trainable_parameters()
        except ImportError:
            print(
                "Warning: `peft` library not found. Running full finetuning. `pip install peft` to use LoRA."
            )

    print(
        f"Finetuning model with {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters."
    )

    # --- Training & Evaluation --- #
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for i, (seqs, tabs, labels) in enumerate(train_loader):
            seqs, tabs, labels = (
                seqs.to(config.device),
                tabs.to(config.device),
                labels.to(config.device),
            )
            optimizer.zero_grad()
            logits = model(seqs, tabs)
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 20 == 0:
                print(
                    f"Epoch {epoch+1}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )
        print(f"Epoch {epoch+1} Avg Train Loss: {total_loss/len(train_loader):.4f}")

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seqs, tabs, labels in test_loader:
                seqs, tabs, labels = (
                    seqs.to(config.device),
                    tabs.to(config.device),
                    labels.to(config.device),
                )
                logits = model(seqs, tabs)
                preds = torch.sigmoid(logits.squeeze())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} Test AUC: {auc:.4f}")

    # --- Save Final Model ---
    final_model_path = os.path.join(config.model_dir, "finetuned_nuformer.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Finetuned nuFormer model saved to {final_model_path}")


if __name__ == "__main__":
    main()
