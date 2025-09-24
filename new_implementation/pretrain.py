
import torch
from transformers import Trainer, TrainingArguments
from tokenizers import Tokenizer
from model import NuFormerForCausalLM, NuFormerConfig
from data_preprocessor import DataPreprocessor
import pandas as pd

def main():
    # Load data and preprocessor
    df = pd.read_csv('dummy_transactions.csv')
    preprocessor = DataPreprocessor("tokenizer.json")
    sequences = preprocessor.process_data(df)

    # Tokenize the data
    tokenizer = preprocessor.tokenizer
    encoded_sequences = tokenizer.encode_batch(sequences)

    # Create dataset
    class NuDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: torch.tensor(val) for key, val in self.encodings[idx].items()}

        def __len__(self):
            return len(self.encodings)

    dataset = NuDataset(encoded_sequences)

    # Model config
    config = NuFormerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=tokenizer.token_to_id("<PAD>"),
        bos_token_id=tokenizer.token_to_id("<SOS>"),
        eos_token_id=tokenizer.token_to_id("<EOS>"),
    )

    # Instantiate model
    model = NuFormerForCausalLM(config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train
    trainer.train()

    # Save model
    model.save_pretrained("./nuformer_pretrained")

if __name__ == "__main__":
    main()
