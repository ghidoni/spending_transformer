
import torch
from transformers import Trainer, TrainingArguments
from tokenizers import Tokenizer
from model import NuFormerForSequenceClassification, NuFormerConfig
from data_preprocessor import DataPreprocessor
import pandas as pd
import numpy as np

def main():
    # Load data and preprocessor
    df = pd.read_csv('dummy_transactions.csv')
    # Add dummy labels for classification
    df['label'] = np.random.randint(0, 2, df.shape[0])
    labels = df.groupby('client_id')['label'].first().tolist()

    preprocessor = DataPreprocessor("tokenizer.json")
    sequences = preprocessor.process_data(df)

    # Tokenize the data
    tokenizer = preprocessor.tokenizer
    encoded_sequences = tokenizer.encode_batch(sequences)

    # Create dataset
    class NuDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    dataset = NuDataset(encoded_sequences, labels)

    # Model config
    config = NuFormerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=tokenizer.token_to_id("<PAD>"),
        bos_token_id=tokenizer.token_to_id("<SOS>"),
        eos_token_id=tokenizer.token_to_id("<EOS>"),
        num_labels=2,
    )

    # Instantiate model from pretrained
    model = NuFormerForSequenceClassification.from_pretrained("./nuformer_pretrained", config=config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results_finetuned",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
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
    model.save_pretrained("./nuformer_finetuned")

if __name__ == "__main__":
    main()
