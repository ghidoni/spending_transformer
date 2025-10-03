import pandas as pd
import numpy as np
import json
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import os
from tqdm import tqdm

# --- Special Tokens --- #

# As mentioned in the paper, numerical features are quantized into bins.
# We use logarithmic bins for the transaction amount, as financial data is often heavy-tailed.
# This is an assumption, as the paper only specifies "21 bins".

VAL_RANGE_BINS = ["0-10", "10-50", "50-100", "100-500", "500-1000", "1000+"]


def get_special_tokens():
    """Generates the list of all special tokens for transaction attributes."""
    tokens = []
    # origin
    tokens.extend(["<account>", "<card>"])
    # Amount sign
    tokens.extend(["<INFLOW>", "<OUTFLOW>"])
    # Amount bins
    tokens.extend([f"<AMOUNT_{i}>" for i in VAL_RANGE_BINS])
    # Date features
    tokens.extend([f"<MONTH_{i}>" for i in range(1, 13)])
    tokens.extend([f"<DAY_{i}>" for i in range(1, 32)])
    tokens.extend([f"<WEEKDAY_{i}>" for i in range(7)])
    # Separator
    tokens.append("[SEP]")
    return tokens


class TransactionTokenizer:
    """
    A tokenizer that handles both the special transaction tokens and the
    natural language from transaction descriptions.
    """

    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.special_tokens = get_special_tokens()
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer.decoder = decoders.BPEDecoder()

    def train(self, descriptions):
        """Train the BPE tokenizer on the transaction descriptions."""
        print("Training BPE tokenizer on transaction descriptions...")
        # We add special tokens to the trainer so they are not split
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]"] + self.special_tokens,
        )
        self.tokenizer.train_from_iterator(descriptions, trainer=trainer)

    def save(self, path):
        """Save the tokenizer to a file."""
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path):
        """Load a tokenizer from a file."""
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(path)
        # Manually set special tokens from the loaded vocab if needed
        instance.special_tokens = [
            token
            for token in instance.tokenizer.get_vocab()
            if token.startswith("<") or token.startswith("[")
        ]
        return instance

    def encode_user_transactions(self, user_transactions_df):
        pass


if __name__ == "__main__":
    # This is an example of how to use the preprocessor
    DATA_DIR = "data"
    MODEL_DIR = "nuformer_model"

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Loading raw transaction data...")
    # TODO: need to think about reading only a fraction of the data
    df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))

    # 1. Train the tokenizer
    # In a real scenario, you might use a much larger sample of descriptions
    tokenizer = TransactionTokenizer(vocab_size=10000)
    tokenizer.train(df["transaction_history"])
    tokenizer.save(os.path.join(MODEL_DIR, "tokenizer.json"))
    print(f"Tokenizer trained and saved to {MODEL_DIR}/tokenizer.json")
