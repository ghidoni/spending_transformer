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
AMOUNT_BINS = [
    0,
    1,
    5,
    10,
    20,
    30,
    50,
    75,
    100,
    150,
    200,
    300,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    5000,
    10000,
    np.inf,
]


def get_special_tokens():
    """Generates the list of all special tokens for transaction attributes."""
    tokens = []
    # Amount sign
    tokens.extend(["<INFLOW>", "<OUTFLOW>"])
    # Amount bins
    tokens.extend([f"<AMOUNT_{i}>" for i in range(len(AMOUNT_BINS))])
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

    def _tokenize_transaction(self, tx):
        """Converts a single transaction (as a dictionary) into a list of tokens."""
        tokens = []
        # 1. Amount Sign
        tokens.append("<INFLOW>" if tx["amount"] >= 0 else "<OUTFLOW>")

        # 2. Amount Bucket
        abs_amount = abs(tx["amount"])
        bin_index = np.digitize(abs_amount, bins=AMOUNT_BINS)
        tokens.append(f"<AMOUNT_{bin_index}>")

        # 3. Date Features
        date = tx["date"]
        tokens.append(f"<MONTH_{date.month}>")
        tokens.append(f"<DAY_{date.day}>")
        tokens.append(f"<WEEKDAY_{date.weekday()}>")

        # 4. Text Description
        desc_tokens = self.tokenizer.encode(tx["description"]).tokens
        tokens.extend(desc_tokens)

        return tokens

    def encode_user_transactions(self, user_transactions_df):
        """
        Encodes a dataframe of a single user's transactions into a single list of token IDs.
        Transactions are separated by the [SEP] token.
        """
        all_token_ids = []
        sep_token_id = self.tokenizer.token_to_id("[SEP]")

        for i, tx in enumerate(user_transactions_df.to_dict("records")):
            tx_tokens = self._tokenize_transaction(tx)
            tx_ids = self.tokenizer.encode(tx_tokens, is_pretokenized=True).ids
            all_token_ids.extend(tx_ids)
            if i < len(user_transactions_df) - 1:
                all_token_ids.append(sep_token_id)

        return all_token_ids


def prepare_pretrain_data(transactions_df, tokenizer, output_dir):
    """
    Groups transactions by user and saves each user's tokenized sequence to a file.
    This is more memory-efficient for large datasets than keeping all sequences in memory.
    """
    print("Preparing pre-training data...")
    user_sequences_dir = os.path.join(output_dir, "user_sequences")
    if not os.path.exists(user_sequences_dir):
        os.makedirs(user_sequences_dir)

    grouped = transactions_df.groupby("user_id")

    for user_id, user_df in tqdm(grouped, desc="Processing users"):
        token_ids = tokenizer.encode_user_transactions(user_df)
        file_path = os.path.join(user_sequences_dir, f"{user_id}.json")
        with open(file_path, "w") as f:
            json.dump(token_ids, f)

    print(f"User sequences saved to {user_sequences_dir}")


if __name__ == "__main__":
    # This is an example of how to use the preprocessor
    DATA_DIR = "dummy_data"
    MODEL_DIR = "nuformer_model"

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Loading raw transaction data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"), parse_dates=["date"])

    # 1. Train the tokenizer
    # In a real scenario, you might use a much larger sample of descriptions
    tokenizer = TransactionTokenizer(vocab_size=10000)
    tokenizer.train(df["description"].dropna().unique())
    tokenizer.save(os.path.join(MODEL_DIR, "tokenizer.json"))
    print(f"Tokenizer trained and saved to {MODEL_DIR}/tokenizer.json")

    # 2. Prepare the data for pre-training
    prepare_pretrain_data(df, tokenizer, DATA_DIR)
