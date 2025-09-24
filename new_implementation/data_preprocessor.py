
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class DataPreprocessor:
    def __init__(self, tokenizer_path=None):
        if tokenizer_path:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
            self.tokenizer.pre_tokenizer = Whitespace()

    def train_tokenizer(self, data, vocab_size=1000, special_tokens=["<PAD>", "<SOS>", "<EOS>", "<EOT>", "<UNK>"]):
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        self.tokenizer.train_from_iterator(data, trainer)

    def save_tokenizer(self, path):
        self.tokenizer.save(path)

    def _get_amount_bucket(self, amount):
        if amount < 100:
            return "0-100"
        elif amount < 500:
            return "100-500"
        else:
            return "500+"

    def create_transaction_string(self, row):
        amount_bucket = self._get_amount_bucket(row['amount'])
        return f"<{row['type']}><amount:{amount_bucket}><hour:{row['hour']}><day:{row['day']}><month:{row['month']}><day_week:{row['day_week']}>{row['merchant']}"

    def process_data(self, df):
        df['transaction_str'] = df.apply(self.create_transaction_string, axis=1)
        sequences = df.groupby('client_id')['transaction_str'].apply(lambda x: "<EOT>".join(x)).tolist()
        return [f"<SOS>{s}<EOS>" for s in sequences]

def main():
    df = pd.read_csv('dummy_transactions.csv')
    preprocessor = DataPreprocessor()
    sequences = preprocessor.process_data(df)
    preprocessor.train_tokenizer(sequences)
    preprocessor.save_tokenizer("tokenizer.json")

if __name__ == "__main__":
    main()
