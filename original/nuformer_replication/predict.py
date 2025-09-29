import torch
import os
import json
import random

from model import CausalTransformer
from data_preprocessor import TransactionTokenizer
from pretrain import PretrainConfig


def predict_next_tokens(
    model, tokenizer, input_sequence, num_tokens_to_predict=5, device="cpu"
):
    """
    Predicts a sequence of next tokens.

    Args:
        model (CausalTransformer): The pre-trained transformer model.
        tokenizer (TransactionTokenizer): The tokenizer.
        input_sequence (list[int]): A list of token IDs.
        num_tokens_to_predict (int): The number of tokens to predict.
        device (str): The device to run the model on.

    Returns:
        tuple[list[str], list[int]]: Predicted tokens (strings) and their IDs.
    """
    model.eval()
    predicted_tokens = []
    predicted_token_ids = []
    current_sequence = input_sequence.copy()

    with torch.no_grad():
        for _ in range(num_tokens_to_predict):
            # Take the last part of the sequence that fits the model's context length
            context_length = model.config.context_length
            input_tensor = torch.tensor(
                [current_sequence[-context_length:]], dtype=torch.long
            ).to(device)

            # Get the model's prediction (logits)
            logits, _, _ = model(input_tensor)

            # We only care about the prediction for the very last token
            next_token_logits = logits[:, -1, :]

            # Get the token with the highest probability
            predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()

            # Add the predicted token to our list and to the current sequence for the next prediction
            current_sequence.append(predicted_token_id)
            predicted_token_ids.append(predicted_token_id)
            predicted_tokens.append(tokenizer.tokenizer.id_to_token(predicted_token_id))

    return predicted_tokens, predicted_token_ids


def tokens_from_ids(tokenizer, token_ids):
    """Convert a list of token IDs to their string representations."""
    return [tokenizer.tokenizer.id_to_token(tok_id) for tok_id in token_ids]


def format_tokens_with_ids(tokens, token_ids):
    """Render tokens and IDs side-by-side as `token(id)` strings."""
    return " ".join(f"{token}({tok_id})" for token, tok_id in zip(tokens, token_ids))


def main():
    """
    Main function to load the model and run predictions.
    """
    config = PretrainConfig()
    device = config.device
    print(f"Using device: {device}")

    # --- Load Model and Tokenizer ---
    model_dir = "nuformer_model"
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    model_config_path = os.path.join(model_dir, "model_config.json")
    pretrained_path = os.path.join(model_dir, "pretrained_transformer.pth")

    if not all(
        os.path.exists(p) for p in [tokenizer_path, model_config_path, pretrained_path]
    ):
        print("Error: Model files not found. Please run pretrain.py first.")
        return

    # Load tokenizer
    tokenizer = TransactionTokenizer.load(tokenizer_path)

    # Load model config and update vocab size
    with open(model_config_path, "r") as f:
        model_config_dict = json.load(f)

    model_config = PretrainConfig(**model_config_dict)
    model_config.vocab_size = tokenizer.tokenizer.get_vocab_size()

    # Create model and load weights
    model = CausalTransformer(model_config).to(device)

    # The pre-trained weights are for the `transformer` submodule
    # We need to load them into `model.transformer`
    state_dict = torch.load(pretrained_path, map_location=device)
    model.transformer.load_state_dict(state_dict)

    print("Model and tokenizer loaded successfully.")

    # --- Prepare Input Data ---
    sequence_dir = os.path.join("dummy_data", "user_sequences")
    if not os.path.exists(sequence_dir):
        print(
            f"Error: User sequences not found in {sequence_dir}. Please run data_preprocessor.py."
        )
        return

    sequence_files = [f for f in os.listdir(sequence_dir) if f.endswith(".json")]
    if not sequence_files:
        print("No user sequences found.")
        return

    # --- Run Predictions for a Few Different Sequences ---
    num_examples = 3
    num_to_predict = 5
    random.shuffle(sequence_files)

    for i, sequence_file in enumerate(sequence_files[:num_examples]):
        with open(os.path.join(sequence_dir, sequence_file), "r") as f:
            full_sequence = json.load(f)

        if len(full_sequence) <= num_to_predict + 10:
            print(f"\nSkipping {sequence_file}: too short for a meaningful prediction.")
            continue

        # Take a portion of the sequence to predict the next tokens
        slice_end = len(full_sequence) - num_to_predict
        input_sequence = full_sequence[:slice_end]
        actual_next_tokens_ids = full_sequence[slice_end : slice_end + num_to_predict]
        actual_next_tokens = tokens_from_ids(tokenizer, actual_next_tokens_ids)

        # Run prediction
        predicted_tokens, predicted_token_ids = predict_next_tokens(
            model,
            tokenizer,
            input_sequence,
            num_tokens_to_predict=num_to_predict,
            device=device,
        )

        # Display results
        input_sequence_ids_tail = input_sequence[-20:]
        input_tokens_display = tokens_from_ids(tokenizer, input_sequence_ids_tail)
        input_tokens_with_ids_display = format_tokens_with_ids(
            input_tokens_display, input_sequence_ids_tail
        )

        actual_tokens_with_ids = format_tokens_with_ids(
            actual_next_tokens, actual_next_tokens_ids
        )
        predicted_tokens_with_ids = format_tokens_with_ids(
            predicted_tokens, predicted_token_ids
        )

        print("\n" + "=" * 60)
        print(f"        PREDICTION EXAMPLE {i + 1}/{num_examples}        ")
        print("=" * 60)
        print(f"User sequence file: {sequence_file}")
        print(f"Input sequence (last 20 tokens):\n{' '.join(input_tokens_display)}")
        print(f"Input sequence with IDs (last 20):\n{input_tokens_with_ids_display}")
        print("-" * 60)
        print(f"ACTUAL next {num_to_predict} tokens:    {' '.join(actual_next_tokens)}")
        print(f"ACTUAL tokens with IDs:      {actual_tokens_with_ids}")
        print(f"PREDICTED next {num_to_predict} tokens: {' '.join(predicted_tokens)}")
        print(f"PREDICTED tokens with IDs:   {predicted_tokens_with_ids}")
        print("=" * 60)


if __name__ == "__main__":
    main()
