import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta
import os

# --- Configuration ---
NUM_USERS = 1000
NUM_TRANSACTIONS = 50000
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"
OUTPUT_DIR = "data"

MERCHANT_DESCRIPTIONS = [
    "Netflix.com",
    "Amazon Prime Video",
    "Spotify AB",
    "Apple Store",
    "Google Play",
    "Uber Trip",
    "Lyft Ride",
    "DoorDash",
    "Grubhub",
    "Instacart",
    "Starbucks",
    "McDonald's",
    "Whole Foods Market",
    "Walmart Supercenter",
    "Target",
    "Shell Gas Station",
    "ExxonMobil",
    "PG&E Utility Payment",
    "Comcast Xfinity",
    "Transfer from savings",
    "Credit Card Payment",
]

# As per the paper, there are 291 tabular features. We'll generate a smaller number for simplicity.
NUM_TABULAR_FEATURES = 50


def get_amount_bin(amount):
    """Categorizes the transaction amount into predefined bins."""
    abs_amount = abs(amount)
    if abs_amount <= 10:
        return "0-10"
    elif abs_amount <= 50:
        return "10-50"
    elif abs_amount <= 100:
        return "50-100"
    elif abs_amount <= 500:
        return "100-500"
    elif abs_amount <= 1000:
        return "500-1000"
    else:
        return "1000+"


def generate_transactions(users):
    """Generates a DataFrame of synthetic user transaction sequences."""
    print("Generating user transaction sequences...")
    user_sequences = []

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    time_diff_seconds = int((end_dt - start_dt).total_seconds())

    for user_id in users:
        num_user_transactions = random.randint(5, 50)

        dates = [
            start_dt + timedelta(seconds=random.randint(0, time_diff_seconds))
            for _ in range(num_user_transactions)
        ]
        dates.sort()

        amounts = np.random.lognormal(mean=3.5, sigma=1.5, size=num_user_transactions)
        amounts[np.random.rand(num_user_transactions) < 0.5] *= -1
        amounts = np.round(amounts, 2)

        descriptions = [
            random.choice(MERCHANT_DESCRIPTIONS) for _ in range(num_user_transactions)
        ]

        sequence_parts = []
        for i in range(num_user_transactions):
            date = dates[i]
            amount = amounts[i]
            description = descriptions[i]

            origin_token = random.choice(["<account>", "<card>"])
            sign_token = "<INFLOW>" if amount >= 0 else "<OUTFLOW>"
            amount_token = f"<AMOUNT_{get_amount_bin(amount)}>"
            month_token = f"<MONTH_{date.month}>"
            day_token = f"<DAY_{date.day}>"
            weekday_token = f"<WEEKDAY_{date.weekday()}>"

            transaction_str = (
                f"{origin_token} {sign_token} {amount_token} "
                f"{month_token} {day_token} {weekday_token} "
                f"{description} [SEP]"
            )
            sequence_parts.append(transaction_str)

        user_sequences.append(
            {"user_id": user_id, "transaction_history": " ".join(sequence_parts)}
        )

    df = pd.DataFrame(user_sequences)
    return df


def generate_downstream_labels(users):
    """Generates a DataFrame with user labels for a downstream task."""
    print("Generating downstream task labels...")
    # In the paper, the same user can appear multiple times with different labels/timestamps
    num_rows = int(len(users) * 1.2)
    user_ids = random.choices(users, k=num_rows)

    labels = np.random.randint(0, 2, size=num_rows)

    df = pd.DataFrame({"user_id": user_ids, "label": labels})
    return df


def generate_tabular_features(users):
    """Generates a DataFrame of synthetic tabular features for each user."""
    print(f"Generating {NUM_TABULAR_FEATURES} tabular features...")
    num_categorical = NUM_TABULAR_FEATURES // 3
    num_numerical = NUM_TABULAR_FEATURES - num_categorical

    data = {"user_id": users}
    # Numerical features
    for i in range(num_numerical):
        data[f"numerical_feat_{i}"] = np.random.randn(len(users))

    # Categorical features
    for i in range(num_categorical):
        num_categories = random.randint(3, 10)
        data[f"categorical_feat_{i}"] = np.random.randint(
            0, num_categories, size=len(users)
        )

    df = pd.DataFrame(data)
    return df


def main():
    """Main function to generate and save all dummy data."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    users = [f"user_{i}" for i in range(NUM_USERS)]

    # Generate and save data
    transactions_df = generate_transactions(users)
    transactions_df.to_csv(os.path.join(OUTPUT_DIR, "transactions.csv"), index=False)

    labels_df = generate_downstream_labels(users)
    tabular_df = generate_tabular_features(users)

    # Join labels and tabular features for the finetuning task
    finetune_df = pd.merge(labels_df, tabular_df, on="user_id", how="left")
    finetune_df.to_csv(os.path.join(OUTPUT_DIR, "finetune_data.csv"), index=False)

    # Save feature info for the model
    feature_info = {
        "numerical_cols": [col for col in tabular_df.columns if "numerical" in col],
        "categorical_cols": [col for col in tabular_df.columns if "categorical" in col],
        "cat_cardinalities": {
            col: int(tabular_df[col].nunique())
            for col in tabular_df.columns
            if "categorical" in col
        },
    }
    with open(os.path.join(OUTPUT_DIR, "feature_info.json"), "w") as f:
        json.dump(feature_info, f, indent=4)

    print(f"\nDummy data successfully generated in '{OUTPUT_DIR}/' directory.")


if __name__ == "__main__":
    main()
