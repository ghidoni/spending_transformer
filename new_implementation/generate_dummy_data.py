
import pandas as pd
import numpy as np

def generate_dummy_data(num_clients=100, num_transactions_per_client=50):
    data = []
    for client_id in range(num_clients):
        for _ in range(num_transactions_per_client):
            amount = np.random.uniform(1, 1000)
            hour = np.random.randint(0, 24)
            day = np.random.randint(1, 29)
            month = np.random.randint(1, 13)
            day_week = np.random.choice(['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'])
            merchant = np.random.choice(['netflix.com', 'amazon.com', 'walmart.com', 'target.com', 'starbucks.com'])
            transaction_type = np.random.choice(['in', 'out'])
            data.append([client_id, amount, hour, day, month, day_week, merchant, transaction_type])

    df = pd.DataFrame(data, columns=['client_id', 'amount', 'hour', 'day', 'month', 'day_week', 'merchant', 'type'])
    df.to_csv('dummy_transactions.csv', index=False)

if __name__ == "__main__":
    generate_dummy_data()
