from pathlib import Path
import random
import numpy as np
import pandas as pd
random.seed(42)
np.random.seed(42)
CATEGORIES = {
    "Food": ["canteen lunch", "dinner", "snacks", "restaurant", "coffee", "groceries"],
    "Travel": ["bus", "metro", "auto", "cab", "train", "fuel"],
    "Education": ["books", "notebook", "online course", "printing", "stationery", "exam fee"],
    "Shopping": ["clothes", "shoes", "bag", "accessories", "online order"],
    "Entertainment": ["movie", "concert", "game", "streaming", "outing"],
    "Utilities": ["mobile recharge", "internet", "electricity", "water"],
}
CATEGORY_WEIGHTS = [0.30, 0.18, 0.18, 0.14, 0.10, 0.10]
BASE_AMOUNT = {
    "Food": 180,
    "Travel": 220,
    "Education": 450,
    "Shopping": 800,
    "Entertainment": 500,
    "Utilities": 350,
}
def generate_dataset(number_of_transactions=600):
    rows = []
    for i in range(number_of_transactions):
        category = random.choices(
            list(CATEGORIES.keys()),
            weights=CATEGORY_WEIGHTS,
            k=1,
        )[0]
        description = random.choice(CATEGORIES[category])
        amount = max(
            40,
            np.random.lognormal(
                np.log(BASE_AMOUNT[category]),
                0.45,
            ),
        )
        rows.append(
            [
                f"E{i + 1:03d}",
                description,
                round(float(amount), 2),
                category,
            ]
        )
    df = pd.DataFrame(
        rows,
        columns=["transaction_id", "description", "amount", "category"],
    )
    anomaly_indices = np.random.choice(
        df.index,
        12,
        replace=False,
    )
    df.loc[anomaly_indices, "amount"] *= np.random.uniform(
        4,
        8,
        len(anomaly_indices),
    )
    df["amount"] = df["amount"].round(2)
    return df
def main():
    base_dir = Path(__file__).resolve().parents[1]
    output_path = base_dir / "data" / "expenses.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(output_path, index=False)
    print("Dataset successfully created.")
    print("Number of transactions:", len(df))
    print("Saved at:", output_path)
if __name__ == "__main__":
    main()
