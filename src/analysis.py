import pandas as pd
def clean_data(df):
    df = df.copy()
    df["transaction_id"] = (
        df["transaction_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    df["description"] = (
        df["description"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )
    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    )
    df = df[
        (df["transaction_id"] != "")
        & (df["description"] != "")
        & (df["amount"] > 0)
    ].copy()
    df = df.drop_duplicates(
        subset="transaction_id",
    )
    return df.reset_index(drop=True)
def spending_summary(df):
    total = float(df["amount"].sum())
    average = float(df["amount"].mean())
    category_spend = (
        df.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )
    if total > 0 and not category_spend.empty:
        top_category = str(category_spend.index[0])
        top_amount = float(category_spend.iloc[0])
        top_percentage = top_amount / total * 100
    else:
        top_category = "N/A"
        top_percentage = 0.0
    return {
        "total": total,
        "average": average,
        "category_spend": category_spend,
        "top_category": top_category,
        "top_percentage": top_percentage,
    }
