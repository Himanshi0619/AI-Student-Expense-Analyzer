from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from analysis import clean_data, spending_summary
from anomaly_detection import detect_anomalies
from insights import build_prompt, local_summary
from train_classifier import train_model
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "expenses.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    print("\n=== AI STUDENT EXPENSE ANALYZER ===")
    print("Records loaded:", len(df))
    df = clean_data(df)
    print("Valid records:", len(df))
    model, accuracy = train_model(
        DATA_FILE,
        show_metrics=True,
    )
    df["predicted_category"] = model.predict(
        df["description"]
    )
    df["category"] = df["predicted_category"]
    summary = spending_summary(df)
    print("\n=== SPENDING SUMMARY ===")
    print(f"Total Spending: ■{summary['total']:,.2f}")
    print(f"Average Transaction: ■{summary['average']:,.2f}")
    print(f"Top Category: {summary['top_category']}")
    print(f"Top Category Share: {summary['top_percentage']:.1f}%")
    print("\nCategory-wise Spending:")
    print(summary["category_spend"])
    df = detect_anomalies(df)
    unusual = df[df["is_unusual_spending"]]
    print("\nUnusual Spending Alerts:", len(unusual))
    summary["category_spend"].plot(
        kind="bar",
        title="Student Spending by Category",
        xlabel="Category",
        ylabel="Amount (■)",
    )
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "spending_by_category.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    output_csv = OUTPUT_DIR / "analyzed_expenses.csv"
    df.to_csv(output_csv, index=False)
    prompt = build_prompt(
        summary,
        len(unusual),
    )
    print("\n=== STRUCTURED AI PROMPT ===")
    print(prompt)
    print("\n=== SAMPLE AI INSIGHT ===")
    print(local_summary(summary, len(unusual)))
    print("\n=== OUTPUT FILES ===")
    print("Chart:", chart_path)
    print("Analyzed CSV:", output_csv)
    print(f"Model accuracy: {accuracy:.2%}")
if __name__ == "__main__":
    main()

