from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
from analysis import clean_data, spending_summary
from anomaly_detection import detect_anomalies
from insights import local_summary
from train_classifier import train_model
st.set_page_config(
    page_title="AI Student Expense Analyzer",
    page_icon="■",
    layout="wide",
)
@st.cache_resource
def get_model():
    training_file = BASE_DIR / "data" / "expenses.csv"
    model, accuracy = train_model(
        training_file,
        show_metrics=False,
    )
    return model, accuracy
st.title("■ AI-Powered Student Expense Analyzer")
st.caption(
    "AI Fundamentals Project | Edunet Foundation × AICTE"
)
st.write(
    """
    Upload your expense data to analyze spending patterns
    using Artificial Intelligence and Machine Learning.
    """
)
uploaded_file = st.file_uploader(
    "Upload Expense CSV",
    type=["csv"],
)
if uploaded_file is None:
    st.info(
        """
        Required columns:
         transaction_id
         description
         amount
        The category column is optional.
        """
    )
    st.stop()
df = pd.read_csv(uploaded_file)
required_columns = {
    "transaction_id",
    "description",
    "amount",
}
missing_columns = (
    required_columns - set(df.columns)
)
if missing_columns:
    st.error(
        "Missing required columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()
df = clean_data(df)
if len(df) < 10:
    st.warning(

        "Please upload at least 10 valid transactions."
    )
    st.stop()
model, accuracy = get_model()
df["predicted_category"] = model.predict(
    df["description"]
)
df["category"] = df["predicted_category"]
summary = spending_summary(df)
df = detect_anomalies(df)
unusual = df[df["is_unusual_spending"]]
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Total Spending",
        f"■{summary['total']:,.0f}",
    )
with col2:
    st.metric(
        "Average Transaction",
        f"■{summary['average']:,.0f}",
    )
with col3:
    st.metric(
        "Top Category",
        summary["top_category"],
    )
with col4:
    st.metric(
        "Unusual Spending Alerts",
        len(unusual),
    )
st.divider()
left, right = st.columns(2)
with left:
    st.subheader("■ Spending by Category")
    fig, ax = plt.subplots()
    summary["category_spend"].plot(
        kind="bar",
        ax=ax,
    )
    ax.set_xlabel("Category")
    ax.set_ylabel("Amount (■)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("■ AI Spending Insight")
    st.info(
        local_summary(
            summary,
            len(unusual),
        )
    )
    st.caption(
        "The insight is based on calculated data. "
        "It does not make financial decisions for the user."
    )
st.divider()
st.subheader("■■ Unusual Spending")
if unusual.empty:

    st.success(
        "No unusual spending pattern was detected."
    )
else:
    display_columns = [
        "transaction_id",
        "description",
        "amount",
        "category",
    ]
    unusual_display = (
        unusual[display_columns]
        .sort_values(
            "amount",
            ascending=False,
        )
    )
    st.dataframe(
        unusual_display,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        """
        These transactions differ noticeably from the
        normal spending pattern. They are alerts for the
        user to check, not fraud predictions.
        """
    )
st.subheader("■ Analyzed Transactions")
st.dataframe(
    df[
        [
            "transaction_id",
            "description",
            "amount",
            "category",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)
st.download_button(
    label="■■ Download Analyzed CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="analyzed_expenses.csv",
    mime="text/csv",
)
with st.expander("Model Information"):
    st.write(
        f"Test-set classification accuracy: {accuracy:.2%}"
    )
    st.write(
        "Text classification: TF-IDF + Logistic Regression"
    )
    st.write(
        "Unusual spending detection: Isolation Forest"
    )

README.md
