import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
def train_model(csv_path, show_metrics=False):
    df = pd.read_csv(csv_path)
    x_train, x_test, y_train, y_test = train_test_split(
        df["description"],
        df["category"],
        test_size=0.20,
        random_state=42,
        stratify=df["category"],
    )
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(
        y_test,
        predictions,
    )
    if show_metrics:
        print(f"Classification Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )
    return model, accuracy
