from sklearn.ensemble import IsolationForest
def detect_anomalies(df):
    df = df.copy()
    detector = IsolationForest(
        n_estimators=100,
        contamination=0.02,
        random_state=42,
    )
    predictions = detector.fit_predict(
        df[["amount"]]
    )
    # Isolation Forest:
    # -1 = unusual
    # +1 = normal
    df["anomaly_label"] = predictions
    df["is_unusual_spending"] = (
        df["anomaly_label"] == -1
    )
    return df
