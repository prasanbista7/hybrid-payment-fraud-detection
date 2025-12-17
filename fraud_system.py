
# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import datetime

# -----------------------------
# Load Data (CSV / Excel)
# -----------------------------

def load_data(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.xlsx'):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format")

# -----------------------------
# Feature Engineering
# -----------------------------

def feature_engineering(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['is_high_amount'] = df['amount'] > 50000
    df['device_tx_count'] = df.groupby('device_id')['transaction_id'].transform('count')
    df['user_tx_count'] = df.groupby('user_id')['transaction_id'].transform('count')
    return df

# -----------------------------
# Rule-Based Risk Scoring
# -----------------------------

def rule_based_score(row):
    score = 0
    if row['amount'] > 50000:
        score += 30
    if row['device_tx_count'] > 5:
        score += 25
    if row['country'] != row['user_country']:
        score += 20
    if row['payment_method'] in ['prepaid_card', 'crypto']:
        score += 15
    return score

# -----------------------------
# Machine Learning Model
# -----------------------------

def train_model(df):
    features = ['amount', 'hour', 'device_tx_count', 'user_tx_count']
    X = df[features]
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\nMODEL EVALUATION")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

    return model

# -----------------------------
# Hybrid Fraud Decision
# -----------------------------

def hybrid_fraud_detection(row, model):
    rule_score = rule_based_score(row)
    ml_features = np.array([
        row['amount'],
        row['hour'],
        row['device_tx_count'],
        row['user_tx_count']
    ]).reshape(1, -1)

    ml_prob = model.predict_proba(ml_features)[0][1] * 100
    final_score = 0.6 * rule_score + 0.4 * ml_prob

    is_fraud = final_score >= 60

    return final_score, is_fraud

# -----------------------------
# Real-Time Fraud Alert (Simulation)
# -----------------------------

def fraud_alert(tx_id, score):
    print(f"🚨 FRAUD ALERT | Transaction {tx_id} | Risk Score: {score:.2f}")

# -----------------------------
# Main Pipeline
# -----------------------------

if __name__ == '__main__':
    df = load_data('transactions.csv')
    df = feature_engineering(df)

    # Train ML model
    model = train_model(df)

    # Apply Hybrid Detection
    scores = []
    flags = []

    for _, row in df.iterrows():
        score, fraud = hybrid_fraud_detection(row, model)
        scores.append(score)
        flags.append(fraud)

        if fraud:
            fraud_alert(row['transaction_id'], score)

    df['risk_score'] = scores
    df['fraud_flag'] = flags

    df.to_csv('fraud_results.csv', index=False)
    print("\nFraud detection completed. Results saved to fraud_results.csv")

