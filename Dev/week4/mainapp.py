from fastapi import FastAPI
from pydantic import BaseModel, Field, conint, confloat
from typing import Optional, List
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="Financial Fraud Detection API",
    description="Detects potential fraud in financial transactions using a trained ML model and SHAP explanations.",
    version="1.0.0"
)

# Load model and supporting files at startup
rf_pipeline = joblib.load("rf_pipeline.joblib")
feature_order = joblib.load("model_features.joblib")
explainer = joblib.load("shap_explainer.joblib")

# Helper functions (as refined above)
def to_scalar(value):
    if isinstance(value, (pd.Series, pd.Index)):
        return value.iloc[0] if hasattr(value, 'iloc') else value[0]
    elif isinstance(value, (np.ndarray, list)):
        return value[0]
    else:
        return value

def get_plain_english_explanation(feature, value):
    value = to_scalar(value)
    explanations = {
        "isOrigNeg": "The sender's account balance would go negative after this transaction.",
        "isOrigZero": "The sender's account started with zero balance.",
        "isDestNeg": "The receiver's account balance would go negative after this transaction.",
        "isDestZero": "The receiver's account started with zero balance.",
        "type_CASH_OUT": "This is a cash withdrawal transaction, which is often higher risk.",
        "type_TRANSFER": "This is a standard account transfer transaction.",
        "errorBalanceOrig": "There is an inconsistency in the sender's balance after the transaction.",
        "errorBalanceDest": "There is an inconsistency in the receiver's balance after the transaction."
    }
    if feature in explanations and value == 1:
        return explanations[feature]
    elif feature == "amt_perc_orig":
        if value > 0.9:
            return f"Almost all ({value:.0%}) of the sender's balance is being transferred."
        elif value > 0.5:
            return f"A large portion of the sender's balance ({value:.0%}) is being transferred."
    elif feature == "amount":
        if value > 1_000_000:
            return f"The transaction amount (${value:,.0f}) is extremely high."
        elif value > 100_000:
            return f"The transaction amount (${value:,.0f}) is quite high."
    elif feature == "step":
        return f"This transaction was performed very early in the transaction sequence (Step {value})."
    elif feature == "newbalanceOrig":
        if value < 0:
            return "After this transaction, the sender's account would be overdrawn (negative balance)."
        elif value == 0:
            return "After this transaction, the sender's account balance would be zero."
    elif feature == "oldbalanceDest":
        if value < 1000:
            return "The receiver's account had a low balance before this transaction."
        elif value > 100_000:
            return "The receiver's account had a very high balance before this transaction."
    if feature.startswith("is") and value == 0:
        if feature == "isOrigZero":
            return "The sender's account had a balance before this transaction."
        elif feature == "isOrigNeg":
            return "The sender's account did not go negative after this transaction."
        elif feature == "isDestZero":
            return "The receiver's account had a balance before this transaction."
        elif feature == "isDestNeg":
            return "The receiver's account did not go negative after this transaction."
    label = feature.replace('_', ' ').capitalize()
    return f"{label}: {value}"

def get_fraud_explanations(features, X_input):
    explanations = []
    try:
        shap_values = explainer.shap_values(X_input)
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap_row = shap_values[1]
            else:
                shap_row = shap_values[0]
        else:
            shap_row = shap_values
        if shap_row.ndim > 1:
            shap_row = shap_row[0]
        shap_row = np.array(shap_row).flatten()
        top_indices = np.argsort(np.abs(shap_row))[-3:][::-1]
        feature_names = list(X_input.columns)
        for idx in top_indices:
            if idx < len(feature_names) and abs(shap_row[idx]) > 1e-6:
                feature = feature_names[idx]
                value = X_input.iloc[0, idx]
                explanations.append(get_plain_english_explanation(feature, value))
    except Exception as e:
        # Fallback
        explanations = ["Model detected suspicious patterns in the transaction."]
    return explanations if explanations else ["Model detected suspicious patterns in the transaction."]

# Input schema for the API
class Transaction(BaseModel):
    step: int = Field(..., ge=1, description="Time step of transaction")
    tx_type: str = Field(..., description="Type of transaction, e.g. 'TRANSFER' or 'CASH_OUT'")
    amount: float = Field(..., ge=0, description="Transaction amount")
    sender_balance: float = Field(..., ge=0, description="Sender's current balance before transaction")
    receiver_balance: float = Field(..., ge=0, description="Receiver's current balance before transaction")

class FraudResult(BaseModel):
    fraud_detected: bool
    fraud_probability: float
    explanations: List[str]
    recommended_actions: List[str]

# Main predict endpoint
@app.post("/predict", response_model=FraudResult)
def predict_fraud(transaction: Transaction):
    # Calculate new balances
    new_sender_balance = transaction.sender_balance - transaction.amount
    new_receiver_balance = transaction.receiver_balance + transaction.amount

    features = {
        'step': transaction.step,
        'amount': transaction.amount,
        'oldbalanceOrg': transaction.sender_balance,
        'newbalanceOrig': new_sender_balance,
        'oldbalanceDest': transaction.receiver_balance,
        'newbalanceDest': new_receiver_balance,
        'errorBalanceOrig': transaction.sender_balance - transaction.amount - new_sender_balance,
        'errorBalanceDest': transaction.receiver_balance + transaction.amount - new_receiver_balance,
        'isOrigZero': int(transaction.sender_balance == 0),
        'isDestZero': int(transaction.receiver_balance == 0),
        'isOrigNeg': int(new_sender_balance < 0),
        'isDestNeg': int(new_receiver_balance < 0),
        'amt_perc_orig': transaction.amount / (transaction.sender_balance + 1),
        'amt_perc_dest': transaction.amount / (transaction.receiver_balance + 1),
        'type_TRANSFER': int(transaction.tx_type == 'TRANSFER'),
        'type_CASH_OUT': int(transaction.tx_type == 'CASH_OUT'),
    }
    for col in feature_order:
        if col not in features:
            features[col] = 0
    X_input = pd.DataFrame([features])[feature_order]

    fraud_probability = rf_pipeline.predict_proba(X_input)[0, 1]
    is_fraud = fraud_probability > 0.5

    explanations = get_fraud_explanations(features, X_input)

    # Recommendation logic
    if is_fraud:
        recommended = [
            "Flag transaction for manual review",
            "Verify customer identity",
            "Check transaction history",
            "Consider temporary account restrictions"
        ]
    else:
        recommended = [
            "Process transaction normally",
            "Continue standard monitoring",
            "No additional verification needed"
        ]

    return FraudResult(
        fraud_detected=is_fraud,
        fraud_probability=fraud_probability,
        explanations=explanations,
        recommended_actions=recommended
    )
# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running smoothly."}
