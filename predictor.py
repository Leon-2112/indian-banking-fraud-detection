
from __future__ import annotations

from datetime import date, time, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "indian_fraud_detection_model_bundle.joblib"

def load_bundle() -> dict[str, Any]:
    """Load the trusted, bundled model and switch inference to CPU."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {MODEL_PATH}. "
            "Make sure the model/ folder is included in the deployment repository."
        )

    bundle = joblib.load(MODEL_PATH)
    required = {
        "selected_model_name", "preprocessor", "model",
        "threshold_f2_validation", "feature_columns",
        "numeric_columns", "categorical_columns", "test_metrics",
        "dataset_note",
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing required keys: {sorted(missing)}")

    # Hosted inference does not require a GPU. This avoids CUDA dependency.
    model = bundle["model"]
    if hasattr(model, "set_params"):
        model.set_params(device="cpu")

    return bundle


def make_feature_row(
    *,
    account_type: str,
    transaction_type: str,
    transaction_amount: float,
    transaction_direction: str,
    merchant_category: str,
    state: str,
    credit_score: float | None,
    has_loan: bool,
    loan_type: str | None,
    emi_amount: float | None,
    channel: str,
    kyc_status: str,
    transaction_date: date,
    transaction_time: time,
) -> pd.DataFrame:
    """Recreate the 19 model features from app inputs."""
    amount = float(transaction_amount)
    if not np.isfinite(amount) or amount < 0:
        raise ValueError("Transaction amount must be a finite, non-negative number.")

    hour = int(transaction_time.hour)
    dt = transaction_date
    loan_flag = int(bool(has_loan))

    if loan_flag == 0:
        normalized_loan_type = "Not applicable"
        emi = 0.0 if emi_amount is None else float(emi_amount)
    else:
        normalized_loan_type = loan_type or "Unknown"
        emi = np.nan if emi_amount is None else float(emi_amount)

    row = {
        "account_type": account_type,
        "transaction_type": transaction_type,
        "transaction_amount": amount,
        "transaction_direction": transaction_direction,
        "merchant_category": merchant_category,
        "state": state,
        "credit_score": np.nan if credit_score is None else float(credit_score),
        "has_loan": loan_flag,
        "loan_type": normalized_loan_type,
        "emi_amount": emi,
        "channel": channel,
        "kyc_status": kyc_status,
        "transaction_hour": hour,
        "transaction_dayofweek": int(dt.weekday()),
        "transaction_month": int(dt.month),
        "is_weekend": int(dt.weekday() >= 5),
        "is_late_night": int(hour >= 22 or hour < 6),
        "amount_log1p": float(np.log1p(amount)),
        "amount_over_5lakh": int(amount > 500_000),
    }
    return pd.DataFrame([row])


def predict_row(bundle: dict[str, Any], row: pd.DataFrame) -> dict[str, Any]:
    """Return model score and threshold flag, using the saved preprocessor."""
    expected = list(bundle["feature_columns"])
    missing = [c for c in expected if c not in row.columns]
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")

    # Enforce the exact original feature order.
    X = row.loc[:, expected]
    transformed = bundle["preprocessor"].transform(X)
    probabilities = bundle["model"].predict_proba(transformed)
    score = float(probabilities[0, 1])
    threshold = float(bundle["threshold_f2_validation"])

    return {
        "score": score,
        "threshold": threshold,
        "flagged": bool(score >= threshold),
    }
