# Indian Banking Fraud Detection — Streamlit Demo

A small Streamlit application wrapping the submitted `indian_fraud_detection_model_bundle.joblib`.

## Important research and safety limitations

- The model was trained on publisher-described synthetic Indian banking transaction data, not verified real bank transactions.
- Its supplied held-out test metrics are weak: ROC-AUC 0.5430, Average Precision 0.01929, precision 0.07277, recall 0.08564.
- The model score is not calibrated and must not be described as a reliable fraud probability.
- This app is an educational demonstration only. Do not use it to block payments, approve/reject financial activity, or make real customer decisions.
- Do not enter real customer, account, card, PIN, OTP, or other sensitive information.

## Run locally

Use Python 3.12 for compatibility with the serialized estimator versions.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

## Deploy using Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the contents of this folder, preserving the `model/` directory.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Create an app and select your repository, branch, and `app.py`.
5. In Advanced settings, select Python 3.12.
6. Deploy.

Streamlit Community Cloud installs dependencies from `requirements.txt`.

## Model inputs

The app reconstructs the exact 19 feature columns expected by the saved preprocessor. The date/time-derived fields are created in `predictor.py`. `loan_type` is set to `Not applicable` when `has_loan=0`, or to the selected/unknown value when `has_loan=1`.

The serialized model bundle is loaded with joblib. Only deploy the exact model file you created and trust.
