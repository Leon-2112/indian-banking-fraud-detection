
from datetime import date, time

import streamlit as st

from predictor import load_bundle, make_feature_row, predict_row

st.set_page_config(
    page_title="Indian Banking Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Indian Banking Fraud Detection")
st.caption("Experimental machine-learning demonstration | Indian banking transaction features")

st.warning(
    "Research demonstration only — not for real banking, payment blocking, "
    "credit decisions, or customer investigations. The training dataset is "
    "publisher-described synthetic data, not verified bank transaction records. "
    "The held-out test performance is weak (ROC-AUC 0.543 and Average Precision "
    "0.0193), so this model should not be treated as a reliable fraud detector."
)

@st.cache_resource(show_spinner="Loading trained model...")
def get_bundle():
    return load_bundle()

try:
    bundle = get_bundle()
except Exception as exc:
    st.error(f"Could not load the model bundle: {exc}")
    st.stop()

metrics = bundle.get("test_metrics", {})
m1, m2, m3, m4 = st.columns(4)
m1.metric("Test ROC-AUC", f"{float(metrics.get('ROC_AUC', float('nan'))):.3f}")
m2.metric("Test Average Precision", f"{float(metrics.get('Average_Precision', float('nan'))):.4f}")
m3.metric("Test precision", f"{float(metrics.get('Precision', float('nan'))):.1%}")
m4.metric("Test recall", f"{float(metrics.get('Recall', float('nan'))):.1%}")

with st.expander("About this model and its limitations"):
    st.write(f"Selected estimator: **{bundle.get('selected_model_name', 'Unknown')}**")
    st.write(f"Decision threshold selected on validation data: **{float(bundle['threshold_f2_validation']):.4f}**")
    st.write(bundle.get("dataset_note", ""))
    st.markdown(
        "- The displayed score is an **uncalibrated model score**, not a verified real-world fraud probability.\n"
        "- The dataset contains synthetic labels and documented rule-based generation patterns.\n"
        "- Historical test performance indicates weak discrimination; flagged/not-flagged outputs are illustrative only.\n"
        "- Do not enter real customer data, account numbers, card numbers, PINs, OTPs, or other sensitive information."
    )

st.subheader("Enter a sample transaction")
st.write("Use fictitious sample values. Required fields below reproduce the saved model's input features.")

account_types = ["Current", "Fixed Deposit", "NRI", "Salary", "Savings"]
transaction_types = [
    "ATM_Withdrawal", "Auto_Debit", "Cheque", "Credit_Card",
    "IMPS", "NEFT", "Net_Banking", "POS", "RTGS", "UPI"
]
directions = ["Credit", "Debit"]
merchant_categories = [
    "E-Commerce", "Education", "Entertainment", "Food & Dining",
    "Fuel", "Government", "Healthcare", "Insurance", "Investment",
    "Peer Transfer", "Real Estate", "Retail", "Salary", "Travel", "Utilities"
]
states = [
    "Delhi", "Gujarat", "Karnataka", "Maharashtra", "Punjab",
    "Rajasthan", "Tamil Nadu", "Telangana", "UP", "West Bengal"
]
channels = ["API", "ATM", "Branch", "Mobile_App", "POS_Terminal", "Web"]
kyc_statuses = ["Expired", "Pending", "Verified"]
loan_types = ["Auto", "Business", "Education", "Gold", "Home", "Personal", "Unknown"]

with st.form("transaction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        transaction_amount = st.number_input(
            "Transaction amount (INR)", min_value=0.0, value=2500.0,
            step=100.0, help="Enter a fictitious amount in Indian rupees."
        )
        transaction_date = st.date_input("Transaction date", value=date(2024, 1, 15))
        transaction_time = st.time_input("Transaction time", value=time(14, 30))
        transaction_type = st.selectbox("Transaction type", transaction_types, index=9)
        transaction_direction = st.selectbox("Transaction direction", directions, index=1)
    with c2:
        account_type = st.selectbox("Account type", account_types, index=4)
        merchant_category = st.selectbox("Merchant category", merchant_categories, index=11)
        state = st.selectbox("State", states, index=3)
        channel = st.selectbox("Channel", channels, index=3)
        kyc_status = st.selectbox("KYC status", kyc_statuses, index=2)
    with c3:
        credit_missing = st.checkbox("Credit score unavailable")
        credit_score_value = st.number_input(
            "Credit score", min_value=300, max_value=900, value=720, step=1,
            disabled=credit_missing
        )
        has_loan = st.checkbox("Customer has an active loan")
        loan_type = st.selectbox("Loan type", loan_types, disabled=not has_loan)
        emi_amount = st.number_input(
            "Monthly EMI (INR)", min_value=0.0, value=0.0 if not has_loan else 5000.0,
            step=100.0, disabled=not has_loan
        )

    submitted = st.form_submit_button("Evaluate transaction", type="primary", use_container_width=True)

if submitted:
    try:
        row = make_feature_row(
            account_type=account_type,
            transaction_type=transaction_type,
            transaction_amount=transaction_amount,
            transaction_direction=transaction_direction,
            merchant_category=merchant_category,
            state=state,
            credit_score=None if credit_missing else credit_score_value,
            has_loan=has_loan,
            loan_type=loan_type if has_loan else "Not applicable",
            emi_amount=emi_amount,
            channel=channel,
            kyc_status=kyc_status,
            transaction_date=transaction_date,
            transaction_time=transaction_time,
        )
        result = predict_row(bundle, row)

        st.divider()
        st.subheader("Model output")
        r1, r2 = st.columns(2)
        r1.metric("Uncalibrated model score", f"{result['score']:.2%}")
        r2.metric("Validation decision threshold", f"{result['threshold']:.2%}")

        if result["flagged"]:
            st.error("Model flag: ABOVE THRESHOLD — illustrative review flag")
        else:
            st.success("Model flag: BELOW THRESHOLD — this does not establish that the transaction is legitimate")

        st.caption(
            "This is an educational model output. Its test-set results are weak, "
            "and the training labels were synthetically generated. Do not use "
            "this result to approve, reject, block, or investigate real transactions."
        )
    except Exception as exc:
        st.exception(exc)

st.divider()
st.caption("College research prototype • Model inference runs on CPU • No transaction is sent to a bank or payment network.")
