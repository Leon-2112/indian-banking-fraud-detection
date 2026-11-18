# Deployment checklist

1. Extract `indian_fraud_streamlit_deployment.zip`.
2. Create a GitHub repository (public is simplest for Community Cloud discovery; use private only if your Streamlit account/repo access permits it).
3. Upload the contents of the extracted folder to the repository root. Confirm `app.py`, `predictor.py`, `requirements.txt`, `runtime.txt`, and `model/indian_fraud_detection_model_bundle.joblib` are at the listed paths.
4. Visit https://share.streamlit.io/ and sign in with GitHub.
5. Choose **Create app**, select your repository, branch, and `app.py`.
6. Open **Advanced settings** and select Python **3.12**.
7. Deploy. The first build can take several minutes while dependencies install.
8. When the app launches, try a fictional sample transaction. Confirm the page displays the model score and threshold.
9. Do not enter real customer or account information.

The app switches the saved XGBoost estimator to CPU inference, so a hosted GPU is not required.
