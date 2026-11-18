# Model Card — Indian Banking Fraud Detection (Educational Demo)

- Estimator: XGBoost binary classifier
- Training data: publisher-described synthetic Indian banking transactions benchmark
- Dataset size reported by training notebook: 550,000 rows
- Selected validation threshold: 0.645349
- Held-out test ROC-AUC: 0.543046
- Held-out test Average Precision: 0.019293
- Held-out test precision: 0.072770
- Held-out test recall: 0.085635
- Held-out test Brier score: 0.224118

## Intended use
Educational demonstration of model deployment and limitations.

## Out-of-scope use
Real-world transaction monitoring, payment authorization/blocking, customer investigations, credit decisions, or regulatory compliance.

## Limitations
Training labels are synthetic. The model demonstrates weak held-out discrimination and low precision/recall. Its score is not calibrated. The training dataset also includes documented synthetic rule-generation patterns.
