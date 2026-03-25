# 🔐 URL Threat Detector

ML-powered web app to classify URLs as safe or malicious.
Built with LightGBM + Streamlit.

## Features
- Real-time URL threat classification
- Dark cyber-themed UI
- Feature extraction (entropy, TLD, special chars, etc.)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model
- Algorithm: LightGBM (tuned with Optuna)
- Dataset: Labelled URL dataset (malicious vs benign)