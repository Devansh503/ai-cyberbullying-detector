# AI Cyberbullying Detection System

An end-to-end, real-time, deployable AI system for cyberbullying detection with ML, DL, and Transformer models. Includes a FastAPI backend, Streamlit frontend, explainability (LIME/SHAP), rephrasing support, analytics dashboard, containerized deployment, and cloud setup guidance.

## Features
- Data ingestion from public sources (Hugging Face datasets; optional Kaggle files)
- Preprocessing: cleaning, tokenization, lemmatization (NLTK)
- Models: Logistic Regression, Linear SVM, LSTM, CNN (PyTorch), DistilBERT, RoBERTa
- Model selection and registry; persists the best model with metadata
- FastAPI REST API endpoints:
  - POST /analyze → prediction, category, severity, confidence, explanations
  - POST /rephrase → safer rephrasing of offensive text
  - GET /dashboard → analytics (category distribution, severity distribution, word frequencies, timeline)
  - GET /report/csv and GET /report/pdf → downloadable reports
- Streamlit UI: analyze text, show severity (color-coded), confidence, rephrasing, dashboard, feedback
- Multi-language support via deep-translator + langdetect
- Dockerized; Procfile for Heroku; guidance for AWS/GCP

## Quickstart

1) Python 3.10+ recommended. On Windows, use PowerShell.

2) Create and activate a virtual environment:
- PowerShell:
  - python -m venv .venv
  - .\.venv\Scripts\Activate.ps1

3) Install dependencies:
- pip install -r requirements.txt

4) (Optional) Download NLTK data once:
- python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"

5) Train models and select the best:
- python scripts/train_all.py

6) Run the API:
- uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2

7) Run the Streamlit frontend in another terminal:
- streamlit run src\frontend\streamlit_app.py

8) Open UI at http://localhost:8501. API at http://localhost:8000/docs

## Environment variables
- APP_ENV: dev|prod (default: dev)
- ENABLE_TRANSLATION: 0|1 (default: 1)
- MODEL_DIR: override model registry path if needed

## Data notes
- Uses public datasets via `datasets` library: `tweet_eval:offensive` and `hatexplain`.
- Kaggle datasets can be added: place CSVs under `data/raw/` and the loader will merge when found.

## Deployment
- Docker: build and run
  - docker build -t cyberbully-api .
  - docker run -p 8000:8000 cyberbully-api
- Heroku: Deploy with Docker or Procfile; see deployment/heroku.md
- AWS/GCP: see deployment/aws.md and deployment/gcp.md

## License
For educational use. Ensure compliance with dataset licenses.
# ai-cyberbullying-detector
An end-to-end, real-time, deployable AI system for cyberbullying detection with ML, DL, and Transformer models. Includes a FastAPI backend, Streamlit frontend, explainability (LIME/SHAP), rephrasing support, analytics dashboard, containerized deployment, and cloud setup guidance.

