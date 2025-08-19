# Spam Detection System (API + DB + Dashboard)

## What you get
- FastAPI service with `/predict`, `/messages`, `/health`
- PostgreSQL storage for predictions
- Streamlit dashboard to monitor spam and test messages
- Uses your existing scikit-learn joblib model

## Quick start (Docker)
1. Place your model at `api/models/spam_classifier_model.joblib` if not already copied.
2. In a terminal:
   ```
   cd spam_system
   docker compose up --build
   ```
3. API: http://localhost:8000/docs
4. App: http://localhost:8501

## Local without Docker
1. Start DB or skip to SQLite:
   - For SQLite, do nothing.
   - For Postgres: set `DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/dbname`
2. API:
   ```
   cd api
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   export MODEL_PATH=./models/spam_classifier_model.joblib
   uvicorn main:app --reload
   ```
3. Streamlit:
   ```
   cd ../streamlit_app
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   export API_BASE=http://localhost:8000
   streamlit run app.py
   ```

## Test
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"Win money now!"}'
curl "http://localhost:8000/messages?limit=20&is_spam=true"
```

## Notes
- If `DATABASE_URL` is unset, the API falls back to SQLite at `./spam.db`.
- Change model threshold by editing `main.py` (0.5 default).
