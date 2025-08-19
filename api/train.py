# api/train.py
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from preprocess import basic_clean

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
CSV_PATH = ROOT_DIR / "spam_ham_dataset.csv"
MODELS_DIR = THIS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25

def build_pipeline(clf):
    return Pipeline(
        steps=[
            ("clean", FunctionTransformer(basic_clean, validate=False)),
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
            ("clf", clf),
        ]
    )

def main():
    print(f"Loading data from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    # common Kaggle columns: 'label' or 'Category' and 'text' or 'Message'
    label_col = "label" if "label" in df.columns else ("Category" if "Category" in df.columns else None)
    text_col = "text" if "text" in df.columns else ("Message" if "Message" in df.columns else None)
    if not label_col or not text_col:
        raise ValueError(f"Expected columns not found. Have: {df.columns.tolist()}")

    # Normalize labels to 1=spam, 0=ham
    y = df[label_col].astype(str).str.lower().map({"spam": 1, "ham": 0}).fillna(df[label_col]).astype(int)
    X = df[text_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "logreg": LogisticRegression(max_iter=200, class_weight="balanced", random_state=RANDOM_STATE),
        "linearsvc_cal": CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=RANDOM_STATE), cv=5),
        "mnb": MultinomialNB(),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, class_weight="balanced_subsample", random_state=RANDOM_STATE
        ),
    }

    print("\n=== Training models ===\n")
    saved = []
    for name, clf in models.items():
        print(f"--- {name} ---")
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        print(classification_report(y_test, preds))

        out_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipe, out_path)
        saved.append(name)

    index = {
        "default": "logreg",
        "models": saved,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(MODELS_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nSaved models to: {MODELS_DIR}")
    print(f"Index written to: {MODELS_DIR / 'index.json'}")
    print(f"Available models: {', '.join(saved)}")
    print("Default model: logreg")

if __name__ == "__main__":
    main()
