import streamlit as st
import pandas as pd
import numpy as np
import re, string, io, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Spam/Ham Predictor", page_icon="📧", layout="centered")

def basic_clean(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    s = x.astype(str).str.lower()
    s = s.str.replace(r"http\S+|www\S+"," ", regex=True)
    s = s.str.replace(r"\d+"," ", regex=True)
    s = s.str.replace(f"[{re.escape(string.punctuation)}]"," ", regex=True)
    s = s.str.replace(r"\s+"," ", regex=True).str.strip()
    return s

def make_features():
    word = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_df=0.95)
    return FeatureUnion([("word_tfidf", word), ("char_tfidf", char)])

def get_models():
    return {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "LinearSVC": LinearSVC(),
        "SGDClassifier": SGDClassifier(loss="log_loss", max_iter=2000),
        "PassiveAggressive": PassiveAggressiveClassifier(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42)
    }

@st.cache_data
def load_frame(file):
    df = pd.read_csv(file, encoding="utf-8", engine="python", on_bad_lines="skip")
    if {"label","text"}.issubset(df.columns):
        df = df[["label","text"]]
    else:
        df = df.rename(columns={df.columns[0]:"label", df.columns[-1]:"text"})[["label","text"]]
    df = df.dropna(subset=["label","text"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower().replace({"1":"spam","0":"ham"})
    df["text"] = df["text"].astype(str)
    return df

@st.cache_resource
def train_once(df, model_name, seed):
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X = df["text"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    pipe = Pipeline([("clean", FunctionTransformer(basic_clean)), ("feat", make_features()), ("clf", get_models()[model_name])])
    pipe.fit(X_train, y_train)
    return {"pipeline": pipe, "label_encoder": le}

with st.sidebar:
    st.title("📧 Spam/Ham Predictor")
    mode = st.radio("Source", ["Use pre-trained .joblib", "Train from CSV"])
    model_name = st.selectbox("Model (when training from CSV)", list(get_models().keys()), index=1)
    seed = st.number_input("Random seed", value=42, step=1)
    joblib_file = None
    csv_file = None
    if mode == "Use pre-trained .joblib":
        joblib_file = st.file_uploader("Upload .joblib", type=["joblib"])
    else:
        csv_file = st.file_uploader("Upload CSV (needs label,text)", type=["csv"])

st.header("Predict: SPAM or NOT SPAM")
txt = st.text_area("Enter a message", height=140, placeholder="Type or paste a message to classify...")
go = st.button("Predict")

state = st.session_state
if "obj" not in state:
    state.obj = None

if mode == "Use pre-trained .joblib" and joblib_file is not None and state.obj is None:
    state.obj = joblib.load(joblib_file)

if mode == "Train from CSV" and csv_file is not None and state.obj is None:
    df = load_frame(csv_file)
    state.obj = train_once(df, model_name, seed)

if go:
    if state.obj is None:
        st.error("Load a .joblib or upload a CSV first.")
    else:
        pipe = state.obj["pipeline"]
        pred = pipe.predict(pd.Series([txt]))[0]
        label = "SPAM" if pred == 1 else "NOT SPAM"
        st.metric("Prediction", label)
