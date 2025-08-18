# streamlit_app/app.py
import os
import requests
import pandas as pd
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

MODEL_NAME_MAP = {
    "Logistic Regression": "logreg",
    "Linear Support Vector Classifier (Calibrated)": "linearsvc_cal",
    "Multinomial Naive Bayes": "mnb",
    "Random Forest Classifier": "rf",
}

st.set_page_config(page_title="Spam Monitor", layout="wide")
t1, t2 = st.tabs(["Predict", "Dashboard"])

with t1:
    st.header("Predict")
    c1, c2 = st.columns(2)
    with c1:
        sender = st.text_input("Sender")
        subject = st.text_input("Subject")
        # model dropdown with human-readable names
        model_human = st.selectbox(
            "Model",
            list(MODEL_NAME_MAP.keys()),
            index=0,
            help="Choose which trained model the API should use for this prediction.",
        )
    text = st.text_area("Email Text", height=220)

    if st.button("Predict"):
        if not text.strip():
            st.error("Enter text")
        else:
            payload = {
                "text": text,
                "subject": subject or None,
                "sender": sender or None,
                "model": MODEL_NAME_MAP[model_human],
            }
            try:
                r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
                if r.ok:
                    data = r.json()
                    # Only show label result (no probability shown to the user)
                    if bool(data.get("label", False)):
                        st.error("Spam")
                    else:
                        st.success("Not Spam")
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(f"Request failed: {e}")

with t2:
    st.header("Dashboard")
    col_a, col_b = st.columns(2)
    with col_a:
        spam_only = st.checkbox("Spam only", value=False)
    with col_b:
        limit = st.slider("Rows", 10, 500, 100, 10)

    url = f"{API_BASE}/messages?limit={limit}" + (f"&is_spam=true" if spam_only else "")
    try:
        r = requests.get(url, timeout=10)
        if r.ok:
            rows = r.json()
            df = pd.DataFrame(rows)
            if not df.empty:
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Total", len(df))
                with m2:
                    st.metric("Spam", int(df["label"].sum()))
                with m3:
                    st.metric("Not Spam", int((~df["label"]).sum()))

                # Keep the probability series for the time plot, but it's fine if the user doesn't focus on it
                if "probability" in df.columns and not df["probability"].isna().all():
                    st.line_chart(df.sort_values("created_at")["probability"])

                st.dataframe(
                    df[["created_at", "sender", "subject", "label", "model_name", "id"]],
                    use_container_width=True,
                )
            else:
                st.info("No data")
        else:
            st.error(r.text)
    except Exception as e:
        st.error(f"Request failed: {e}")
