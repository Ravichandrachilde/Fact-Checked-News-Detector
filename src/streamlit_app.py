import streamlit as st
import os
import pandas as pd
from transformers import pipeline
from huggingface_hub import hf_hub_download 
import requests  

# Load model from HF Model repo
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="Ravichandrachilde/fact-checked-news-detector-model")

classifier = load_model()

# Function to load CSVs from HF Dataset repo
@st.cache_data
def load_csv_from_repo(filename, repo_id="Ravichandrachilde/fake-news-classification-dataset"):
    try:
        csv_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Could not load {filename}: {e}")
        return pd.DataFrame()

def augment_with_api(query="fake news", num_articles=10):
    api_key = os.getenv("NEWS_API_KEY", "") 
    if not api_key:
        st.warning("Add NEWS_API_KEY secret in Space Settings for real augmentation.")
        return pd.DataFrame()
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&pageSize={num_articles}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json()["articles"]
            data = []
            for article in articles:
                statement = (article.get("title", "") + " " + article.get("description", "")).strip()
                if statement:
                    label = 0 if "fake" in statement.lower() else 1 
                    data.append({"statement": statement, "label": "Fake" if label == 0 else "True"})
            df = pd.DataFrame(data)
            return df
        else:
            st.error(f"API failed: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()
@st.cache_data
def check_with_google_factcheck(query):
    api_key = os.getenv("FACT_CHECK_API_KEY", "")
    if not api_key:
        return None, "Add FACT_CHECK_API_KEY in environment variables to enable Google Fact Check API integration."
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={requests.utils.quote(query)}&pageSize=10&key={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            claims = data.get("claims", [])
            if not claims:
                return None, "No fact-checks found for this statement."
            # Aggregate ratings
            ratings = []
            for claim in claims:
                for review in claim.get("claimReview", []):
                    rating = review.get("textualRating", "").lower()
                    if rating:
                        ratings.append(rating)
            if not ratings:
                return None, "No specific ratings found in fact-checks."
            # Count true-like and false-like ratings
            true_keywords = ["true", "accurate", "correct", "mostly true"]
            false_keywords = ["false", "fake", "misleading", "incorrect", "debunked", "mostly false"]
            true_count = sum(1 for r in ratings if any(word in r for word in true_keywords))
            false_count = sum(1 for r in ratings if any(word in r for word in false_keywords))
            if true_count > false_count:
                return "True", f"Fact-check ratings suggest mostly true ({true_count} true-like, {false_count} false-like)."
            elif false_count > true_count:
                return "Fake", f"Fact-check ratings suggest mostly fake ({false_count} false-like, {true_count} true-like)."
            else:
                return None, "Fact-check ratings are mixed or inconclusive."
        else:
            return None, f"API request failed with status code: {response.status_code}"
    except Exception as e:
        return None, f"Error accessing Fact Check API: {e}"

st.title("📰 Fact Checked News Detector")
st.write("Powered by DistilBERT—detects fake news in statements. Try single text or batch CSV. Now integrated with Google Fact Check API for cross-verification.")

# Sidebar: Load Demo Data from Your HF Dataset Repo
with st.sidebar:
    st.header("📊 Load Demo Data")
    if st.button("Load Sample Fake News CSV"):
        fake_df = load_csv_from_repo("Fake.csv")
        if not fake_df.empty:
            st.success(f"Loaded {len(fake_df)} fake samples!")
            st.dataframe(fake_df.head())
    if st.button("Load Sample True News CSV"):
        true_df = load_csv_from_repo("True.csv")
        if not true_df.empty:
            st.success(f"Loaded {len(true_df)} true samples!")
            st.dataframe(true_df.head())

    st.header("🔄 Augment Data")
    query = st.text_input("Search query", "fake news")
    num_articles = st.slider("Articles to fetch", 5, 50, 10)
    if st.button("Pull Fresh Data"):
        new_data = augment_with_api(query, num_articles)
        if not new_data.empty:
            st.success(f"Added {len(new_data)} articles!")
            st.dataframe(new_data.head())
            csv = new_data.to_csv(index=False).encode("utf-8")
            st.download_button("Download Augmented CSV", csv, "augmented.csv", "text/csv")

    st.header("🔑 API Keys")
    st.caption("Set NEWS_API_KEY and FACT_CHECK_API_KEY in environment variables for API features.")

# Single prediction
st.header("Single Detection")
text = st.text_area("Enter news statement:", height=150, placeholder="E.g., 'The moon landing was faked...'")
if st.button("Detect Fake News") and text.strip():
    # model's prediction
    with st.spinner("Analyzing with model..."):
        result = classifier(text)[0]
        model_label = "True" if result["label"] == "LABEL_1" else "Fake"
        model_score = result['score']
    
    st.info(f"Model Prediction: {'🟢 True' if model_label == 'True' else '🔴 Fake'} (Confidence: {model_score:.2%})")

    # fact-check result
    with st.spinner("Cross-checking with Google Fact Check API..."):
        fact_label, fact_msg = check_with_google_factcheck(text)

    # final label
    if fact_label is not None:
        final_label = fact_label
        st.success(f"Final Prediction (Overridden by Fact-Check): {'🟢 True' if final_label == 'True' else '🔴 Fake'}")
        st.caption(fact_msg)
    else:
        final_label = model_label
        st.success(f"Final Prediction (from model): {'🟢 True' if final_label == 'True' else '🔴 Fake'}")
        st.caption(fact_msg)

    if final_label == "Fake":
        st.warning("⚠️ Potential fake news—always verify with trusted sources!")

# Batch prediction
st.header("Batch Detection (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV (must have 'statement' column)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "statement" in df.columns:
        with st.spinner(f"Processing {len(df)} statements..."):
            preds = classifier(df["statement"].tolist())
            df["prediction"] = ["🟢 True" if p["label"] == "LABEL_1" else "🔴 Fake" for p in preds]
            df["confidence"] = [f"{p['score']:.2%}" for p in preds]
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results CSV", csv, "results.csv", "text/csv")
    else:
        st.error("CSV must have a 'statement' column. Add it as title + text.")

st.markdown("---")
st.caption("Model trained on ~44k balanced samples | Dataset: Ravichandrachilde/fake-news-dataset | Integrated with Google Fact Check API for improved accuracy on recent news.")
