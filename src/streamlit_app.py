import streamlit as st
import os
import requests
from transformers import pipeline

# Load model from HF Model repo
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="Ravichandrachilde/fact-checked-news-detector-model")

classifier = load_model()

@st.cache_data
def check_with_google_factcheck(query):
    api_key = os.getenv("FACT_CHECK_API_KEY", "")
    if not api_key:
        return None, "Add FACT_CHECK_API_KEY in environment variables to enable Google Fact Check API integration."
    
    # Google Fact Check API endpoint
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={requests.utils.quote(query)}&pageSize=10&key={api_key}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            claims = data.get("claims", [])
            
            if not claims:
                return None, "No specific fact-checks found for this statement."
            
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

# UI
st.title("📰 Fact Checked News Detector")
st.write("Powered by DistilBERT—detects fake news in statements. Integrated with Google Fact Check API for cross-verification.")

# Sidebar for API Keys
with st.sidebar:
    st.header("🔑 Settings")
    st.info("Ensure you have set `FACT_CHECK_API_KEY` in your environment variables/secrets.")

# Main Detection Area
st.header("Analyze News Statement")
text = st.text_area("Enter text to check:", height=150, placeholder="E.g., 'The moon landing was faked by the government...'")

if st.button("Detect Veracity") and text.strip():
    
    # 1. Model Prediction
    with st.spinner("Analyzing with AI Model..."):
        result = classifier(text)[0]
        model_label = "True" if result["label"] == "LABEL_1" else "Fake"
        model_score = result['score']
    
    st.info(f"**AI Model Prediction:** {'🟢 Real' if model_label == 'True' else '🔴 Fake'} (Confidence: {model_score:.2%})")

    # 2. Google Fact Check
    with st.spinner("Cross-referencing Google Fact Check database..."):
        fact_label, fact_msg = check_with_google_factcheck(text)

    # 3. Final Logic
    st.markdown("---")
    if fact_label is not None:
        final_label = fact_label
        st.subheader(f"Final Verdict: {'🟢 Real' if final_label == 'True' else '🔴 Fake'}")
        st.caption(f"Reason: Overridden by official fact-check data. ({fact_msg})")
    else:
        final_label = model_label
        st.subheader(f"Final Verdict: {'🟢 Real' if final_label == 'True' else '🔴 Fake'}")
        st.caption(f"Reason: Based on AI model analysis. ({fact_msg})")

    if final_label == "Fake":
        st.warning("⚠️ This statement shows signs of misinformation. Verify with trusted news sources.")

st.markdown("---")
st.caption("Model: Ravichandrachilde/fact-checked-news-detector-model | Data Sources: Google Fact Check Tools API")
