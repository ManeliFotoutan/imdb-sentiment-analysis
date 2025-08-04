import joblib
from scipy.sparse import hstack
from utils import preprocess_text

# Load trained vectorizer, LDA, and model
def load_models():
    vectorizer = joblib.load("vectorizer.pkl")
    lda = joblib.load("lda.pkl")
    model = joblib.load("sentiment_model.pkl")
    return vectorizer, lda, model

# Predict sentiment for a given input text
def predict_sentiment(text, vectorizer, lda, model):
    cleaned = preprocess_text(text)
    tfidf_vec = vectorizer.transform([cleaned])
    lda_vec = lda.transform(tfidf_vec)
    combined_vec = hstack([tfidf_vec, lda_vec])
    return model.predict(combined_vec)[0]
