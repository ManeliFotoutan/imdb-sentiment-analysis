import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy.sparse import hstack
import joblib
from utils import preprocess_text

# Load IMDB data from given directory structure
def load_imdb_data(dir_path):
    data = {"review": [], "sentiment": []}
    for label in ['pos', 'neg']:
        for split in ['train', 'test']:
            folder = os.path.join(dir_path, split, label)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                with open(os.path.join(folder, fname), encoding="utf-8") as f:
                    data["review"].append(f.read())
                    data["sentiment"].append('positive' if label == 'pos' else 'negative')
    return pd.DataFrame(data)

# Train model and save vectorizer, LDA, and classifier
def train_and_save_models(data_dir):
    data = load_imdb_data(data_dir)

    # Clean text data
    data['cleaned_review'] = data['review'].apply(preprocess_text)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_review'], data['sentiment'], test_size=0.2, random_state=42)

    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.1)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Apply LDA topic modeling
    lda = LatentDirichletAllocation(n_components=10, random_state=42)
    X_train_lda = lda.fit_transform(X_train_tfidf)

    # Combine TF-IDF and LDA features
    X_train_combined = hstack([X_train_tfidf, X_train_lda])

    # Train Linear Support Vector Classifier
    model = LinearSVC(C=100, random_state=42)
    model.fit(X_train_combined, y_train)

    # Save trained components
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(lda, "lda.pkl")
    joblib.dump(model, "sentiment_model.pkl")

    # Evaluate model on test set
    X_test_tfidf = vectorizer.transform(X_test)
    X_test_lda = lda.transform(X_test_tfidf)
    X_test_combined = hstack([X_test_tfidf, X_test_lda])
    y_pred = model.predict(X_test_combined)

    report = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label='positive')

    return report, acc, f1
