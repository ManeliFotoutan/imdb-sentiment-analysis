#  Sentiment Analysis on IMDB Reviews

This project performs **sentiment analysis** on movie reviews using **Natural Language Processing (NLP)** and **Machine Learning** techniques such as TF-IDF, LDA, and a Linear Support Vector Classifier. It includes an intuitive **Tkinter-based GUI** for training and prediction without requiring command-line interaction.

## Dataset Download

You can download the dataset used in this project from the following Google Drive link:

📁 [Download Dataset](https://drive.google.com/drive/folders/1bUW8eantdXLDUmtr3Y355Zf4B3XsSYMG?usp=sharing)


##  Key Concepts and Techniques

###  Text Preprocessing

Performed using `NLTK`, includes:
- Removal of HTML tags and URLs
- Removal of non-alphabetic characters
- Lowercasing all text
- Stopword removal
- Lemmatization

###  Feature Extraction

- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Converts cleaned text into weighted numeric vectors indicating word importance.
- **LDA (Latent Dirichlet Allocation)**: Extracts hidden topics from the text and combines with TF-IDF features.

###  Classification

- **LinearSVC (Support Vector Classifier)**:
  - High efficiency for large sparse datasets (like text)
  - Maximizes margin between classes (positive/negative)
  - Uses `hinge loss` for optimization

---

##  Installation

Ensure you have **Python 3.6+** installed.
###  Install required libraries:
```bash
pip install scikit-learn nltk pandas joblib
```

### Then download NLTK resources:
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
## Usage
### 1. Train the Model
You can train the model using either the terminal or the GUI.

 Terminal:
```bash
python train_model.py
```
 GUI:
```bash
python gui_train.py
```
This will:

Load and preprocess the dataset

Perform vectorization and topic modeling

Train the LinearSVC classifier

Save the models (vectorizer.pkl, lda.pkl, sentiment_model.pkl)

### 2. Predict Sentiment (GUI)
Launch the prediction GUI with:

```bash
python gui_predict.py
```
Then:

Paste or write a movie review

Click the “🔮 Predict Sentiment” button

View the result: Positive or Negative

## Evaluation Metrics
The model is evaluated using:

Accuracy

F1-Score

Classification Report

## Saved Artifacts
After training, the following files are saved:

vectorizer.pkl – TF-IDF vectorizer

lda.pkl – Topic model

sentiment_model.pkl – Trained LinearSVC model

They are reused during prediction to ensure consistency.

## Sample Workflow (Pipeline)
Input: Raw IMDB review

Preprocessing: Cleaning → Tokenizing → Lemmatizing

Vectorization: Convert to TF-IDF

Topic Modeling: Add LDA-based features

Prediction: Classify using LinearSVC

Output: Sentiment Label (Positive / Negative)
