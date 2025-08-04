import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download tokenizer model for word/sentence tokenization
# nltk.download('punkt')

# Download WordNet lexical database for lemmatization
# nltk.download('wordnet')

# Download list of common English stopwords
# nltk.download('stopwords')

# Preprocess text: remove HTML tags, URLs, non-letter characters, and apply lemmatization
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove non-letter characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Convert all text to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lemmatize and remove stopwords and short words
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    
    return " ".join(tokens)


#More precisely Preprocess text:

# import nltk
# from nltk.corpus import stopwords, wordnet
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk import pos_tag
# import re

# # Map NLTK POS tags to WordNet POS tags
# def get_wordnet_pos(treebank_tag):
#     if treebank_tag.startswith('J'):
#         return wordnet.ADJ  # Adjective
#     elif treebank_tag.startswith('V'):
#         return wordnet.VERB  # Verb
#     elif treebank_tag.startswith('N'):
#         return wordnet.NOUN  # Noun
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV  # Adverb
#     else:
#         return wordnet.NOUN  # Default to noun

# # Clean and preprocess the input text
# def preprocess_text(text):
#     lemmatizer = WordNetLemmatizer()
#     stop_words = set(stopwords.words('english'))

#     # Remove HTML tags
#     text = re.sub(r"<.*?>", " ", text)

#     # Remove URLs
#     text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

#     # Remove non-alphabetic characters
#     text = re.sub(r"[^a-zA-Z\s]", " ", text)

#     # Convert to lowercase
#     text = text.lower()

#     # Tokenize the text
#     tokens = word_tokenize(text)

#     # POS tagging
#     pos_tags = pos_tag(tokens)

#     # Lemmatize tokens using POS info and remove stopwords and short tokens
#     lemmatized_tokens = []
#     for token, tag in pos_tags:
#         if token not in stop_words and len(token) > 2 and token.isalpha():
#             pos = get_wordnet_pos(tag)
#             lemma = lemmatizer.lemmatize(token, pos)
#             lemmatized_tokens.append(lemma)

#     return " ".join(lemmatized_tokens)
