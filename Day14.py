# Day 14: Community NLP Assignment – 3 Step Challenge
# Step 1: Cleaning & Tokenization
# Step 2: Text Representation (BoW & TF-IDF)
# Step 3: Reusable NLP Pipeline

import re
import string
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Basic Stopwords List
STOPWORDS = {
    "is", "an", "the", "of", "and", "in", "to", "are",
    "very", "a", "on", "for", "with", "this", "that"
}

# ================= STEP 1: TEXT CLEANING & TOKENIZATION =================
def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_words(text):
    return text.split()


def remove_stopwords(tokens):
    return [word for word in tokens if word not in STOPWORDS]


# ================= STEP 2: TEXT REPRESENTATION =================
def word_frequency(tokens):
    return Counter(tokens)


def get_top_words(freq_dict, n=10):
    return freq_dict.most_common(n)


def bag_of_words(corpus):
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, bow_matrix


def tfidf_representation(corpus):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    return tfidf_vectorizer, tfidf_matrix


# ================= STEP 3: REUSABLE NLP PIPELINE =================
def nlp_pipeline(text):
    """
    Reusable NLP Pipeline:
    1. Cleaning
    2. Tokenization
    3. Stopword Removal
    """
    cleaned = clean_text(text)
    tokens = tokenize_words(cleaned)
    filtered_tokens = remove_stopwords(tokens)
    return filtered_tokens


# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    print("=== Day 14: Community NLP 3-Step Challenge ===\n")

    # Load Dataset
    with open("dataset.txt", "r", encoding="utf-8") as file:
        documents = file.readlines()

    documents = [doc.strip() for doc in documents if doc.strip()]

    print("Original Documents:\n")
    for doc in documents:
        print(doc)
    print("\n" + "="*60)

    # Apply NLP Pipeline
    all_tokens = []
    processed_docs = []

    print("\nStep 1: Cleaning & Tokenization\n")
    for i, doc in enumerate(documents, 1):
        tokens = nlp_pipeline(doc)
        processed_docs.append(" ".join(tokens))
        all_tokens.extend(tokens)

        print(f"Document {i} Cleaned Tokens:")
        print(tokens)
        print("-"*50)

    # Step 2: Word Frequency Analysis
    print("\nStep 2: Word Frequency Distribution\n")
    freq = word_frequency(all_tokens)
    print("Top 10 Frequent Words:")
    print(get_top_words(freq, 10))

    # Bag of Words
    print("\nBag of Words Representation:\n")
    bow_vectorizer, bow_matrix = bag_of_words(processed_docs)
    print("Vocabulary:", bow_vectorizer.get_feature_names_out())
    print("BoW Matrix:\n", bow_matrix.toarray())

    # TF-IDF
    print("\nTF-IDF Representation:\n")
    tfidf_vectorizer, tfidf_matrix = tfidf_representation(processed_docs)
    print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
    print("TF-IDF Matrix:\n", tfidf_matrix.toarray())

    print("\n=== Comparison ===")
    print("BoW counts word frequency.")
    print("TF-IDF gives importance to rare but meaningful words.")
