# Day 10: NLP Feature Extraction - Bag-of-Words & TF-IDF

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# ------------------------------
# Sample Text Dataset
# ------------------------------

documents = [
    "Natural language processing is amazing",
    "Machine learning and NLP are closely related",
    "Deep learning advances natural language understanding",
    "NLP is used in machine learning applications"
]

print("Original Documents:\n")
for doc in documents:
    print("-", doc)

# ==============================
# 1️⃣ Bag of Words
# ==============================

print("\n--- Bag of Words Representation ---")

bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(documents)

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=bow_vectorizer.get_feature_names_out()
)

print("\nBoW Feature Names:\n", bow_vectorizer.get_feature_names_out())
print("\nBoW Matrix:\n")
print(bow_df)

# ==============================
# 2️⃣ TF-IDF
# ==============================

print("\n--- TF-IDF Representation ---")

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("\nTF-IDF Feature Names:\n", tfidf_vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix:\n")
print(tfidf_df)

# ==============================
# Observations
# ==============================

print("\n--- Observations ---")
print("1. BoW shows raw word counts.")
print("2. TF-IDF reduces importance of common words.")
print("3. TF-IDF gives higher weight to rare but meaningful words.")
