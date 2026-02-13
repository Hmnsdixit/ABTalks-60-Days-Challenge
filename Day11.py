# Day 11: Cosine Similarity Between Two Documents (NLP)

import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Step 1: Input Documents
# -----------------------------
doc1 = "Python is widely used in AI and Machine Learning"
doc2 = "Python is popular for AI, ML, and Data Science"

print("Document 1:", doc1)
print("Document 2:", doc2)

# -----------------------------
# Step 2: Text Preprocessing Function
# -----------------------------
def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text

clean_doc1 = preprocess(doc1)
clean_doc2 = preprocess(doc2)

print("\nCleaned Document 1:", clean_doc1)
print("Cleaned Document 2:", clean_doc2)

# -----------------------------
# Step 3: Convert Text to Vectors (TF-IDF)
# -----------------------------
documents = [clean_doc1, clean_doc2]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("\nTF-IDF Feature Names:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())

# -----------------------------
# Step 4: Compute Cosine Similarity
# -----------------------------
similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

print("\nCosine Similarity Score:", similarity_score[0][0])

# -----------------------------
# Step 5: Interpretation
# -----------------------------
score = similarity_score[0][0]

if score > 0.7:
    print("Interpretation: Documents are Highly Similar")
elif score > 0.4:
    print("Interpretation: Documents are Moderately Similar")
else:
    print("Interpretation: Documents are Low Similarity")
