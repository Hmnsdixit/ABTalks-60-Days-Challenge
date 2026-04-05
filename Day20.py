# Day 20: Embedding Similarity Calculator

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

# -------------------------------
# Step 1: Sample Texts
# -------------------------------
text1 = "Machine learning is amazing"
text2 = "AI and machine learning are powerful"

# -------------------------------
# Step 2: Simple Embedding (Bag of Words Vector)
# -------------------------------
def text_to_vector(text, vocabulary):
    vector = [text.lower().split().count(word) for word in vocabulary]
    return np.array(vector)

# Create vocabulary
vocab = list(set(text1.lower().split() + text2.lower().split()))

# Convert text to vectors
vec1 = text_to_vector(text1, vocab)
vec2 = text_to_vector(text2, vocab)

print("Vocabulary:", vocab)
print("\nVector 1:", vec1)
print("Vector 2:", vec2)

# -------------------------------
# Step 3: Cosine Similarity
# -------------------------------
cos_sim = cosine_similarity([vec1], [vec2])[0][0]

# -------------------------------
# Step 4: Euclidean Distance
# -------------------------------
eu_dist = euclidean(vec1, vec2)

# -------------------------------
# Step 5: Output
# -------------------------------
print("\n🔹 Cosine Similarity:", round(cos_sim, 3))
print("🔹 Euclidean Distance:", round(eu_dist, 3))

# -------------------------------
# Step 6: Interpretation
# -------------------------------
print("\n📊 Interpretation:")
if cos_sim > 0.7:
    print("High similarity between texts")
elif cos_sim > 0.4:
    print("Moderate similarity")
else:
    print("Low similarity")