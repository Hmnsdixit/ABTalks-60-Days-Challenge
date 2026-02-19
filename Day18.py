# Day 18 - Optimization for Text Similarity using SciPy
# ABTalks 60 Days Coding Challenge

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Step 1: Sample Text Data
# -------------------------------
doc1 = "Natural Language Processing is a powerful field of AI"
doc2 = "NLP is an important part of Artificial Intelligence"

documents = [doc1, doc2]

print("Original Documents:")
print("Doc1:", doc1)
print("Doc2:", doc2)

# -------------------------------
# Step 2: Convert Text to Vectors (TF-IDF)
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents).toarray()

vec1 = X[0]
vec2 = X[1]

print("\nTF-IDF Vector Shape:", X.shape)

# -------------------------------
# Step 3: Cosine Similarity Function
# -------------------------------
def cosine_sim(v1, v2):
    return cosine_similarity([v1], [v2])[0][0]

# Initial similarity (Before Optimization)
initial_similarity = cosine_sim(vec1, vec2)
print("\nInitial Cosine Similarity:", round(initial_similarity, 4))

# -------------------------------
# Step 4: Define Cost Function
# (We minimize 1 - similarity)
# -------------------------------
def cost_function(weights):
    weighted_vec1 = vec1 * weights
    similarity = cosine_sim(weighted_vec1, vec2)
    cost = 1 - similarity  # Minimize cost = Maximize similarity
    return cost

# -------------------------------
# Step 5: Initialize Weights
# -------------------------------
initial_weights = np.ones(len(vec1))

# -------------------------------
# Step 6: Optimization using SciPy
# -------------------------------
result = minimize(
    cost_function,
    initial_weights,
    method='L-BFGS-B'
)

optimized_weights = result.x

# -------------------------------
# Step 7: Apply Optimized Weights
# -------------------------------
optimized_vec1 = vec1 * optimized_weights
optimized_similarity = cosine_sim(optimized_vec1, vec2)

# -------------------------------
# Step 8: Results
# -------------------------------
print("\n--- Optimization Results ---")
print("Optimization Success:", result.success)
print("Final Cost:", round(result.fun, 4))

print("\nSimilarity Before Optimization:", round(initial_similarity, 4))
print("Similarity After Optimization:", round(optimized_similarity, 4))

print("\nConclusion:")
if optimized_similarity > initial_similarity:
    print("Optimization improved the text similarity alignment.")
else:
    print("Similarity did not improve significantly, but optimization was applied.")
