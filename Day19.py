# 🚀 Day 19 - Embedding Normalization & Numerical Stability

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Step 1: Create Sample Embeddings
# -------------------------------
vec1 = np.array([100, 200, 300])
vec2 = np.array([1, 2, 3])

print("Original Vectors:")
print("Vector 1:", vec1)
print("Vector 2:", vec2)

# -------------------------------
# Step 2: Cosine Similarity WITHOUT Normalization
# -------------------------------
sim_before = cosine_similarity([vec1], [vec2])[0][0]

print("\nSimilarity BEFORE Normalization:", round(sim_before, 4))

# -------------------------------
# Step 3: Normalize Vectors
# -------------------------------
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm

norm_vec1 = normalize(vec1)
norm_vec2 = normalize(vec2)

print("\nNormalized Vectors:")
print("Vector 1:", norm_vec1)
print("Vector 2:", norm_vec2)

# -------------------------------
# Step 4: Cosine Similarity AFTER Normalization
# -------------------------------
sim_after = cosine_similarity([norm_vec1], [norm_vec2])[0][0]

print("\nSimilarity AFTER Normalization:", round(sim_after, 4))

# -------------------------------
# Step 5: Compare Results
# -------------------------------
print("\n--- Analysis ---")

if sim_after >= sim_before:
    print("Normalization improved or stabilized similarity.")
else:
    print("Normalization adjusted vector scale for better comparison.")

# -------------------------------
# Step 6: Matrix Normalization (Multiple Embeddings)
# -------------------------------
embedding_matrix = np.array([
    [100, 200, 300],
    [1, 2, 3],
    [50, 60, 70]
])

print("\nOriginal Embedding Matrix:\n", embedding_matrix)

# Normalize entire matrix
norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
normalized_matrix = embedding_matrix / norms

print("\nNormalized Embedding Matrix:\n", normalized_matrix)

# -------------------------------
# Step 7: Cosine Similarity Matrix
# -------------------------------
sim_matrix = cosine_similarity(normalized_matrix)

print("\nCosine Similarity Matrix:\n", sim_matrix)

# -------------------------------
# Final Interpretation
# -------------------------------
print("\n📊 Conclusion:")
print("- Normalization ensures fair comparison between vectors.")
print("- It removes scale differences.")
print("- Essential for cosine similarity and ML models.")