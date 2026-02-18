# 🚀 Day 16: NLP + Linear Algebra (Dot Product & Matrix Multiplication)

import numpy as np

# -------------------------------
# Step 1: Create Sample Embedding Vectors
# -------------------------------
# Suppose these are word/sentence embeddings
embedding_1 = np.array([0.2, 0.5, 0.1, 0.7])
embedding_2 = np.array([0.3, 0.6, 0.2, 0.8])

print("Embedding Vector 1:", embedding_1)
print("Embedding Vector 2:", embedding_2)

# -------------------------------
# Step 2: Compute Dot Product
# -------------------------------
dot_product = np.dot(embedding_1, embedding_2)

print("\n🔹 Dot Product:", dot_product)

# Explanation
print("Dot Product Meaning: Higher value = More similar vectors")

# -------------------------------
# Step 3: Create Embedding Matrix (Multiple Texts)
# -------------------------------
embedding_matrix = np.array([
    [0.2, 0.5, 0.1, 0.7],  # Text 1
    [0.3, 0.6, 0.2, 0.8],  # Text 2
    [0.9, 0.1, 0.4, 0.3]   # Text 3
])

print("\nEmbedding Matrix:\n", embedding_matrix)
print("Matrix Shape:", embedding_matrix.shape)

# -------------------------------
# Step 4: Matrix Multiplication (Similarity Matrix)
# -------------------------------
similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

print("\n🔹 Similarity Matrix (Dot Product based):")
print(similarity_matrix)

# -------------------------------
# Step 5: Cosine Similarity (Vectorized)
# -------------------------------
def cosine_similarity_matrix(matrix):
    # Normalize vectors
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / norm
    
    # Cosine similarity = A . A^T
    cosine_sim = np.dot(normalized_matrix, normalized_matrix.T)
    return cosine_sim

cos_sim_matrix = cosine_similarity_matrix(embedding_matrix)

print("\n🔹 Cosine Similarity Matrix:")
print(cos_sim_matrix)

# -------------------------------
# Step 6: Interpretation
# -------------------------------
print("\n📊 Interpretation:")
print("Diagonal values = 1 (same vectors)")
print("Higher value = More similar texts")
print("Lower value = Less similar texts")
