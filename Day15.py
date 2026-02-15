# Day 15: Community NLP Challenge – Vectorized Text Embeddings
# Focus: Vectorization, Embedding Matrix, Dot Product, Cosine Similarity (NO LOOPS)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=== Day 15: Vectorized Text Embeddings (NLP) ===\n")

# ================= STEP 1: INPUT TEXT DOCUMENTS =================
documents = [
    "NLP is an exciting field of artificial intelligence",
    "Machine learning and NLP are closely related",
    "Text embeddings convert words into numerical vectors",
    "Deep learning models use vectorized text representations"
]

print("Original Documents:\n")
for i, doc in enumerate(documents, 1):
    print(f"Doc {i}: {doc}")

# ================= STEP 2: CONVERT TEXT TO EMBEDDINGS (TF-IDF) =================
# This creates numerical embeddings (vectorized representation)
vectorizer = TfidfVectorizer()
embedding_matrix = vectorizer.fit_transform(documents)

print("\n=== Embedding Matrix (TF-IDF Vectors) ===")
print("Shape of Embedding Matrix:", embedding_matrix.shape)
print("(Rows = Documents, Columns = Vocabulary Size)\n")

# Convert sparse matrix to dense matrix (for NumPy operations)
embeddings = embedding_matrix.toarray()
print("Dense Embedding Matrix:\n", embeddings)

# ================= STEP 3: VECTOR STACKING (MATRIX FORM) =================
print("\n=== Vectorized Matrix Representation ===")
print("Matrix Shape:", embeddings.shape)

# ================= STEP 4: DOT PRODUCT (Matrix Multiplication) =================
print("\n=== Dot Product Matrix (Vectorized) ===")
dot_product_matrix = np.dot(embeddings, embeddings.T)
print(dot_product_matrix)

# ================= STEP 5: COSINE SIMILARITY (VECTOR METHOD - NO LOOPS) =================
print("\n=== Cosine Similarity Matrix (Vectorized) ===")
cosine_sim_matrix = cosine_similarity(embeddings)
print(cosine_sim_matrix)

# ================= STEP 6: MEAN EMBEDDING (Average Vector) =================
print("\n=== Mean Embedding Vector (All Documents) ===")
mean_embedding = np.mean(embeddings, axis=0)
print(mean_embedding)

# ================= STEP 7: SIMILARITY INTERPRETATION =================
print("\n=== Similarity Interpretation ===")
print("Cosine similarity closer to 1 = More similar documents")
print("Cosine similarity closer to 0 = Less similar documents")

# Find most similar document pair (excluding self similarity)
similarity_no_diag = cosine_sim_matrix.copy()
np.fill_diagonal(similarity_no_diag, 0)

most_similar_indices = np.unravel_index(
    np.argmax(similarity_no_diag), similarity_no_diag.shape
)

print(f"\nMost Similar Documents: Doc {most_similar_indices[0]+1} and Doc {most_similar_indices[1]+1}")
print("Similarity Score:", cosine_sim_matrix[most_similar_indices])
