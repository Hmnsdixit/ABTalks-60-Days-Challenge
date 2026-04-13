from langchain_community.embeddings import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Initialize embedding model
embeddings = OllamaEmbeddings(model="phi")

# Sample texts
documents = [
    "AI is transforming the world",
    "Machine learning is a subset of AI",
    "I love playing cricket",
    "Deep learning is powerful"
]

# Generate embeddings
doc_embeddings = embeddings.embed_documents(documents)

# Compare similarity
query = "Artificial Intelligence"
query_embedding = embeddings.embed_query(query)

# Calculate similarity
scores = cosine_similarity([query_embedding], doc_embeddings)

print("🔍 Query:", query)
print("\n📊 Similarity Scores:")

for doc, score in zip(documents, scores[0]):
    print(f"{doc} → {round(score, 2)}")