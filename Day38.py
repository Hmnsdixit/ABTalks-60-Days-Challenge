# Day 38 - FAISS Vector Database Example

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize embeddings (using local Ollama model)
embeddings = OllamaEmbeddings(model="phi")

# Sample documents
documents = [
    "Artificial Intelligence is the future",
    "Machine Learning is a subset of AI",
    "Deep Learning is powerful",
    "I love playing cricket",
    "Data Science is amazing"
]

# Create FAISS vector store
vectorstore = FAISS.from_texts(documents, embeddings)

# Query
query = "What is AI?"

# Perform similarity search
results = vectorstore.similarity_search(query)

print("🔍 Query:", query)
print("\n📊 Top Similar Results:\n")

for i, doc in enumerate(results):
    print(f"{i+1}. {doc.page_content}")