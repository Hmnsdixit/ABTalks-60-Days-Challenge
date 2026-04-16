from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# 🔹 Step 1:   Documents
documents = [
    "Artificial Intelligence is the future of technology",
    "Machine Learning is a subset of AI",
    "Deep Learning is a powerful technique",
    "I love playing cricket",
    "Data Science is amazing"
]

# 🔹 Step 2: Load embeddings (Ollama - local)
embeddings = OllamaEmbeddings(model="phi")

# 🔹 Step 3: Create vector DB
vectorstore = FAISS.from_texts(documents, embeddings)

print("🤖 Semantic Search System Ready! Type 'exit' to quit.\n")

# 🔹 Step 4: Query loop
while True:
    query = input("🔍 Enter your query: ")

    if query.lower() == "exit":
        print("👋 Exiting...")
        break

    # 🔹 Step 5: Similarity search
    results = vectorstore.similarity_search(query, k=3)

    print("\n📊 Top Results:\n")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content}")
    
    print("-" * 40)