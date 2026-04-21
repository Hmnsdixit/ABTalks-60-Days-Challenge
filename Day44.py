# 🔹 Day 44: RAG System Review Tool

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# 🔹 Sample documents (same as previous days)
documents = [
    "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
    "Machine Learning is a subset of AI that learns from data.",
    "Deep Learning uses neural networks for complex tasks.",
    "FAISS is used for efficient similarity search in vector databases.",
    "RAG combines retrieval with generation to improve answer accuracy.",
    "Embeddings convert text into numerical vectors for similarity search."
]

# 🔹 Load models
embeddings = OllamaEmbeddings(model="phi")
vectorstore = FAISS.from_texts(documents, embeddings)
llm = Ollama(model="phi")

print("📊 RAG System Review Tool Ready! Type 'exit' to quit.\n")

while True:
    query = input("🔍 Enter your query: ")

    if query.lower() == "exit":
        print("👋 Exiting...")
        break

    # 🔹 Retrieve docs
    docs = vectorstore.similarity_search(query, k=3)

    print("\n📄 Retrieved Documents:")
    for d in docs:
        print("-", d.page_content)

    # 🔹 Combine context
    context = "\n".join([doc.page_content for doc in docs])

    # 🔹 Review prompt (IMPORTANT)
    prompt = f"""
You are an AI system reviewer.

Analyze the RAG system output and provide:
1. Relevance of retrieved documents (Good / Average / Poor)
2. Quality of answer (Accurate / Needs Improvement)
3. Suggest 1 improvement

Context:
{context}

User Query:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    print("\n🧠 Review:")
    print(response.strip())
    print("-" * 50)