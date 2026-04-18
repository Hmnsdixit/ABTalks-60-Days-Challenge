from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# 🔹 Documents (strong definitions)
documents = [
    "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
    "Machine Learning is a subset of AI that learns from data.",
    "Deep Learning uses neural networks for complex tasks.",
    "FAISS is used for efficient similarity search in vector databases.",
    "RAG combines retrieval with generation to improve answer accuracy.",
    "Python is widely used in AI development."
]

# 🔹 Load models
embeddings = OllamaEmbeddings(model="phi")
vectorstore = FAISS.from_texts(documents, embeddings)
llm = Ollama(model="phi")

print("📚 Document-based Q&A System Ready! Type 'exit' to quit.\n")

while True:
    query = input("🔍 Ask your question: ")

    if query.lower() == "exit":
        print("👋 Exiting...")
        break

    # 🔹 Step 1: FAISS retrieval
    docs = vectorstore.similarity_search(query, k=3)

    # 🔹 Step 2: KEYWORD FILTER (🔥 MAIN FIX)
    keyword_docs = [d for d in documents if query.lower().split()[0] in d.lower()]

    # 🔹 Step 3: Merge + remove duplicates
    final_docs = list(dict.fromkeys(keyword_docs + [d.page_content for d in docs]))[:3]

    print("\n📄 Relevant Documents:")
    for d in final_docs:
        print("-", d)

    # 🔹 Context
    context = "\n".join(final_docs)

    # 🔹 Strict prompt
    prompt = f"""
You are an AI assistant.

Rules:
- Answer ONLY from the context
- Keep answer short (1 line)
- Do NOT explain extra
- If not found, say: "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    clean = response.strip().split("\n")[0]

    print("\n🤖 Answer:", clean)
    print("-" * 50)