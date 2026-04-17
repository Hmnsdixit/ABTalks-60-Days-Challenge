from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# 🔹 Step 1: Documents (improve quality)
documents = [
    "Artificial Intelligence is transforming industries.",
    "Machine Learning is a subset of AI that learns from data.",
    "Deep Learning uses neural networks for complex tasks.",
    "FAISS is used for efficient similarity search.",
    "RAG improves LLM responses using external knowledge."
]

# 🔹 Step 2: Load embedding model
embeddings = OllamaEmbeddings(model="phi")

# 🔹 Step 3: Create vector DB with better search
vectorstore = FAISS.from_texts(documents, embeddings)

# 🔹 Step 4: Load LLM
llm = Ollama(model="phi")

print("🚀 RAG Optimization System Ready! Type 'exit' to quit.\n")

# 🔹 Step 5: Query loop
while True:
    query = input("🔍 Enter your query: ")

    if query.lower() == "exit":
        print("👋 Exiting...")
        break

    # 🔹 Step 6: Optimized retrieval (top 2 relevant docs)
    docs = vectorstore.similarity_search(query, k=2)

    # 🔹 Step 7: Combine context
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are an AI assistant.

    Use the following context to answer accurately:

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # 🔹 Step 8: Generate response
    response = llm.invoke(prompt)

    print("\n📄 Retrieved Context:")
    for d in docs:
        print("-", d.page_content)

    print("\n🤖 Answer:", response)
    print("-" * 50)