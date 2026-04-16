from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# 🔹 Step 1: Documents
documents = [
    "Artificial Intelligence is the future of technology",
    "Machine Learning is a subset of AI",
    "Deep Learning is a powerful technique",
    "Data Science involves statistics and programming",
    "Python is widely used in AI development"
]

# 🔹 Step 2: Load embedding model
embeddings = OllamaEmbeddings(model="phi")

# 🔹 Step 3: Create vector database
vectorstore = FAISS.from_texts(documents, embeddings)

# 🔹 Step 4: Load LLM
llm = Ollama(model="phi")

print("🤖 RAG Chatbot Ready! Type 'exit' to quit.\n")

# 🔹 Step 5: Query loop
while True:
    query = input("🔍 Ask something: ")

    if query.lower() == "exit":
        print("👋 Exiting...")
        break

    # 🔹 Step 6: Retrieve relevant docs
    docs = vectorstore.similarity_search(query, k=2)

    context = "\n".join([doc.page_content for doc in docs])

    # 🔹 Step 7: Create prompt
    prompt = f"""
    Use the following context to answer the question:

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # 🔹 Step 8: Generate response
    response = llm.invoke(prompt)

    print("\n📊 Relevant Context:")
    for d in docs:
        print("-", d.page_content)

    print("\n🤖 Answer:", response)
    print("-" * 50)